"""Phase 6 (spec §10; X1-X4, T17): the assessment engine shared by the video-file / webcam demo (demo/live.py) and the
Streamlit app (demo/app.py). Everything reuses the training-time code, so a demo rep is processed exactly like a
dataset rep:

  PoseStream      MediaPipe Pose (Tasks API, VIDEO mode, the Phase 1 model and settings, P1, P2) per frame; for E06
                  also MediaPipe Hands on crops around both Pose wrists (P1c)
  OnlineSession   the buffered keypoints -> Phase 1 preprocessing (30 fps grid, interpolation, smoothing, root / torso
                  normalisation, P5) -> the exercise's rep signal and adaptive segmentation (P6, P6c) -> every rep that
                  has ended is assessed once by Module III (the camera view selects the per-view bias and bands, X4)
  ExerciseModel   the per-exercise checkpoint (T17; default Proposed, lambda 1.0): its per-time CVDC features (for the
                  completed Module III), and, as a supplementary output, the rep score (rep head), set score and items
                  (video head, T13) and the cosine distance of the rep embedding to the PT demo's reference embedding
                  (E5, mirrored for the other side); the fold is the one that held the subject out when the video is a
                  dataset video (X2), and the demo warns when the subject was in the model's training split

X11 (2026-10-22): **the assessment is Module III's score, as in the proposal (Fig. 3)**. With the Proposed model
(eval.proposed_default) loaded, it is the completed Module III (M19 + M20; configs/completion.yaml): every rep is
aligned to the PT template in the CVDC embedding space of the model's fold, the joint angles are compared along that
alignment, and the embedding distance and the angular velocity are scored and localised, with the library of that
fold (the completed run's ``<EX>_fold<k>``). Without a model, or with another network, it is the rule-based Module
III with the deployable library (M8-M18). The network's own scores (rep head, video head) are a **supplementary**
output, labelled as such: the instrument used to compare the models, not the pipeline's result.
No frame, crop or face is stored by the engine (keypoints only, D10).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cvdc.assess.angles import ASSESS_ANGLES
from cvdc.assess.cvdc_join import align_steps, encode_sequences, frame_distance, frame_path, ref_template_packs
from cvdc.assess.module3 import Assessor, assess_config, set_errors, video_report
from cvdc.assess.template import load_library
from cvdc.config import data_path, exercise_cfg, load_config, resolve

COMPLETION_CONFIG = "configs/completion.yaml"  # X11: the completed Module III's settings and fold libraries
SUPPLEMENTARY = ("supplementary: the network's learned score head, used to compare the models; not the pipeline's "
                 "output (the assessment is Module III's score)")
from cvdc.data.extract import HAND_CONF, HAND_SIDES, detect_hand
from cvdc.data.labels import MQ_ITEMS
from cvdc.data.preprocess import preprocess_arrays, resample_fixed
from cvdc.data.reps import segment_signal, video_info
from cvdc.data.skeleton import COCO17_MIRROR, JOINT_SETS, mirror_coco17
from cvdc.eval.baselines import hand_angle_2d_arrays
from cvdc.models.model import CVDCModel
from cvdc.train.run import method_config
from cvdc.train.trainer import fold_plan


# ----------------------------------------------------------------------------------------------- pose
class PoseStream:
    """MediaPipe Pose per frame (VIDEO mode) with the Phase 1 settings; ``hands`` adds MediaPipe Hands on wrist crops
    (E06, P1c). Keeps every frame's landmarks: lm2d (33, 5), world (33, 3), detected, hand2d (2, 21, 3)."""

    def __init__(self, cfg, hands=False, model=None):
        os.environ.setdefault("GLOG_minloglevel", "2")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        pc = cfg["pose"]
        conf = {k: pc[k] for k in ("min_pose_detection_confidence", "min_pose_presence_confidence",
                                   "min_tracking_confidence")}
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolve(model or pc["model"]))),
            running_mode=vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=False, **conf))
        self.hc = pc.get("hands") if hands else None
        self.hand_lms = []
        if self.hc:
            for _ in HAND_SIDES:
                self.hand_lms.append(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(resolve(pc["hand_model"]))),
                    running_mode=vision.RunningMode.VIDEO, num_hands=self.hc["num_hands"],
                    **{k: self.hc[k] for k in HAND_CONF})))
        self.reset()

    def reset(self):
        self.t, self.lm2d, self.world, self.detected, self.hand2d = [], [], [], [], []
        self.frame_hw, self._last_ms, self._t0 = None, -1, None

    def close(self):
        for x in [self.pose, *self.hand_lms]:
            x.close()

    def process(self, t, img, frame_hw=None):
        """One upright RGB frame at time t (s). Returns the (33, 5) normalised landmarks or None."""
        self._t0 = t if self._t0 is None else self._t0
        ms = max(int(round((t - self._t0) * 1000)), self._last_ms + 1)
        self._last_ms = ms
        self.frame_hw = tuple(frame_hw or img.shape[:2])
        r = self.pose.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                     data=np.ascontiguousarray(img)), ms)
        if r.pose_landmarks:
            lm = np.array([[p.x, p.y, p.z, p.visibility, p.presence] for p in r.pose_landmarks[0]], np.float32)
            w = np.array([[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]], np.float32)
        else:
            lm, w = np.full((33, 5), np.nan, np.float32), np.full((33, 3), np.nan, np.float32)
        h = np.full((2, 21, 3), np.nan, np.float32)
        for k, (_, je, jw) in enumerate(HAND_SIDES):
            if self.hand_lms and np.isfinite(lm[jw, :2]).all():
                res = detect_hand(self.hand_lms[k], img, ms, lm[je, :2], lm[jw, :2], self.hc)
                if res is not None:
                    h[k] = res[0]
        self.t.append(t - self._t0)
        self.lm2d.append(lm)
        self.world.append(w)
        self.detected.append(bool(r.pose_landmarks))
        self.hand2d.append(h)
        return None if not r.pose_landmarks else lm

    def arrays(self):
        return (np.asarray(self.t, np.float64), np.stack(self.lm2d), np.stack(self.world),
                np.asarray(self.detected), np.stack(self.hand2d))


# ----------------------------------------------------------------------------------------------- model
def dataset_subject(cfg, video_id):
    """The subject of a dataset video id (index lookup), or None for an outside video."""
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    r = idx[idx.video_id == video_id]
    return None if r.empty else str(r.subject.iloc[0])


class ExerciseModel:
    """The per-exercise network (T17) of one GroupKFold-5 fold, with its target scalers, the embedding centre of its
    training fold (E13) and the PT demo's reference embedding (E5) for both sides."""

    def __init__(self, cfg, exercise, method="proposed_j1", fold=None, subject=None, device="cpu", tag="perex_gkf5"):
        self.cfg, self.ex, self.method, self.dev = cfg, exercise, method, device
        mc = method_config(cfg, method)
        if mc["model"]["input"] != "xy":
            raise ValueError(f"the demo runs 2D-input models (b1, proposed_j1, proposed_lp), not {method!r}")
        self.mc = mc
        reps = pd.read_csv(data_path(cfg, "reps_csv"))
        plans = fold_plan(cfg, reps[reps.exercise == exercise])
        self.warning = None
        if subject is not None and fold is None:
            fold = next((p["fold"] for p in plans if subject in p["test"]), None)
            if fold is None:
                self.warning = f"{subject} has no fold in {exercise}; using fold 0"
        self.fold = 0 if fold is None else int(fold)
        self.plan = next(p for p in plans if p["fold"] == self.fold)
        if subject is not None and subject in self.plan["train"]:
            self.warning = (f"LEAKAGE: subject {subject} was in the training split of {exercise} {method} fold "
                            f"{self.fold}; use a held-out subject or the fold that held it out")
        run = resolve(cfg["train"]["runs_dir"]) / tag / exercise / method / f"fold{self.fold}"
        self.checkpoint = run / "finetune" / "best.pt"
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"no checkpoint {self.checkpoint} (train {exercise} first)")
        ck = torch.load(self.checkpoint, map_location=device, weights_only=True)
        from cvdc.losses.supervised import TargetScaler
        self.model = CVDCModel(mc).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.scaler = TargetScaler().load_state_dict(ck["scaler"], device)
        self.vscaler = TargetScaler().load_state_dict(ck["video_scaler"], device) if ck.get("video_scaler") else None
        self.items, self.video_items = ck["items"], ck.get("video_items") or []
        with np.load(run / "embeddings_final.npz") as z:
            self.centre = z["z_train_mean"].astype(np.float64)
        self.joints = JOINT_SETS[cfg["preprocess"]["joint_set"]]
        self.T = cfg["preprocess"]["T_rep"]
        self.ex_id = torch.tensor([self.model.exercises.index(exercise)], device=device)
        self.ref = self._reference()

    def _pack(self, pre, start, end, mirror=False):
        xy = resample_fixed(pre["xy"][start:end + 1][:, self.joints], self.T)
        if mirror:
            xy = mirror_coco17(xy)
        return torch.from_numpy(np.asarray(xy, np.float32))[None].to(self.dev)

    @torch.no_grad()
    def _z(self, x):
        return self.model(x.permute(0, 3, 1, 2).contiguous(), exercise=self.ex_id.expand(len(x)), project=False,
                          heads=False)["z"].float()

    def _reference(self):
        """Mean embedding of the primary PT demo's reps, as recorded (its side) and mirrored (the other side)."""
        cfg = self.cfg
        ref_idx = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
        rr = pd.read_csv(data_path(cfg, "reps_ref_csv"))
        row = ref_idx[(ref_idx.exercise == self.ex) & (ref_idx.role == "primary")].iloc[0]
        with np.load(resolve(row.pre_path)) as z:
            pre = {"xy": z["xy"]}
        reps = rr[(rr.ref_id == row.ref_id) & (rr.exercise == self.ex)]
        side = reps.side.iloc[0]
        out = {}
        for s, mir in ((side, False), ("left" if side == "right" else "right", True)):
            zs = [self._z(self._pack(pre, r.start, r.end, mir)) for r in reps.itertuples()]
            out[s] = torch.cat(zs).mean(0).cpu().numpy().astype(np.float64)
        return out

    def sequence(self, pre, start, end):
        """(T', D) per-time CVDC features of one rep (STGCN.sequence), centred with the fold's training mean (M19)."""
        x = self._pack(pre, start, end).permute(0, 3, 1, 2).contiguous()
        return encode_sequences(self.model, x, self.mc, self.dev)[0] - self.centre

    def ref_sequences(self, ref_meta):
        """{patient side: (T', D)} the PT template reps of Module III (``ref_meta['spans']``) encoded and averaged,
        mirrored for the other side, centred (exactly as scripts/full/completion/encode.py)."""
        from cvdc.assess.cvdc_join import ref_sequences
        packs = ref_template_packs(self.cfg, ref_meta, self.T)
        return {k: v - self.centre for k, v in ref_sequences(self.model, packs, self.mc, self.dev).items()}

    @torch.no_grad()
    def rep(self, pre, start, end, side):
        """Rep head outputs and the distance to the PT reference of one rep (frames start..end of the 30 fps grid)."""
        x = self._pack(pre, start, end)
        o = self.model(x.permute(0, 3, 1, 2).contiguous(), exercise=self.ex_id, project=False, heads=True)
        inv = self.scaler.inverse(o)
        z = o["z"].float()[0].cpu().numpy().astype(np.float64)
        a, b = z - self.centre, self.ref[side] - self.centre
        d = 1.0 - float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        items = {k: float(inv["items"][0, j]) for j, k in enumerate(self.items)} if "items" in inv else {}
        return {"score": float(inv["score"][0]), "items": items, "z": z,
                "embedding_distance": {"model": self.method, "fold": self.fold, "cosine_centred": round(d, 4)}}

    @torch.no_grad()
    def video(self, zs, n_reps_set):
        """Video head (T13) on the reps so far: set score, items and the movement-quality score (D13)."""
        if self.model.video_head is None or not zs:
            return None
        z = torch.tensor(np.stack(zs), dtype=torch.float32, device=self.dev)
        seg = torch.zeros(len(z), dtype=torch.long, device=self.dev)
        count = torch.tensor([float(n_reps_set)], device=self.dev)
        vo = self.model.video(z, seg, 1, count, self.ex_id)
        inv = self.vscaler.inverse(vo)
        items = {k: float(inv["items"][0, j]) for j, k in enumerate(self.video_items)}
        mq = 100.0 * np.mean([items[k] for k in MQ_ITEMS if k in items]) if items else None
        return {"set_score": float(inv["score"][0]), "items": items, "movement_quality": mq}

    def info(self):
        return {"exercise": self.ex, "method": self.method, "fold": self.fold, "checkpoint": str(self.checkpoint),
                "test_subjects": self.plan["test"], "warning": self.warning}


# ----------------------------------------------------------------------------------------------- session
class OnlineSession:
    """Online assessment of one set: feed frames (PoseStream) and call ``update`` regularly. Every update re-runs the
    batch pipeline on everything so far (preprocessing, the moving side from the whole buffer, segmentation), so the
    side and the rep boundaries are those the dataset pipeline would give for the video up to now; reps that end at
    least ``margin_s`` before the newest frame are assessed. Unchanged reps (same boundaries and side) come from a
    cache; ``final=True`` recomputes every rep, so the final report equals the batch result for the whole video."""

    def __init__(self, cfg, exercise, view="F", model=None, library_dir=None, margin_s=0.6, pipeline="auto"):
        """``pipeline``: 'completed' (Module III with the CVDC model, X11), 'rules' (Module III without a network) or
        'auto' = completed when the Proposed model is loaded and its fold library exists, otherwise rules."""
        self.cfg, self.ex, self.view, self.model, self.margin_s = cfg, exercise, view, model, margin_s
        self.ex_cfg = exercise_cfg(cfg, exercise)
        self.fps = cfg["preprocess"]["fps"]
        self.mode, self.R = "rules", None
        if pipeline != "rules" and model is not None and model.method == cfg["eval"]["proposed_default"]:
            cc = load_config(resolve(COMPLETION_CONFIG))
            lp = resolve(cc["templates"]["library_dir"]) / f"{exercise}_fold{model.fold}"
            if lp.with_suffix(".npz").exists():
                self.mode = "completed"
                self.ac = assess_config(cc, exercise)
                ref, lib, meta = load_library(lp)
                self.R = model.ref_sequences(ref)
            elif pipeline == "completed":
                raise FileNotFoundError(f"{lp}.npz: run python -m cvdc.assess.run --config {COMPLETION_CONFIG}")
        elif pipeline == "completed":
            raise ValueError(f"the completed Module III needs the {cfg['eval']['proposed_default']} model")
        if self.mode == "rules":
            self.ac = assess_config(cfg, exercise)
            ref, lib, meta = load_library(resolve(library_dir or cfg["templates"]["library_dir"]) / exercise)
        self.assessor = Assessor(ref, lib, self.ac)
        self.ref, self.lib, self.lib_meta = ref, lib, meta
        self.pipeline_label = (
            f"Module III with the CVDC model ({model.method}, fold {model.fold}): alignment in the embedding space, "
            "joint angles, angular velocity and embedding distance (M19, M20)" if self.mode == "completed" else
            "Module III, rule-based (joint angles against the PT template; no network in the score)")
        self.reset()

    def reset(self):
        self.reps, self.entries, self.debug, self.zs = [], [], [], []
        self.side, self.pre, self.signal, self.frame_hw, self.polarity = None, None, None, None, "peak"
        self._cache, self._seen = {}, set()

    def _extra(self, pose_arrays, n, side):
        """E06: the Hands wrist angle of the hand the side camera sees (P1c); other exercises: (None, side)."""
        if self.ac.get("angle_source") != "hands":
            return None, side
        t, lm, _, _, h2 = pose_arrays
        if self.view in (self.ac.get("angle_views") or []):
            hs = self.ac["near_hand"][self.view]
            return {self.ac["angle"]: hand_angle_2d_arrays(t, lm, self.frame_hw, h2[:, 0 if hs == "left" else 1],
                                                           hs, n)}, hs
        return {self.ac["angle"]: np.full(n, np.nan)}, side

    def _joined(self, pre, r, side):
        """M19 / X11: a transform that attaches the alignment found in the CVDC embedding space and the embedding
        distance channel to the rep's signals (as cvdc.assess.run does for the batch evaluation)."""
        S = self.model.sequence(pre, r["start"], r["end"])
        sp, d = align_steps(S, self.R[side])
        n_steps, T = S.shape[0], self.model.T

        def transform(sig):
            return dict(sig, ch={**sig["ch"], "emb_dist": frame_distance(sp, d, sig["n"], n_steps, T)},
                        frame_path=frame_path(sp, sig["n"], self.ref["n"], n_steps, T))
        return transform

    def _assess(self, pre, arrs, n, r, side, index):
        extra, aside = self._extra(arrs, n, side)
        mo = self.model.rep(pre, r["start"], r["end"], side) if self.model else None
        tr = self._joined(pre, r, side) if self.mode == "completed" else None
        entry, dbg = self.assessor.rep(pre["world"], r["start"], r["end"], aside, self.view, index,
                                       mo["embedding_distance"] if mo else None, extra, transform=tr)
        if mo:  # supplementary (X11): the network's own outputs, not the assessment
            entry["model_score"] = round(mo["score"], 1)
            entry["model_items"] = {k: round(v, 3) for k, v in mo["items"].items()}
        entry["time_s"] = [round(r["start"] / self.fps, 2), round(r["end"] / self.fps, 2)]
        return entry, dbg, (mo["z"] if mo else None)

    def update(self, pose, final=False):
        """Re-run the pipeline on everything so far; returns the reps reported for the first time."""
        t, lm, world, det, h2 = arrs = pose.arrays()
        if det.sum() < 2 * self.fps or t[-1] < 2.0:
            return []
        self.frame_hw = pose.frame_hw
        pre = preprocess_arrays(lm, world, t, det, pose.frame_hw, self.cfg["preprocess"])
        ang, info = video_info(pre, self.ex_cfg)
        side = info["side"]
        reps, self.polarity = segment_signal(ang[side], self.fps, self.ex_cfg["reps"])
        n = len(pre["t"])
        horizon = n - 1 if final else n - 1 - int(round(self.margin_s * self.fps))
        done = [r for r in reps if r["end"] <= horizon]
        entries, debug, zs, new = [], [], [], []
        for i, r in enumerate(done):
            key = (r["start"], r["end"], side)
            if final or key not in self._cache:
                self._cache[key] = self._assess(pre, arrs, n, r, side, i)
            e, dbg, z = self._cache[key]
            e = {**e, "rep": i}
            entries.append(e)
            debug.append(dbg)
            if z is not None:
                zs.append(z)
            if key not in self._seen:
                self._seen.add(key)
                new.append({**e, "updated": i < len(self.entries)})  # an earlier rep whose boundaries moved
        self.pre, self.signal, self.side = pre, ang[side], side
        self.reps, self.entries, self.debug, self.zs = done, entries, debug, zs
        return new

    def set_report(self, source=""):
        """The per-set JSON (spec §8 / M13 shape) + the model's set-level outputs when a model is loaded."""
        exp = (self.ex_cfg.get("eaaq") or {}).get("expected_reps")
        meta = {"exercise": f"{self.ex}_{self.ex_cfg['name']}", "video_id": source, "view": self.view,
                "angle_source": (self.ac["angle_source"] if self.ac["angle_source"] != "hands"
                                 else "2d_mediapipe_hands_side_views + 3d_mediapipe_world"),
                "template": {"source": self.cfg["templates"]["source"], "file": self.lib_meta.get("file"),
                             "library": self.lib_meta.get("fitted_on"),
                             **({"alignment": self.lib_meta.get("alignment")} if self.mode == "completed" else {})},
                "n_reps_set": len(self.entries), "expected_reps": exp, "mode": "demo", "pipeline": self.pipeline_label}
        rep = video_report(meta, self.entries, set_errors(len(self.entries), exp), self.ac)
        rep["set_score_source"] = "module3_with_cvdc" if self.mode == "completed" else "module3_rules"  # X11
        if self.model:
            rep["model"] = {"role": SUPPLEMENTARY, **self.model.info(),
                            **(self.model.video(self.zs, len(self.entries)) or {})}
        return rep

    def live_angle(self, world_frame, side=None):
        """The main angle of one raw frame (for the live plot; E06's Hands angle is drawn from the reps only)."""
        main = self.ac["angle"]
        if main not in ASSESS_ANGLES or world_frame is None or not np.isfinite(world_frame).all():
            return np.nan
        return float(ASSESS_ANGLES[main](np.asarray(world_frame, float)[None], side or self.side or "right")[0])


def save_report(rep, out_dir, name):
    """JSON + a per-rep CSV of a session report (keypoint-derived numbers only)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    (out / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    rows = [dict(rep=r["rep"], start_s=r.get("time_s", [None, None])[0], end_s=r.get("time_s", [None, None])[1],
                 score=r["score"], quality=r["quality"], model_score=r.get("model_score"),
                 distance=(r.get("embedding_distance") or {}).get("cosine_centred"),
                 errors="; ".join(f"{e['type']}@{e.get('joint', '')}" for e in r["errors"])) for r in rep["reps"]]
    pd.DataFrame(rows).to_csv(out / f"{stem}.csv", index=False)
    return out / f"{stem}.json"
