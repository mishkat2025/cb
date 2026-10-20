"""Network inputs and labels as tensors: the fixed-length packs (cvdc/data/pack.py) joined with the video labels
of data/index.csv. Rep level (default, T1): every rep inherits its video's score, items and quality class as a
weak label. Video level: whole videos resampled to T_video.

Input modes (``model.input``, A1): ``xy`` (2D keypoints), ``xyv`` (2D + visibility), ``xyz`` (3D MediaPipe world
landmarks, B2) or ``xyz_ktpformer`` (3D lifted by KTPFormer from the 2D keypoints, B2; A10). The pack is small
(7,095 reps x 64 frames x 17 joints), so it is held on the device.

Videos (step a2, T13): at rep level every rep also knows its video (``video_of``), and every video its reps
(``v_reps``), its training rep count (``v_nreps``), its set rep count (Phase 1 ``n_reps_set``, ``v_count``), its
labels and its stratum (exercise x expert / non-expert), for the set-level head and the per-video loss weights (T6).
"""
import numpy as np
import pandas as pd
import torch

from ..config import data_path, exercise_codes, resolve

ITEMS = ["PF1", "PF2", "PF4", "CF1", "CF2", "CF3"]  # D5: PF3 is not a target


LIFTED = {"xyz_ktpformer": "ktpformer"}


def exercise_label_ok(cfg, meta):
    """(n,) bool: False for rows of (exercise, subject) pairs in ``data.exercise_label_exclude`` (D8: the movement
    shows a wrong exercise, e.g. E03 P02), which the exercise head neither learns nor is scored on."""
    excl = {(e, s) for e, subs in (cfg["data"].get("exercise_label_exclude") or {}).items() for s in subs}
    return np.array([(e, s) not in excl for e, s in zip(meta.exercise, meta.subject)], bool)


def to_input(xy, vis, world, mode, lifted=None):
    """(N, T, V, *) arrays/tensors -> (N, T, V, C) for the input mode (``lifted``: the lifter's 3D)."""
    if mode == "xy":
        return xy
    if mode == "xyv":
        return torch.cat([xy, vis[..., None]], -1)
    if mode == "xyz":
        return world
    if mode in LIFTED:
        if lifted is None:
            raise ValueError(f"model.input {mode!r} needs the lifted pack: python -m cvdc.data.pack --lift {LIFTED[mode]}")
        return lifted
    raise ValueError(f"model.input {mode!r}")


class PackData:
    def __init__(self, cfg, unit="rep", device="cpu", lift=None):
        """lift: also load the lifted 3D pack of this lifter (e.g. 'ktpformer'); None = only if
        ``model.input`` needs it."""
        pc = cfg["preprocess"]
        T = pc["T_rep"] if unit == "rep" else pc["T_video"]
        name = f"{'reps' if unit == 'rep' else 'videos'}_{pc['joint_set']}_T{T}"
        pdir = resolve(pc["pack_dir"])
        meta = pd.read_csv(pdir / f"{name}.csv")
        if unit == "rep":
            reps = pd.read_csv(data_path(cfg, "reps_csv"))
            if not (reps.video_id.values == meta.video_id.values).all():
                raise RuntimeError("data/reps.csv and the rep pack are out of sync")
            meta = reps
        idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
        lab = ["exercise", "subject", "group", "view", "variation", "gender", "group_id", "score", "quality", *ITEMS]
        meta = meta.drop(columns=[c for c in lab if c in meta.columns and c != "video_id"]).merge(
            idx[["video_id", *lab]], on="video_id", how="left", validate="many_to_one")
        self.unit, self.T, self.device, self.name = unit, T, device, name
        self.meta = meta.reset_index(drop=True)
        with np.load(pdir / f"{name}.npz") as z:
            self.xy, self.vis, self.world = (torch.from_numpy(z[k]).to(device) for k in ("xy", "vis", "world"))
        lift = lift or LIFTED.get(cfg["model"]["input"])
        self.lifted = None
        if lift:
            with np.load(pdir / f"{name}_{lift}.npz") as z:
                self.lifted = torch.from_numpy(z["world"]).to(device)
            if self.lifted.shape != self.world.shape:
                raise RuntimeError(f"lifted pack {name}_{lift} does not match the main pack")
        codes = exercise_codes(cfg)
        qn = cfg["labels"]["quality_names"]
        m = self.meta
        self.exercise = torch.tensor(m.exercise.map(codes.index).to_numpy(), device=device)
        # D8: subjects whose movement shows a wrong exercise keep their exercise as input (exercise embedding) but
        # get no exercise-head target (-1 = masked in the loss)
        self.exercise_target = torch.where(torch.tensor(exercise_label_ok(cfg, m), device=device), self.exercise,
                                           torch.full_like(self.exercise, -1))
        self.quality = torch.tensor(m.quality.map(lambda q: qn.index(q) if q in qn else -1).to_numpy(), device=device)
        self.score = torch.tensor(m.score.to_numpy(np.float32), device=device)
        self.items = {k: torch.tensor(m[k].to_numpy(np.float32), device=device) for k in ITEMS}
        self.group_code = torch.tensor(pd.factorize(m.group_id)[0], device=device)  # (exercise, subject, variation)
        # strata (exercise x group) and videos (T13, T14)
        self.strata = sorted({(e, g) for e, g in zip(m.exercise, m.group)})
        spos = {k: i for i, k in enumerate(self.strata)}
        self.stratum = torch.tensor([spos[k] for k in zip(m.exercise, m.group)], device=device)
        vcode, self.video_ids = pd.factorize(m.video_id)
        self.video_of = torch.tensor(vcode, device=device)
        self.vmeta = vm = m.drop_duplicates("video_id").reset_index(drop=True)  # in code order (first appearance)
        order = np.argsort(vcode, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(vcode, minlength=len(vm)))]
        self.v_reps = [torch.tensor(order[a:b], device=device) for a, b in zip(bounds[:-1], bounds[1:])]
        self.v_nreps = torch.tensor([len(r) for r in self.v_reps], device=device)
        self.v_score = torch.tensor(vm.score.to_numpy(np.float32), device=device)
        self.v_items = {k: torch.tensor(vm[k].to_numpy(np.float32), device=device) for k in ITEMS}
        self.v_exercise = torch.tensor(vm.exercise.map(codes.index).to_numpy(), device=device)
        self.v_stratum = torch.tensor([spos[k] for k in zip(vm.exercise, vm.group)], device=device)
        self.v_count = None
        cpath = ((cfg.get("heads") or {}).get("video_head") or {}).get("count_csv")
        if unit == "rep" and cpath and resolve(cpath).exists():
            cnt = pd.read_csv(resolve(cpath)).set_index("video_id").n_reps_set
            vals = cnt.reindex(vm.video_id)
            if vals.isna().any():
                raise RuntimeError(f"{cpath}: no n_reps_set for {list(vm.video_id[vals.isna().to_numpy()])[:5]}")
            self.v_count = torch.tensor(vals.to_numpy(np.float32), device=device)

    def __len__(self):
        return len(self.meta)

    def inputs(self, idx, mode):
        return to_input(self.xy[idx], self.vis[idx], self.world[idx], mode,
                        None if self.lifted is None else self.lifted[idx])

    def targets(self, idx, items):
        return {"score": self.score[idx], "quality": self.quality[idx], "exercise": self.exercise_target[idx],
                "items": torch.stack([self.items[k][idx] for k in items], -1) if items else None}

    def subject_mask(self, subjects, views=None):
        """Items of ``subjects``; with ``views``, only those filmed from these views (cross-view protocol, E24)."""
        m = self.meta.subject.isin(set(subjects)).to_numpy()
        if views is not None:
            m = m & self.meta.view.isin(set(views)).to_numpy()
        return torch.tensor(m, device=self.device)

    def video_targets(self, vids, items):
        """Video-level labels (score 0-100, items 0-1) for video codes ``vids``."""
        return {"score": self.v_score[vids],
                "items": torch.stack([self.v_items[k][vids] for k in items], -1) if items else None}

    def videos_of(self, items):
        """(video codes, local segment index of every item) for a set of rep items."""
        return torch.unique(self.video_of[items], return_inverse=True)

    def video_items(self, vids):
        """(rep items of the videos ``vids`` in order, segment index 0..len(vids)-1 of every item)."""
        parts = [self.v_reps[v] for v in vids.tolist()]
        seg = torch.cat([torch.full((len(r),), j, device=self.device) for j, r in enumerate(parts)])
        return torch.cat(parts), seg

    def subject_videos(self, subjects):
        return torch.tensor(np.flatnonzero(self.vmeta.subject.isin(set(subjects)).to_numpy()), device=self.device)

    def item_masks(self, items, min_subjects=5, min_sd=0.05):
        """T14: (n_strata, len(items)) bool, True where the item varies within the stratum (from the video labels;
        cvdc.eval.baselines.item_activity)."""
        from ..eval.baselines import item_activity

        a = item_activity(self.vmeta, items, min_subjects, min_sd).set_index(["exercise", "group", "item"]).varies
        return torch.tensor([[bool(a[(e, g, k)]) for k in items] for e, g in self.strata], device=self.device)
