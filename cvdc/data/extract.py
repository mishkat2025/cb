"""MediaPipe PoseLandmarker keypoint extraction (Tasks API, VIDEO mode), one .npz per video (P1); for the exercises in
``pose.hands_for`` (E06) also MediaPipe HandLandmarker on square crops around both Pose wrists (P1c).

Saved per video (keypoints only, no pixels):
  lm2d   (T,33,5) float32  x, y normalised to the upright original frame, z (MediaPipe image z), visibility, presence
  world  (T,33,3) float32  world landmarks in metres (hip-centred)
  t      (T,)     float64  presentation time in s, relative to the first frame
  t0     ()       float64  container time of the first frame (for audio/video sync)
  detected (T,)   bool
  frame_hw (2,)   upright original frame size (H, W)
  settings        json string (model, max_side, pad_square, mediapipe version)
Hands (``pose.hand_dir``/<id>.npz), side 0 = the body's left (Pose wrist 15), 1 = right (Pose wrist 16):
  hand2d (T,2,21,3) float32  x, y normalised like lm2d, z (MediaPipe relative depth, scaled like x)
  hand_world (T,2,21,3)      MediaPipe hand world landmarks (m, hand-centred)
  wrist_dist (T,2)           distance of the hand's wrist landmark to the Pose wrist, in 2D forearm lengths
  handedness (T,2) int8      MediaPipe's label of the kept hand (0 left, 1 right; -1 = no hand), handedness_score
  t (T,), sides, settings
Undetected frames are NaN. An existing file with the same settings is skipped (resumable); if only the hands are
missing, the cached Pose landmarks drive the crops (the frames are decoded again).

usage: python -m cvdc.data.extract [--config C] [--workers N] [--only ID ...] [--limit N] [--no-refs]
                                   [--max-side PX] [--pad-square 0|1] [--kp-dir DIR] [--ref-kp-dir DIR]
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import data_root, load_config, resolve, video_root

UPPER = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips
HAND_SIDES = (("left", 13, 15), ("right", 14, 16))  # body side, Pose elbow, Pose wrist (MediaPipe 33 indices)
HAND_CONF = ("min_hand_detection_confidence", "min_hand_presence_confidence", "min_tracking_confidence")


def _settings(model, max_side, pad_square, conf):
    import mediapipe as mp
    return dict(model=Path(model).name, max_side=max_side, pad_square=bool(pad_square),
                mediapipe=mp.__version__, **conf)


def _hand_settings(hand_model, max_side, hc):
    import mediapipe as mp
    return dict(model=Path(hand_model).name, max_side=max_side, mediapipe=mp.__version__, **hc)


def _cached(path, settings) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return json.loads(str(z["settings"])) == settings
    except Exception:  # noqa: BLE001 - an unreadable cache file is recomputed
        return False


def _save(out, **arrays):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, out)


def hand_crop(elbow, wrist, W, H, hc):
    """Square crop around one Pose wrist (elbow, wrist: normalised x, y). Returns (2x3 affine frame -> crop, centre,
    half side, 2D forearm length) in processed-frame pixels, or None. Centre = wrist + ahead x (wrist - elbow), side =
    crop_scale x the 2D forearm length (at least 48 px)."""
    e, w = np.asarray(elbow, float) * (W, H), np.asarray(wrist, float) * (W, H)
    if not (np.isfinite(e).all() and np.isfinite(w).all()):
        return None
    fa = float(np.linalg.norm(w - e))
    c = w + hc["ahead"] * (w - e)
    half = max(24.0, hc["crop_scale"] * fa / 2)
    k = hc["crop_px"] / (2 * half)
    M = np.array([[k, 0, -(c[0] - half) * k], [0, k, -(c[1] - half) * k]], np.float32)
    return M, c, half, fa


def pick_hand(cands, wrist_px, c, half, fa, W, H):
    """cands: [(landmarks (21,3) crop-normalised, world (21,3), handedness label, score)]. Maps each hand back to the
    processed frame and keeps the one whose wrist landmark (0) is nearest the Pose wrist. Returns (xyz normalised to
    the frame (21,3), world, distance in forearm lengths, label 0 left | 1 right, score) or None."""
    best = None
    for lm, wl, lab, sc in cands:
        a = np.asarray(lm, np.float64) * (2 * half)
        a[:, :2] += np.asarray(c) - half
        d = float(np.linalg.norm(a[0, :2] - wrist_px)) / max(fa, 1e-6)
        if best is None or d < best[2]:
            best = (np.c_[a[:, 0] / W, a[:, 1] / H, a[:, 2] / W], np.asarray(wl, np.float64), d,
                    0 if str(lab).lower().startswith("l") else 1, float(sc))
    return best


def detect_hand(lm, img, ms, elbow, wrist, hc):
    """One side's HandLandmarker (VIDEO mode) on the crop around that side's Pose wrist."""
    import cv2
    import mediapipe as mp
    H, W = img.shape[:2]
    g = hand_crop(elbow, wrist, W, H, hc)
    if g is None:
        return None
    M, c, half, fa = g
    n = hc["crop_px"]
    crop = cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)), ms)
    cands = [([[q.x, q.y, q.z] for q in h], [[q.x, q.y, q.z] for q in wl], hd[0].category_name, hd[0].score)
             for h, wl, hd in zip(r.hand_landmarks, r.hand_world_landmarks, r.handedness)]
    return pick_hand(cands, np.asarray(wrist, float) * (W, H), c, half, fa, W, H) if cands else None


def extract_one(video, out, model, max_side, pad_square, conf, hand_out=None, hand_model=None, hc=None) -> dict:
    """Pose keypoints, and with ``hand_out`` the hands on crops around both Pose wrists, in one decode pass."""
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from .video import iter_frames

    out = Path(out)
    settings = _settings(model, max_side, pad_square, conf)
    need_pose = not _cached(out, settings)
    hsettings = _hand_settings(hand_model, max_side, hc) if hand_out else None
    need_hand = bool(hand_out) and not _cached(hand_out, hsettings)
    if not need_pose and not need_hand:
        return dict(out=str(out), status="cached")
    if need_hand and pad_square:
        raise ValueError("hand crops assume pose.pad_square: false")
    if need_hand and need_pose and hc.get("full_res"):
        # hands on full-resolution crops: Pose first at max_side (unchanged), then a hands-only pass at full resolution
        r1 = extract_one(video, out, model, max_side, pad_square, conf)
        r2 = extract_one(video, out, model, max_side, pad_square, conf, hand_out, hand_model, hc)
        return dict(r2, det=r1.get("det"), sec=r1.get("sec", 0) + r2.get("sec", 0))
    decode_side = None if (need_hand and not need_pose and hc.get("full_res")) else max_side
    cached_lm = cached_t = None
    if not need_pose:
        with np.load(out) as z:
            cached_lm, cached_t = z["lm2d"], z["t"]
    t_start = time.time()
    lm2d, world, ts, det = [], [], [], []
    hands = {k: [] for k in ("hand2d", "hand_world", "wrist_dist", "handedness", "handedness_score")}
    geom, last_ms, t0 = None, -1, None
    pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO,
        num_poses=1, output_segmentation_masks=False, **conf)) if need_pose else None
    hand_lms = []
    if need_hand:  # one landmarker per side, so each keeps its own tracking state
        for _ in HAND_SIDES:
            hand_lms.append(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(hand_model)), running_mode=vision.RunningMode.VIDEO,
                num_hands=hc["num_hands"], **{k: hc[k] for k in HAND_CONF})))
    try:
        for i, (t, img, geom) in enumerate(iter_frames(video, decode_side, pad_square)):
            t0 = t if t0 is None else t0
            ms = max(int(round((t - t0) * 1000)), last_ms + 1)
            last_ms = ms
            if need_pose:
                r = pose_lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), ms)
                if r.pose_landmarks:
                    a = np.array([[p.x, p.y, p.z, p.visibility, p.presence] for p in r.pose_landmarks[0]], np.float32)
                    oy, ox, S = geom["pad"]
                    if S:  # padded square -> coordinates of the unpadded (= original, upright) frame
                        ph, pw = geom["proc_hw"]
                        a[:, 0] = (a[:, 0] * S - ox) / pw
                        a[:, 1] = (a[:, 1] * S - oy) / ph
                        a[:, 2] = a[:, 2] * S / pw
                    lm2d.append(a)
                    world.append(np.array([[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]], np.float32))
                    det.append(True)
                else:
                    lm2d.append(np.full((33, 5), np.nan, np.float32))
                    world.append(np.full((33, 3), np.nan, np.float32))
                    det.append(False)
                cur = lm2d[-1]
            else:
                if i >= len(cached_lm):
                    raise RuntimeError(f"{video}: more frames than its cached pose file ({len(cached_lm)})")
                cur = cached_lm[i]
            ts.append(t - t0)
            if need_hand:
                h = dict(hand2d=np.full((2, 21, 3), np.nan, np.float32), hand_world=np.full((2, 21, 3), np.nan, np.float32),
                         wrist_dist=np.full(2, np.nan, np.float32), handedness=np.full(2, -1, np.int8),
                         handedness_score=np.zeros(2, np.float32))
                for k, (_, je, jw) in enumerate(HAND_SIDES):
                    res = detect_hand(hand_lms[k], img, ms, cur[je, :2], cur[jw, :2], hc)
                    if res is not None:
                        for name, v in zip(("hand2d", "hand_world", "wrist_dist", "handedness", "handedness_score"), res):
                            h[name][k] = v
                for name, v in h.items():
                    hands[name].append(v)
    finally:
        for x in [pose_lm, *hand_lms]:
            if x is not None:
                x.close()
    if not ts:
        return dict(out=str(out), status="no frames")
    if not need_pose and (len(ts) != len(cached_lm) or np.abs(np.asarray(ts) - cached_t).max() > 1e-6):
        raise RuntimeError(f"{video}: decoded frames do not match its cached pose file")
    res = dict(out=str(out), status="done", frames=len(ts), sec=time.time() - t_start)
    if need_pose:
        _save(out, lm2d=np.stack(lm2d), world=np.stack(world), t=np.asarray(ts, np.float64), t0=np.float64(t0),
              detected=np.asarray(det), frame_hw=np.asarray(geom["orig_hw"]), settings=json.dumps(settings))
        res["det"] = float(np.mean(det))
    if need_hand:
        H = {k: np.stack(v) for k, v in hands.items()}
        _save(hand_out, **H, t=np.asarray(ts, np.float64), sides=np.asarray(["left", "right"]),
              settings=json.dumps(hsettings))
        res["hand_det"] = [round(float(x), 3) for x in (H["handedness"] >= 0).mean(0)]
    return res


def summarize_kp(path) -> dict:
    """Per-video extraction stats from a keypoint file."""
    with np.load(path) as z:
        t, det, lm = z["t"], z["detected"], z["lm2d"]
        hw = z["frame_hw"]
    dt = np.diff(t)
    vis = lm[:, UPPER, 3]
    return dict(n_frames=len(t), duration_s=float(t[-1] + (np.median(dt) if len(dt) else 0)),
                fps_est=float(1 / np.median(dt)) if len(dt) else np.nan,
                dt_max_ms=float(dt.max() * 1000) if len(dt) else np.nan,
                det_rate=float(det.mean()), upper_vis=float(np.nanmean(vis)) if det.any() else 0.0,
                upper_vis_ok=float(np.nanmean(vis >= 0.5)) if det.any() else 0.0,
                hips_vis_ok=float(np.mean(np.nan_to_num(lm[:, [23, 24], 3]).min(1) >= 0.5)),
                frame_h=int(hw[0]), frame_w=int(hw[1]))


def summarize_hands(path) -> dict:
    """Per-video hand stats: share of frames with a hand per body side, both sides, and the wrist distance."""
    with np.load(path) as z:
        lab, d = z["handedness"], z["wrist_dist"]
    found = lab >= 0
    out = dict(hand_frames=len(lab), hand_both=float(found.all(1).mean()))
    for k, s in enumerate(("left", "right")):
        out[f"hand_det_{s}"] = float(found[:, k].mean())
        out[f"wrist_dist_{s}_p50"] = float(np.nanmedian(d[:, k])) if found[:, k].any() else np.nan
        out[f"wrist_dist_{s}_p90"] = float(np.nanpercentile(d[:, k], 90)) if found[:, k].any() else np.nan
        # MediaPipe labels handedness for a mirrored (selfie) image, so a body-left hand is labelled "right"
        out[f"label_mirrored_{s}"] = float((lab[found[:, k], k] == 1 - k).mean()) if found[:, k].any() else np.nan
    return out


def run_jobs(jobs, model, max_side, pad_square, conf, workers, hand_model=None, hc=None):
    """jobs: list of (name, video_path, out_path[, hand_out_path or None]). Largest files first for load balance."""
    jobs = sorted(jobs, key=lambda j: -Path(j[1]).stat().st_size)
    t0, n_done = time.time(), 0
    with ProcessPoolExecutor(workers) as pool:
        futs = {pool.submit(extract_one, j[1], j[2], model, max_side, pad_square, conf,
                            j[3] if len(j) > 3 else None, hand_model, hc): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["status"] == "done":
                n_done += 1
                det = f"det {r['det']:6.1%}" if "det" in r else "pose cached"
                hd = f"  hands L/R {r['hand_det']}" if "hand_det" in r else ""
                print(f"[{i}/{len(jobs)}] {futs[f]:28s} {r['frames']:5d} fr  {det}{hd}  {r['sec']:6.1f}s", flush=True)
            elif r["status"] != "cached":
                print(f"[{i}/{len(jobs)}] {futs[f]:28s} {r['status']}", flush=True)
    print(f"extracted {n_done} new, {len(jobs) - n_done} cached, {time.time() - t0:.0f}s total", flush=True)


def detection_table(summ: pd.DataFrame) -> pd.DataFrame:
    """Per exercise x group x view x variation detection rates (spec: log per-variation detection)."""
    g = summ.groupby(["exercise", "group", "view", "variation"])
    return g.agg(videos=("video_id", "size"), det_rate_mean=("det_rate", "mean"), det_rate_min=("det_rate", "min"),
                 videos_det_below_90=("det_rate", lambda s: int((s < 0.9).sum())),
                 upper_vis_ok=("upper_vis_ok", "mean"), hips_vis_ok=("hips_vis_ok", "mean")).round(3).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--only", nargs="*", help="video ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-refs", action="store_true")
    ap.add_argument("--max-side", type=int)
    ap.add_argument("--pad-square", type=int, choices=[0, 1])
    ap.add_argument("--kp-dir")
    ap.add_argument("--ref-kp-dir")
    ap.add_argument("--hand-dir", help="override pose.hand_dir (e.g. a test of other hand settings)")
    ap.add_argument("--hands-full-res", action="store_true", help="hand crops from full-resolution frames")
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary tables")
    a = ap.parse_args()
    cfg = load_config(a.config)
    p = cfg["pose"]
    max_side = a.max_side if a.max_side is not None else p["max_side"]
    pad = bool(a.pad_square) if a.pad_square is not None else bool(p["pad_square"])
    conf = {k: p[k] for k in ("min_pose_detection_confidence", "min_pose_presence_confidence", "min_tracking_confidence")}
    root, vroot = data_root(cfg), video_root(cfg)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    ref = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    hands_for = set(p.get("hands_for") or [])
    need_hands = sorted(set(idx.exercise) & hands_for)
    hand_model = resolve(p["hand_model"]) if need_hands else None
    if need_hands and (not hand_model.exists() or "hands" not in p or "hand_dir" not in p):
        raise SystemExit(f"{need_hands} need MediaPipe Hands: {p['hand_model']} and pose.hands / pose.hand_dir "
                         "(configs/data_full.yaml). Download the model or remove them from data.include_exercises.")
    kp_dir = resolve(a.kp_dir or p["kp_dir"])
    ref_dir = resolve(a.ref_kp_dir or p["ref_kp_dir"])
    hand_dir = resolve(a.hand_dir or p["hand_dir"]) if need_hands else None
    if a.hands_full_res:
        p["hands"] = {**p["hands"], "full_res": True}
    ref_hand_dir = resolve(p.get("ref_hand_dir") or p["hand_dir"]) if need_hands else None
    if a.only:
        idx = idx[idx.video_id.isin(a.only)]
    if a.limit:
        idx = idx.head(a.limit)
    jobs = [(r.video_id, str(vroot / r.rel_path), str(kp_dir / f"{r.video_id}.npz"),
             str(hand_dir / f"{r.video_id}.npz") if r.exercise in hands_for else None) for r in idx.itertuples()]
    if not a.no_refs and not a.only:
        ref = ref[ref.exists]
        ref_hands = set(ref.ref_id[ref.exercise.isin(hands_for)]) if need_hands else set()
        ref = ref.drop_duplicates("ref_id")
        jobs += [(r.ref_id, str(root / r.rel_path), str(ref_dir / f"{r.ref_id}.npz"),
                  str(ref_hand_dir / f"{r.ref_id}.npz") if r.ref_id in ref_hands else None) for r in ref.itertuples()]
    if not a.summary:
        print(f"{len(jobs)} videos -> {kp_dir} (max_side={max_side}, pad_square={pad}, workers={a.workers or p['workers']}; "
              f"hands for {need_hands or 'none'}: {sum(j[3] is not None for j in jobs)} videos)")
        run_jobs(jobs, resolve(p["model"]), max_side, pad, conf, a.workers or p["workers"],
                 hand_model, p.get("hands"))

    rows = []
    for r in idx.itertuples():
        f = kp_dir / f"{r.video_id}.npz"
        if f.exists():
            row = dict(video_id=r.video_id, exercise=r.exercise, group=r.group, view=r.view,
                       variation=r.variation, **summarize_kp(f))
            hf = hand_dir / f"{r.video_id}.npz" if hand_dir is not None else None
            if hf is not None and r.exercise in hands_for and hf.exists():
                row.update(summarize_hands(hf))
            rows.append(row)
    if not rows:
        return
    summ = pd.DataFrame(rows)
    out_dir = resolve(cfg["outputs"]["phase1"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if (a.kp_dir is None and a.hand_dir is None) else "_" + Path(a.hand_dir or kp_dir).name
    summ.to_csv(out_dir / f"extract_summary{tag}.csv", index=False)
    det = detection_table(summ)
    det.to_csv(out_dir / f"detection_by_variation{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(summ)}/{len(idx)} videos have keypoints; frames {summ.n_frames.sum()}; "
          f"mean detection {summ.det_rate.mean():.1%}; videos with detection < 90%: {(summ.det_rate < 0.9).sum()}")
    print(summ.groupby(["exercise", "variation"]).det_rate.agg(["count", "mean", "min"]).round(3).unstack(0).to_string())
    if "hand_both" in summ:
        h = summ[summ.hand_both.notna()]
        print(f"\nhands ({len(h)} videos): share of frames with a hand, by view (left / right body side / both)")
        print(h.groupby("view")[["hand_det_left", "hand_det_right", "hand_both", "wrist_dist_left_p50",
                                 "wrist_dist_right_p50"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
