"""MediaPipe PoseLandmarker keypoint extraction (Tasks API, VIDEO mode), one .npz per video (P1).

Saved per video (keypoints only, no pixels):
  lm2d   (T,33,5) float32  x, y normalised to the upright original frame, z (MediaPipe image z), visibility, presence
  world  (T,33,3) float32  world landmarks in metres (hip-centred)
  t      (T,)     float64  presentation time in s, relative to the first frame
  t0     ()       float64  container time of the first frame (for audio/video sync)
  detected (T,)   bool
  frame_hw (2,)   upright original frame size (H, W)
  settings        json string (model, max_side, pad_square, mediapipe version)
Undetected frames are NaN. An existing file with the same settings is skipped (resumable).

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

from ..config import data_root, load_config, resolve

UPPER = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips


def _settings(model, max_side, pad_square, conf):
    import mediapipe as mp
    return dict(model=Path(model).name, max_side=max_side, pad_square=bool(pad_square),
                mediapipe=mp.__version__, **conf)


def extract_one(video, out, model, max_side, pad_square, conf) -> dict:
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from .video import iter_frames

    out = Path(out)
    settings = _settings(model, max_side, pad_square, conf)
    if out.exists():
        try:
            with np.load(out) as z:
                if json.loads(str(z["settings"])) == settings:
                    return dict(out=str(out), status="cached")
        except Exception:  # noqa: BLE001 - unreadable cache file is recomputed
            pass
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO,
        num_poses=1, output_segmentation_masks=False, **conf)
    t_start = time.time()
    lm2d, world, ts, det = [], [], [], []
    geom, last_ms, t0 = None, -1, None
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for t, img, geom in iter_frames(video, max_side, pad_square):
            t0 = t if t0 is None else t0
            ms = max(int(round((t - t0) * 1000)), last_ms + 1)
            last_ms = ms
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), ms)
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
            ts.append(t - t0)
    if not ts:
        return dict(out=str(out), status="no frames")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, lm2d=np.stack(lm2d), world=np.stack(world), t=np.asarray(ts, np.float64),
                            t0=np.float64(t0), detected=np.asarray(det), frame_hw=np.asarray(geom["orig_hw"]),
                            settings=json.dumps(settings))
    os.replace(tmp, out)
    return dict(out=str(out), status="done", frames=len(ts), det=float(np.mean(det)), sec=time.time() - t_start)


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


def run_jobs(jobs, model, max_side, pad_square, conf, workers):
    """jobs: list of (name, video_path, out_path). Largest files first for load balance."""
    jobs = sorted(jobs, key=lambda j: -Path(j[1]).stat().st_size)
    t0, n_done = time.time(), 0
    with ProcessPoolExecutor(workers) as pool:
        futs = {pool.submit(extract_one, v, o, model, max_side, pad_square, conf): n for n, v, o in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["status"] == "done":
                n_done += 1
                print(f"[{i}/{len(jobs)}] {futs[f]:28s} {r['frames']:5d} fr  det {r['det']:6.1%}  {r['sec']:6.1f}s",
                      flush=True)
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
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary tables")
    a = ap.parse_args()
    cfg = load_config(a.config)
    p = cfg["pose"]
    max_side = a.max_side if a.max_side is not None else p["max_side"]
    pad = bool(a.pad_square) if a.pad_square is not None else bool(p["pad_square"])
    conf = {k: p[k] for k in ("min_pose_detection_confidence", "min_pose_presence_confidence", "min_tracking_confidence")}
    root = data_root(cfg)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    ref = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    need_hands = sorted(set(idx.exercise) & set(p.get("hands_for") or []))
    if need_hands:
        raise NotImplementedError(f"{need_hands} need MediaPipe Hands ({p['hand_model']}), which is not approved/"
                                  "downloaded yet (PROGRESS.md §4 item 7). Remove them from data.include_exercises.")
    kp_dir = resolve(a.kp_dir or p["kp_dir"])
    ref_dir = resolve(a.ref_kp_dir or p["ref_kp_dir"])
    if a.only:
        idx = idx[idx.video_id.isin(a.only)]
    if a.limit:
        idx = idx.head(a.limit)
    jobs = [(r.video_id, str(root / r.rel_path), str(kp_dir / f"{r.video_id}.npz")) for r in idx.itertuples()]
    if not a.no_refs and not a.only:
        ref = ref[ref.exists].drop_duplicates("ref_id")
        jobs += [(r.ref_id, str(root / r.rel_path), str(ref_dir / f"{r.ref_id}.npz")) for r in ref.itertuples()]
    if not a.summary:
        print(f"{len(jobs)} videos -> {kp_dir} (max_side={max_side}, pad_square={pad}, workers={a.workers or p['workers']})")
        run_jobs(jobs, resolve(p["model"]), max_side, pad, conf, a.workers or p["workers"])

    rows = []
    for r in idx.itertuples():
        f = kp_dir / f"{r.video_id}.npz"
        if f.exists():
            rows.append(dict(video_id=r.video_id, exercise=r.exercise, group=r.group, view=r.view,
                             variation=r.variation, **summarize_kp(f)))
    if not rows:
        return
    summ = pd.DataFrame(rows)
    out_dir = resolve(cfg["outputs"]["phase1"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if (a.kp_dir is None) else "_" + Path(kp_dir).name
    summ.to_csv(out_dir / f"extract_summary{tag}.csv", index=False)
    det = detection_table(summ)
    det.to_csv(out_dir / f"detection_by_variation{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(summ)}/{len(idx)} videos have keypoints; frames {summ.n_frames.sum()}; "
          f"mean detection {summ.det_rate.mean():.1%}; videos with detection < 90%: {(summ.det_rate < 0.9).sum()}")
    print(summ.groupby(["exercise", "variation"]).det_rate.agg(["count", "mean", "min"]).round(3).unstack(0).to_string())


if __name__ == "__main__":
    main()
