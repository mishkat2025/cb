"""Keypoint preprocessing (P5): interpolate missing and briefly low-visibility joints, resample to a uniform
grid from real timestamps, light smoothing, root-centring and torso scaling; fixed-length resampling for the
network. Long low-visibility runs (self-occlusion, e.g. the far arm in side views) keep MediaPipe's estimate.

The uniform-grid sequence (default 30 fps, full length) is the "original" sequence kept for DTW; fixed-T
copies are built from it by ``resample_fixed`` (see cvdc.data.pack).

Saved per video (data/cache/pre/<id>.npz):
  t (T,)            uniform time grid, s
  xy (T,33,2)       2D, root-centred and divided by the median 2D torso length (image axes: x right, y down)
  vis (T,33)        MediaPipe visibility (0 where no pose was detected)
  world (T,33,3)    3D world landmarks, root-centred and divided by the median 3D torso length
  gap_s (T,33)      time to the nearest used observation of the joint (0 = observed, > 0 = interpolated)
  root_px (T,2), torso_px, torso_m, frame_hw, root (name)

usage: python -m cvdc.data.preprocess [--config C] [--workers N] [--only ID ...]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..config import load_config, resolve
from .skeleton import J

L_SH, R_SH, L_HIP, R_HIP = J["left_shoulder"], J["right_shoulder"], J["left_hip"], J["right_hip"]


def resample_uniform(t, X, valid, fps):
    """Linear resampling of X (N,J,C) at times t (N,) onto a uniform grid, per joint, using only valid
    samples (valid: (N,J) bool). Returns (t_u, X_u, gap_s). Edges are held; joints never valid are NaN."""
    t = np.asarray(t, float)
    n_u = int(np.floor(t[-1] * fps + 1e-6)) + 1
    t_u = np.arange(n_u) / fps
    Xu = np.full((n_u,) + X.shape[1:], np.nan, np.float64)
    gap = np.full((n_u, X.shape[1]), np.inf)
    for j in range(X.shape[1]):
        ok = valid[:, j]
        if not ok.any():
            continue
        tj = t[ok]
        for c in range(X.shape[2]):
            Xu[:, j, c] = np.interp(t_u, tj, X[ok, j, c])
        k = np.clip(np.searchsorted(tj, t_u), 1, len(tj) - 1) if len(tj) > 1 else np.zeros(n_u, int)
        gap[:, j] = np.minimum(np.abs(t_u - tj[k - 1]), np.abs(t_u - tj[k])) if len(tj) > 1 else np.abs(t_u - tj[0])
    gap[gap < 0.5 / fps] = 0.0
    return t_u, Xu, gap


def smooth(X, window, polyorder):
    """Savitzky-Golay along time (axis 0). Skipped when the sequence is shorter than the window."""
    w = min(window, len(X) if len(X) % 2 else len(X) - 1)
    if w <= polyorder or np.isnan(X).all():
        return X
    out = X.copy()
    ok = ~np.isnan(X).any(axis=0)  # joints/coords observed at least once are NaN-free after resampling
    out[:, ok] = savgol_filter(X[:, ok], w, polyorder, axis=0)
    return out


def root_point(P, root):
    if root == "mid_hip":
        return (P[:, L_HIP] + P[:, R_HIP]) / 2
    if root == "mid_shoulder":
        return (P[:, L_SH] + P[:, R_SH]) / 2
    raise ValueError(root)


def torso_length(P):
    d = np.linalg.norm((P[:, L_SH] + P[:, R_SH]) / 2 - (P[:, L_HIP] + P[:, R_HIP]) / 2, axis=-1)
    return float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan


def resample_fixed(X, T):
    """Linear resampling of X (N, ...) to exactly T steps along axis 0 (endpoints kept)."""
    X = np.asarray(X)
    n = len(X)
    if n == T:
        return X.copy()
    pos = np.linspace(0, n - 1, T)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (pos - i0).reshape((T,) + (1,) * (X.ndim - 1))
    return (1 - w) * X[i0] + w * X[i1]


def short_runs(mask, t, max_s):
    """Per column of mask (N,J): True on runs of True whose time span (between the neighbouring samples
    outside the run) is <= max_s."""
    out = np.zeros_like(mask)
    n = len(t)
    for j in range(mask.shape[1]):
        m = np.r_[False, mask[:, j], False].astype(np.int8)
        starts, ends = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1) - 1
        for s, e in zip(starts, ends):
            if t[min(e + 1, n - 1)] - t[max(s - 1, 0)] <= max_s:
                out[s:e + 1, j] = True
    return out


def preprocess_arrays(lm2d, world, t, detected, frame_hw, pc) -> dict:
    """Frames without a detected pose are always interpolated. Low-visibility joints are interpolated when
    the low-visibility run is short (<= max_gap_s); longer runs (e.g. the far arm in side views) keep
    MediaPipe's estimate, flagged by the visibility channel."""
    H, W = frame_hw
    xy = lm2d[..., :2].astype(np.float64) * np.array([W, H])
    vis = np.where(detected[:, None], np.nan_to_num(lm2d[..., 3]), 0.0)
    low = detected[:, None] & (vis < pc["visibility_threshold"])
    valid = detected[:, None] & ~short_runs(low, np.asarray(t, float), pc["max_gap_s"])
    fps = pc["fps"]
    t_u, xy_u, gap = resample_uniform(t, xy, valid, fps)
    _, w3_u, _ = resample_uniform(t, world.astype(np.float64), valid, fps)
    _, vis_u, _ = resample_uniform(t, vis[..., None], np.ones_like(valid), fps)
    sm = pc["smooth"]
    if sm and sm.get("method") == "savgol":
        xy_u, w3_u = smooth(xy_u, sm["window"], sm["polyorder"]), smooth(w3_u, sm["window"], sm["polyorder"])
    root_px = root_point(xy_u, pc["root"])
    torso_px = torso_length(xy_u)
    root_3d = root_point(w3_u, pc["root"])
    torso_m = torso_length(w3_u)
    xy_n = (xy_u - root_px[:, None]) / (torso_px if pc["scale"] == "torso" else 1.0)
    w3_n = (w3_u - root_3d[:, None]) / (torso_m if pc["scale_3d"] == "torso" else 1.0)
    never = np.isnan(xy_n).all(axis=(0, 2))  # joints never observed (e.g. out of frame): root, visibility 0
    xy_n[:, never], w3_n[:, never] = 0.0, 0.0
    return dict(t=t_u, xy=xy_n.astype(np.float32), vis=vis_u[..., 0].astype(np.float32),
                world=w3_n.astype(np.float32), gap_s=np.minimum(gap, 999).astype(np.float16),
                root_px=root_px.astype(np.float32), torso_px=np.float64(torso_px), torso_m=np.float64(torso_m),
                frame_hw=np.asarray(frame_hw), root=np.asarray(pc["root"]))


def preprocess_file(kp_path, out_path, pc) -> dict:
    with np.load(kp_path) as z:
        det_rate = float(z["detected"].mean())
        if det_rate < pc["min_detection_rate"]:
            Path(out_path).unlink(missing_ok=True)  # no stale file from an earlier run
            return dict(det_rate=det_rate, excluded="low_detection")
        d = preprocess_arrays(z["lm2d"], z["world"], z["t"], z["detected"], z["frame_hw"], pc)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **d)
    gap = d["gap_s"].astype(np.float32)
    up = [11, 12, 13, 14, 15, 16, 23, 24]
    return dict(det_rate=det_rate, excluded="", n_frames_30=len(d["t"]), torso_px=d["torso_px"], torso_m=d["torso_m"],
                interp_frac_upper=float((gap[:, up] > 0).mean()),
                long_gap_frac_upper=float((gap[:, up] > pc["max_gap_s"]).mean()),
                lowvis_frac_upper=float((d["vis"][:, up] < pc["visibility_threshold"]).mean()),
                never_seen_joints=int((d["vis"].max(0) == 0).sum()))


def _job(args):
    vid, kp, out, pc = args
    try:
        return dict(video_id=vid, **preprocess_file(kp, out, pc))
    except Exception as e:  # noqa: BLE001 - reported in the summary
        return dict(video_id=vid, error=repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pc = cfg["preprocess"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    ref = pd.read_csv(resolve(cfg["data"]["ref_index_csv"])).drop_duplicates("ref_id")
    jobs = [(r.video_id, resolve(r.kp_path), resolve(r.pre_path), pc) for r in idx.itertuples()]
    jobs += [(r.ref_id, resolve(r.kp_path), resolve(r.pre_path), pc) for r in ref.itertuples()]
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only]
    missing = [j[0] for j in jobs if not Path(j[1]).exists()]
    jobs = [j for j in jobs if Path(j[1]).exists()]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(_job, jobs, chunksize=4))
    S = pd.DataFrame(rows)
    out = resolve(cfg["outputs"]["phase1"])
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "preprocess_summary.csv", index=False)
    err = S[S.get("error", pd.Series(dtype=object)).notna()] if "error" in S else S.iloc[:0]
    low = S[S.excluded == "low_detection"] if "excluded" in S else S.iloc[:0]
    print(f"preprocessed {len(S) - len(err) - len(low)} videos; excluded (detection < {pc['min_detection_rate']:.0%}): "
          f"{len(low)} {list(low.video_id)}; errors {len(err)}; missing keypoints {len(missing)}")
    for r in err.itertuples():
        print("  ERROR", r.video_id, r.error)
    ok = S.drop(columns=["error"], errors="ignore").dropna(subset=["n_frames_30"])
    print(ok[["interp_frac_upper", "long_gap_frac_upper", "lowvis_frac_upper", "never_seen_joints"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
