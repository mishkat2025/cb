"""Repetition segmentation from each exercise's primary joint angle (P6).

A rep is rest -> peak -> rest of the primary angle (``polarity: peak``; ``valley`` flips the signal).
The angle is median-filtered (``median_s``); peaks need a minimum prominence and spacing; each rep starts where the angle last leaves its rest level
before the peak and ends where it first returns after the peak (rest level = within ``rest_frac`` of the
amplitude above the neighbouring minimum), padded by ``pad_s`` without crossing that minimum. Reps with an
amplitude below ``min_amplitude`` on either side (e.g. a video that ends with the arm still raised)
or a duration outside [min_rep_s, max_rep_s] are dropped.

Times are frame indices on the uniform preprocessing grid (cfg preprocess.fps).

usage: python -m cvdc.data.reps [--config C] [--workers N]
Writes data/reps.csv (one row per rep), outputs/phase1/reps_per_video.csv and summary tables.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from ..assess.angles import UNITS, primary_angle
from ..config import exercise_cfg, load_config, resolve


def fill_nan(x):
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    if not ok.any():
        return None
    x[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), x[ok])
    return x


def segment_reps(angle, fps, prominence=40, min_distance_s=1.0, min_amplitude=40, rest_frac=0.1,
                 pad_s=0.1, min_rep_s=0.8, max_rep_s=15.0, polarity="peak", median_s=0.3, **_):
    """Return a list of reps: dict(start, peak, end, ...) with frame indices into ``angle``. ``prominence`` and
    ``min_amplitude`` are in the signal's unit (degrees for angles, torso lengths for wrist_path_pc1).
    A running median (``median_s``) first removes short pose glitches (1-3 frame spikes or dips)."""
    s = fill_nan(angle)
    if s is None or len(s) < 3:
        return []
    k = int(round(median_s * fps)) // 2 * 2 + 1
    if k > 1:
        s = median_filter(s, size=k, mode="nearest")
    if polarity == "valley":
        s = -s
    peaks, _ = find_peaks(s, prominence=prominence, distance=max(1, int(round(min_distance_s * fps))))
    pad = int(round(pad_s * fps))
    reps = []
    for i, p in enumerate(peaks):
        lo = peaks[i - 1] if i > 0 else 0
        hi = peaks[i + 1] if i + 1 < len(peaks) else len(s) - 1
        vl = lo + int(np.argmin(s[lo:p + 1]))
        vr = p + int(np.argmin(s[p:hi + 1]))
        amp_l, amp_r = s[p] - s[vl], s[p] - s[vr]
        if min(amp_l, amp_r) < min_amplitude:
            continue
        rest_l, rest_r = s[vl] + rest_frac * amp_l, s[vr] + rest_frac * amp_r
        start = vl + int(np.flatnonzero(s[vl:p + 1] <= rest_l)[-1])
        end = p + int(np.flatnonzero(s[p:vr + 1] <= rest_r)[0])
        start, end = max(vl, start - pad), min(vr, end + pad)
        dur = (end - start) / fps
        if not (min_rep_s <= dur <= max_rep_s):
            continue
        top_l, top_r = s[vl] + 0.9 * amp_l, s[vr] + 0.9 * amp_r
        up90 = start + int(np.flatnonzero(s[start:p + 1] >= top_l)[0])
        down90 = p + int(np.flatnonzero(s[p:end + 1] >= top_r)[-1])
        sign = -1 if polarity == "valley" else 1
        reps.append(dict(start=start, peak=int(p), end=end, duration_s=dur,
                         peak_val=sign * s[p], rest_start_val=sign * s[vl], rest_end_val=sign * s[vr],
                         amplitude=float(min(amp_l, amp_r)),
                         raise_s=(up90 - start) / fps, hold_s=(down90 - up90) / fps, lower_s=(end - down90) / fps))
    return reps


def moving_side(angles: dict, side_cfg="auto"):
    """'auto' picks the side whose primary angle has the larger p5-p95 range."""
    rng = {s: float(np.nanpercentile(a, 95) - np.nanpercentile(a, 5)) if np.isfinite(a).any() else 0.0
           for s, a in angles.items()}
    side = max(rng, key=rng.get) if side_cfg == "auto" else side_cfg
    return side, rng


def video_angles(pre, pa_cfg):
    P = pre["world"] if pa_cfg.get("source", "world") == "world" else pre["xy"]
    return {s: primary_angle(P, pa_cfg["name"], s) for s in ("right", "left")}


def reps_for_file(pre_path, ex_cfg, fps):
    with np.load(pre_path) as z:
        pre = {k: z[k] for k in ("world", "xy", "t")}
    pa = ex_cfg["primary_angle"]
    ang = video_angles(pre, pa)
    side, rng = moving_side(ang, pa.get("side", "auto"))
    reps = segment_reps(ang[side], fps, **ex_cfg["reps"])
    other = "left" if side == "right" else "right"
    info = dict(side=side, range_moving=rng[side], range_other=rng[other], unit=UNITS[pa["name"]], n_reps=len(reps),
                angle_source=f"{'3d_mediapipe_world' if pa.get('source', 'world') == 'world' else '2d_image'}",
                primary_angle=pa["name"], n_frames_30=len(pre["t"]))
    return info, reps


def _job(args):
    key, pre_path, ex_cfg, fps = args
    try:
        info, reps = reps_for_file(pre_path, ex_cfg, fps)
        return key, info, reps, None
    except Exception as e:  # noqa: BLE001
        return key, None, [], repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cfg = load_config(a.config)
    fps = cfg["preprocess"]["fps"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    ref = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    jobs = [(("video", r.video_id), resolve(r.pre_path), exercise_cfg(cfg, r.exercise), fps) for r in idx.itertuples()]
    jobs += [(("ref", r.ref_id, r.exercise), resolve(r.pre_path), exercise_cfg(cfg, r.exercise), fps)
             for r in ref.itertuples()]
    jobs = [j for j in jobs if j[1].exists()]
    per_video, rep_rows, ref_rows = [], [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for key, info, reps, err in pool.map(_job, jobs, chunksize=4):
            if err:
                print("ERROR", key, err)
                continue
            if key[0] == "video":
                per_video.append(dict(video_id=key[1], **info))
                rep_rows += [dict(video_id=key[1], rep=i, side=info["side"], **r) for i, r in enumerate(reps)]
            else:
                ref_rows += [dict(ref_id=key[1], exercise=key[2], rep=i, side=info["side"], **r) for i, r in enumerate(reps)]
                print(f"reference {key[1]} ({key[2]} config): side {info['side']}, {info['n_reps']} reps, "
                      f"range {info['range_moving']:.2f} {info['unit']} (other arm {info['range_other']:.2f})")
    V = pd.DataFrame(per_video).merge(idx[["video_id", "exercise", "subject", "group", "view", "variation", "score",
                                           "PF3", "PF4"]], on="video_id")
    R = pd.DataFrame(rep_rows).merge(idx[["video_id", "exercise", "subject", "group", "view", "variation"]],
                                     on="video_id")
    R.to_csv(resolve("data/reps.csv"), index=False)
    pd.DataFrame(ref_rows).to_csv(resolve("data/reps_ref.csv"), index=False)
    out = resolve(cfg["outputs"]["phase1"])
    V.to_csv(out / "reps_per_video.csv", index=False)
    print(f"\n{len(R)} reps in {len(V)} videos; videos with 0 reps: {(V.n_reps == 0).sum()}")


if __name__ == "__main__":
    main()
