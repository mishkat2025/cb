"""Proposal completion (2026-10-22), quick measure 3: temporal stability (proposal D7.6, "Temporal stability: jitter
variance, frame-to-frame trajectory smoothness").

Skeleton sources, all on the 30 fps grid of Phase 1 (P5) and in torso lengths (2D: the 2D torso, 3D: the 3D torso):
  2d_raw        MediaPipe 2D landmarks as detected (linear interpolation onto the grid, no smoothing)
  2d_input      the networks' 2D input (Phase 1: gap interpolation + Savitzky-Golay 7 frames)
  3d_raw        MediaPipe world landmarks as detected (interpolated onto the grid)
  3d_mediapipe  MediaPipe world landmarks after Phase 1 preprocessing (Module III's angles come from these)
  3d_ktpformer  KTPFormer lifted 3D (B2's input)
Joints: shoulders, elbows, wrists, hips. Per video:
  accel         mean acceleration magnitude |second difference| x 30^2 (torso lengths / s^2): the usual jitter measure
  jitter_var    variance over time of the acceleration magnitude, mean over the joints ((torso / s^2)^2)
  rest_jitter   RMS of the residual from a 0.5 s Savitzky-Golay fit (order 2) on frames outside every rep (torso
                lengths): noise when the person holds still
  sparc         smoothness of the moving wrist's speed profile per rep (spectral arc length, Balasubramanian et al.
                2015; fc 10 Hz, amplitude threshold 0.05; closer to 0 = smoother), median over the video's reps
Model outputs: the within-video SD of the rep-level predicted total score (rep head, LOSO) and of Module III's rep
score (GroupKFold-5 libraries): how much the assessment of one set jumps from rep to rep.

usage: python scripts/full/completion/temporal_stability.py [--workers 12]
Writes <results>/completion/temporal/{videos.csv, by_source.csv, by_condition.csv, prediction_stability.csv,
summary.md}.
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "full" / "phase3"))
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase4"))
from cvdc.config import data_path, load_config, resolve  # noqa: E402

EXS = ["E01", "E02", "E03", "E04"]
MP = {"l_sh": 11, "r_sh": 12, "l_el": 13, "r_el": 14, "l_wr": 15, "r_wr": 16, "l_hip": 23, "r_hip": 24}
COCO = {"l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8, "l_wr": 9, "r_wr": 10, "l_hip": 11, "r_hip": 12}
SOURCES = ["2d_raw", "2d_input", "3d_raw", "3d_mediapipe", "3d_ktpformer"]
FPS = 30.0
VARS = ["FL", "ML", "LL", "LJ", "HJ", "O", "LR"]


def sparc(speed, fs=FPS, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length of a speed profile (Balasubramanian et al. 2015)."""
    v = np.asarray(speed, float)
    v = v[np.isfinite(v)]
    if len(v) < 8 or not np.any(v > 0):
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(v))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    M = np.abs(np.fft.fft(v, nfft))
    M = M / M.max()
    sel = f <= fc
    f, M = f[sel], M[sel]
    idx = np.flatnonzero(M >= amp_th)
    f, M = f[idx[0]:idx[-1] + 1], M[idx[0]:idx[-1] + 1]
    if len(f) < 2:
        return np.nan
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(M) ** 2)))


def on_grid(t_src, X, t_grid, ok):
    """Linear interpolation of X (n, J, C) at times t_grid from the frames ``ok``; NaN outside the observed span."""
    out = np.full((len(t_grid),) + X.shape[1:], np.nan)
    if ok.sum() < 2:
        return out
    ts, Xs = t_src[ok], X[ok].reshape(ok.sum(), -1)
    inside = (t_grid >= ts[0]) & (t_grid <= ts[-1])
    flat = np.stack([np.interp(t_grid[inside], ts, Xs[:, k]) for k in range(Xs.shape[1])], 1)
    out[inside] = flat.reshape((inside.sum(),) + X.shape[1:])
    return out


def measures(P, rest_mask, reps, side_wrist):
    """P: (n, J, C) torso-normalised positions on the 30 fps grid (joints in MP / COCO order of JOINTS)."""
    acc = np.linalg.norm(np.diff(P, 2, axis=0), axis=-1) * FPS ** 2  # (n-2, J)
    out = {"accel": float(np.nanmean(acc)), "jitter_var": float(np.nanmean(np.nanvar(acc, axis=0)))}
    res = []
    for a, b in _runs(rest_mask):
        if b - a + 1 >= 15:
            seg = P[a:b + 1]
            if np.isfinite(seg).all():
                res.append((seg - savgol_filter(seg, 15, 2, axis=0)).reshape(len(seg), -1))
    out["rest_jitter"] = float(np.sqrt(np.mean(np.concatenate(res) ** 2))) if res else np.nan
    sp = []
    for (s, e, side) in reps:
        w = P[s:e + 1, side_wrist[side]]
        v = np.linalg.norm(np.diff(w, axis=0), axis=-1) * FPS
        sp.append(sparc(v))
    out["sparc"] = float(np.nanmedian(sp)) if sp and np.isfinite(sp).any() else np.nan
    return out


def _runs(mask):
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def video_job(args):
    vid, kp_path, pre_path, lift_path, reps = args
    names = list(MP)
    with np.load(pre_path) as z:
        t = z["t"].astype(float)
        xy_in, w_in = z["xy"][:, [MP[k] for k in names]].astype(float), z["world"][:, [MP[k] for k in names]].astype(float)
        torso_px, torso_m, hw = float(z["torso_px"]), float(z["torso_m"]), z["frame_hw"]
    with np.load(kp_path) as z:
        tk, det = z["t"].astype(float), z["detected"].astype(bool)
        lm = z["lm2d"][:, [MP[k] for k in names], :2].astype(float)
        wr = z["world"][:, [MP[k] for k in names]].astype(float)
    lm = lm * np.array([hw[1], hw[0]]) / torso_px  # normalised image coordinates -> pixels -> 2D torso lengths
    src = {"2d_raw": on_grid(tk, lm, t, det), "2d_input": xy_in, "3d_raw": on_grid(tk, wr / torso_m, t, det),
           "3d_mediapipe": w_in}
    if lift_path:
        with np.load(lift_path) as z:
            src["3d_ktpformer"] = z["world"][:, [COCO[k] for k in names]].astype(float)
    n = len(t)
    rest = np.ones(n, bool)
    rr = []
    for s, e, side in reps:
        rest[max(0, s):e + 1] = False
        rr.append((int(s), int(e), side))
    wrist = {"left": names.index("l_wr"), "right": names.index("r_wr")}
    rows = []
    for k, P in src.items():
        rows.append(dict(video_id=vid, source=k, **measures(P[:n], rest, rr, wrist)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    cfg = load_config(REPO / a.config)
    res = resolve(cfg["eval"]["results_dir"])
    out = res / "completion" / "temporal"
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    idx = idx[idx.exercise.isin(EXS) & idx.pre_path.notna()]
    idx = idx[[resolve(p).exists() for p in idx.pre_path]]  # P7: videos with < 50% detection have no Phase 1 file
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    lift_dir = resolve(cfg["lifter"]["out_dir"])
    rmap = {v: list(zip(d.start, d.end, d.side)) for v, d in reps.groupby("video_id")}
    jobs = []
    for r in idx.itertuples():
        lp = lift_dir / f"{r.video_id}.npz"
        jobs.append((r.video_id, str(resolve(r.kp_path)), str(resolve(r.pre_path)), str(lp) if lp.exists() else "",
                     rmap.get(r.video_id, [])))
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(video_job, jobs, chunksize=8)):
            rows += r
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)} videos ({time.time() - t0:.0f} s)", flush=True)
    V = pd.DataFrame(rows).merge(idx[["video_id", "exercise", "subject", "group", "view", "variation"]], on="video_id")
    V.to_csv(out / "videos.csv", index=False)
    met = ["accel", "jitter_var", "rest_jitter", "sparc"]
    S = V.groupby(["source", "view"])[met].median().reset_index()
    S.to_csv(out / "by_source.csv", index=False)
    C = V.groupby(["source", "variation"])[met].median().reset_index()
    C.to_csv(out / "by_condition.csv", index=False)

    # model outputs: rep-to-rep variation of the predicted score within one video
    import report_perex as RP
    _, pr, _ = RP.load_runs(resolve(cfg["train"]["runs_dir"]) / "perex_loso", EXS,
                            ["b1", "b2_ktpformer", "proposed_j1", "proposed_lp"])
    pr = pr[pr.groupby(["method", "video_id"]).video_id.transform("size") >= 3]
    ps = pr.groupby(["method", "video_id", "group"]).score_pred.std().reset_index(name="sd")
    m3 = pd.read_csv(resolve(cfg["assess"]["json_dir"]).parent / "module3_reps.csv")
    m3 = m3[m3.exercise.isin(EXS)]
    m3 = m3[m3.groupby("video_id").video_id.transform("size") >= 3]
    ms = m3.groupby(["video_id", "group"]).ref_score.std().reset_index(name="sd")
    ms["method"] = "module3"
    PS = pd.concat([ps, ms], ignore_index=True).groupby(["method", "group"]).sd.agg(
        ["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75), "size"]).reset_index()
    PS.columns = ["method", "group", "median_sd", "q25", "q75", "videos"]
    PS.to_csv(out / "prediction_stability.csv", index=False)
    write_summary(out, V, S, C, PS)
    print(f"written to {out} ({time.time() - t0:.0f} s)")


def write_summary(out, V, S, C, PS):
    L = ["# Proposal completion: temporal stability (E01–E04)\n",
         "Generated by `scripts/full/completion/temporal_stability.py`. Medians over videos. Positions in torso "
         "lengths on the 30 fps grid; accel = mean |acceleration| (torso/s²); jitter var = its variance over time; "
         "rest jitter = RMS deviation from a 0.5 s smooth fit while the person holds still (torso lengths); SPARC = "
         "smoothness of the moving wrist's speed per rep (closer to 0 = smoother).\n"]
    names = {"2d_raw": "2D MediaPipe, raw", "2d_input": "2D network input (smoothed)", "3d_raw": "3D MediaPipe, raw",
             "3d_mediapipe": "3D MediaPipe, preprocessed (Module III)", "3d_ktpformer": "3D KTPFormer (B2 input)"}
    L.append("\n## By skeleton source and camera\n")
    L.append("| Source | view | accel | jitter var | rest jitter | SPARC |")
    L.append("|---|---|---|---|---|---|")
    for r in S.itertuples():
        L.append(f"| {names[r.source]} | {r.view} | {r.accel:.2f} | {r.jitter_var:.1f} | {r.rest_jitter:.4f} | "
                 f"{r.sparc:.2f} |")
    L.append("\n## By recording condition (all cameras)\n")
    vs = [v for v in VARS if v in set(C.variation)]
    L.append("| Source | " + " | ".join(vs) + " |")
    L.append("|" + "---|" * (1 + len(vs)))
    for s in SOURCES:
        g = C[C.source == s].set_index("variation")
        if g.empty:
            continue
        L.append(f"| {names[s]}: accel / rest jitter | " + " | ".join(
            f"{g.loc[v, 'accel']:.2f} / {g.loc[v, 'rest_jitter']:.4f}" for v in VARS if v in g.index) + " |")
    L.append("\n## Rep-to-rep stability of the assessment\n")
    L.append("Within-video SD of the rep-level predicted total score (points; videos with ≥ 3 reps; networks: LOSO "
             "rep head; Module III: rep score, GroupKFold-5 libraries). Median [IQR].\n")
    L.append("| Method | group | median SD [IQR] | videos |")
    L.append("|---|---|---|---|")
    for r in PS.itertuples():
        L.append(f"| {r.method} | {r.group} | {r.median_sd:.2f} [{r.q25:.2f}, {r.q75:.2f}] | {r.videos} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
