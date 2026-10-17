"""Synchronization test v3: mid-level crossing timing (sharp events) + per-view signal candidates.

For each video: candidates = PC1, PC2 of both wrists' (x,y) relative to mid-shoulder (torso-normalised)
and the wrists' distance from their rest pose. For a pair, the first view uses its PC1 and the second
view uses whichever candidate correlates best (same procedure for controls).
Timing test: times where the z-scored signal crosses 0 (with hysteresis +-0.5), refined by linear
interpolation; after the best lag, matched crossings (same direction, <0.5 s apart) give residuals.
  resid_sd_ms     : SD of residuals after removing a constant offset
  resid_sd_lin_ms : SD after removing offset + linear drift (clock-rate mismatch)
Simultaneous recordings -> residuals at pose-noise level (tens of ms); separate recordings -> tempo
differences -> hundreds of ms.
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

KP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FS = 30.0
PAT = re.compile(r"^(E\d{2})_(P\d+)_A([FLR])_V(\w+)_G([MF])$")
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 15, 16, 23, 24


def fill(x):
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return None
    i = np.arange(len(x))
    x[~ok] = np.interp(i[~ok], i[ok], x[ok])
    return x


def smooth(s, t):
    tt = np.arange(0, t[-1] + 1e-9, 1 / FS)
    s = np.interp(tt, t, median_filter(s, 5))
    return savgol_filter(s, 9, 2) if len(s) > 11 else s


def load(path):
    d = np.load(path)
    img, t, det = d["img"], d["t"], d["detected"]
    h, w = d["frame_hw"]
    xy = img[..., :2] * np.array([w, h])
    mid_sh = (xy[:, L_SH] + xy[:, R_SH]) / 2
    torso = np.nanmedian(np.linalg.norm(mid_sh - (xy[:, L_HIP] + xy[:, R_HIP]) / 2, axis=1))
    rel = np.concatenate([xy[:, L_WR] - mid_sh, xy[:, R_WR] - mid_sh], 1) / torso
    cols = [fill(c) for c in rel.T]
    if any(c is None for c in cols):
        return None, dict(det=float(det.mean()), dur=float(t[-1]))
    rel = np.stack(cols, 1)
    z = rel - rel.mean(0)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    rest = np.median(rel[np.argsort(np.linalg.norm(np.gradient(rel, axis=0), axis=1))[: len(rel) // 5]], 0)
    dist = np.linalg.norm((rel - rest).reshape(-1, 2, 2), axis=2).max(1)
    cand = {"pc1": smooth(z @ vt[0], t), "pc2": smooth(z @ vt[1], t), "dist": smooth(dist, t)}
    return cand, dict(det=float(det.mean()), dur=float(t[-1]))


def xcorr(a, b, min_overlap=0.6):
    n, m = len(a), len(b)
    need = int(min_overlap * min(n, m))
    A = (a - a.mean()) / (a.std() + 1e-9)
    best = (-np.inf, 0, 1)
    for lag in range(-(n - need), m - need + 1):
        a0, b0 = max(0, -lag), max(0, lag)
        k = min(n - a0, m - b0)
        x, y = A[a0:a0 + k], b[b0:b0 + k]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if abs(r) > best[0]:
            best = (abs(r), lag, np.sign(r))
    return best


def crossings(s):
    z = (s - np.median(s)) / (s.std() + 1e-9)
    ev, state = [], None
    for i in range(1, len(z)):
        if z[i] > 0.5:
            new = "hi"
        elif z[i] < -0.5:
            new = "lo"
        else:
            continue
        if state is not None and new != state:
            j = i
            while j > 0 and np.sign(z[j]) == np.sign(z[i]):
                j -= 1
            t0 = j + (0 - z[j]) / (z[j + 1] - z[j]) if z[j + 1] != z[j] else j  # interpolate zero
            ev.append((t0 / FS, +1 if new == "hi" else -1))
        state = new
    return ev


def timing(a, b, lag, sgn):
    ea = crossings(a)
    eb = [(t - lag / FS, d * sgn) for t, d in crossings(b)]
    res, ts = [], []
    for t, dirn in ea:
        cands = [tb for tb, db in eb if db == dirn and abs(tb - t) < 0.5]
        if cands:
            tb = min(cands, key=lambda x: abs(x - t))
            res.append(tb - t)
            ts.append(t)
    if len(res) < 4:
        return np.nan, np.nan, len(res), len(ea), len(eb)
    res, ts = np.array(res), np.array(ts)
    sd = res.std()
    p = np.polyfit(ts, res, 1)
    sd_lin = (res - np.polyval(p, ts)).std()
    return sd * 1000, sd_lin * 1000, len(res), len(ea), len(eb)


sig, info = {}, {}
for f in sorted(KP.glob("*.npz")):
    k = PAT.match(f.stem).groups()
    sig[k], info[k] = load(f)
by_group = {}
for k in sorted(sig):
    by_group.setdefault((k[0], k[1], k[3]), {})[k[2]] = k


def pair(kind, a, b):
    r = dict(kind=kind, ex=a[0], pid=a[1], var_a=a[3], var_b=b[3], view_a=a[2], view_b=b[2],
             det_min=min(info[a]["det"], info[b]["det"]), dur_a=info[a]["dur"], dur_b=info[b]["dur"])
    if sig[a] is None or sig[b] is None:
        return r
    x = sig[a]["pc1"]
    best = max(((xcorr(x, y), name, y) for name, y in sig[b].items()), key=lambda z: z[0][0])
    (rr, lag, sgn), name, y = best
    sd, sd_lin, nm, na, nb = timing(x, y, lag, sgn)
    r.update(r=rr, lag_s=lag / FS, cand_b=name, resid_sd_ms=sd, resid_sd_lin_ms=sd_lin,
             matched=nm, cross_a=na, cross_b=nb)
    return r


rows = []
for v in by_group.values():
    for va, vb in itertools.combinations(sorted(v), 2):
        rows.append(pair("same_group", v[va], v[vb]))
gp = {}
for g in by_group:
    gp.setdefault((g[0], g[1]), []).append(g)
for gs in gp.values():
    for g1, g2 in itertools.combinations(gs, 2):
        for va in by_group[g1]:
            for vb in by_group[g2]:
                rows.append(pair("control", by_group[g1][va], by_group[g2][vb]))
df = pd.DataFrame(rows)
df.to_csv(OUT / "sync_pairs_v3.csv", index=False)
pd.set_option("display.width", 230)

c, s = df[df.kind == "control"].copy(), df[df.kind == "same_group"].copy()
print("=== CONTROLS (separate recordings), n =", len(c))
for col in ("r", "resid_sd_ms", "resid_sd_lin_ms"):
    q = c[col].dropna()
    print(f"  {col:16s} n={len(q):3d}  min {q.min():7.1f}  p5 {q.quantile(.05):7.1f}  median {q.median():7.1f}  max {q.max():7.1f}")
print("  |lag| s: median %.2f  min %.2f" % (c.lag_s.abs().median(), c.lag_s.abs().min()))
print("  controls by exercise (median resid_sd_lin_ms):", c.groupby("ex").resid_sd_lin_ms.median().round(0).to_dict())

print("\n=== SAME-GROUP pairs")
cols = ["ex", "pid", "var_a", "view_a", "view_b", "det_min", "r", "lag_s", "cand_b", "resid_sd_ms",
        "resid_sd_lin_ms", "matched", "cross_a", "cross_b"]
print(s[cols].round(3).to_string(index=False))

thr = float(c.resid_sd_lin_ms.min())
print(f"\nrule: pair synchronized if >=4 matched crossings and resid_sd_lin_ms < {thr:.0f} ms "
      "(= the smallest value seen in any control pair)")
s["sync"] = (s.matched >= 4) & (s.resid_sd_lin_ms < thr)
s["testable"] = s.matched >= 4
v = s.groupby(["ex", "pid", "var_a"]).agg(
    testable=("testable", "sum"), sync_pairs=("sync", "sum"), min_r=("r", "min"),
    max_lag_s=("lag_s", lambda x: x.abs().max()), worst_resid_lin_ms=("resid_sd_lin_ms", "max"),
    best_resid_lin_ms=("resid_sd_lin_ms", "min"), det_min=("det_min", "min")).reset_index()
v["verdict"] = np.select(
    [v.testable == 0, v.sync_pairs == 3, v.sync_pairs > 0],
    ["untestable (<2 reps)", "synced (3/3 pairs)", "partly synced"], "no evidence of sync")
print(v.round(1).to_string(index=False))
print("\nverdicts:", v.verdict.value_counts().to_dict())
v.to_csv(OUT / "sync_groups_v3.csv", index=False)
