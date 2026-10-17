"""Classify every multi-view group as synced / not-synced / unknown (Step 0 prototype).

Order of evidence (as requested):
 1. audio    : FLR group whose three GCC-PHAT lags (+-3 s window) are additive,
               |lag_FL + lag_LR - lag_FR| <= 20 ms, none at the window edge  -> synced (high)
 2. metadata : creation_time. Recording windows that cannot overlap
               (|dC| > dur_a + dur_b)                                        -> not-synced (high)
               all pairs |dC| <= 3 s (no control pair ever is)               -> synced (medium; lag
               from audio, to be refined by motion in Phase 1)
 3. motion   : rep-timing residual < 42 ms on every pair (sampled groups only)-> synced (medium)
 otherwise unknown.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])
pd.set_option("display.width", 230)
p = pd.read_csv(D / "av_pairs.csv")
mo = pd.read_csv(D / "sync_pairs_v3.csv")
mo = mo[mo.kind == "same_group"][["ex", "pid", "var_a", "view_a", "view_b", "lag_s", "resid_sd_lin_ms", "matched"]]

# ---- LR provenance
lr = p[p.kind == "lr_check"].sort_values("audio_peak_ratio", ascending=False).groupby(["ex", "pid", "var_a", "view_a"]).head(1)
print("=== LR videos: best-matching other variation (same person, same view)")
print(lr[["ex", "pid", "view_a", "var_b", "dur_a", "dur_b", "audio_lag_s", "audio_peak_ratio", "d_creation_s"]].round(3).to_string(index=False))
print("LR videos whose audio is a copy of another variation (peak ratio > 10):", int((lr.audio_peak_ratio > 10).sum()), "/", len(lr))

s = p[p.kind == "same_group"].copy()
s = s.merge(mo, on=["ex", "pid", "var_a", "view_a", "view_b"], how="left")
s["meta_close"] = s.d_creation_s.abs() <= 3
s["meta_apart"] = s.d_creation_s.abs() > (s.dur_a + s.dur_b)
s["motion_ok"] = (s.resid_sd_lin_ms < 42) & (s.matched >= 4)

rows = []
for (ex, pid, var), g in s.groupby(["ex", "pid", "var_a"]):
    views = sorted(set(g.view_a) | set(g.view_b))
    lag = {(a, b): l for a, b, l in zip(g.view_a, g.view_b, g.audio_lag3_s)}
    r = dict(ex=ex, pid=pid, var=var, views="".join(views), expert=int(pid[1:]) <= 24,
             max_abs_dC_s=g.d_creation_s.abs().max(), audio_triangle_err_ms=np.nan,
             motion_tested=int(g.resid_sd_lin_ms.notna().sum()), motion_ok=int(g.motion_ok.sum()))
    edge = (g.audio_lag3_s.abs() > 2.9).any()
    if views == ["F", "L", "R"] and not g.audio_lag3_s.isna().any():
        r["audio_triangle_err_ms"] = 1000 * abs(lag[("F", "L")] + lag[("L", "R")] - lag[("F", "R")])
    if r["audio_triangle_err_ms"] <= 20 and not edge:
        cls, method, conf = "synced", "audio", "high"
    elif g.meta_apart.any():
        cls, method, conf = "not-synced", "metadata", "high"
    elif g.meta_close.all():
        cls, method, conf = "synced", "metadata", "medium"
    elif r["motion_tested"] and g.motion_ok.all():
        cls, method, conf = "synced", "motion", "medium"
    else:
        cls, method, conf = "unknown", "-", "-"
    r.update(cls=cls, method=method, confidence=conf)
    if "F" in views:
        r.update({f"lag_{v}_vs_F_s": lag.get(("F", v), np.nan) for v in ("L", "R")})
    rows.append(r)
G = pd.DataFrame(rows)
G.to_csv(D / "sync_groups_classified.csv", index=False)

print("\n=== Group classification (all multi-view groups)")
print(G.groupby(["ex", "expert", "cls", "method"]).size().unstack(["cls", "method"], fill_value=0))
print("\naudio triangle error (ms) on FLR groups, expert:",
      G[G.expert].audio_triangle_err_ms.describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict())
fl = G[G.expert & G.audio_triangle_err_ms.notna()]
print("share of expert FLR groups with triangle error <= 20 ms:", round((fl.audio_triangle_err_ms <= 20).mean(), 3),
      " by exercise:", fl.groupby("ex").audio_triangle_err_ms.apply(lambda x: round((x <= 20).mean(), 3)).to_dict())
print("audio-synced groups: |lag| vs F  median %.2f s, max %.2f s" % (
    np.nanmedian(np.abs(G.loc[G.method == "audio", ["lag_L_vs_F_s", "lag_R_vs_F_s"]].values)),
    np.nanmax(np.abs(G.loc[G.method == "audio", ["lag_L_vs_F_s", "lag_R_vs_F_s"]].values))))

# motion vs audio lag on motion-confirmed pairs (per-device audio/video offset?)
mc = s[s.motion_ok]
mc = mc.assign(video_minus_audio_ms=1000 * (mc.lag_s - mc.audio_lag3_s))
print("\nmotion-confirmed pairs: video lag - audio lag (ms) by view pair:")
print(mc.groupby(["view_a", "view_b"]).video_minus_audio_ms.describe()[["count", "mean", "std", "min", "max"]].round(0))
print("\nsampled groups (pose available): class vs motion evidence")
smp = G[G.motion_tested > 0]
print(smp[["ex", "pid", "var", "views", "max_abs_dC_s", "audio_triangle_err_ms", "motion_ok", "motion_tested", "cls", "method"]].round(1).to_string(index=False))
