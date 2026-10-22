"""Compare audio / metadata / motion sync evidence; calibrate on control pairs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])
pd.set_option("display.width", 230)
v = pd.read_csv(D / "av_videos.csv")
p = pd.read_csv(D / "av_pairs.csv")
p["expert"] = p.pid.str[1:].astype(int) <= 24

print("=== per-video container facts")
print(v.groupby("ex").agg(videos=("file", "size"), has_audio=("has_audio", "mean"), creation=("creation", lambda s: s.notna().mean())).round(3))
v["model"] = v.model.fillna("")
print("phone model tag (only some phones write it):")
print(v.groupby(["ex", "angle"]).model.agg(lambda s: s.replace("", "untagged").value_counts().to_dict()))
v["year"] = pd.to_datetime(v.creation, unit="s").dt.year
v["expert"] = v.pid.str[1:].astype(int) <= 24
print("recording year by group:", v.groupby(["expert", "year"]).size().to_dict())

c, s = p[p.kind == "control"], p[p.kind == "same_group"]
print("\n=== AUDIO GCC-PHAT peak ratio (peak / 2nd peak)")
for name, d in (("control", c), ("same_group", s)):
    q = d.audio_peak_ratio.dropna()
    print(f"  {name:10s} n={len(q):4d}  p50 {q.median():.2f}  p95 {q.quantile(.95):.2f}  p99 {q.quantile(.99):.2f}  max {q.max():.2f}")
    print(f"  {'':10s} peak z: p50 {d.audio_peak_z.median():.1f}  p99 {d.audio_peak_z.quantile(.99):.1f}  max {d.audio_peak_z.max():.1f}")
thr_ratio = c.audio_peak_ratio.max()
print(f"  -> audio threshold = max control ratio = {thr_ratio:.2f}")
s = s.assign(audio_sync=s.audio_peak_ratio > thr_ratio)
print("  same-group pairs above threshold, by exercise/group:")
print(s.groupby(["ex", "expert"]).agg(pairs=("audio_sync", "size"), audio_sync=("audio_sync", "sum"),
                                      lag_med=("audio_lag_s", lambda x: x[s.loc[x.index, "audio_sync"]].abs().median()),
                                      lag_max=("audio_lag_s", lambda x: x[s.loc[x.index, "audio_sync"]].abs().max())).round(3))
print("  by view pair:", s.groupby(["view_a", "view_b"]).audio_sync.mean().round(3).to_dict())

print("\n=== METADATA creation_time difference |dC| (s)")
for name, d in (("control", c), ("same_group", s)):
    q = d.d_creation_s.abs().dropna()
    print(f"  {name:10s} n={len(q):4d}  min {q.min():.0f}  p5 {q.quantile(.05):.0f}  p50 {q.median():.0f}  p95 {q.quantile(.95):.0f}  max {q.max():.0f}")
print("  same-group |dC| distribution:", s.d_creation_s.abs().clip(upper=10).value_counts().sort_index().to_dict(), "(10 = >=10)")
print("  controls with |dC| <= 3 s:", int((c.d_creation_s.abs() <= 3).sum()))
a = s[s.audio_sync]
# convention check: creation = start  => dC ~ -lag ; creation = end => dC - (dur_b - dur_a) ~ -lag
e_start = (a.d_creation_s + a.audio_lag_s).abs()
e_end = (a.d_creation_s - (a.dur_b - a.dur_a) + a.audio_lag_s).abs()
print(f"  on audio-synced pairs: |dC + lag| median {e_start.median():.2f} s (creation=start)   "
      f"|dC - dDur + lag| median {e_end.median():.2f} s (creation=end)")
print("  audio-synced pairs: |dC| max", a.d_creation_s.abs().max())
ns = s[~s.audio_sync]
print("  NOT-audio-synced same-group pairs |dC| distribution:", ns.d_creation_s.abs().clip(upper=10).value_counts().sort_index().to_dict())

print("\n=== AUDIO vs MOTION on the pose-sampled pairs")
m = pd.read_csv(D / "sync_pairs_v3.csv")
m = m[m.kind == "same_group"]
j = m.merge(s, left_on=["ex", "pid", "var_a", "view_a", "view_b"], right_on=["ex", "pid", "var_a", "view_a", "view_b"], how="left")
j["lag_diff_s"] = (j.lag_s - j.audio_lag_s).abs()
cols = ["ex", "pid", "var_a", "view_a", "view_b", "audio_peak_ratio", "audio_lag_s", "lag_s", "lag_diff_s", "resid_sd_lin_ms", "d_creation_s"]
print(j[cols].round(3).to_string(index=False))
ok = j[j.audio_sync]
print(f"\naudio-synced sampled pairs: {len(ok)}; motion lag within 0.1 s of audio lag: {(ok.lag_diff_s < 0.1).sum()}; "
      f"motion test passed: {(ok.resid_sd_lin_ms < 42).sum()}")
print("motion-passed pairs that audio did NOT confirm:", int(((j.resid_sd_lin_ms < 42) & ~j.audio_sync).sum()))
s.to_csv(D / "av_pairs_same_group_eval.csv", index=False)
