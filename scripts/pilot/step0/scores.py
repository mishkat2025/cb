"""Parse 'Physiotherapist Exercise Marking.csv' (3 rows per video) and join to the local inventory."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV, INV, META = map(Path, sys.argv[1:4])
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

raw = pd.read_csv(CSV, header=None, skiprows=2, dtype=str, encoding="utf-8-sig")
raw.columns = ["video", "judge", "PF1", "PF2", "PF3", "PF4", "CF1", "CF2", "CF3",
               "total", "scaled100", "avg", "avg_round"][: raw.shape[1]]
raw["video"] = raw["video"].ffill().str.strip().str.upper()
q = ["PF1", "PF2", "PF3", "PF4", "CF1", "CF2", "CF3"]
for c in q + ["total", "scaled100", "avg", "avg_round"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["judge"] = raw["judge"].str.strip()

print("rows:", len(raw), " unique video names:", raw.video.nunique())
print("judges:", raw.judge.value_counts().to_dict())
rows_per_video = raw.groupby("video").size()
print("rows per video:", rows_per_video.value_counts().to_dict())
print("videos with !=3 rows:", rows_per_video[rows_per_video != 3].to_dict())
dupe = raw.groupby("video").judge.apply(lambda s: s.duplicated().any())
print("videos with a duplicated judge:", list(dupe[dupe].index)[:10])

PAT = re.compile(r"^(E\d{2})_(P\d{1,3})_A([FLR])_V(FL|ML|LL|LJ|HJ|O|LR)_G([MF])$")
bad = [v for v in raw.video.unique() if not PAT.match(v)]
print("CSV names failing the filename regex:", len(bad), bad[:20])

print("\nper-question value sets (all judges):")
for c in q:
    print(f"  {c}: {sorted(raw[c].dropna().unique().tolist())}  NaN={raw[c].isna().sum()}")

# consistency of the provided totals/averages
raw["sum7"] = raw[q].sum(axis=1, min_count=1)
print("\nrows where total != sum of 7 questions:", int((raw.total - raw.sum7).abs().gt(1e-6).sum()))
print("rows where scaled100 != total/7*100:", int((raw.scaled100 - raw.total / 7 * 100).abs().gt(1e-3).sum()))

per_vid = raw.groupby("video").agg(
    avg_round=("avg_round", "first"), avg=("avg", "first"), mean_scaled=("scaled100", "mean"),
    judge_sd=("scaled100", "std"), **{f"{c}_mean": (c, "mean") for c in q})
print("videos where provided avg != mean(judge scaled):",
      int((per_vid.avg - per_vid.mean_scaled).abs().gt(1e-3).sum()))
print("videos with missing avg:", int(per_vid.avg_round.isna().sum()))

m = per_vid.index.to_series().str.extract(PAT)
m.columns = ["ex", "pid", "angle", "var", "gender"]
per_vid = per_vid.join(m)
per_vid["pnum"] = per_vid.pid.str[1:].astype(float)
per_vid["group"] = np.where(per_vid.pnum <= 24, "expert", "non-expert")

print("\n=== whole-dataset score summary (all 9 exercises, from CSV)")
s = per_vid.avg_round
print(f"n={len(s)} mean {s.mean():.1f} median {s.median():.1f} sd {s.std():.1f}  "
      f"==100: {(s == 100).mean():.1%}  >=80: {(s >= 80).mean():.1%}  50-79: {s.between(50, 79.999).mean():.1%}  <50: {(s < 50).mean():.1%}")
print(per_vid.groupby(["group", "gender"]).avg_round.agg(["count", "mean"]).round(1))
print(per_vid.groupby("ex").avg_round.agg(["count", "mean", "median", "std", "min", "max"]).round(1))

# join to local files
inv = pd.read_csv(INV)
inv["video"] = inv.file.str.replace(r"\.mp4$", "", regex=True).str.upper()
meta = pd.read_csv(META)
inv = inv.merge(meta[["file", "fps", "frames", "duration_s", "width", "height"]], on="file", how="left")
j = inv.merge(per_vid.reset_index(), on="video", how="left", suffixes=("", "_csv"))
print("\n=== local videos:", len(inv), " with a CSV score:", int(j.avg_round.notna().sum()),
      " missing:", j.loc[j.avg_round.isna(), "video"].tolist())
local_ex = sorted(inv.ex.unique())
csv_local = per_vid[per_vid.ex.isin(local_ex)]
print("CSV rows for local exercises without a local file:",
      sorted(set(csv_local.index) - set(inv.video)))

for ex, g in j.groupby("ex"):
    print(f"\n######## {ex}")
    for grp, gg in g.groupby("group"):
        s = gg.avg_round
        print(f"  {grp:10s} n={len(s):3d} subjects={gg.pid.nunique():2d}  mean {s.mean():5.1f}  median {s.median():5.1f}  "
              f"sd {s.std():4.1f}  min {s.min():5.1f}  ==100 {(s == 100).mean():5.1%}  >=95 {(s >= 95).sum():3d}")
    bins = pd.cut(g.avg_round, [-0.1, 49.999, 79.999, 100], labels=["Incorrect<50", "Partial50-79", "Correct>=80"])
    print("  default quality bins:", bins.value_counts().reindex(["Correct>=80", "Partial50-79", "Incorrect<50"]).to_dict())
    print("  histogram (10-pt bins):", np.histogram(g.avg_round, bins=range(0, 110, 10))[0].tolist())
    print("  mean score per variation:", g.groupby("var").avg_round.mean().round(1).to_dict())
    print("  mean score per angle   :", g.groupby("angle").avg_round.mean().round(1).to_dict())
    print("  per-question mean (0-1):", g[[f"{c}_mean" for c in q]].mean().round(3).to_dict())
    print("  per-question frac==1   :", {c: round(float((g[f'{c}_mean'] >= 0.999).mean()), 3) for c in q})
    print("  judge sd (0-100) mean/max:", round(g.judge_sd.mean(), 2), round(g.judge_sd.max(), 2))
    # same (pid, variation) group: do views share one score?
    gs = g.groupby(["pid", "var"]).avg_round.agg(["nunique", "count"])
    multi = gs[gs["count"] > 1]
    print(f"  multi-view groups whose views have identical scores: {(multi['nunique'] == 1).sum()}/{len(multi)}")
    spread = g.groupby(["pid", "var"]).avg_round.agg(lambda s: s.max() - s.min())[gs["count"] > 1]
    print(f"  within-group score range: mean {spread.mean():.1f}  max {spread.max():.1f}")
    print(f"  duration by group (s):", g.groupby("group").duration_s.describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict("index"))
    print("  score vs duration Spearman:", round(g[["avg_round", "duration_s"]].corr("spearman").iloc[0, 1], 3))

j.to_csv(INV.with_name("inventory_scored.csv"), index=False)
per_vid.to_csv(INV.with_name("scores_per_video.csv"))
