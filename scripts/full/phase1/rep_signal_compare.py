"""Choose the rep signal of E03, E06 and E09 among Step 0's candidates by cross-view rep agreement (user, 2026-10-20).

Criterion (fixed before running; no score or label is used):
  * groups: every synced expert F+L+R group of the exercise whose three views have keypoints (Phase 1 sync report),
    leaving out only the subjects excluded from the exercise labels for a wrong exercise (``data.exercise_label_exclude``:
    E03 P02, D8);
  * per view, the rep count of the candidate signal segmented on its own (no boundary transfer), with the pipeline's
    settings: adaptive thresholds and polarity 'peak' (P6c), after orienting the candidate per side so that it rises
    with the exercise's reference movement (E03: the forearm going up, i.e. -forearm elevation; E06: the 2D wrist height
    = the arm raised; E09: the trunk-thigh angle = the trunk raised);
  * measure: the share of groups whose three counts lie within +-1; ties broken by the share of exactly equal counts,
    then in favour of the current signal.
Candidates: Step 0's per region (scripts/full/step0/pose_analysis.py ``candidate``): upper limb (E03) shoulder elevation,
elbow flexion, forearm elevation and azimuth, wrist-path PC1, 2D wrist height, all-joint 2D PC1; wrist (E06) the Pose-point
wrist deviation 3D / 2D, elbow flexion, forearm elevation, 2D wrist height, all-joint PC1; trunk (E09) hip and knee
flexion (moving side and summed), trunk-thigh angle, 2D trunk inclination, 2D shoulder height, all-joint PC1. Sided
signals use the side with the larger p5-p95 range, as in Step 0 and the pipeline.
Writes <outputs.phase1>/rep_signal_compare.csv (per exercise and candidate) and rep_signal_compare_counts.csv.

usage: python scripts/full/phase1/rep_signal_compare.py [--config configs/data_full.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "full" / "step0"))
from cvdc.config import exercise_cfg, load_config, resolve  # noqa: E402
from cvdc.data.reps import load_pre, orient, segment_signal  # noqa: E402
from pose_analysis import candidate, signals  # noqa: E402  (Step 0's candidate signals)

FPS = 30
REGION = {"E03": "upper", "E06": "wrist", "E09": "trunk"}
CURRENT = {"E03": "wpc1", "E06": "wy", "E09": "hip"}


def reference(s, ex, side):
    """The signal each candidate is oriented by (rises with the movement away from rest)."""
    if ex == "E03":
        return -s[f"fa_up_{side[0]}"]
    if ex == "E06":
        return s[f"wy_{side[0]}"]
    return s["trunk_leg"]


def ref_side(s, ex):
    """Side of the reference for candidates without sides: the moving limb by Step 0's family range."""
    fam = {"E03": "wpc1", "E06": "wy", "E09": "hip"}[ex]
    rng = {c: np.nanpercentile(s[f"{fam}_{c}"], 95) - np.nanpercentile(s[f"{fam}_{c}"], 5) for c in "lr"}
    return "left" if rng["l"] > rng["r"] else "right"


def counts(pre, ex, rc):
    s = signals(pre)
    cand = candidate(s, REGION[ex])
    rs = ref_side(s, ex)
    out = {}
    for k, x in cand.items():
        if k.endswith("@side"):
            continue
        side = {"l": "left", "r": "right"}.get(cand.get(f"{k}@side"), rs)
        x = orient(np.asarray(x, float), reference(s, ex, side))
        out[k] = len(segment_signal(x, FPS, rc)[0]) if np.isfinite(x).sum() >= 2 * FPS else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_full.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    G = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    excl = {(e, s) for e, subs in (cfg["data"].get("exercise_label_exclude") or {}).items() for s in subs}
    rows = []
    for ex in REGION:
        rc = exercise_cfg(cfg, ex)["reps"]
        e = idx[(idx.exercise == ex) & (idx.group == "expert")]
        e = e[[(ex, s) not in excl for s in e.subject]]
        for gid, g in e.groupby("group_id"):
            if gid not in G.index or G.loc[gid, "cls"] != "synced" or sorted(g.view) != ["F", "L", "R"]:
                continue
            if not g.pre_path.map(lambda p: resolve(p).exists()).all():
                continue
            for r in g.itertuples():
                for k, n in counts(load_pre(resolve(r.pre_path)), ex, rc).items():
                    rows.append(dict(exercise=ex, group_id=gid, view=r.view, signal=k, n=n))
    C = pd.DataFrame(rows)
    out = resolve(cfg["outputs"]["phase1"])
    C.to_csv(out / "rep_signal_compare_counts.csv", index=False)
    per = C.groupby(["exercise", "signal", "group_id"]).n.agg(["min", "max", "count", "median"]).reset_index()
    per = per[per["count"] == 3]
    T = per.groupby(["exercise", "signal"]).apply(lambda d: pd.Series(dict(
        groups=len(d), within1=float(((d["max"] - d["min"]) <= 1).mean()), exact=float((d["max"] == d["min"]).mean()),
        median_count=float(d["median"].median()))), include_groups=False).reset_index()
    T["current"] = T.signal == T.exercise.map(CURRENT)
    T = T.sort_values(["exercise", "within1", "exact", "current"], ascending=[True, False, False, False])
    T["rank"] = T.groupby("exercise").cumcount() + 1
    T.to_csv(out / "rep_signal_compare.csv", index=False)
    pd.set_option("display.width", 200)
    print("criterion: share of synced expert F+L+R groups with the three per-view counts within +-1 "
          "(ties: exact, then the current signal); excluded subjects:", sorted(excl))
    print(T.round(3).to_string(index=False))
    W = T[T["rank"] == 1]
    print("\nwinner per exercise:", {r.exercise: f"{r.signal} ({r.within1:.1%}, current: "
                                     f"{CURRENT[r.exercise]} {T[(T.exercise == r.exercise) & T.current].within1.iloc[0]:.1%})"
                                     for r in W.itertuples()})


if __name__ == "__main__":
    main()
