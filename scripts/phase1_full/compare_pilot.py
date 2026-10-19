"""Consistency of the full-scope Phase 1 with the pilot (E01/E02) and with the full-data Step 0 (E03-E09).

1. E01/E02 preprocessed files, full scope vs pilot: identical arrays? (Same keypoint files, same preprocessing code;
   if identical, the pilot's KTPFormer lifts can be reused.)
2. E01/E02 sync classes, full vs pilot: the groups whose class or method changed (expected: only the start-time rule,
   S8b, and its knock-on effects), and the per-video rep counts in unchanged groups.
3. E03-E09: sync classes, Phase 1 vs Step 0 (`outputs/step0_full/sync_groups_classified.csv`); cross-view rep
   agreement on all synced expert F+L+R groups (per-view segmentation, the Step 0 criterion: three counts within +-1)
   next to the Step 0 sample value.

usage: python scripts/phase1_full/compare_pilot.py [--config configs/data_full.yaml] [--pilot-config configs/data.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvdc.config import data_path, load_config, resolve  # noqa: E402

STEP0_WITHIN1 = {"E03": 0.90, "E04": 0.64, "E05": 0.82, "E06": 0.75, "E07": 0.70, "E08": 0.92, "E09": 0.78}


def same_npz(a, b, keys=("t", "xy", "vis", "world")):
    with np.load(a) as za, np.load(b) as zb:
        return all(np.array_equal(za[k], zb[k], equal_nan=True) for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_full.yaml")
    ap.add_argument("--pilot-config", default="configs/data.yaml")
    ap.add_argument("--only", choices=["pre", "all"], default="all")
    a = ap.parse_args()
    cfg, pcfg = load_config(a.config), load_config(a.pilot_config)
    out = resolve(cfg["outputs"]["phase1"])
    pd.set_option("display.width", 220)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    pidx = pd.read_csv(resolve(pcfg["data"]["index_csv"]))

    # 1. preprocessed files
    rows = []
    for r in pidx.itertuples():
        f = idx.loc[idx.video_id == r.video_id, "pre_path"]
        pa, fa = resolve(r.pre_path), resolve(f.iloc[0]) if len(f) else None
        rows.append(dict(video_id=r.video_id, pilot_exists=pa.exists(), full_exists=bool(fa and fa.exists()),
                         identical=bool(pa.exists() and fa and fa.exists() and same_npz(pa, fa))))
    fr = pd.read_csv(resolve(cfg["data"]["ref_index_csv"])).drop_duplicates("ref_id").set_index("ref_id").pre_path
    for r in pd.read_csv(resolve(pcfg["data"]["ref_index_csv"])).drop_duplicates("ref_id").itertuples():
        pa, fa = resolve(r.pre_path), resolve(fr[r.ref_id])
        rows.append(dict(video_id=r.ref_id, pilot_exists=pa.exists(), full_exists=fa.exists(),
                         identical=bool(pa.exists() and fa.exists() and same_npz(pa, fa))))
    P = pd.DataFrame(rows)
    P.to_csv(out / "compare_pilot_pre.csv", index=False)
    both = P[P.pilot_exists & P.full_exists]
    print(f"=== 1. E01/E02 preprocessed files: {len(both)} in both; identical {int(both.identical.sum())}; "
          f"pilot only {int((P.pilot_exists & ~P.full_exists).sum())}; full only {int((~P.pilot_exists & P.full_exists).sum())}")
    if a.only == "pre":
        return

    # 2. E01/E02 sync and reps
    G = pd.read_csv(resolve(cfg["sync"]["report_csv"]))
    Gp = pd.read_csv(resolve(pcfg["sync"]["report_csv"]))
    m = Gp.merge(G, on="group_id", suffixes=("_pilot", "_full"))
    ch = m[(m.cls_pilot != m.cls_full) | (m.method_pilot != m.method_full)]
    print(f"\n=== 2. E01/E02 sync groups: {len(m)} in both; class or method changed in {len(ch)}:")
    print(ch[["group_id", "cls_pilot", "method_pilot", "cls_full", "method_full", "aligned_views_full"]]
          .to_string(index=False) if len(ch) else "  none")
    V = pd.read_csv(out / "reps_per_video.csv")
    Vp = pd.read_csv(resolve(pcfg["outputs"]["phase1"]) / "reps_per_video.csv")
    mv = Vp.merge(V, on="video_id", suffixes=("_pilot", "_full"))
    mv["changed_group"] = mv.group_id_pilot.isin(ch.group_id)
    for col in ("n_reps", "n_reps_set", "n_reps_own", "side"):
        same = (mv[f"{col}_pilot"] == mv[f"{col}_full"])
        print(f"  {col}: identical in {same[~mv.changed_group].mean():.1%} of {int((~mv.changed_group).sum())} videos "
              f"of unchanged groups; {same[mv.changed_group].mean() if mv.changed_group.any() else np.nan:.1%} of "
              f"{int(mv.changed_group.sum())} videos of changed groups")
    R, Rp = pd.read_csv(data_path(cfg, "reps_csv")), pd.read_csv(data_path(pcfg, "reps_csv"))
    k = ["video_id", "rep", "start", "end"]
    mr = Rp[k].merge(R[R.exercise.isin(["E01", "E02"])][k], how="outer", indicator=True)
    print(f"  rep boundaries (video, rep, start, end): identical {int((mr._merge == 'both').sum())}; pilot only "
          f"{int((mr._merge == 'left_only').sum())}; full only {int((mr._merge == 'right_only').sum())}")

    # 3. E03-E09 vs Step 0
    S0 = pd.read_csv(resolve("outputs/step0_full/sync_groups_classified.csv"))
    S0["method"] = S0.method.replace({"creation_time": "metadata", "creation_time_start": "metadata_start"})
    g3 = G[~G.exercise.isin(["E01", "E02"])]
    mm = S0.merge(g3, on="group_id", suffixes=("_step0", "_p1"))
    print(f"\n=== 3. E03-E09 sync classes, Phase 1 vs Step 0: {len(mm)} groups; same class {(mm.cls_step0 == mm.cls_p1).mean():.1%}")
    dd = mm[(mm.cls_step0 != mm.cls_p1) | (mm.method_step0 != mm.method_p1)]
    if len(dd):
        print(dd.groupby(["cls_step0", "method_step0", "cls_p1", "method_p1"]).size().to_string())
        dd[["group_id", "cls_step0", "method_step0", "cls_p1", "method_p1", "motion_confirmed_pairs", "n_pairs"]].to_csv(
            out / "compare_step0_sync_changes.csv", index=False)
    rows = []
    pf1 = idx.groupby(["exercise", "subject"]).PF1.mean()
    low = {k for k, v in pf1.items() if v < 0.5}   # left out as in Step 0 (E03 P02, E09 P21)
    for ex, gx in V[V.exercise.isin(STEP0_WITHIN1)].merge(G[["group_id", "cls"]], on="group_id").groupby("exercise"):
        e = gx[(gx.group == "expert") & (gx.cls == "synced")]
        e = e[[(ex, s) not in low for s in e.subject]]
        per = e.groupby("group_id").agg(nv=("view", "nunique"), lo=("n_reps_own", "min"), hi=("n_reps_own", "max"),
                                        lo_t=("n_reps", "min"), hi_t=("n_reps", "max"), med=("n_reps_set", "median"))
        per = per[per.nv == 3]
        rows.append(dict(exercise=ex, groups=len(per), within1_perview=((per.hi - per.lo) <= 1).mean(),
                         exact_perview=(per.hi == per.lo).mean(), step0_sample_within1=STEP0_WITHIN1[ex],
                         identical_after_transfer=(per.hi_t == per.lo_t).mean(), median_set_reps=per.med.median()))
    T = pd.DataFrame(rows).round(3)
    T.to_csv(out / "compare_step0_rep_agreement.csv", index=False)
    print("\ncross-view rep agreement, all synced expert F+L+R groups (per-view segmentation, before transfer):")
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
