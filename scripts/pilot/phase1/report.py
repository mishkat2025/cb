"""Phase 1 summary tables (printed and saved to outputs/phase1/): detection and visibility per view and
variation, hip visibility (root choice), interpolation, rep counts vs PF4 and across views, rep timing,
sync classes and CVDC pair counts.
usage: python scripts/pilot/phase1/report.py [--config configs/data.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cvdc.config import data_path, load_config, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    out = resolve(cfg["outputs"]["phase1"])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    ex = pd.read_csv(out / "extract_summary.csv")
    pre = pd.read_csv(out / "preprocess_summary.csv")
    V = pd.read_csv(out / "reps_per_video.csv")
    R = pd.read_csv(data_path(cfg, "reps_csv"))
    R = R.merge(V[["video_id", "n_frames_30"]].rename(columns={"n_frames_30": "n_frames"}), on="video_id")
    A = idx.merge(ex.drop(columns=["exercise", "group", "view", "variation"]), on="video_id", how="left") \
           .merge(pre.drop(columns=["det_rate"], errors="ignore"), on="video_id", how="left") \
           .merge(V[["video_id", "side", "n_reps", "n_reps_set", "n_reps_own", "rep_source", "best_view",
                     "range_moving", "range_other", "unit"]], on="video_id", how="left")
    A.to_csv(out / "phase1_per_video.csv", index=False)

    print("=== 1. keypoints:", int(A.n_frames.notna().sum()), "/", len(A), "videos;",
          f"frames {int(A.n_frames.sum())}; mean frame detection {A.det_rate.mean():.1%}")
    t1 = A.groupby(["exercise", "variation"]).agg(videos=("video_id", "size"), det_mean=("det_rate", "mean"),
                                                  det_min=("det_rate", "min"),
                                                  below90=("det_rate", lambda s: int((s < 0.9).sum()))).round(3)
    print(t1.to_string())
    t1.to_csv(out / "table_detection_by_variation.csv")

    print("\n=== 2. hips visible (both >= 0.5) per view, and low-visibility share of upper-body joints")
    t2 = A.groupby(["exercise", "group", "view"]).agg(videos=("video_id", "size"), hips_vis_ok=("hips_vis_ok", "mean"),
                                                      hips_min=("hips_vis_ok", "min"), upper_lowvis=("lowvis_frac_upper", "mean"),
                                                      interp=("interp_frac_upper", "mean"),
                                                      long_gap=("long_gap_frac_upper", "mean")).round(3)
    print(t2.to_string())
    t2.to_csv(out / "table_visibility_by_view.csv")

    print("\n=== 3. reps per video")
    t3 = A.groupby(["exercise", "group", "view"]).n_reps.describe()[["count", "mean", "min", "25%", "50%", "75%", "max"]]
    print(t3.round(2).to_string())
    t3.to_csv(out / "table_reps_by_view.csv")
    print("set rep count (best view of a synced group, else own):")
    print(A.groupby(["exercise", "group"]).n_reps_set.describe()[["count", "mean", "min", "50%", "max"]].round(2).to_string())
    print("moving arm (auto):", A.groupby(["exercise", "group"]).side.value_counts().unstack(fill_value=0).to_dict("index"))
    print("videos with 0 reps:", A[A.n_reps == 0].groupby(["exercise", "group", "view"]).size().to_dict())

    print("\n=== 4. set rep count vs PF4 (Repetition item, 0.1 steps)")
    rows = []
    for (e, g), d in A[A.n_reps_set.notna()].groupby(["exercise", "group"]):
        rho = spearmanr(d.n_reps_set, d.PF4).statistic if d.PF4.nunique() > 1 else np.nan
        full = d[d.PF4 >= 0.999]
        part = d[d.PF4 < 0.999]
        rows.append(dict(exercise=e, group=g, videos=len(d), spearman_reps_PF4=rho,
                         PF4_eq_1=len(full), of_which_reps_ge_10=(full.n_reps_set >= 10).mean() if len(full) else np.nan,
                         PF4_lt_1=len(part), abs_diff_reps_vs_10xPF4_median=(part.n_reps_set - 10 * part.PF4).abs().median()
                         if len(part) else np.nan, within1_of_10xPF4=((part.n_reps_set - 10 * part.PF4).abs() <= 1).mean()
                         if len(part) else np.nan))
    t4 = pd.DataFrame(rows).round(3)
    print(t4.to_string(index=False))
    t4.to_csv(out / "table_reps_vs_PF4.csv", index=False)

    print("\n=== 5. cross-view rep agreement (multi-view groups; expert groups = one performance)")
    G = pd.read_csv(resolve(cfg["sync"]["report_csv"]))[["group_id", "cls"]]
    m = A[A.n_reps.notna()].merge(G, on="group_id", how="left").fillna({"cls": "single"})
    m = m[m.groupby("group_id").video_id.transform("size") >= 2]
    rows = []
    for (e, cls), d in m.groupby(["exercise", "cls"]):
        for col, label in (("n_reps_own", "per-view segmentation (before)"), ("n_reps", "training reps (after)"),
                           ("n_reps_set", "set rep count (after)")):
            g = d.groupby("group_id")[col].agg(["min", "max"])
            rows.append(dict(exercise=e, group_cls=cls, counts=label, groups=len(g),
                             equal=(g["min"] == g["max"]).mean(), within1=((g["max"] - g["min"]) <= 1).mean()))
    t5 = pd.DataFrame(rows).round(3)
    print(t5.to_string(index=False))
    t5.to_csv(out / "table_reps_cross_view.csv", index=False)

    print("\n=== 5b. what changed per group (views summed)")
    d = m.assign(seg_change=(m.n_reps_set - m.n_reps_own).abs(), dropped=m.n_reps_set - m.n_reps)
    ch = d.groupby(["exercise", "cls", "group_id"]).agg(
        seg_change=("seg_change", "sum"), dropped=("dropped", "sum"), before=("n_reps_own", "sum"),
        after=("n_reps", "sum")).reset_index()
    t5b = ch.groupby(["exercise", "cls"]).agg(
        groups=("group_id", "size"), reps_before=("before", "sum"), reps_after=("after", "sum"),
        groups_count_changed=("seg_change", lambda x: int((x > 0).sum())), count_change_median=("seg_change", "median"),
        count_change_p90=("seg_change", lambda x: x.quantile(.9)), count_change_max=("seg_change", "max"),
        groups_with_dropped=("dropped", lambda x: int((x > 0).sum())), reps_dropped=("dropped", "sum")).round(1)
    print("count_change = sum over views of |set count - per-view count| (segmentation differences);")
    print("reps_dropped = reps not kept because some aligned view did not record them (summed over views)")
    print(t5b.to_string())
    t5b.to_csv(out / "table_reps_changes.csv")
    tv = d[d.rep_source.isin(["best_view", "transfer"])].groupby(["exercise", "rep_source", "view"]).agg(
        videos=("video_id", "size"), mean_count_change=("seg_change", "mean"), reps_dropped=("dropped", "sum"))
    print(tv.round(2).to_string())
    # identical boundaries by construction: a rep_uid has the same aligned start/end in every view (+-1 frame)
    sr = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    Rs = R[R.rep_source.isin(["best_view", "transfer"])].copy()
    fps = cfg["preprocess"]["fps"]
    offs = np.array([sr.loc[gid, f"offset_{v}_s"] for gid, v in zip(Rs.group_id, Rs.view)])
    Rs["a_start"], Rs["a_end"] = Rs.start / fps - offs, Rs.end / fps - offs
    spread = Rs.groupby("rep_uid")[["a_start", "a_end"]].agg(lambda x: x.max() - x.min()).max(axis=1) * fps
    inner = Rs.groupby("rep_uid").apply(lambda x: ((x.start > 0) & (x.end < x.n_frames - 1)).all(), include_groups=False)
    print(f"boundary check: {len(spread)} shared reps; max spread of aligned boundaries across views "
          f"(reps not clipped at a recording edge): {spread[inner[spread.index]].max():.2f} frames")

    print("\n=== 6. rep timing (s): duration, raise (10->90%), hold, lower (90->10%)")
    R = R.merge(idx[["video_id", "group"]], on="video_id", how="left", suffixes=("", "_i"))
    t6 = R.groupby(["exercise", "group"])[["duration_s", "raise_s", "hold_s", "lower_s", "amplitude", "peak_val"]] \
          .quantile([0.05, 0.5, 0.95]).unstack().round(2)
    print(t6.to_string())
    t6.to_csv(out / "table_rep_timing.csv")
    hi = R[R.video_id.isin(idx.video_id[idx.score >= 95])]
    t6b = hi.groupby("exercise")[["duration_s", "raise_s", "hold_s", "lower_s"]].quantile([0.05, 0.5, 0.95]).unstack().round(2)
    print(f"expert-data speed reference (reps of videos scoring >= 95; {hi.video_id.nunique()} videos):")
    print(t6b.to_string())
    t6b.to_csv(out / "table_rep_timing_score95.csv")
    T = cfg["preprocess"]["T_rep"]
    print(f"rep length in frames at {cfg['preprocess']['fps']} fps: p5/p50/p95 =",
          np.percentile(R.duration_s * cfg["preprocess"]["fps"], [5, 50, 95]).round(0), f"(T_rep = {T})")

    rp = resolve(cfg["sync"]["report_csv"])
    if rp.exists():
        G = pd.read_csv(rp)
        P = pd.read_csv(resolve(cfg["sync"]["pairs_csv"]))
        print("\n=== 7. sync classes (groups with >= 2 views)")
        t7 = G.groupby(["exercise", "group", "cls", "method"]).size().unstack(["cls", "method"], fill_value=0)
        print(t7.to_string())
        t7.to_csv(out / "table_sync_classes.csv")
        P["exercise"] = P.group_id.str[:3]
        t8 = P.groupby(["exercise", "pair_source", "lag_source"]).size().unstack(fill_value=0)
        print("\nview pairs by CVDC source / lag source:\n" + t8.to_string())
        t8.to_csv(out / "table_sync_pair_sources.csv")


if __name__ == "__main__":
    main()
