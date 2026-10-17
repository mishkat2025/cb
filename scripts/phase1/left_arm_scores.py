"""Supervisor answer 2: are expert scores (total, PF, CF) lower for left-arm performances?

Unit = one performance (exercise, subject, variation): its views are averaged, because they are the same
take. Arm side = the group's moving arm (best view of a synced group, else the video's own; majority over
views if they differ). PF = mean of the judge-mean items PF1-PF4 (0-1), CF = mean of CF1-CF3.
Tests per exercise and score: Mann-Whitney U, one-sided (left < right), Holm-corrected over the
2 exercises x 3 scores; rank-biserial effect size; within-subject check on subjects with both sides.
Decision (configured in Phase 4): add an ``arm_side`` flag to the Module III output only if some score is
significantly lower for left-arm performances after Holm correction.
usage: python scripts/phase1/left_arm_scores.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvdc.config import load_config, resolve  # noqa: E402

PF, CF = ["PF1", "PF2", "PF3", "PF4"], ["CF1", "CF2", "CF3"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    cfg = load_config()
    out = resolve(cfg["outputs"]["phase1"])
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    V = pd.read_csv(out / "reps_per_video.csv")[["video_id", "side"]]
    d = idx[idx.group == "expert"].merge(V, on="video_id", how="left")
    d["PF"], d["CF"] = d[PF].mean(axis=1), d[CF].mean(axis=1)
    perf = d.groupby(["exercise", "subject", "group_id"]).agg(
        side=("side", lambda s: s.mode().iloc[0] if s.notna().any() else np.nan), views=("view", "size"),
        score=("score", "mean"), PF=("PF", "mean"), CF=("CF", "mean"),
        **{c: (c, "mean") for c in PF + CF}).reset_index().dropna(subset=["side"])
    perf.to_csv(out / "left_arm_performances.csv", index=False)
    rows = []
    for ex, g in perf.groupby("exercise"):
        L, R = g[g.side == "left"], g[g.side == "right"]
        for col in ("score", "PF", "CF"):
            u = mannwhitneyu(L[col], R[col], alternative="less")
            both = g.groupby(["subject", "side"])[col].mean().unstack().dropna()
            w = wilcoxon(both["left"], both["right"], alternative="less") if len(both) >= 3 else None
            rows.append(dict(exercise=ex, score=col, n_left=len(L), n_right=len(R),
                             subjects_left=L.subject.nunique(), mean_left=L[col].mean(), mean_right=R[col].mean(),
                             median_left=L[col].median(), median_right=R[col].median(),
                             diff_mean=L[col].mean() - R[col].mean(), p_mwu_less=u.pvalue,
                             rank_biserial=1 - 2 * u.statistic / (len(L) * len(R)),
                             within_subjects=len(both), within_diff_mean=(both["left"] - both["right"]).mean()
                             if len(both) else np.nan, p_within_less=w.pvalue if w is not None else np.nan))
    T = pd.DataFrame(rows)
    T["p_holm"] = holm(T.p_mwu_less)
    T["lower_for_left"] = T.p_holm < 0.05
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    print("\nleft-arm performances per subject:",
          perf[perf.side == "left"].groupby(["exercise", "subject"]).size().to_dict())
    items = perf.groupby(["exercise", "side"])[PF + CF].mean().round(3)
    print("\nitem means (0-1) by side:\n" + items.to_string())
    T.to_csv(out / "left_arm_scores.csv", index=False)
    print(f"\ndecision: arm_side flag {'NEEDED' if T.lower_for_left.any() else 'not needed'} "
          f"(Holm-corrected one-sided Mann-Whitney, alpha 0.05)")


if __name__ == "__main__":
    main()
