"""Phase 3 step (a2), part 1: label diagnostics and set-level baselines (no training).

Per stratum (E01 expert, E02 expert, E01 non-expert; the evaluated videos with >= 1 training rep, E15):
  items          variance of the total score and of each EAAQ item (PF1-PF4, CF1-CF3), how often an item is off its
                 modal value (videos, subjects), and whether it varies within the group (T14)
  decomposition  the share of the within-group score variance each item explains. The total score is exactly
                 (100 / 7) x the sum of the 7 items, so share_k = Cov(score, (100/7) item_k) / Var(score) splits it
                 exactly (shares sum to 1); the single-item R^2 is shown next to it
  set features   rep count (Phase 1 set count), mean rep duration, mean ROM of the main angle (E01 3D shoulder
                 elevation, E02 3D horizontal adduction)
  baselines      within-group regressions on the same GroupKFold-5 folds as the models (E20): rep-count-only,
                 duration-within, ROM-within, set-features (all three), next to predict-the-mean, group-only and
                 duration-only (E16), all fitted on the training fold's videos; video level, 95% CIs from the
                 subject bootstrap (E17)
Writes outputs/results/phase3a2/: label_items.csv, score_decomposition.csv, set_features_summary.csv,
baseline_metrics.csv, baseline_predictions.csv, label_diagnostics.md.

usage: python scripts/pilot/phase3/label_diagnostics.py [--out DIR] [--boot 2000] [--refresh-features]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.eval.baselines import (ALL_ITEMS, SET_FEATURES, STRATA, VIDEO_BASELINES, item_activity,  # noqa: E402
                                 score_decomposition, set_features, video_baselines, video_table)
from cvdc.eval.metrics import interval, item_weights, regression, subject_draws  # noqa: E402
from cvdc.train.trainer import fold_plan  # noqa: E402

TARGETS = ["score", "PF1", "PF2", "PF4", "CF1", "CF2", "CF3"]  # D5: PF3 is not a target
LABEL = {"base_mean": "predict-the-mean", "base_group": "group-only (within-group mean)",
         "base_duration": "duration-only (per exercise, E16)", "base_repcount": "rep-count-only (within group)",
         "base_duration_within": "mean rep duration (within group)", "base_rom_within": "ROM of the main angle (within group)",
         "base_setfeat": "set-features: count + duration + ROM (within group)"}


def fmt(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def stratum_metrics(v, preds, targets, active, n_boot, seed, ci):
    """Within-group metrics of every baseline (video level, pooled over folds) with subject-bootstrap CIs."""
    rows = []
    for sname, ex, g in STRATA:
        d = v[(v.exercise == ex) & (v.group == g)].sort_values("video_id")
        subj = d.drop_duplicates("subject")[["subject", "group"]].sort_values("subject")
        M = subject_draws(subj.subject.to_numpy(), subj.group.to_numpy(), n_boot, seed)
        W = np.vstack([np.ones(len(d)), item_weights(d.subject.to_numpy(), subj.subject.to_numpy(), M)])
        for m in VIDEO_BASELINES:
            pm = preds[preds.method == m].set_index("video_id").loc[d.video_id]
            for t in targets:
                if t != "score" and t not in active[(ex, g)]:
                    continue
                y, p = d[t].to_numpy(float), pm[f"{t}_pred"].to_numpy(float)
                r = regression(y, p, W)
                var = (W * (y - (W * y).sum(1, keepdims=True) / W.sum(1, keepdims=True)) ** 2).sum(1) / W.sum(1)
                with np.errstate(divide="ignore", invalid="ignore"):  # a draw can hold one constant subject
                    r2 = 1 - r["rmse"] ** 2 / var
                row = dict(stratum=sname, method=m, target=t, n=len(d), subjects=len(subj))
                for k, val in {**r, "r2_oof": r2}.items():
                    row[k] = val[0]
                    row[f"{k}_lo"], row[f"{k}_hi"] = interval(val[1:], ci)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/results/phase3a2")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--refresh-features", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    ec, ia = cfg["eval"], cfg["heads"]["item_activity"]
    n_boot, ci = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"]
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)

    v = video_table(cfg, set_features(cfg, refresh=a.refresh_features))
    reps = pd.read_csv(resolve("data/reps.csv"))
    plans = fold_plan(cfg, reps)  # the models' folds (rep-level subjects, E1)

    act = item_activity(v, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    act.to_csv(out / "label_items.csv", index=False)
    dec = score_decomposition(v)
    dec.to_csv(out / "score_decomposition.csv", index=False)
    active = {(e, g): list(d[d.varies & d.item.isin(TARGETS)].item) for (e, g), d in act.groupby(["exercise", "group"])}

    frows = []
    for sname, ex, g in STRATA:
        d = v[(v.exercise == ex) & (v.group == g)]
        for f in SET_FEATURES:
            x = d[f]
            ok = x.notna()
            rho = pd.Series(x[ok]).rank().corr(d.score[ok].rank())
            frows.append(dict(stratum=sname, feature=f, mean=x.mean(), sd=x.std(ddof=0), min=x.min(), max=x.max(),
                              missing=int((~ok).sum()), spearman_with_score=rho))
        frows.append(dict(stratum=sname, feature="angle_defined_frac", mean=d.angle_defined_frac.mean(),
                          sd=d.angle_defined_frac.std(ddof=0), min=d.angle_defined_frac.min(),
                          max=d.angle_defined_frac.max(), missing=0, spearman_with_score=np.nan))
    F = pd.DataFrame(frows)
    F.to_csv(out / "set_features_summary.csv", index=False)

    preds = video_baselines(v, plans, TARGETS)
    preds.to_csv(out / "baseline_predictions.csv", index=False)
    assert preds.groupby("method").video_id.nunique().eq(len(v)).all(), "every video must be predicted once"
    B = stratum_metrics(v, preds, TARGETS, active, n_boot, ec["bootstrap"]["seed"], ci)
    B.to_csv(out / "baseline_metrics.csv", index=False)

    # ---- markdown
    L = []
    w = L.append
    w("# Phase 3 step (a2): label diagnostics and set-level baselines (no training)")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/pilot/phase3/label_diagnostics.py`. Units: the {len(v)} "
      "evaluated videos (≥ 1 training rep, E15). Variances are population variances (ddof 0) within the stratum. "
      "Score 0–100; items 0–1; 'SD (pts)' = (100/7) × SD, the item's spread in score points. The total score is "
      f"exactly (100/7) × Σ items (largest deviation {dec.max_reconstruction_error.max():.4f} points, rounding), so "
      "the item shares (Cov(score, (100/7) item) / Var(score)) add up to 100%. 'Varies' (T14): off the stratum's "
      f"modal value in videos of ≥ {ia['min_subjects_off_mode']} subjects (the number of folds) and SD ≥ {ia['min_sd']}.")
    w("")
    for sname, ex, g in STRATA:
        d = v[(v.exercise == ex) & (v.group == g)]
        dd = dec[dec.stratum == sname].set_index("item")
        aa = act[act.stratum == sname].set_index("item")
        w(f"## {sname}: {len(d)} videos, {d.subject.nunique()} subjects; score variance {dd.score_var.iloc[0]:.1f} "
          f"(SD {dd.score_sd.iloc[0]:.2f})")
        w("")
        w("| Item | mean | SD | SD (pts) | variance | share of score variance | single-item R² | off-mode videos / subjects | varies (T14) |")
        w("|---|---|---|---|---|---|---|---|---|")
        for k in ALL_ITEMS:
            r, s = aa.loc[k], dd.loc[k]
            note = "" if k != "PF3" else " (not a target, D5)"
            w(f"| {k}{note} | {r['mean']:.3f} | {r.sd:.3f} | {r.sd_points:.2f} | {r.sd ** 2:.4f} | {100 * s.share:.1f}% | "
              f"{fmt(s.r2_single)} | {r.off_mode_videos} / {r.off_mode_subjects} | {'yes' if r.varies else '**no**'} |")
        pf = dd.loc[["PF1", "PF2", "PF3", "PF4"], "share"].sum()
        rep = dd.loc[["PF3", "PF4"], "share"].sum()
        w("")
        w(f"PF items together {100 * pf:.1f}% (repetition items PF3 + PF4: {100 * rep:.1f}%), CF items "
          f"{100 * (1 - pf):.1f}%.")
        w("")
    w("## Set-level features (per video)")
    w("")
    w("Rep count = Phase 1 set count (`n_reps_set`, best view, P6b); rep duration = mean over the video's training "
      "reps; ROM = mean over those reps of max − min of the main angle (E01 3D shoulder elevation; E02 3D horizontal "
      "adduction, undefined where the upper arm is near vertical, M7a). Spearman with the score is in-sample.")
    w("")
    w("| Stratum | feature | mean | SD | min | max | missing | Spearman with score |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in F.iterrows():
        w(f"| {r.stratum} | {r.feature} | {fmt(r['mean'])} | {fmt(r.sd)} | {fmt(r['min'])} | {fmt(r['max'])} | "
          f"{r.missing} | {fmt(r.spearman_with_score)} |")
    w("")
    w("## Within-group baselines (GroupKFold-5, same folds as the models, video level)")
    w("")
    w("RMSE [95% CI] / Spearman ρ / out-of-fold R² (1 − MSE / within-stratum variance; ≤ 0 means no better than "
      "the stratum's pooled mean). Score 0–100, items 0–1. Only items that vary within the group (T14) are shown.")
    w("")
    for sname, ex, g in STRATA:
        tt = ["score", *active[(ex, g)]]
        w(f"### {sname}")
        w("")
        w("| Baseline | " + " | ".join(tt) + " |")
        w("|---|" + "---|" * len(tt))
        for m in VIDEO_BASELINES:
            cells = []
            for t in tt:
                r = B[(B.stratum == sname) & (B.method == m) & (B.target == t)].iloc[0]
                d = 2 if t == "score" else 3
                cells.append(f"{fmt(r.rmse, d)} [{fmt(r.rmse_lo, d)}, {fmt(r.rmse_hi, d)}] / {fmt(r.spearman)} / {fmt(r.r2_oof)}")
            w(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
        no = act[(act.stratum == sname) & ~act.varies & act.item.isin(TARGETS)]
        if len(no):
            w("")
            w("Not predicted (≈ 0 within-group variance, T14): " + "; ".join(
                f"{r.item} (SD {r.sd:.3f}, off-mode in {r.off_mode_videos} videos of {r.off_mode_subjects} subjects)"
                for r in no.itertuples()) + ".")
        w("")
    (out / "label_diagnostics.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"written to {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
