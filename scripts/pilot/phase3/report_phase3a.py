"""Phase 3 step (a) report (E14): GroupKFold-5 results of B1, B2-MediaPipe, B2-KTPFormer and Proposed.

Reads runs/<tag>/<method>/fold<k>/ (cvdc.train.run): test_predictions.csv (rep level), embeddings_*.npz, done.json
and the stage summaries. Test predictions of the 5 folds are pooled (E1). Reports:
  score metrics   RMSE, MAE, Pearson, Spearman for the total score (0-100) and CF1-CF3 (0-1), within groups first
                  (E01 expert, E02 expert, E01 non-expert), then pooled (E01 both groups; all), at video level
                  (mean of the video's rep predictions; headline, E15) and rep level
  baselines       fitted on each training fold (E10, E16): predict-the-mean, group-only (mean per exercise x group),
                  duration-only (per-exercise least squares on the rep duration of data/reps.csv); aggregated to
                  video level like the models
  noise floor     E3 (score; CF items with the same method, from data/index.csv)
  paired deltas   RMSE of each method minus the group-only baseline and minus B1, same bootstrap draws
  classification  derived quality class (quality head; also the binned predicted score): macro F1, Cohen's kappa;
                  exercise accuracy (secondary, E6)
  VCI (E13)       centred 256-d (headline), raw 256-d, 128-d projection; within exercise x group and pooled; for
                  the random-init encoder (reference), Proposed after CVDC pretraining, and every final model
  CIs (E17)       95% percentile intervals from a subject-level bootstrap (2,000 draws, subjects resampled within
                  expert / non-expert)
  training        stopping epochs and wall time per method and fold
Writes CSV tables and summary.md to <eval.results_dir>/<tag> (default outputs/results/phase3a).

usage: python scripts/pilot/phase3/report_phase3a.py [--tag phase3a] [--out DIR] [--boot 2000] [--methods ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.eval.metrics import (add_sums, classification, interval, item_weights, regression, subject_draws,  # noqa: E402
                               vci_from_sums, vci_sums)
from cvdc.train.trainer import fold_plan  # noqa: E402

METHODS = ["b1", "b2_mediapipe", "b2_ktpformer", "proposed"]
BASELINES = ["base_mean", "base_group", "base_duration"]
LABEL = {"b1": "B1 (2D xy, scratch)", "b2_mediapipe": "B2-MediaPipe (world xyz, scratch)",
         "b2_ktpformer": "B2-KTPFormer (lifted xyz, scratch)", "proposed": "Proposed (xy, CVDC + fine-tune)",
         "base_mean": "baseline: predict-the-mean", "base_group": "baseline: group-only",
         "base_duration": "baseline: duration-only"}
SHORT = {"b1": "B1", "b2_mediapipe": "B2-MP", "b2_ktpformer": "B2-KTP", "proposed": "Proposed",
         "base_mean": "mean", "base_group": "group-only", "base_duration": "duration-only"}
TARGETS = ["score", "CF1", "CF2", "CF3"]
STRATA = [("E01 expert", "E01", "expert"), ("E02 expert", "E02", "expert"), ("E01 non-expert", "E01", "non-expert"),
          ("E01 pooled", "E01", None), ("pooled", None, None)]
WITHIN = ["E01 expert", "E02 expert", "E01 non-expert"]


def stratum_mask(df, ex, grp):
    m = np.ones(len(df), bool)
    if ex:
        m &= (df.exercise == ex).to_numpy()
    if grp:
        m &= (df.group == grp).to_numpy()
    return m


def noise_floors(index, items=("score", "CF1", "CF2", "CF3")):
    """E3: RMSE of each view's label against its own (exercise, subject, variation) group mean, over expert groups
    with >= 2 views (simultaneous recordings of one performance)."""
    d = index[index.group == "expert"]
    rows = []
    for ex, de in d.groupby("exercise"):
        g = de.groupby("group_id")
        n = g.size()
        keep = n[n >= 2].index
        row = {"exercise": ex, "groups": len(keep)}
        for k in items:
            v = g[k].var(ddof=0)[keep]
            row[k] = float(np.sqrt((v * n[keep]).sum() / n[keep].sum()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("exercise")


def fit_baselines(meta, plans, items):
    """Rep-level predictions of the three baselines for every test rep, fitted on the fold's training reps."""
    out = []
    for p in plans:
        tr = meta[meta.subject.isin(p["train"])]
        te = meta[meta.subject.isin(p["test"])]
        for name in BASELINES:
            d = te[["item"]].copy()
            d["fold"], d["method"] = p["fold"], name
            for k in ["score", *items]:
                if name == "base_mean":
                    d[f"{k}_pred"] = tr[k].mean()
                elif name == "base_group":
                    gm = tr.groupby(["exercise", "group"])[k].mean()
                    d[f"{k}_pred"] = [gm.get((e, g), tr[k].mean()) for e, g in zip(te.exercise, te.group)]
                else:
                    pred = np.full(len(te), np.nan)
                    for ex, de in tr.groupby("exercise"):
                        b, a = np.polyfit(de.duration_s.to_numpy(), de[k].to_numpy(), 1)
                        sel = (te.exercise == ex).to_numpy()
                        pred[sel] = a + b * te.duration_s.to_numpy()[sel]
                    d[f"{k}_pred"] = pred
            out.append(d)
    return pd.concat(out, ignore_index=True)


def load_runs(root, methods):
    preds, done = [], {}
    for m in methods:
        for f in sorted((root / m).glob("fold*/done.json")):
            fd = f.parent
            d = json.loads(f.read_text(encoding="utf-8"))
            done[(m, d["fold"])] = d
            p = pd.read_csv(fd / "test_predictions.csv")
            p["fold"], p["method"] = d["fold"], m
            preds.append(p)
    return pd.concat(preds, ignore_index=True), done


def to_video(df, meta_cols):
    """Mean of the rep predictions per video (and of the class probabilities); labels are per video."""
    pred_cols = [c for c in df.columns if c.endswith("_pred") and c not in ("quality_pred", "exercise_pred")]
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    agg = df.groupby(["method", "fold", "video_id"], sort=False)[pred_cols + prob_cols].mean().reset_index()
    agg["n_reps"] = df.groupby(["method", "fold", "video_id"], sort=False).size().to_numpy()
    agg = agg.merge(meta_cols, on="video_id", how="left", validate="many_to_one")
    if any(c.startswith("p_quality_") for c in prob_cols):
        q = sorted(c for c in prob_cols if c.startswith("p_quality_"))
        agg["quality_pred"] = np.where(agg[q[0]].notna(), agg[q].fillna(0).to_numpy().argmax(1), -1)
        e = sorted(c for c in prob_cols if c.startswith("p_ex_"))
        agg["exercise_pred"] = np.where(agg[e[0]].notna(), np.array([c[5:] for c in e])[agg[e].fillna(0).to_numpy().argmax(1)], "")
    return agg


def fmt(v, d=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def fci(v, lo, hi, d=2):
    return f"{fmt(v, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--tag", default="phase3a")
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--methods", nargs="+", default=METHODS, help="subset of the methods (all must have the same folds)")
    a = ap.parse_args()
    t_start = time.time()
    cfg = load_config(a.config, a.model_config)
    ec = cfg["eval"]
    n_boot = a.boot or ec["bootstrap"]["n"]
    ci = ec["bootstrap"]["ci"]
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out = Path(a.out) if a.out else resolve(ec["results_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    smoke = a.tag.endswith("_smoke")

    # rep-level meta = PackData's (data/reps.csv joined with data/index.csv), without loading the packs
    index = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    reps = pd.read_csv(resolve("data/reps.csv"))
    lab = ["exercise", "subject", "group", "view", "variation", "group_id", "score", "quality", "CF1", "CF2", "CF3"]
    meta = reps.drop(columns=[c for c in lab if c in reps.columns]).merge(index[["video_id", *lab]], on="video_id",
                                                                         how="left", validate="many_to_one")
    meta["item"] = np.arange(len(meta))
    plans = fold_plan(cfg, meta, smoke=smoke)
    qn = cfg["labels"]["quality_names"]
    meta["quality_i"] = meta.quality.map(qn.index)
    bins = cfg["labels"]["quality_bins"]
    assert (np.digitize(meta.score, bins) == meta.quality_i).all(), "quality bins do not reproduce the labels"
    items = ["CF1", "CF2", "CF3"]

    preds, done = load_runs(root, a.methods)
    methods = [m for m in METHODS if m in set(preds.method)]
    folds = sorted({f for (_, f) in done})
    plans = [p for p in plans if p["fold"] in folds]
    for (m, f), d in done.items():  # the runs used exactly these folds and inner splits
        p = next(q for q in plans if q["fold"] == f)
        for k in ("train", "inner_train", "inner_val", "test"):
            assert sorted(d["subjects"][k]) == sorted(p[k]), (m, f, k)
    base = fit_baselines(meta, plans, items)
    keep = ["item", "fold", "method", "score_pred", *[f"{k}_pred" for k in items]]
    allp = pd.concat([preds[keep + [c for c in preds.columns if c.startswith("p_") or c in ("quality_pred", "exercise_pred")]],
                      base], ignore_index=True)
    allp = allp.merge(meta[["item", "video_id", "subject", "group", "exercise", "view", "variation", "duration_s",
                            "score", "quality_i", *items]], on="item", how="left", validate="many_to_one")
    allp["quality_from_score"] = np.digitize(allp.score_pred, bins)
    all_methods = methods + BASELINES
    n_items = allp.groupby("method").size()
    assert n_items.nunique() == 1, f"methods predict different test sets: {n_items.to_dict()}"

    vmeta = index[["video_id", "subject", "group", "exercise", "view", "variation", "score", "quality", *items]].copy()
    vmeta["quality_i"] = vmeta.quality.map(qn.index)
    rep_level = allp
    vid_level = to_video(allp.drop(columns=["subject", "group", "exercise", "view", "variation", "score",
                                            "quality_i", *items, "duration_s"]), vmeta)
    vid_level["quality_from_score"] = np.digitize(vid_level.score_pred, bins)
    floors = noise_floors(index)
    floors.to_csv(out / "noise_floor.csv")

    # ---- score metrics with paired bootstrap CIs
    score_rows, delta_rows, fold_rows, cls_rows = [], [], [], []
    for level, df in (("video", vid_level), ("rep", rep_level)):
        key = "video_id" if level == "video" else "item"
        for sname, ex, grp in STRATA:
            d = df[stratum_mask(df, ex, grp)]
            if d.empty:
                continue
            ref = d[d.method == all_methods[0]].sort_values(key)
            subj_tab = ref.drop_duplicates("subject")[["subject", "group"]].sort_values("subject")
            M = subject_draws(subj_tab.subject.to_numpy(), subj_tab.group.to_numpy(), n_boot, ec["bootstrap"]["seed"])
            W = np.vstack([np.ones(len(ref)), item_weights(ref.subject.to_numpy(), subj_tab.subject.to_numpy(), M)])
            reps_by = {}
            for m in all_methods:
                dm = d[d.method == m].sort_values(key)
                assert (dm[key].to_numpy() == ref[key].to_numpy()).all()
                for t in TARGETS:
                    r = regression(dm[t].to_numpy(), dm[f"{t}_pred"].to_numpy(), W)
                    reps_by[(m, t)] = r
                    row = dict(level=level, stratum=sname, method=m, target=t, n=len(dm), subjects=len(subj_tab))
                    for k, v in r.items():
                        row[k] = v[0]
                        row[f"{k}_lo"], row[f"{k}_hi"] = interval(v[1:], ci)
                    # E3 floors are measured on expert groups (same performance, several views); the non-expert
                    # views are separate performances, so the floor is only a reference outside expert strata
                    fl = ec["noise_floor"] if t == "score" else floors[t].to_dict()
                    row["noise_floor"] = fl.get(ex, np.nan) if ex and grp == "expert" else np.nan
                    row["noise_floor_ref"] = " / ".join(fmt(fl[e], 2 if t == "score" else 3) for e in ([ex] if ex else sorted(fl)))
                    score_rows.append(row)
                # per-fold score RMSE (pooled metrics are primary; this shows the spread)
                for f, dmf in dm.groupby("fold"):
                    e = dmf.score_pred - dmf.score
                    fold_rows.append(dict(level=level, stratum=sname, method=m, fold=f, n=len(dmf),
                                          rmse=float(np.sqrt((e ** 2).mean())), mae=float(e.abs().mean())))
                # classification (secondary)
                for src in ("quality_pred", "quality_from_score"):
                    if src == "quality_pred" and m in BASELINES:
                        continue
                    c = classification(dm.quality_i.to_numpy(), dm[src].to_numpy(), W, len(qn),
                                       labels=sorted(set(dm.quality_i) | set(dm[src])))
                    row = dict(level=level, stratum=sname, method=m, prediction=src, n=len(dm))
                    for k, v in c.items():
                        row[k] = v[0]
                        row[f"{k}_lo"], row[f"{k}_hi"] = interval(v[1:], ci)
                    cls_rows.append(row)
                if m not in BASELINES:
                    acc = (dm.exercise_pred.to_numpy() == dm.exercise.to_numpy()).astype(float)
                    v = (W * acc).sum(1) / W.sum(1)
                    lo, hi = interval(v[1:], ci)
                    cls_rows.append(dict(level=level, stratum=sname, method=m, prediction="exercise_pred", n=len(dm),
                                         accuracy=v[0], accuracy_lo=lo, accuracy_hi=hi))
            for m in methods:
                for vs in ["base_group", "base_mean", "b1"]:
                    if vs == m or vs not in all_methods:
                        continue
                    for t in TARGETS:
                        for k in ("rmse", "spearman"):
                            dv = reps_by[(m, t)][k] - reps_by[(vs, t)][k]
                            lo, hi = interval(dv[1:], ci)
                            delta_rows.append(dict(level=level, stratum=sname, method=m, vs=vs, target=t, metric=k,
                                                   delta=dv[0], lo=lo, hi=hi, share_below_0=float(np.mean(dv[1:] < 0))))
    S = pd.DataFrame(score_rows)
    D = pd.DataFrame(delta_rows)
    F = pd.DataFrame(fold_rows)
    C = pd.DataFrame(cls_rows)
    S.to_csv(out / "score_metrics.csv", index=False)
    D.to_csv(out / "paired_deltas.csv", index=False)
    F.to_csv(out / "per_fold_score.csv", index=False)
    C.to_csv(out / "classification.csv", index=False)

    # ---- VCI (E13) on held-out subjects
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    assert (members.item.to_numpy() == meta.item.to_numpy()).all()
    gcode = pd.factorize(meta.group_id)[0]
    subjects = sorted(meta.subject.unique())
    spos = {s: i for i, s in enumerate(subjects)}
    sgroup = meta.drop_duplicates("subject").set_index("subject").group
    strata_vci = [(n, e, g) for n, e, g in STRATA]
    vci_rows, vci_fold_rows = [], []
    sums = {}
    for m in methods:
        for stage in ("init", "pretrained", "final"):
            for f in folds:
                fp = root / m / f"fold{f}" / f"embeddings_{stage}.npz"
                if not fp.exists():
                    continue
                with np.load(fp) as z:
                    it, emb = z["item"], {"centred": z["z"], "raw": z["z"], "proj": z["proj"]}
                    center = z["z_train_mean"]
                mm = members.iloc[it]
                for sname, ex, grp in strata_vci:
                    sel = stratum_mask(mm, ex, grp)
                    if not sel.any():
                        continue
                    for var, e in emb.items():
                        s = vci_sums(e[sel], mm.set_id.to_numpy()[sel], mm.view.to_numpy()[sel], gcode[it][sel],
                                     np.array([spos[x] for x in mm.subject.to_numpy()[sel]]), len(subjects),
                                     center if var == "centred" else None)
                        k = (m, stage, sname, var)
                        sums[k] = add_sums(sums.get(k), s)
                        v = vci_from_sums(s)
                        vci_fold_rows.append(dict(method=m, stage=stage, stratum=sname, variant=var, fold=f, **v,
                                                  same_pairs=int(s["same_cnt"].sum() // 2)))
    for (m, stage, sname, var), s in sums.items():
        present = np.flatnonzero((s["same_cnt"] > 0) | (s["other_cnt"].sum(1) > 0))
        Mfull = np.zeros((n_boot, len(subjects)))
        Mfull[:, present] = subject_draws(present, [sgroup[subjects[i]] for i in present], n_boot,
                                          ec["bootstrap"]["seed"])
        v = vci_from_sums(s)
        b = vci_from_sums(s, Mfull)
        lo, hi = interval(b["vci"], ci)
        fv = pd.DataFrame(vci_fold_rows)
        fv = fv[(fv.method == m) & (fv.stage == stage) & (fv.stratum == sname) & (fv.variant == var)].vci
        vci_rows.append(dict(method=m, stage=stage, stratum=sname, variant=var, vci=v["vci"], vci_lo=lo, vci_hi=hi,
                             same=v["same"], other=v["other"], same_pairs=int(s["same_cnt"].sum() // 2),
                             subjects=len(present), fold_mean=fv.mean(), fold_sd=fv.std(ddof=1), folds=len(fv)))
    V = pd.DataFrame(vci_rows)
    V.to_csv(out / "vci.csv", index=False)
    pd.DataFrame(vci_fold_rows).to_csv(out / "vci_per_fold.csv", index=False)

    # ---- stopping epochs and runtimes
    trows = []
    for (m, f), d in sorted(done.items()):
        for stage in ("cvdc_stage", "supervised_stage"):
            if stage in d:
                s = d[stage]
                row = dict(method=m, fold=f, stage=stage.split("_")[0], best_epoch=s["best_epoch"],
                           stopped_epoch=s["stopped_epoch"], max_epochs=s["max_epochs"],
                           early_stopped=s["early_stopped"], best_val=s["best_val"], stage_s=s["wall_s"])
                if stage == "supervised_stage":
                    # diagnostic: inner-validation score RMSE at the kept epoch vs the inner-training mean on the
                    # same reps (both rep level; the test fold is not involved)
                    h = pd.read_csv(root / m / f"fold{f}" / "finetune" / "history.csv")
                    row["val_rmse_best"] = float(h.loc[h.epoch == s["best_epoch"], "val/score_rmse"].iloc[0])
                    row["val_rmse_min_any_epoch"] = float(h["val/score_rmse"].min())
                    sub = d["subjects"]
                    mu = meta[meta.subject.isin(sub["inner_train"])].score.mean()
                    yv = meta[meta.subject.isin(sub["inner_val"])].score
                    row["val_rmse_inner_train_mean"] = float(np.sqrt(((yv - mu) ** 2).mean()))
                trows.append(row)
        trows.append(dict(method=m, fold=f, stage="total (incl. embeddings, prediction)", stage_s=d["wall_s"],
                          peak_gpu_gb=d.get("peak_gpu_gb")))
    T = pd.DataFrame(trows)
    T.to_csv(out / "stopping_runtime.csv", index=False)

    write_summary(out, S, D, F, C, V, T, floors, methods, folds, n_boot, ci, smoke, meta, vid_level, cfg)
    print(f"report written to {out} in {time.time() - t_start:.0f} s")


def write_summary(out, S, D, F, C, V, T, floors, methods, folds, n_boot, ci, smoke, meta, vid, cfg):
    L = []
    w = L.append
    w("# Phase 3 step (a): GroupKFold-5, main methods" + (" (SMOKE RUN: plumbing check, not a result)" if smoke else ""))
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/pilot/phase3/report_phase3a.py`. Folds: {folds} of "
      "stratified GroupKFold-5 (`group_kfold(meta, 5, seed=0)`, rep-level subjects); test predictions of all folds "
      "pooled (E1). Every method uses the same folds, inner validation splits (T8), preprocessing, rep-level "
      "training (T1), epoch budget (CVDC ≤ 100, fine-tuning ≤ 60, patience 10), schedule and encoder (ST-GCN, "
      "3.07 M parameters). CVDC sources (Proposed): sync, audio, rep, synthetic.")
    w("")
    w(f"95% CIs: subject-level bootstrap, {n_boot} draws, subjects resampled with replacement within expert / "
      "non-expert (E17); the same draws for every method, so method differences are paired. "
      "Score on 0–100, CF items on their 0–1 scale. Noise floor = inter-view label-noise RMSE (E3; computed on "
      "expert groups, so it has no counterpart for E01 non-experts, whose views are separate performances).")
    w("")
    w("**Limitations (PROGRESS §0):** only 2 exercises, both shoulder movements (exercise classification near-trivial); "
      "E02 has no non-experts; the expert/non-expert contrast exists only in E01 and is confounded (score ~80 vs ~45, "
      "clip length, views, recording year, location, phones, recording mode), so within-group results are primary; "
      "3 views, not 4, with synchronisation reconstructed in software; no error-type labels; scores skewed high, "
      "label-noise floor 5.28 (E01) / 4.38 (E02); 50 subjects at rep level (P39 has no rep), 24 in E02; E01 "
      "non-experts are almost all single-view for CVDC (synthetic views only, D12). GroupKFold-5, not LOSO.")
    w("")
    nv = vid[vid.method == methods[0]] if methods else vid
    w(f"Test units pooled over folds: {len(nv)} videos with ≥ 1 rep, {len(meta[meta.subject.isin(set(nv.subject))])} reps.")
    w("")
    w("## Headline: total score within groups, video level")
    w("")
    w("RMSE [95% CI] / Spearman ρ. Within a group the group-only baseline is the within-group mean of the training "
      "fold, the bar a model must clear (E10).")
    w("")
    w("| Method | " + " | ".join(WITHIN) + " |")
    w("|---|" + "---|" * len(WITHIN))
    for m in [*methods, *BASELINES]:
        cells = []
        for sname in WITHIN:
            r = S[(S.level == "video") & (S.stratum == sname) & (S.target == "score") & (S.method == m)]
            cells.append("–" if r.empty else f"{fci(r.rmse.iloc[0], r.rmse_lo.iloc[0], r.rmse_hi.iloc[0])} / {fmt(r.spearman.iloc[0])}")
        w(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    w("| label-noise floor (E3) | 5.28 | 4.38 | – (expert-based 5.28) |")
    w("")
    w("## Headline: VCI, centred 256-d, within groups (final encoders; random-init reference in parentheses)")
    w("")
    w("| Method | " + " | ".join(WITHIN) + " | pooled |")
    w("|---|" + "---|" * (len(WITHIN) + 1))
    for m in methods:
        cells = []
        for sname in [*WITHIN, "pooled"]:
            d = V[(V.method == m) & (V.stratum == sname) & (V.variant == "centred")]
            fin, ini = d[d.stage == "final"], d[d.stage == "init"]
            pre = d[d.stage == "pretrained"]
            c = "–" if fin.empty else fci(fin.vci.iloc[0], fin.vci_lo.iloc[0], fin.vci_hi.iloc[0], 3)
            if not ini.empty:
                c += f" ({fmt(ini.vci.iloc[0], 3)})"
            if not pre.empty:
                c += f"; after CVDC only: {fmt(pre.vci.iloc[0], 3)}"
            cells.append(c)
        w(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    w("")
    for level, title in (("video", "Total score, video level (headline, E15)"), ("rep", "Total score, rep level")):
        w(f"## {title}")
        w("")
        w("| Stratum | Method | n | RMSE [95% CI] | noise floor | MAE | Pearson r [CI] | Spearman ρ [CI] |")
        w("|---|---|---|---|---|---|---|---|")
        for sname, *_ in STRATA:
            d = S[(S.level == level) & (S.stratum == sname) & (S.target == "score")]
            for _, r in d.iterrows():
                fl = fmt(r.noise_floor) if np.isfinite(r.noise_floor) else f"({r.noise_floor_ref})¹"
                w(f"| {sname} | {LABEL[r.method]} | {r.n} | {fci(r.rmse, r.rmse_lo, r.rmse_hi)} | "
                  f"{fl} | {fmt(r.mae)} | {fci(r.pearson, r.pearson_lo, r.pearson_hi)} | "
                  f"{fci(r.spearman, r.spearman_lo, r.spearman_hi)} |")
        w("")
        w("¹ Floor measured on expert groups only (E3); shown for reference where the stratum is not expert-only.")
        w("")
    w("## Paired RMSE differences (total score): method − group-only baseline, and method − B1")
    w("")
    w("Negative = lower RMSE than the comparison. CI from the paired subject bootstrap; 'share < 0' is the share of "
      "bootstrap draws in which the method's RMSE is lower.")
    w("")
    w("| Level | Stratum | Method | − group-only [95% CI] (share < 0) | − B1 [95% CI] (share < 0) |")
    w("|---|---|---|---|---|")
    for level in ("video", "rep"):
        for sname, *_ in STRATA:
            for m in methods:
                g = D[(D.level == level) & (D.stratum == sname) & (D.method == m) & (D.target == "score") & (D.metric == "rmse")]
                cg = g[g.vs == "base_group"]
                cb = g[g.vs == "b1"]
                sg = f"{fci(cg.delta.iloc[0], cg.lo.iloc[0], cg.hi.iloc[0])} ({cg.share_below_0.iloc[0]:.2f})" if len(cg) else "–"
                sb = f"{fci(cb.delta.iloc[0], cb.lo.iloc[0], cb.hi.iloc[0])} ({cb.share_below_0.iloc[0]:.2f})" if len(cb) else "–"
                w(f"| {level} | {sname} | {SHORT[m]} | {sg} | {sb} |")
    w("")
    w("## CF1–CF3 (0–1 scale), video level")
    w("")
    w("Floor in parentheses (expert strata). **The CF floor is 0**: CF1–CF3 (and PF1, PF2) are identical across the "
      "views of every expert performance; only PF3/PF4 (repetitions) differ between views, so the score floor comes "
      "entirely from the repetition items, which a single rep cannot show (`noise_floor.csv`).")
    w("")
    w("| Stratum | Method | CF1 RMSE (floor) / ρ | CF2 RMSE (floor) / ρ | CF3 RMSE (floor) / ρ |")
    w("|---|---|---|---|---|")
    for sname, *_ in STRATA:
        d = S[(S.level == "video") & (S.stratum == sname)]
        for m in d.method.unique():
            cells = []
            for t in ("CF1", "CF2", "CF3"):
                r = d[(d.method == m) & (d.target == t)].iloc[0]
                flc = f" ({fmt(r.noise_floor, 3)})" if np.isfinite(r.noise_floor) else ""
                cells.append(f"{fmt(r.rmse, 3)} [{fmt(r.rmse_lo, 3)}, {fmt(r.rmse_hi, 3)}]{flc} / {fmt(r.spearman)}")
            w(f"| {sname} | {SHORT[m]} | " + " | ".join(cells) + " |")
    w("")
    w("## Viewpoint Consistency Index (E13), held-out subjects")
    w("")
    w("Headline: mean-centred 256-d embedding (centre = the training fold's mean embedding), within exercise × group. "
      "VCI = mean cosine of the same rep seen from another view − mean cosine to other (exercise, subject, variation) "
      "groups of the same stratum, on plain real views; pairs pooled over the 5 folds; CI by subject bootstrap. "
      "'init' = the random-init encoder of the same seed (reference); 'pretrained' = Proposed after CVDC, before "
      "fine-tuning. For B1/B2 the 128-d projection head is untrained (random), so their 'proj' VCI is a random "
      "projection of the trained embedding. For E01 non-experts the 'same rep' pairs are rep-matched pairs of views "
      "recorded one after another (index pairing, A5), not one performance seen twice.")
    w("")
    w("| Stratum | Method | stage | centred 256-d [95% CI] | same / other | raw 256-d | projection 128-d | same-rep view pairs |")
    w("|---|---|---|---|---|---|---|---|")
    for sname, *_ in STRATA:
        for m in methods:
            for stage in ("init", "pretrained", "final"):
                d = V[(V.method == m) & (V.stage == stage) & (V.stratum == sname)]
                if d.empty or (stage == "init" and m == "proposed"):
                    continue
                c = d[d.variant == "centred"].iloc[0]
                raw = d[d.variant == "raw"].iloc[0]
                pr = d[d.variant == "proj"].iloc[0]
                w(f"| {sname} | {SHORT[m]} | {stage} | {fci(c.vci, c.vci_lo, c.vci_hi, 3)} | {fmt(c.same, 3)} / "
                  f"{fmt(c.other, 3)} | {fmt(raw.vci, 3)} | {fmt(pr.vci, 3)} | {c.same_pairs} |")
    w("")
    w("(B1's 'init' encoder is identical to Proposed's random init: same seed and input.)")
    w("")
    w("## Classification (secondary, E6)")
    w("")
    w("Derived quality class (≥ 80 / 50–79 / < 50, D6): the quality head (video level: mean class probabilities), and "
      "the binned predicted score. Exercise accuracy from the exercise head.")
    w("")
    w("| Level | Stratum | Method | quality head: macro F1 / κ | binned score: macro F1 / κ | exercise accuracy |")
    w("|---|---|---|---|---|---|")
    for level in ("video", "rep"):
        for sname, *_ in STRATA:
            for m in [*methods, *BASELINES]:
                d = C[(C.level == level) & (C.stratum == sname) & (C.method == m)]
                if d.empty:
                    continue
                qh = d[d.prediction == "quality_pred"]
                qs = d[d.prediction == "quality_from_score"].iloc[0]
                ea = d[d.prediction == "exercise_pred"]
                sqh = f"{fci(qh.macro_f1.iloc[0], qh.macro_f1_lo.iloc[0], qh.macro_f1_hi.iloc[0])} / {fmt(qh.kappa.iloc[0])}" if len(qh) else "–"
                sea = fci(ea.accuracy.iloc[0], ea.accuracy_lo.iloc[0], ea.accuracy_hi.iloc[0], 3) if len(ea) else "–"
                w(f"| {level} | {sname} | {SHORT[m]} | {sqh} | {fci(qs.macro_f1, qs.macro_f1_lo, qs.macro_f1_hi)} / "
                  f"{fmt(qs.kappa)} | {sea} |")
    w("")
    w("## Training: stopping epochs and wall time")
    w("")
    w("Early stopping on the inner validation subjects (T8). best = epoch kept; stopped = last epoch run.")
    w("")
    w("| Method | stage | best epochs (folds 0–4) | stopped epochs | early-stopped folds | wall time per fold (min) | total (min) |")
    w("|---|---|---|---|---|---|---|")
    for m in methods:
        for stage in ("cvdc", "supervised"):
            d = T[(T.method == m) & (T.stage == stage)]
            if d.empty:
                continue
            w(f"| {SHORT[m]} | {stage} | {', '.join(str(int(x)) for x in d.best_epoch)} | "
              f"{', '.join(str(int(x)) for x in d.stopped_epoch)} | {int(d.early_stopped.sum())}/{len(d)} | "
              f"{', '.join(f'{x / 60:.1f}' for x in d.stage_s)} | {d.stage_s.sum() / 60:.1f} |")
        d = T[(T.method == m) & T.stage.str.startswith("total")]
        w(f"| {SHORT[m]} | **all stages** | | | | {', '.join(f'{x / 60:.1f}' for x in d.stage_s)} | "
          f"**{d.stage_s.sum() / 60:.1f}** (peak GPU {d.peak_gpu_gb.max():.2f} GB) |")
    w("")
    w("Diagnostic (inner validation subjects only, rep level): score RMSE of the kept epoch, the lowest score RMSE "
      "of any epoch, and the RMSE of the inner training subjects' mean score on the same reps. The kept epoch is "
      "chosen by the whole validation loss (T8a), not by the score RMSE.")
    w("")
    w("| Method | fold | best epoch | val RMSE at best | lowest val RMSE (any epoch) | val RMSE of the inner-train mean |")
    w("|---|---|---|---|---|---|")
    for m in methods:
        for _, r in T[(T.method == m) & (T.stage == "supervised")].iterrows():
            w(f"| {SHORT[m]} | {r.fold} | {int(r.best_epoch)} | {fmt(r.val_rmse_best)} | {fmt(r.val_rmse_min_any_epoch)} | "
              f"{fmt(r.val_rmse_inner_train_mean)} |")
    w("")
    w("Tables: `score_metrics.csv`, `paired_deltas.csv`, `per_fold_score.csv`, `classification.csv`, `vci.csv`, "
      "`vci_per_fold.csv`, `stopping_runtime.csv`, `noise_floor.csv`.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
