"""Phase 3 step (a2) report: the fixed score setup on GroupKFold-5 (B1, Proposed variants, B2-MediaPipe,
B2-KTPFormer).

Reads runs/<tag>/<method>/fold<k>/ (cvdc.train.run): test_predictions_video.csv (video head, T13; headline),
test_predictions.csv (rep head; averaged per video), embeddings_*.npz and done.json. Test predictions of the 5 folds
are pooled (E1). Reports, within groups first (E01 expert, E02 expert, E01 non-expert), then pooled:
  score       RMSE, MAE, Pearson, Spearman of the total score, video level, for the video head and the rep head
  items       the EAAQ items that vary within the group (T14): video head (PF1, PF2, PF4, CF1-CF3) and rep head
              (CF1-CF3); items that do not vary are listed with their spread instead
  baselines   video level, fitted on the training fold's videos (E16, E20): predict-the-mean, group-only,
              duration-only, rep-count-only and set-features (rep count + mean rep duration + ROM, within group)
  deltas      paired RMSE differences vs group-only, set-features and B1, same bootstrap draws (E17)
  VCI         centred 256-d (headline), raw, projection; within groups and pooled; random init, after CVDC
              (the step (a) encoders, shared by the Proposed variants) and after fine-tuning
  training    kept / stopping epochs (T8b) and wall time per method and fold
Writes CSV tables and summary.md to --out (default <eval.results_dir>/<tag> = outputs/results/phase3a2).

usage: python scripts/pilot/phase3/report_phase3a2.py [--tag phase3a2] [--out DIR] [--boot 2000] [--methods ...]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_baselines, video_table  # noqa: E402
from cvdc.eval.metrics import (add_sums, interval, item_weights, regression, subject_draws, vci_from_sums,  # noqa: E402
                               vci_sums)
from cvdc.train.trainer import fold_plan  # noqa: E402
from report_phase3a import fci, fmt, noise_floors, stratum_mask  # noqa: E402

METHODS = ["b1", "proposed", "proposed_lp", "proposed_j01", "proposed_j1", "b2_mediapipe", "b2_ktpformer"]
BASELINES = ["base_mean", "base_group", "base_duration", "base_repcount", "base_setfeat"]
LABEL = {"b1": "B1 (2D xy, scratch)", "proposed": "Proposed (iii): CVDC + full fine-tune",
         "proposed_lp": "Proposed (i): frozen CVDC encoder + heads", "proposed_j01": "Proposed (ii): fine-tune + 0.1·L_cvdc",
         "proposed_j1": "Proposed (ii): fine-tune + 1.0·L_cvdc", "b2_mediapipe": "B2-MediaPipe (world xyz, scratch)",
         "b2_ktpformer": "B2-KTPFormer (lifted xyz, scratch)", "base_mean": "baseline: predict-the-mean",
         "base_group": "baseline: group-only", "base_duration": "baseline: duration-only",
         "base_repcount": "baseline: rep-count-only", "base_setfeat": "baseline: set-features"}
SHORT = {"b1": "B1", "proposed": "Prop (iii) FT", "proposed_lp": "Prop (i) frozen", "proposed_j01": "Prop (ii) λ0.1",
         "proposed_j1": "Prop (ii) λ1", "b2_mediapipe": "B2-MP", "b2_ktpformer": "B2-KTP", "base_mean": "mean",
         "base_group": "group-only", "base_duration": "duration-only", "base_repcount": "rep-count-only",
         "base_setfeat": "set-features"}
VIDEO_TARGETS = ["score", "PF1", "PF2", "PF4", "CF1", "CF2", "CF3"]
REP_TARGETS = ["score", "CF1", "CF2", "CF3"]
STRATA = [("E01 expert", "E01", "expert"), ("E02 expert", "E02", "expert"), ("E01 non-expert", "E01", "non-expert"),
          ("E01 pooled", "E01", None), ("pooled", None, None)]
WITHIN = ["E01 expert", "E02 expert", "E01 non-expert"]


def load_runs(root, methods):
    vid, rep, done = [], [], {}
    for m in methods:
        for f in sorted((root / m).glob("fold*/done.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            done[(m, d["fold"])] = d
            pv = pd.read_csv(f.parent / "test_predictions_video.csv")
            pr = pd.read_csv(f.parent / "test_predictions.csv")
            for p, lst in ((pv, vid), (pr, rep)):
                p["fold"], p["method"] = d["fold"], m
                lst.append(p)
    return pd.concat(vid, ignore_index=True), pd.concat(rep, ignore_index=True), done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--tag", default="phase3a2")
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    a = ap.parse_args()
    t_start = time.time()
    cfg = load_config(a.config, a.model_config)
    ec, ia = cfg["eval"], cfg["heads"]["item_activity"]
    n_boot, ci = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"]
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out = Path(a.out) if a.out else resolve(ec["results_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    smoke = a.tag.endswith("_smoke")

    v = video_table(cfg)
    reps = pd.read_csv(resolve("data/reps.csv"))
    index = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    plans = fold_plan(cfg, reps, smoke=smoke)
    act = item_activity(v, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}

    pv, pr, done = load_runs(root, a.methods)
    methods = [m for m in METHODS if m in set(pv.method)]
    folds = sorted({f for (_, f) in done})
    plans = [p for p in plans if p["fold"] in folds]
    for (m, f), d in done.items():  # every run used exactly these folds and inner splits
        p = next(q for q in plans if q["fold"] == f)
        for k in ("train", "inner_train", "inner_val", "test"):
            assert sorted(d["subjects"][k]) == sorted(p[k]), (m, f, k)
    test_videos = v[v.subject.isin({s for p in plans for s in p["test"]})]

    # ---- one long table of video-level predictions: (method, source) x video
    rows = []
    vp = pv.copy()
    vp["source"] = "video head"
    rows.append(vp[["method", "source", "fold", "video_id", *[f"{t}_pred" for t in VIDEO_TARGETS]]])
    rp = pr.groupby(["method", "fold", "video_id"])[[f"{t}_pred" for t in REP_TARGETS]].mean().reset_index()
    rp["source"] = "rep head"
    rows.append(rp)
    base = video_baselines(v, plans, VIDEO_TARGETS)
    base["source"] = "baseline"
    rows.append(base)
    P = pd.concat(rows, ignore_index=True).merge(
        v[["video_id", "subject", "group", "exercise", "view", "variation", *VIDEO_TARGETS]], on="video_id",
        how="left", validate="many_to_one")
    n = P.groupby(["method", "source"]).video_id.nunique()
    assert (n == len(test_videos)).all(), f"every method must predict every test video once: {n.to_dict()}"
    assert not P.duplicated(["method", "source", "video_id"]).any()
    P.to_csv(out / "video_predictions.csv", index=False)
    entries = [(m, "video head") for m in methods] + [(m, "rep head") for m in methods] + \
              [(b, "baseline") for b in BASELINES]
    floors = noise_floors(index, ["score", "PF1", "PF2", "PF4", "CF1", "CF2", "CF3"])
    floors.to_csv(out / "noise_floor.csv")

    # ---- metrics with paired subject-bootstrap CIs (E17)
    score_rows, delta_rows, fold_rows = [], [], []
    for sname, ex, grp in STRATA:
        d = P[stratum_mask(P, ex, grp)]
        ref = d[(d.method == entries[0][0]) & (d.source == entries[0][1])].sort_values("video_id")
        subj = ref.drop_duplicates("subject")[["subject", "group"]].sort_values("subject")
        M = subject_draws(subj.subject.to_numpy(), subj.group.to_numpy(), n_boot, ec["bootstrap"]["seed"])
        W = np.vstack([np.ones(len(ref)), item_weights(ref.subject.to_numpy(), subj.subject.to_numpy(), M)])
        res = {}
        for m, src in entries:
            dm = d[(d.method == m) & (d.source == src)].sort_values("video_id")
            assert (dm.video_id.to_numpy() == ref.video_id.to_numpy()).all()
            for t in (REP_TARGETS if src == "rep head" else VIDEO_TARGETS):
                varies = t == "score" or (grp is not None and t in active[(ex, grp)]) or \
                    (grp is None and any(t in active[k] for k in active if ex is None or k[0] == ex))
                r = regression(dm[t].to_numpy(), dm[f"{t}_pred"].to_numpy(), W)
                res[(m, src, t)] = r
                row = dict(stratum=sname, method=m, source=src, target=t, n=len(dm), subjects=len(subj),
                           varies_within_group=varies)
                for k, val in r.items():
                    row[k] = val[0]
                    row[f"{k}_lo"], row[f"{k}_hi"] = interval(val[1:], ci)
                fl = floors[t].to_dict() if t in floors else {}
                row["noise_floor"] = fl.get(ex, np.nan) if ex and grp == "expert" else np.nan
                score_rows.append(row)
            for f, dmf in dm.groupby("fold"):
                e = dmf.score_pred - dmf.score
                fold_rows.append(dict(stratum=sname, method=m, source=src, fold=f, n=len(dmf),
                                      rmse=float(np.sqrt((e ** 2).mean()))))
        for m, src in entries:
            if src == "baseline":
                continue
            for vs, vsrc in (("base_group", "baseline"), ("base_setfeat", "baseline"), ("base_repcount", "baseline"),
                             ("b1", src), ("proposed", src)):
                if (vs, vsrc) == (m, src) or (vs, vsrc) not in [(x, y) for x, y in entries]:
                    continue
                for t in (REP_TARGETS if src == "rep head" else VIDEO_TARGETS):
                    if (vs, vsrc, t) not in res:
                        continue
                    for k in ("rmse", "spearman"):
                        dv = res[(m, src, t)][k] - res[(vs, vsrc, t)][k]
                        lo, hi = interval(dv[1:], ci)
                        delta_rows.append(dict(stratum=sname, method=m, source=src, vs=vs, vs_source=vsrc, target=t,
                                               metric=k, delta=dv[0], lo=lo, hi=hi,
                                               share_below_0=float(np.mean(dv[1:] < 0))))
    S, D, F = pd.DataFrame(score_rows), pd.DataFrame(delta_rows), pd.DataFrame(fold_rows)
    S.to_csv(out / "score_metrics.csv", index=False)
    D.to_csv(out / "paired_deltas.csv", index=False)
    F.to_csv(out / "per_fold_score.csv", index=False)

    # ---- VCI (E13) on held-out subjects
    lab = ["exercise", "subject", "group", "view", "variation", "group_id"]
    meta = reps.drop(columns=[c for c in lab if c in reps.columns]).merge(index[["video_id", *lab]], on="video_id",
                                                                         how="left", validate="many_to_one")
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    assert (members.item.to_numpy() == np.arange(len(meta))).all()
    gcode = pd.factorize(meta.group_id)[0]
    subjects = sorted(meta.subject.unique())
    spos = {s: i for i, s in enumerate(subjects)}
    sgroup = meta.drop_duplicates("subject").set_index("subject").group
    sums, vci_fold_rows = {}, []
    for m in methods:
        for stage in ("init", "pretrained", "final"):
            for f in folds:
                fp = root / m / f"fold{f}" / f"embeddings_{stage}.npz"
                if not fp.exists():
                    continue
                with np.load(fp) as z:
                    it, emb, center = z["item"], {"centred": z["z"], "raw": z["z"], "proj": z["proj"]}, z["z_train_mean"]
                mm = members.iloc[it]
                for sname, ex, grp in STRATA:
                    sel = stratum_mask(mm, ex, grp)
                    for var, e in emb.items():
                        s = vci_sums(e[sel], mm.set_id.to_numpy()[sel], mm.view.to_numpy()[sel], gcode[it][sel],
                                     np.array([spos[x] for x in mm.subject.to_numpy()[sel]]), len(subjects),
                                     center if var == "centred" else None)
                        k = (m, stage, sname, var)
                        sums[k] = add_sums(sums.get(k), s)
                        vci_fold_rows.append(dict(method=m, stage=stage, stratum=sname, variant=var, fold=f,
                                                  **vci_from_sums(s)))
    fv = pd.DataFrame(vci_fold_rows)
    vrows = []
    for (m, stage, sname, var), s in sums.items():
        present = np.flatnonzero((s["same_cnt"] > 0) | (s["other_cnt"].sum(1) > 0))
        Mf = np.zeros((n_boot, len(subjects)))
        Mf[:, present] = subject_draws(present, [sgroup[subjects[i]] for i in present], n_boot, ec["bootstrap"]["seed"])
        pt, b = vci_from_sums(s), vci_from_sums(s, Mf)
        lo, hi = interval(b["vci"], ci)
        x = fv[(fv.method == m) & (fv.stage == stage) & (fv.stratum == sname) & (fv.variant == var)].vci
        vrows.append(dict(method=m, stage=stage, stratum=sname, variant=var, vci=pt["vci"], vci_lo=lo, vci_hi=hi,
                          same=pt["same"], other=pt["other"], same_pairs=int(s["same_cnt"].sum() // 2),
                          fold_mean=x.mean(), fold_sd=x.std(ddof=1), folds=len(x)))
    V = pd.DataFrame(vrows)
    V.to_csv(out / "vci.csv", index=False)
    fv.to_csv(out / "vci_per_fold.csv", index=False)

    # ---- kept / stopping epochs, runtimes, inner-validation diagnostic
    trows = []
    for (m, f), d in sorted(done.items()):
        for stage in ("cvdc_stage", "supervised_stage"):
            if stage not in d:
                continue
            s = d[stage]
            row = dict(method=m, fold=f, stage=stage.split("_")[0], best_epoch=s["best_epoch"],
                       stopped_epoch=s["stopped_epoch"], max_epochs=s["max_epochs"], early_stopped=s["early_stopped"],
                       best_val=s["best_val"], stage_s=s["wall_s"], reused=bool(s.get("reused_from")))
            if stage == "supervised_stage":
                h = pd.read_csv(root / m / f"fold{f}" / "finetune" / "history.csv")
                hb = h[h.epoch == s["best_epoch"]].iloc[0]
                row.update(val_video_head_rmse_best=hb.get("val/video_head_rmse", np.nan),
                           val_rep_head_video_rmse_best=hb.get("val/rep_head_video_rmse", np.nan),
                           val_video_head_rmse_min=h["val/video_head_rmse"].min() if "val/video_head_rmse" in h else np.nan)
                sub = d["subjects"]
                vt, vv = v[v.subject.isin(sub["inner_train"])], v[v.subject.isin(sub["inner_val"])]
                row["val_rmse_inner_train_mean"] = float(np.sqrt(((vv.score - vt.score.mean()) ** 2).mean()))
                gm = vt.groupby(["exercise", "group"]).score.mean()
                gp = np.array([gm[(e, g)] for e, g in zip(vv.exercise, vv.group)])
                row["val_rmse_inner_train_group_mean"] = float(np.sqrt(((vv.score - gp) ** 2).mean()))
            trows.append(row)
        trows.append(dict(method=m, fold=f, stage="total", stage_s=d["wall_s"], peak_gpu_gb=d.get("peak_gpu_gb")))
    T = pd.DataFrame(trows)
    T.to_csv(out / "stopping_runtime.csv", index=False)

    write_summary(out, S, D, V, T, act, methods, folds, n_boot, smoke, test_videos, cfg)
    print(f"report written to {out} in {time.time() - t_start:.0f} s")


def cell(S, sname, m, src, t, d=2, rho=True):
    r = S[(S.stratum == sname) & (S.method == m) & (S.source == src) & (S.target == t)]
    if r.empty:
        return "–"
    r = r.iloc[0]
    c = fci(r.rmse, r.rmse_lo, r.rmse_hi, d)
    return c + (f" / {fmt(r.spearman)}" if rho else "")


def delta_cell(D, sname, m, src, vs, t="score"):
    g = D[(D.stratum == sname) & (D.method == m) & (D.source == src) & (D.vs == vs) & (D.target == t) &
          (D.metric == "rmse")]
    if g.empty:
        return "–"
    g = g.iloc[0]
    return f"{fci(g.delta, g.lo, g.hi)} ({g.share_below_0:.2f})"


def write_summary(out, S, D, V, T, act, methods, folds, n_boot, smoke, tv, cfg):
    tc = cfg["train"]
    es = tc["early_stopping"]
    L = []
    w = L.append
    w("# Phase 3 step (a2): fixed score setup, GroupKFold-5" + (" (SMOKE RUN: plumbing check, not a result)" if smoke else ""))
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/pilot/phase3/report_phase3a2.py`. Folds {folds} of the same "
      "stratified GroupKFold-5 as step (a) (`group_kfold(meta, 5, seed=0)`), same inner validation splits (T8), "
      "preprocessing, encoder (ST-GCN, 3.07 M parameters), optimiser and schedule (T9). Test predictions pooled over "
      f"the folds (E1); {len(tv)} test videos with ≥ 1 rep. Changes against step (a) (T6, T8b, T13, T14, E20):")
    w("")
    w("- **Video head** (T13): mean + attention pooling of all rep embeddings of a video plus the set rep count "
      "(Phase 1 `n_reps_set`) → the video score and the items PF1, PF2, PF4, CF1–CF3. The rep head (score, CF1–CF3, "
      "quality, exercise) stays; L_sup = L_rep + L_video. Batches hold whole videos (≥ 256 reps).")
    w("- **T6**: every video weighs the same in L_rep (rep weight 1 / n_reps of its video).")
    w(f"- **T8b**: early stopping on the EMA-smoothed (β = {es['ema']}) inner-validation objective, no epoch before "
      f"{es['min_epochs']} kept, patience {es['patience']}, at most {tc['max_epochs_sup']} epochs.")
    w(f"- **T14**: an item is learned and reported in a group only if it varies there (off its modal value in videos "
      f"of ≥ {cfg['heads']['item_activity']['min_subjects_off_mode']} subjects, SD ≥ {cfg['heads']['item_activity']['min_sd']}).")
    w("- **Proposed variants** share one CVDC encoder per fold (the step (a) pretraining, reused after checking its "
      "subjects): (i) frozen encoder + heads, (ii) full fine-tuning with L_sup + λ·L_cvdc (λ 0.1, 1.0), (iii) full "
      "fine-tuning (as in step (a)).")
    w("- **Baselines** are now fitted on the training fold's videos (each video once, like T6): predict-the-mean, "
      "group-only (within-group mean), duration-only (per exercise, E16), rep-count-only and set-features "
      "(rep count + mean rep duration + ROM of the main angle), the last two within group (E20).")
    w("")
    w(f"95% CIs: subject bootstrap, {n_boot} draws within expert / non-expert, the same draws for every method "
      "(paired). Score 0–100, items 0–1. **Limitations (PROGRESS §0)** as in step (a): 2 shoulder exercises; E02 "
      "experts only; the expert/non-expert contrast is confounded, so within-group results are primary; label-noise "
      "floor 5.28 (E01) / 4.38 (E02), carried entirely by the repetition items; GroupKFold-5, not LOSO.")
    w("")
    w("## Headline: total score within groups, video level")
    w("")
    w("RMSE [95% CI] / Spearman ρ. 'video head' = the set-level head (T13); 'rep head' = the mean of the video's "
      "rep predictions (step (a)'s headline). A model has to beat the group-only baseline (the within-group mean) "
      "to show that it judges movement within a group.")
    w("")
    w("| Method | head | " + " | ".join(WITHIN) + " |")
    w("|---|---|" + "---|" * len(WITHIN))
    for src in ("video head", "rep head"):
        for m in methods:
            w(f"| {LABEL[m]} | {src} | " + " | ".join(cell(S, s, m, src, "score") for s in WITHIN) + " |")
    for b in BASELINES:
        w(f"| {LABEL[b]} | – | " + " | ".join(cell(S, s, b, "baseline", "score") for s in WITHIN) + " |")
    w("| label-noise floor (E3) | | 5.28 | 4.38 | – (expert-based 5.28) |")
    w("")
    sa = resolve("outputs/results/phase3a/score_metrics.csv")
    if sa.exists():
        A = pd.read_csv(sa)
        A = A[(A.level == "video") & (A.target == "score")]
        w("## Change against step (a): total score, video level, RMSE / Spearman ρ")
        w("")
        w("Step (a): rep head only, every rep weight 1, early stopping on the raw validation loss (patience 10). "
          "Step (a)'s group-only baseline was fitted on reps (E16), so it differs slightly from today's.")
        w("")
        w("| Method | " + " | ".join(f"{s}: step (a) → a2 rep head → a2 video head" for s in WITHIN) + " |")
        w("|---|" + "---|" * len(WITHIN))
        for m in [x for x in ("b1", "proposed", "b2_mediapipe", "b2_ktpformer", "base_group")
                  if x in A.method.values and (x in methods or x.startswith("base"))]:
            cells = []
            for sname in WITHIN:
                r = A[(A.stratum == sname) & (A.method == m)]
                old_c = "–" if r.empty else f"{fmt(r.rmse.iloc[0])} / {fmt(r.spearman.iloc[0])}"
                if m.startswith("base"):
                    cur = S[(S.stratum == sname) & (S.method == m) & (S.source == "baseline") & (S.target == "score")]
                    cells.append(f"{old_c} → {fmt(cur.rmse.iloc[0])} (video-fitted)")
                    continue
                parts = [old_c]
                for src in ("rep head", "video head"):
                    cur = S[(S.stratum == sname) & (S.method == m) & (S.source == src) & (S.target == "score")]
                    parts.append("–" if cur.empty else f"{fmt(cur.rmse.iloc[0])} / {fmt(cur.spearman.iloc[0])}")
                cells.append(" → ".join(parts))
            w(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
        w("")
    w("## Paired RMSE differences, total score, video level (method − baseline)")
    w("")
    w("Negative = lower RMSE than the comparison; CI from the paired subject bootstrap; (share of draws < 0).")
    w("")
    w("| Stratum | Method | head | − group-only | − set-features | − B1 (same head) |")
    w("|---|---|---|---|---|---|")
    for sname in WITHIN:
        for src in ("video head", "rep head"):
            for m in methods:
                w(f"| {sname} | {SHORT[m]} | {src} | {delta_cell(D, sname, m, src, 'base_group')} | "
                  f"{delta_cell(D, sname, m, src, 'base_setfeat')} | "
                  f"{'–' if m == 'b1' else delta_cell(D, sname, m, src, 'b1')} |")
    w("")
    w("## Items within groups, video level")
    w("")
    w("RMSE [95% CI] / Spearman ρ on the 0–1 scale; only items that vary within the group (T14). Video head for "
      "PF and CF items; the rep head's CF predictions (mean over the video's reps) are in `score_metrics.csv`. "
      "Noise floor (E3, expert groups): CF1–CF3, PF1, PF2 = 0 (identical across the views of a performance); PF4 > 0.")
    w("")
    for sname, ex, g in STRATA[:3]:
        a = act[(act.exercise == ex) & (act.group == g)]
        items = [t for t in VIDEO_TARGETS[1:] if a.set_index("item").varies.get(t, False)]
        w(f"### {sname}")
        w("")
        w("| Method | " + " | ".join(items) + " |")
        w("|---|" + "---|" * len(items))
        for m in methods:
            w(f"| {SHORT[m]} | " + " | ".join(cell(S, sname, m, "video head", t, 3) for t in items) + " |")
        for b in BASELINES:
            w(f"| {SHORT[b]} | " + " | ".join(cell(S, sname, b, "baseline", t, 3) for t in items) + " |")
        no = a[~a.varies & a.item.isin(VIDEO_TARGETS)]
        if len(no):
            w("")
            w("Not predicted or reported: ≈ 0 within-group variance (T14): " + "; ".join(
                f"{r.item} (SD {r.sd:.3f}, off its modal value in {r.off_mode_videos} videos of {r.off_mode_subjects} "
                f"subjects)" for r in no.itertuples()) + ".")
        w("")
    w("## Viewpoint Consistency Index after fine-tuning (centred 256-d, within groups)")
    w("")
    w("VCI = mean cosine of the same rep seen from another view − mean cosine to other (exercise, subject, "
      "variation) groups of the same stratum; held-out subjects, plain real views, pooled over folds, subject-"
      "bootstrap CI (E13, E19). 'after CVDC' = the shared CVDC encoder before supervised training.")
    w("")
    w("| Method | " + " | ".join(WITHIN) + " | pooled |")
    w("|---|" + "---|" * (len(WITHIN) + 1))

    def vcell(m, stage, s):
        d = V[(V.method == m) & (V.stage == stage) & (V.stratum == s) & (V.variant == "centred")]
        return "–" if d.empty else fci(d.vci.iloc[0], d.vci_lo.iloc[0], d.vci_hi.iloc[0], 3)

    ref = methods[0]
    w(f"| random init (reference) | " + " | ".join(vcell(ref, "init", s) for s in [*WITHIN, "pooled"]) + " |")
    pre = next((m for m in methods if not V[(V.method == m) & (V.stage == "pretrained")].empty), None)
    if pre:
        w(f"| after CVDC (shared by the Proposed variants) | " +
          " | ".join(vcell(pre, "pretrained", s) for s in [*WITHIN, "pooled"]) + " |")
    for m in methods:
        w(f"| {LABEL[m]} | " + " | ".join(vcell(m, "final", s) for s in [*WITHIN, "pooled"]) + " |")
    w("")
    w("Raw 256-d and 128-d projection variants: `vci.csv`.")
    w("")
    w("## Pooled results (secondary): total score, video level")
    w("")
    w("| Method | head | E01 pooled RMSE / Pearson r | pooled RMSE / Pearson r |")
    w("|---|---|---|---|")
    for m, src in [(m, "video head") for m in methods] + [(b, "baseline") for b in BASELINES]:
        cells = []
        for s in ("E01 pooled", "pooled"):
            r = S[(S.stratum == s) & (S.method == m) & (S.source == src) & (S.target == "score")].iloc[0]
            cells.append(f"{fci(r.rmse, r.rmse_lo, r.rmse_hi)} / {fmt(r.pearson)}")
        w(f"| {LABEL[m]} | {src if src != 'baseline' else '–'} | " + " | ".join(cells) + " |")
    w("")
    w("## Training: kept and stopping epochs (T8b), wall time")
    w("")
    w(f"Supervised stage: kept epoch = lowest EMA-smoothed inner-validation objective at epoch ≥ {es['min_epochs']}; "
      f"stop after {es['patience']} epochs without improvement; max {tc['max_epochs_sup']}. CVDC: the step (a) runs "
      "(reused).")
    w("")
    w("| Method | kept epochs (folds 0–4) | stopped epochs | early-stopped folds | wall time per fold (min) | total (min) |")
    w("|---|---|---|---|---|---|")
    for m in methods:
        d = T[(T.method == m) & (T.stage == "supervised")]
        tt = T[(T.method == m) & (T.stage == "total")]
        w(f"| {SHORT[m]} | {', '.join(str(int(x)) for x in d.best_epoch)} | "
          f"{', '.join(str(int(x)) for x in d.stopped_epoch)} | {int(d.early_stopped.sum())}/{len(d)} | "
          f"{', '.join(f'{x / 60:.1f}' for x in tt.stage_s)} | {tt.stage_s.sum() / 60:.1f} (peak GPU "
          f"{tt.peak_gpu_gb.max():.2f} GB) |")
    c = T[T.stage == "cvdc"]
    if len(c):
        r0 = c[c.method == c.method.iloc[0]]
        w(f"| CVDC (reused from step (a)) | {', '.join(str(int(x)) for x in r0.best_epoch)} | "
          f"{', '.join(str(int(x)) for x in r0.stopped_epoch)} | {int(r0.early_stopped.sum())}/{len(r0)} | – | – |")
    w("")
    w("Diagnostic (inner validation subjects, video level): video-head score RMSE at the kept epoch, the lowest over "
      "all epochs, and the RMSE of the inner-training videos' mean and within-group mean on the same videos.")
    w("")
    w("| Method | fold | kept epoch | val RMSE video head (kept) | rep head (kept) | lowest video head | inner-train mean | inner-train group mean |")
    w("|---|---|---|---|---|---|---|---|")
    for m in methods:
        for _, r in T[(T.method == m) & (T.stage == "supervised")].iterrows():
            w(f"| {SHORT[m]} | {r.fold} | {int(r.best_epoch)} | {fmt(r.val_video_head_rmse_best)} | "
              f"{fmt(r.val_rep_head_video_rmse_best)} | {fmt(r.val_video_head_rmse_min)} | "
              f"{fmt(r.val_rmse_inner_train_mean)} | {fmt(r.val_rmse_inner_train_group_mean)} |")
    w("")
    w("Tables: `score_metrics.csv` (all strata, heads, targets; with `varies_within_group`), `paired_deltas.csv`, "
      "`per_fold_score.csv`, `video_predictions.csv`, `vci.csv`, `vci_per_fold.csv`, `stopping_runtime.csv`, "
      "`noise_floor.csv`; label diagnostics and set-feature baselines: `label_diagnostics.md`.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
