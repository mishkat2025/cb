"""Phase 3 on the full scope (all nine exercises, GroupKFold-10; user, eleventh review 2026-10-20): report.

Reads runs/full/<tag>/<method>/fold<k>/ (cvdc.train.run with configs/data_full.yaml): test_predictions_video.csv
(video head, T13: the headline), test_predictions.csv (rep head; averaged per video), embeddings_*.npz, done.json.
The test predictions of the 10 folds are pooled (E1). Same settings as the pilot's step (a2) (T6, T8b, T13, T14, T15).
Reported within groups first (E9): every exercise x group stratum (E01 expert, E01 non-expert, ..., E09 non-expert)
and pooled over the exercises within each group ("experts, all exercises", "non-experts, all exercises"); pooled over
both groups second.
  score       total score: RMSE, MAE, Pearson, Spearman, video level, video head and rep head
  MQ          the movement-quality score (D13) = 100 x mean(PF1, PF2, CF1, CF2, CF3) from the video head's items;
              items that do not vary in the video's group (T14: not learned there) are set to their training-fold
              mean in that group (as in the cross-view protocol, E24)
  items       the EAAQ items that vary within the stratum (T14), video head
  baselines   video level, fitted on the training fold's videos (E16, E20): predict-the-mean, group-only (within-group
              mean), duration-only (per exercise), rep-count-only and set-features (rep count + mean rep duration +
              ROM of the exercise's main angle, within group)
  deltas      paired RMSE / Spearman differences vs group-only, set-features, rep-count-only and B1 (E17)
  classes     exercise classification (9 classes; rep head, mean class probabilities per video, E18): accuracy,
              macro precision / recall / F1, Cohen's kappa, confusion matrix, per-exercise recall; with and without
              the subjects excluded from the exercise labels (D8: E03 P02); derived quality class (D6) within groups
  VCI         centred 256-d (headline), raw, projection, within every exercise x group stratum (E13, E19): random
              init, after CVDC (the encoder shared by the Proposed variants), after fine-tuning
  group probe the E11 tables written by scripts/phase3_full/group_probe.py, if present
  training    kept / stopping epochs (T8a, T8b) and wall time per method and fold
Writes CSV tables and summary.md to --out (default <eval.results_dir>/<tag> = outputs/full/results/full_gkf10).

usage: python scripts/phase3_full/report_full.py [--config configs/data_full.yaml] [--tag full_gkf10] [--boot 2000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.config import data_path, exercise_codes, load_config, resolve  # noqa: E402
from cvdc.data.dataset import exercise_label_ok  # noqa: E402
from cvdc.data.labels import MQ_ITEMS, movement_quality  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_baselines, video_table  # noqa: E402
from cvdc.eval.metrics import (add_sums, classification, interval, item_weights, regression, subject_draws,  # noqa: E402
                               vci_from_sums, vci_sums)
from cvdc.train.trainer import fold_plan  # noqa: E402

METHODS = ["b1", "b2_mediapipe", "b2_ktpformer", "proposed_j1", "proposed_lp"]
BASELINES = ["base_group", "base_setfeat", "base_repcount", "base_duration", "base_mean"]
LABEL = {"b1": "B1 (2D xy, scratch)", "b2_mediapipe": "B2-MediaPipe (world xyz, scratch)",
         "b2_ktpformer": "B2-KTPFormer (lifted xyz, scratch)", "proposed_j1": "Proposed (CVDC + fine-tune, λ = 1.0)",
         "proposed_lp": "Proposed alt. (frozen CVDC encoder + heads)", "base_mean": "baseline: predict-the-mean",
         "base_group": "baseline: group-only (within-group mean)", "base_duration": "baseline: duration-only",
         "base_repcount": "baseline: rep-count-only", "base_setfeat": "baseline: set-features"}
SHORT = {"b1": "B1", "b2_mediapipe": "B2-MP", "b2_ktpformer": "B2-KTP", "proposed_j1": "Prop λ1",
         "proposed_lp": "Prop frozen", "base_mean": "mean", "base_group": "group-only", "base_duration": "duration",
         "base_repcount": "rep-count", "base_setfeat": "set-feat."}
VIDEO_TARGETS = ["score", "PF1", "PF2", "PF4", "CF1", "CF2", "CF3", "MQ"]
REP_TARGETS = ["score", "CF1", "CF2", "CF3"]
POOLED_WITHIN = [("experts, all exercises", None, "expert"), ("non-experts, all exercises", None, "non-expert")]


def stratum_mask(df, ex, grp):
    m = np.ones(len(df), bool)
    if ex:
        m &= (df.exercise == ex).to_numpy()
    if grp:
        m &= (df.group == grp).to_numpy()
    return m


def make_strata(v):
    within = [(f"{e} {g}", e, g) for e, g in sorted({(e, g) for e, g in zip(v.exercise, v.group)})]
    both = sorted(e for e, d in v.groupby("exercise") if d.group.nunique() == 2)
    return within, POOLED_WITHIN, [(f"{e} pooled", e, None) for e in both] + [("all pooled", None, None)]


def fmt(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def fci(x, lo, hi, d=2):
    return f"{fmt(x, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def noise_floors(index, items):
    """E3: RMSE of each view's label against its own (exercise, subject, variation) group mean, over expert groups
    with >= 2 views (simultaneous recordings of one performance)."""
    rows = []
    for ex, de in index[index.group == "expert"].groupby("exercise"):
        g = de.groupby("group_id")
        n = g.size()
        keep = n[n >= 2].index
        row = {"exercise": ex, "groups": len(keep)}
        for k in items:
            var = g[k].var(ddof=0)[keep]
            row[k] = float(np.sqrt((var * n[keep]).sum() / n[keep].sum()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("exercise")


def load_runs(root, methods):
    vid, rep, done = [], [], {}
    for m in methods:
        for f in sorted((root / m).glob("fold*/done.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            done[(m, d["fold"])] = d
            for name, lst in (("test_predictions_video.csv", vid), ("test_predictions.csv", rep)):
                p = pd.read_csv(f.parent / name)
                p["fold"], p["method"] = d["fold"], m
                lst.append(p)
    if not done:
        raise SystemExit(f"no finished runs in {root}")
    return pd.concat(vid, ignore_index=True), pd.concat(rep, ignore_index=True), done


def add_mq(pv, v, plans, active):
    """MQ_pred from the video head's items; items inactive in the video's group (T14) -> their training-fold mean
    in that group (E24)."""
    pv = pv.copy()
    plan = {p["fold"]: p for p in plans}
    for k in MQ_ITEMS:
        pred = pv[f"{k}_pred"].to_numpy(float).copy()
        for f, idx in pv.groupby("fold").groups.items():
            tr = v[v.subject.isin(plan[f]["train"])]
            rows = pv.loc[idx]
            for (ex, g), d in tr.groupby(["exercise", "group"]):
                if k in active[(ex, g)]:
                    continue
                sel = rows.index[(rows.exercise == ex) & (rows.group == g)]
                pred[pv.index.get_indexer(sel)] = d[k].mean()
        pv[f"{k}_pred_mq"] = pred
    pv["MQ_pred"] = 100.0 * pv[[f"{k}_pred_mq" for k in MQ_ITEMS]].mean(axis=1)
    return pv


def draws_for(d, n_boot, seed):
    subj = d.drop_duplicates("subject")[["subject", "group"]].sort_values("subject")
    M = subject_draws(subj.subject.to_numpy(), subj.group.to_numpy(), n_boot, seed)
    return subj.subject.to_numpy(), M


def regression_tables(P, strata, entries, active, floors, n_boot, seed, ci):
    score_rows, delta_rows = [], []
    for sname, ex, grp in strata:
        d = P[stratum_mask(P, ex, grp)]
        ref = d[(d.method == entries[0][0]) & (d.source == entries[0][1])].sort_values("video_id")
        subs, M = draws_for(ref, n_boot, seed)
        W = np.vstack([np.ones(len(ref)), item_weights(ref.subject.to_numpy(), subs, M)])
        exs = [ex] if ex else sorted(set(ref.exercise))
        grps = [grp] if grp else ["expert", "non-expert"]
        # pooled over exercises within a group: also a Spearman of the values centred on each video's within-group
        # training mean (the group-only prediction), so between-exercise differences do not count as ranking
        centre = None
        if ex is None and grp is not None:
            bg = d[(d.method == "base_group") & (d.source == "baseline")].sort_values("video_id")
            centre = {t: bg[f"{t}_pred"].to_numpy() for t in ("score", "MQ")}
        res = {}
        for m, src in entries:
            dm = d[(d.method == m) & (d.source == src)].sort_values("video_id")
            assert (dm.video_id.to_numpy() == ref.video_id.to_numpy()).all(), (sname, m, src)
            for t in (REP_TARGETS if src == "rep head" else VIDEO_TARGETS):
                if t not in ("score", "MQ") and (ex is None or grp is None):
                    continue  # items: per exercise x group only
                if t == "MQ" and src == "rep head":
                    continue
                varies = t == "score" or any(t in active.get((e, g), ()) for e in exs for g in grps) or \
                    (t == "MQ" and any(set(MQ_ITEMS) & set(active.get((e, g), ())) for e in exs for g in grps))
                r = regression(dm[t].to_numpy(), dm[f"{t}_pred"].to_numpy(), W)
                res[(m, src, t)] = r
                row = dict(stratum=sname, exercise=ex or "all", group=grp or "both", method=m, source=src, target=t,
                           n=len(dm), subjects=len(subs), varies_within_group=bool(varies))
                for k, val in r.items():
                    row[k] = val[0]
                    row[f"{k}_lo"], row[f"{k}_hi"] = interval(val[1:], ci)
                row["noise_floor"] = floors.loc[ex, t] if ex and grp == "expert" and t in floors else np.nan
                if centre is not None and t in centre:
                    rc = regression(dm[t].to_numpy() - centre[t], dm[f"{t}_pred"].to_numpy() - centre[t], W)["spearman"]
                    row["spearman_centred"] = rc[0]
                    row["spearman_centred_lo"], row["spearman_centred_hi"] = interval(rc[1:], ci)
                score_rows.append(row)
        for m, src in entries:
            if src == "baseline":
                continue
            for vs, vsrc in (("base_group", "baseline"), ("base_setfeat", "baseline"), ("base_repcount", "baseline"),
                             ("b1", src)):
                if (vs, vsrc) == (m, src):
                    continue
                for t in (REP_TARGETS if src == "rep head" else VIDEO_TARGETS):
                    if (m, src, t) not in res or (vs, vsrc, t) not in res:
                        continue
                    for k in ("rmse", "spearman"):
                        dv = res[(m, src, t)][k] - res[(vs, vsrc, t)][k]
                        lo, hi = interval(dv[1:], ci)
                        delta_rows.append(dict(stratum=sname, method=m, source=src, vs=vs, target=t, metric=k,
                                               delta=dv[0], lo=lo, hi=hi, share_below_0=float(np.mean(dv[1:] < 0))))
    return pd.DataFrame(score_rows), pd.DataFrame(delta_rows)


def video_classes(pr, codes):
    """Video level: mean rep-head class probabilities (E18); exercise argmax over the 9 exercises."""
    pe = [f"p_ex_{c}" for c in codes]
    pq = sorted(c for c in pr.columns if c.startswith("p_quality_"))
    g = pr.groupby(["method", "fold", "video_id"], as_index=False)[pe + pq].mean()
    g["exercise_pred"] = np.asarray(codes)[g[pe].to_numpy().argmax(1)]
    g["quality_pred"] = g[pq].to_numpy().argmax(1)
    return g


def classification_tables(pr, pv, v, cfg, methods, n_boot, seed, ci):
    codes = exercise_codes(cfg)
    qn, bins = cfg["labels"]["quality_names"], cfg["labels"]["quality_bins"]
    G = video_classes(pr, codes).merge(v[["video_id", "subject", "group", "exercise", "view", "quality", "score"]],
                                       on="video_id", how="left", validate="many_to_one")
    G["label_ok"] = exercise_label_ok(cfg, G)
    G["ex_true"] = G.exercise.map(codes.index)
    G["ex_pred"] = G.exercise_pred.map(codes.index)
    G["q_true"] = G.quality.map(qn.index)
    sp = pv[["method", "video_id", "score_pred"]]
    G = G.merge(sp, on=["method", "video_id"], how="left", validate="one_to_one")
    G["q_from_score"] = np.digitize(G.score_pred, bins)
    rows, conf, recall = [], [], []
    ex_strata = [("all videos (D8 subjects excluded)", None, None, True), ("all videos (incl. D8 subjects)", None, None, False),
                 ("experts (D8 excluded)", None, "expert", True), ("non-experts", None, "non-expert", True)]
    ex_strata += [(f"view {w} (D8 excluded)", w, None, True) for w in "FLR"]
    for sname, view, grp, excl in ex_strata:
        base = G[G.method == methods[0]]
        sel = np.ones(len(base), bool)
        if view:
            sel &= (base.view == view).to_numpy()
        if grp:
            sel &= (base.group == grp).to_numpy()
        if excl:
            sel &= base.label_ok.to_numpy()
        ref = base[sel].sort_values("video_id")
        if ref.empty:
            continue
        subs, M = draws_for(ref, n_boot, seed)
        W = np.vstack([np.ones(len(ref)), item_weights(ref.subject.to_numpy(), subs, M)])
        for m in methods:
            dm = G[(G.method == m) & G.video_id.isin(set(ref.video_id))].sort_values("video_id")
            assert (dm.video_id.to_numpy() == ref.video_id.to_numpy()).all()
            c = classification(dm.ex_true.to_numpy(), dm.ex_pred.to_numpy(), W, len(codes), labels=range(len(codes)))
            row = dict(task="exercise", stratum=sname, method=m, n=len(dm), subjects=len(subs))
            for k, val in c.items():
                row[k] = val[0]
                row[f"{k}_lo"], row[f"{k}_hi"] = interval(val[1:], ci)
            rows.append(row)
            if sname.startswith("all videos"):
                cm = pd.crosstab(pd.Categorical(dm.exercise, codes), pd.Categorical(dm.exercise_pred, codes),
                                 dropna=False)
                cm.index.name, cm.columns.name = "true", "predicted"
                conf.append(cm.assign(method=m, stratum=sname).reset_index())
                for e in codes:
                    de = dm[dm.exercise == e]
                    recall.append(dict(stratum=sname, method=m, exercise=e, n=len(de),
                                       recall=float((de.exercise_pred == e).mean()),
                                       top_confusion=de.exercise_pred[de.exercise_pred != e].value_counts().index[0]
                                       if (de.exercise_pred != e).any() else ""))
    # derived quality class (D6), within groups pooled over exercises and per stratum
    q_strata = [(n, e, g) for n, e, g in POOLED_WITHIN] + [(f"{e} {g}", e, g) for e, g in
                                                         sorted({(e, g) for e, g in zip(v.exercise, v.group)})]
    for sname, ex, grp in q_strata:
        base = G[G.method == methods[0]]
        ref = base[stratum_mask(base, ex, grp)].sort_values("video_id")
        if ref.empty:
            continue
        subs, M = draws_for(ref, n_boot, seed)
        W = np.vstack([np.ones(len(ref)), item_weights(ref.subject.to_numpy(), subs, M)])
        for m in methods:
            dm = G[(G.method == m) & G.video_id.isin(set(ref.video_id))].sort_values("video_id")
            for src in ("quality_pred", "q_from_score"):
                lab = sorted(set(dm.q_true) | set(dm[src]))
                c = classification(dm.q_true.to_numpy(), dm[src].to_numpy(), W, len(qn), labels=lab)
                row = dict(task=f"quality ({'quality head' if src == 'quality_pred' else 'binned video-head score'})",
                           stratum=sname, method=m, n=len(dm), subjects=len(subs))
                for k, val in c.items():
                    row[k] = val[0]
                    row[f"{k}_lo"], row[f"{k}_hi"] = interval(val[1:], ci)
                rows.append(row)
    return pd.DataFrame(rows), pd.concat(conf, ignore_index=True), pd.DataFrame(recall), G


def vci_tables(root, methods, folds, cfg, strata, n_boot, seed, ci):
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    index = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    lab = ["exercise", "subject", "group", "view", "variation", "group_id"]
    meta = reps.drop(columns=[c for c in lab if c in reps.columns]).merge(index[["video_id", *lab]], on="video_id",
                                                                         how="left", validate="many_to_one")
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    assert (members.item.to_numpy() == np.arange(len(meta))).all()
    gcode = pd.factorize(meta.group_id)[0]
    subjects = sorted(meta.subject.unique())
    spos = {s: i for i, s in enumerate(subjects)}
    sgroup = meta.drop_duplicates("subject").set_index("subject").group
    sums, fold_rows = {}, []
    for m in methods:
        for stage in ("init", "pretrained", "final"):
            for f in folds:
                fp = root / m / f"fold{f}" / f"embeddings_{stage}.npz"
                if not fp.exists():
                    continue
                with np.load(fp) as z:
                    it, emb, center = z["item"], {"centred": z["z"], "raw": z["z"], "proj": z["proj"]}, z["z_train_mean"]
                mm = members.iloc[it]
                for sname, ex, grp in strata:
                    sel = stratum_mask(mm, ex, grp)
                    if not sel.any():
                        continue
                    for var, e in emb.items():
                        s = vci_sums(e[sel], mm.set_id.to_numpy()[sel], mm.view.to_numpy()[sel], gcode[it][sel],
                                     np.array([spos[x] for x in mm.subject.to_numpy()[sel]]), len(subjects),
                                     center if var == "centred" else None)
                        k = (m, stage, sname, var)
                        sums[k] = add_sums(sums.get(k), s)
                        fold_rows.append(dict(method=m, stage=stage, stratum=sname, variant=var, fold=f,
                                              **vci_from_sums(s)))
    fv = pd.DataFrame(fold_rows)
    rows = []
    for (m, stage, sname, var), s in sums.items():
        present = np.flatnonzero((s["same_cnt"] > 0) | (s["other_cnt"].sum(1) > 0))
        Mf = np.zeros((n_boot, len(subjects)))
        Mf[:, present] = subject_draws(present, [sgroup[subjects[i]] for i in present], n_boot, seed)
        pt, b = vci_from_sums(s), vci_from_sums(s, Mf)
        lo, hi = interval(b["vci"], ci)
        x = fv[(fv.method == m) & (fv.stage == stage) & (fv.stratum == sname) & (fv.variant == var)].vci
        rows.append(dict(method=m, stage=stage, stratum=sname, variant=var, vci=pt["vci"], vci_lo=lo, vci_hi=hi,
                         same=pt["same"], other=pt["other"], same_pairs=int(s["same_cnt"].sum() // 2),
                         subjects_with_pairs=int((s["same_cnt"] > 0).sum()), fold_mean=x.mean(),
                         fold_sd=x.std(ddof=1), folds=len(x)))
    return pd.DataFrame(rows), fv


def training_table(root, done, v):
    rows = []
    for (m, f), d in sorted(done.items()):
        for stage in ("cvdc_stage", "supervised_stage"):
            if stage not in d:
                continue
            s = d[stage]
            row = dict(method=m, fold=f, stage=stage.split("_")[0], best_epoch=s["best_epoch"],
                       stopped_epoch=s["stopped_epoch"], max_epochs=s["max_epochs"], early_stopped=s["early_stopped"],
                       best_val=s["best_val"], stage_s=s["wall_s"], reused_from=s.get("reused_from", ""))
            if stage == "supervised_stage":
                h = pd.read_csv(root / m / f"fold{f}" / "finetune" / "history.csv")
                hb = h[h.epoch == s["best_epoch"]].iloc[0]
                row["val_video_head_rmse_best"] = hb.get("val/video_head_rmse", np.nan)
                sub = d["subjects"]
                vt, vv = v[v.subject.isin(sub["inner_train"])], v[v.subject.isin(sub["inner_val"])]
                gm = vt.groupby(["exercise", "group"]).score.mean()
                gp = np.array([gm.get((e, g), vt.score.mean()) for e, g in zip(vv.exercise, vv.group)])
                row["val_rmse_inner_train_mean"] = float(np.sqrt(((vv.score - vt.score.mean()) ** 2).mean()))
                row["val_rmse_inner_train_group_mean"] = float(np.sqrt(((vv.score - gp) ** 2).mean()))
            rows.append(row)
        rows.append(dict(method=m, fold=f, stage="total", stage_s=d["wall_s"], peak_gpu_gb=d.get("peak_gpu_gb")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_full.yaml")
    ap.add_argument("--model-config")
    ap.add_argument("--tag", default="full_gkf10")
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config, a.model_config)
    ec, ia = cfg["eval"], cfg["heads"]["item_activity"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out = Path(a.out) if a.out else resolve(ec["results_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    smoke = a.tag.endswith("_smoke")

    v = video_table(cfg)
    v["MQ"] = movement_quality(v)
    index = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    index["MQ"] = movement_quality(index)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    plans = fold_plan(cfg, reps, smoke=smoke)
    act = item_activity(v, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    act.to_csv(out / "item_activity.csv", index=False)
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}

    pv, pr, done = load_runs(root, a.methods)
    methods = [m for m in METHODS if m in set(pv.method)]
    folds = sorted({f for (_, f) in done})
    plans = [p for p in plans if p["fold"] in folds]
    for (m, f), d in done.items():  # every run used exactly these folds and inner splits
        p = next(q for q in plans if q["fold"] == f)
        for k in ("train", "inner_train", "inner_val", "test"):
            assert sorted(d["subjects"][k]) == sorted(p[k]), (m, f, k)
    for m in methods:
        assert sorted(f for (mm, f) in done if mm == m) == folds, f"{m}: folds {sorted(f for (mm, f) in done if mm == m)}"
    tv = v[v.subject.isin({s for p in plans for s in p["test"]})]

    # ---- one long table of video-level predictions: (method, source) x video
    pv = add_mq(pv, v, plans, active)
    vp = pv[["method", "fold", "video_id", *[f"{t}_pred" for t in VIDEO_TARGETS]]].copy()
    vp["source"] = "video head"
    rp = pr.groupby(["method", "fold", "video_id"])[[f"{t}_pred" for t in REP_TARGETS]].mean().reset_index()
    rp["source"] = "rep head"
    base = video_baselines(v, plans, VIDEO_TARGETS)
    base = base[base.method.isin(BASELINES)]
    base["source"] = "baseline"
    P = pd.concat([vp, rp, base], ignore_index=True).merge(
        v[["video_id", "subject", "group", "exercise", "view", "variation", *VIDEO_TARGETS]], on="video_id",
        how="left", validate="many_to_one")
    n = P.groupby(["method", "source"]).video_id.nunique()
    assert (n == len(tv)).all(), f"every method must predict every test video once: {n.to_dict()}"
    assert not P.duplicated(["method", "source", "video_id"]).any()
    P.to_csv(out / "video_predictions.csv", index=False)
    floors = noise_floors(index, ["score", "MQ", "PF1", "PF2", "PF4", "CF1", "CF2", "CF3"])
    floors.to_csv(out / "noise_floor.csv")

    within, pooled_within, pooled = make_strata(tv)
    strata = within + pooled_within + pooled
    entries = [(m, "video head") for m in methods] + [(m, "rep head") for m in methods] + \
              [(b, "baseline") for b in BASELINES]
    S, D = regression_tables(P, strata, entries, active, floors, n_boot, seed, ci)
    S.to_csv(out / "score_metrics.csv", index=False)
    D.to_csv(out / "paired_deltas.csv", index=False)
    print(f"regression tables in {time.time() - t0:.0f} s", flush=True)

    C, CM, R, G = classification_tables(pr, pv, v, cfg, methods, n_boot, seed, ci)
    C.to_csv(out / "classification.csv", index=False)
    CM.to_csv(out / "confusion_exercise.csv", index=False)
    R.to_csv(out / "exercise_recall.csv", index=False)
    G.to_csv(out / "video_classes.csv", index=False)
    print(f"classification in {time.time() - t0:.0f} s", flush=True)

    V, FV = vci_tables(root, methods, folds, cfg, strata, n_boot, seed, ci)
    V.to_csv(out / "vci.csv", index=False)
    FV.to_csv(out / "vci_per_fold.csv", index=False)
    print(f"VCI in {time.time() - t0:.0f} s", flush=True)

    T = training_table(root, done, v)
    T.to_csv(out / "stopping_runtime.csv", index=False)
    write_summary(out, S, D, C, CM, R, V, T, act, floors, methods, folds, n_boot, smoke, tv, cfg, within,
                  pooled_within, pooled)
    print(f"report written to {out} in {time.time() - t0:.0f} s")


# --------------------------------------------------------------------------------------------- summary
def s_row(S, sname, m, src, t):
    r = S[(S.stratum == sname) & (S.method == m) & (S.source == src) & (S.target == t)]
    return None if r.empty else r.iloc[0]


def cell(S, sname, m, src, t, d=2, ci=True, rho=True, centred=False):
    r = s_row(S, sname, m, src, t)
    if r is None:
        return "–"
    c = fci(r.rmse, r.rmse_lo, r.rmse_hi, d) if ci else fmt(r.rmse, d)
    if centred and "spearman_centred" in r and np.isfinite(r.spearman_centred):
        return c + f" / {fmt(r.spearman)} / {fci(r.spearman_centred, r.spearman_centred_lo, r.spearman_centred_hi)}"
    return c + (f" / {fmt(r.spearman)}" if rho else "")


def d_row(D, sname, m, src, vs, t, metric="rmse"):
    g = D[(D.stratum == sname) & (D.method == m) & (D.source == src) & (D.vs == vs) & (D.target == t) &
          (D.metric == metric)]
    return None if g.empty else g.iloc[0]


def d_cell(D, sname, m, src, vs, t, metric="rmse", d=2):
    g = d_row(D, sname, m, src, vs, t, metric)
    if g is None:
        return "–"
    s = fci(g.delta, g.lo, g.hi, d)
    return f"**{s}**" if (g.lo > 0 or g.hi < 0) else s


def write_summary(out, S, D, C, CM, R, V, T, act, floors, methods, folds, n_boot, smoke, tv, cfg, within,
                  pooled_within, pooled):
    tc, es = cfg["train"], cfg["train"]["early_stopping"]
    L = []
    w = L.append
    wn = [s for s, _, _ in within]
    pw = [s for s, _, _ in pooled_within]
    w("# Phase 3, full scope (E01–E09): GroupKFold-10" + (" (SMOKE RUN: plumbing check, not a result)" if smoke else ""))
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/phase3_full/report_full.py`. Folds {folds} of a "
      f"stratified GroupKFold-{cfg['eval']['k']} over the 58 subjects (`group_kfold(meta, {cfg['eval']['k']}, seed=0)`, "
      "subjects dealt round-robin within expert / non-expert: 2–3 experts and 3–4 non-experts per test fold); inner "
      "validation = 20% of each group's training subjects (T8a). Test predictions pooled over the folds (E1); "
      f"{len(tv)} test videos with ≥ 1 rep. Same settings as the pilot's step (a2): video head (T13), every video "
      f"weighted once (T6), targets standardised on the inner training subjects (T7), early stopping on the EMA "
      f"(β {es['ema']}) inner-validation objective, ≥ {es['min_epochs']} epochs, patience {es['patience']}, max "
      f"{tc['max_epochs_sup']} (T8b); CVDC pretraining per fold on the inner training subjects only (max "
      f"{tc['max_epochs_cvdc']} epochs, T8a), shared by the two Proposed variants (T15); items learned and reported "
      f"only where they vary (T14: off-mode in ≥ {cfg['heads']['item_activity']['min_subjects_off_mode']} subjects, "
      f"SD ≥ {cfg['heads']['item_activity']['min_sd']}). E03 P02 is masked out of the exercise labels (D8).")
    w("")
    w(f"95% CIs: subject bootstrap ({n_boot} draws within expert / non-expert, the same draws for every method, so "
      "differences are paired; E17). Score and MQ on 0–100, items 0–1. **Bold** = a paired CI that excludes 0. "
      "Limitations: the expert / non-expert contrast is confounded (clip length, views, year, phones, recording "
      "mode), so within-group results are primary; E02, E04, E06, E07 have experts only; E07–E09 all male; scores "
      "skewed; PF3/PF4 differ between the views of one performance (label-noise floor); GroupKFold-10, not LOSO.")
    w("")
    # ---------------- headline: total score, pooled within groups
    w("## 1. Total score within groups, video level")
    w("")
    w("### 1a. Pooled over the exercises within each group")
    w("")
    w("RMSE [95% CI] / Spearman ρ / centred ρ [95% CI]. 'video head' = the set-level head (T13); 'rep head' = the mean "
      "of the video's rep predictions. The group-only baseline predicts each exercise × group's training mean, so the "
      "plain pooled ρ also contains between-exercise differences (the group-only baseline's own ρ shows how much); "
      "**centred ρ** ranks label and prediction after subtracting that within-group training mean from both, i.e. "
      "the ranking within exercises (undefined for the constant baselines).")
    w("")
    w("| Method | head | " + " | ".join(pw) + " |")
    w("|---|---|" + "---|" * len(pw))
    for src in ("video head", "rep head"):
        for m in methods:
            w(f"| {LABEL[m]} | {src} | " + " | ".join(cell(S, s, m, src, "score", centred=True) for s in pw) + " |")
    for b in BASELINES:
        w(f"| {LABEL[b]} | – | " + " | ".join(cell(S, s, b, "baseline", "score", centred=True) for s in pw) + " |")
    w("")
    w("Paired RMSE difference, video head − group-only [95% CI] (negative = better than the within-group mean):")
    w("")
    w("| Method | " + " | ".join(pw) + " |")
    w("|---|" + "---|" * len(pw))
    for m in methods:
        w(f"| {SHORT[m]} | " + " | ".join(d_cell(D, s, m, "video head", "base_group", "score") for s in pw) + " |")
    w("")
    w("### 1b. Per exercise × group (video head)")
    w("")
    w("RMSE / Spearman ρ; noise floor = the inter-view label-noise floor of the expert groups (E3).")
    w("")
    cols = methods + ["base_group", "base_setfeat", "base_repcount", "base_duration", "base_mean"]
    w("| Stratum (videos / subjects) | " + " | ".join(SHORT[c] for c in cols) + " | floor |")
    w("|---|" + "---|" * (len(cols) + 1))
    for s in wn:
        r0 = s_row(S, s, methods[0], "video head", "score")
        cells = [cell(S, s, c, "video head" if c in methods else "baseline", "score", ci=False) for c in cols]
        w(f"| {s} ({r0.n} / {r0.subjects}) | " + " | ".join(cells) + f" | {fmt(r0.noise_floor)} |")
    w("")
    w("Paired RMSE difference, video head − group-only [95% CI]:")
    w("")
    w("| Stratum | " + " | ".join(SHORT[m] for m in methods) + " |")
    w("|---|" + "---|" * len(methods))
    for s in wn:
        w(f"| {s} | " + " | ".join(d_cell(D, s, m, "video head", "base_group", "score") for m in methods) + " |")
    w("")
    w("Paired RMSE difference, video head − set-features [95% CI]:")
    w("")
    w("| Stratum | " + " | ".join(SHORT[m] for m in methods) + " |")
    w("|---|" + "---|" * len(methods))
    for s in wn + pw:
        w(f"| {s} | " + " | ".join(d_cell(D, s, m, "video head", "base_setfeat", "score") for m in methods) + " |")
    w("")
    w("Paired Spearman difference vs B1, video head [95% CI] (positive = ranks better than B1):")
    w("")
    others = [m for m in methods if m != "b1"]
    w("| Stratum | " + " | ".join(SHORT[m] for m in others) + " |")
    w("|---|" + "---|" * len(others))
    for s in wn + pw:
        w(f"| {s} | " + " | ".join(d_cell(D, s, m, "video head", "b1", "score", "spearman") for m in others) + " |")
    w("")
    # how many strata beat the group mean
    w("### 1c. Count of strata where a model beats the within-group mean")
    w("")
    w("Number of the exercise × group strata whose paired RMSE CI vs group-only lies entirely below 0 / entirely "
      f"above 0 / includes 0 (of {len(wn)}), total score and MQ, video head:")
    w("")
    w("| Method | total score: below / above / includes 0 | MQ: below / above / includes 0 |")
    w("|---|---|---|")
    for m in methods:
        cells = []
        for t in ("score", "MQ"):
            g = [d_row(D, s, m, "video head", "base_group", t) for s in wn]
            g = [x for x in g if x is not None]
            cells.append(f"{sum(x.hi < 0 for x in g)} / {sum(x.lo > 0 for x in g)} / "
                         f"{sum(x.lo <= 0 <= x.hi for x in g)}")
        w(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
    w("")
    # ---------------- MQ
    w("## 2. Movement-quality score (MQ, D13) within groups, video level")
    w("")
    w("MQ = 100 × mean(PF1, PF2, CF1, CF2, CF3), from the video head's items (items that do not vary in the group, "
      "T14, at their training-fold mean there); the baselines are fitted on MQ itself. RMSE [95% CI] / ρ / centred ρ "
      "[95% CI] (as in 1a).")
    w("")
    w("| Method | " + " | ".join(pw) + " |")
    w("|---|" + "---|" * len(pw))
    for m in methods:
        w(f"| {LABEL[m]} | " + " | ".join(cell(S, s, m, "video head", "MQ", centred=True) for s in pw) + " |")
    for b in BASELINES:
        w(f"| {LABEL[b]} | " + " | ".join(cell(S, s, b, "baseline", "MQ", centred=True) for s in pw) + " |")
    w("")
    w("Per exercise × group, RMSE / ρ (strata where no MQ item varies are marked ·):")
    w("")
    w("| Stratum | " + " | ".join(SHORT[c] for c in cols) + " | floor |")
    w("|---|" + "---|" * (len(cols) + 1))
    for s in wn:
        r0 = s_row(S, s, methods[0], "video head", "MQ")
        mark = "" if r0.varies_within_group else " ·"
        cells = [cell(S, s, c, "video head" if c in methods else "baseline", "MQ", ci=False) for c in cols]
        w(f"| {s}{mark} | " + " | ".join(cells) + f" | {fmt(r0.noise_floor)} |")
    w("")
    w("Paired MQ RMSE difference, video head − group-only [95% CI]:")
    w("")
    w("| Stratum | " + " | ".join(SHORT[m] for m in methods) + " |")
    w("|---|" + "---|" * len(methods))
    for s in wn + pw:
        w(f"| {s} | " + " | ".join(d_cell(D, s, m, "video head", "base_group", "MQ") for m in methods) + " |")
    w("")
    # ---------------- items
    w("## 3. EAAQ items (video head), only where they vary (T14)")
    w("")
    w("Number of (stratum, item) cells where the paired RMSE CI vs group-only lies below 0, out of the varying "
      "cells; full table in `score_metrics.csv` / `paired_deltas.csv`.")
    w("")
    items = ["PF1", "PF2", "PF4", "CF1", "CF2", "CF3"]
    w("| Method | " + " | ".join(items) + " | all |")
    w("|---|" + "---|" * (len(items) + 1))
    for m in methods:
        tot, cells = [0, 0], []
        for t in items:
            g = [d_row(D, s, m, "video head", "base_group", t) for s in wn]
            vary = [s for s in wn if (r := s_row(S, s, m, "video head", t)) is not None and r.varies_within_group]
            g = [x for x, s in zip(g, wn) if x is not None and s in vary]
            k = sum(x.hi < 0 for x in g)
            tot[0] += k
            tot[1] += len(g)
            cells.append(f"{k} / {len(g)}")
        w(f"| {SHORT[m]} | " + " | ".join(cells) + f" | {tot[0]} / {tot[1]} |")
    w("")
    w("Items active per stratum (T14): " + "; ".join(
        f"{s}: {', '.join(x for x in ['PF1', 'PF2', 'PF4', 'CF1', 'CF2', 'CF3'] if act[(act.stratum == s) & (act.item == x)].varies.any()) or 'none'}"
        for s in wn) + ".")
    w("")
    # ---------------- classification
    w("## 4. Exercise classification (9 classes), video level")
    w("")
    w("Rep head, mean class probabilities per video, argmax over the nine exercises (E18). Accuracy / macro "
      "precision / recall / F1 [95% CI] / Cohen's κ. 'D8 excluded' leaves out E03 P02's videos (wrong exercise; "
      "not in the exercise labels).")
    w("")
    Ce = C[C.task == "exercise"]
    for sname in Ce.stratum.unique():
        w(f"**{sname}**")
        w("")
        w("| Method | accuracy | macro P | macro R | macro F1 | κ |")
        w("|---|---|---|---|---|---|")
        for m in methods:
            r = Ce[(Ce.stratum == sname) & (Ce.method == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            w(f"| {SHORT[m]} ({r.n} videos) | {fci(r.accuracy, r.accuracy_lo, r.accuracy_hi, 3)} | "
              f"{fmt(r.macro_precision, 3)} | {fmt(r.macro_recall, 3)} | {fci(r.macro_f1, r.macro_f1_lo, r.macro_f1_hi, 3)} "
              f"| {fmt(r.kappa, 3)} |")
        w("")
        if sname != "all videos (D8 subjects excluded)":
            continue
        w("Per-exercise recall (D8 excluded):")
        w("")
        codes = exercise_codes(cfg)
        w("| Method | " + " | ".join(codes) + " |")
        w("|---|" + "---|" * len(codes))
        Rs = R[R.stratum == sname]
        for m in methods:
            w(f"| {SHORT[m]} | " + " | ".join(
                fmt(Rs[(Rs.method == m) & (Rs.exercise == e)].recall.iloc[0], 2) for e in codes) + " |")
        w("")
        for m in [x for x in (cfg["eval"]["proposed_default"], "b1") if x in methods]:
            cm = CM[(CM.method == m) & (CM.stratum == sname)].drop(columns=["method", "stratum"]).set_index("true")
            w(f"Confusion matrix, {LABEL[m]} (rows true, columns predicted):")
            w("")
            w("| true \\ predicted | " + " | ".join(cm.columns) + " |")
            w("|---|" + "---|" * len(cm.columns))
            for e, row in cm.iterrows():
                w(f"| {e} | " + " | ".join(str(int(x)) for x in row) + " |")
            w("")
    Cq = C[C.task.str.startswith("quality")]
    w("**Derived quality class (D6; secondary)**, macro F1 [95% CI] / κ, pooled over exercises within groups:")
    w("")
    w("| Method | source | " + " | ".join(pw) + " |")
    w("|---|---|" + "---|" * len(pw))
    for m in methods:
        for task in sorted(Cq.task.unique()):
            cells = []
            for s in pw:
                r = Cq[(Cq.task == task) & (Cq.stratum == s) & (Cq.method == m)]
                cells.append("–" if r.empty else f"{fci(r.macro_f1.iloc[0], r.macro_f1_lo.iloc[0], r.macro_f1_hi.iloc[0])}"
                             f" / {fmt(r.kappa.iloc[0])}")
            w(f"| {SHORT[m]} | {task[9:-1]} | " + " | ".join(cells) + " |")
    w("")
    # ---------------- VCI
    w("## 5. Viewpoint Consistency Index within each exercise × group (centred 256-d)")
    w("")
    w("VCI = mean cosine of the same rep seen from another view − mean cosine to other (exercise, subject, "
      "variation) groups of the same stratum; held-out subjects, plain real views, pooled over the folds, subject-"
      "bootstrap CI (E13, E19). 'after CVDC' = the shared CVDC encoder before supervised training (= the frozen "
      "variant). Same-rep pairs: sync / audio pairs in expert strata, rep-matched pairs only in non-expert strata. "
      "The rows pooled over exercises are secondary: their 'other groups' include other exercises, so they also "
      "measure exercise separation (E13).")
    w("")
    ref_init = "b1" if "b1" in methods else methods[0]
    pre = next((m for m in methods if not V[(V.method == m) & (V.stage == "pretrained")].empty), None)
    vcols = [("random init", ref_init, "init")] + ([("after CVDC", pre, "pretrained")] if pre else []) + \
        [(SHORT[m], m, "final") for m in methods if m != "proposed_lp"]

    def vcell(m, stage, s):
        d = V[(V.method == m) & (V.stage == stage) & (V.stratum == s) & (V.variant == "centred")]
        return "–" if d.empty or not np.isfinite(d.vci.iloc[0]) else fci(d.vci.iloc[0], d.vci_lo.iloc[0],
                                                                           d.vci_hi.iloc[0], 3)

    w("| Stratum (same-rep pairs) | " + " | ".join(c[0] for c in vcols) + " |")
    w("|---|" + "---|" * len(vcols))
    for s in wn + pw + ["all pooled"]:
        d = V[(V.method == ref_init) & (V.stage == "init") & (V.stratum == s) & (V.variant == "centred")]
        npairs = int(d.same_pairs.iloc[0]) if len(d) else 0
        w(f"| {s} ({npairs}) | " + " | ".join(vcell(m, st, s) for _, m, st in vcols) + " |")
    w("")
    w("The frozen variant's final VCI equals 'after CVDC' by construction. Random-init references of the B2 inputs, "
      "the raw 256-d and the 128-d projection variants: `vci.csv`.")
    w("")
    # ---------------- group probe
    gp = out / "group_probe_summary.csv"
    w("## 6. Expert / non-expert group probe (E11)")
    w("")
    if gp.exists():
        G = pd.read_csv(gp)
        w("Linear probes on each method's frozen video embeddings (mean of the video's rep embeddings z from the "
          "fold's final checkpoint; probes trained on the fold's training subjects, scored on its test subjects; "
          "regularisation chosen by an inner subject-grouped CV). **Group probe**: logistic regression expert vs "
          "non-expert per exercise that has both groups (E01, E03, E05, E08, E09), balanced accuracy and AUC pooled "
          "over those exercises; **view-matched**: the same on front-view FL videos only (where both groups were "
          "filmed alike). **Score probe**: ridge regression of the total score within each exercise × group, "
          "out-of-fold Spearman and R² relative to the within-group training mean (R² > 0 = better than the "
          "group-only baseline), pooled within group. Script: `scripts/phase3_full/group_probe.py`.")
        w("")
        w("| Encoder | group AUC | group bal. acc. | view-matched AUC | view-matched bal. acc. | score probe experts ρ / R² | score probe non-experts ρ / R² |")
        w("|---|---|---|---|---|---|---|")
        for r in G.itertuples():
            w(f"| {r.label} | {fci(r.auc, r.auc_lo, r.auc_hi, 3)} | {fci(r.bacc, r.bacc_lo, r.bacc_hi, 3)} | "
              f"{fci(r.vm_auc, r.vm_auc_lo, r.vm_auc_hi, 3)} | {fci(r.vm_bacc, r.vm_bacc_lo, r.vm_bacc_hi, 3)} | "
              f"{fmt(r.sp_exp_rho)} [{fmt(r.sp_exp_rho_lo)}, {fmt(r.sp_exp_rho_hi)}] / {fmt(r.sp_exp_r2)} | "
              f"{fmt(r.sp_non_rho)} [{fmt(r.sp_non_rho_lo)}, {fmt(r.sp_non_rho_hi)}] / {fmt(r.sp_non_r2)} |")
        w("")
        w("Per exercise: `group_probe_per_exercise.csv`; per stratum score probes: `score_probe.csv`.")
    else:
        w("Not run yet (`scripts/phase3_full/group_probe.py`).")
    w("")
    # ---------------- pooled (secondary)
    w("## 7. Pooled over both groups (secondary), total score, video head")
    w("")
    ps = [s for s, _, _ in pooled]
    w("RMSE / Pearson r. Pooled metrics mostly measure the expert / non-expert separation (E9, E10).")
    w("")
    w("| Method | " + " | ".join(ps) + " |")
    w("|---|" + "---|" * len(ps))
    for m, src in [(m, "video head") for m in methods] + [(b, "baseline") for b in BASELINES]:
        cells = []
        for s in ps:
            r = s_row(S, s, m, src, "score")
            cells.append("–" if r is None else f"{fmt(r.rmse)} / {fmt(r.pearson)}")
        w(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
    w("")
    # ---------------- training
    w("## 8. Training: kept and stopping epochs, wall time")
    w("")
    w("| Method | stage | kept epochs (folds 0–9) | stopped epochs | early-stopped | wall time total (h) |")
    w("|---|---|---|---|---|---|")
    for m in methods:
        for st in ("cvdc", "supervised"):
            d = T[(T.method == m) & (T.stage == st)]
            if d.empty:
                continue
            reused = d.reused_from.fillna("").astype(str).str.len().gt(0).all() if "reused_from" in d else False
            w(f"| {SHORT[m]} | {st}{' (reused from ' + Path(str(d.reused_from.iloc[0])).parts[-3] + ')' if reused else ''} | "
              f"{', '.join(str(int(x)) for x in d.best_epoch)} | {', '.join(str(int(x)) for x in d.stopped_epoch)} | "
              f"{int(d.early_stopped.sum())}/{len(d)} | {d.stage_s.sum() / 3600:.2f} |")
        tt = T[(T.method == m) & (T.stage == "total")]
        w(f"| {SHORT[m]} | total | | | | {tt.stage_s.sum() / 3600:.2f} (peak GPU {tt.peak_gpu_gb.max():.1f} GB) |")
    w("")
    sup = T[T.stage == "supervised"]
    w("Inner-validation diagnostic (video-head score RMSE at the kept epoch vs the inner-training mean / within-group "
      "mean on the same validation videos), median over folds: " + "; ".join(
          f"{SHORT[m]} {fmt(sup[sup.method == m].val_video_head_rmse_best.median())} vs "
          f"{fmt(sup[sup.method == m].val_rmse_inner_train_mean.median())} / "
          f"{fmt(sup[sup.method == m].val_rmse_inner_train_group_mean.median())}" for m in methods) + ".")
    w("")
    w("Tables: `score_metrics.csv`, `paired_deltas.csv`, `video_predictions.csv`, `classification.csv`, "
      "`confusion_exercise.csv`, `exercise_recall.csv`, `video_classes.csv`, `vci.csv`, `vci_per_fold.csv`, "
      "`stopping_runtime.csv`, `noise_floor.csv`, `item_activity.csv`, `group_probe_*.csv`, `score_probe.csv`.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
