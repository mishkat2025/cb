"""Full scope, one model per exercise (T17; twelfth review item 3): the per-exercise report.

Reads runs/full/<tag>/<EX>/<method>/fold<k>/ (cvdc.train.run_per_exercise): test_predictions_video.csv (video head,
T13; headline), test_predictions.csv (rep head), embeddings_*.npz, done.json; and, when present,
<results>/<tag>/embedding_distance_reps.csv (scripts/full/phase3/embedding_distance_perex.py) and the group-probe
tables (scripts/full/phase3/group_probe.py). Every exercise has its own GroupKFold-5 folds; test predictions are
pooled over its folds (E1). Within groups first (E9): every exercise x group stratum; pooled over the exercises within
each group second (each video predicted by its own exercise's model). Same metrics as the pilot's step (a2):
  score / MQ  RMSE, MAE, Pearson, Spearman (video head and rep head; MQ from the video head's items, D13), against
              predict-the-mean, group-only, duration-only, rep-count-only and set-features (fitted per exercise and fold
              on the training videos, E16, E20); paired differences vs group-only, set-features, rep-count-only and B1
  VCI         centred 256-d (headline), raw, projection, within every exercise x group (E13, E19): random init, after
              CVDC (shared by the Proposed variants), after fine-tuning
  distance    view stability of the distance to the PT reference over synced expert performances (rep level, E22):
              ICC(1) of the distance standardised per (method, exercise, fold, group) with the training subjects
  group probe E11 (from group_probe.py)
  training    kept / stopping epochs and wall time per exercise and method
Strata and exercises with fewer than ``eval.per_exercise_low_confidence_subjects`` subjects are flagged (⚠, T17).

usage: python scripts/full/phase3/report_perex.py [--config configs/per_exercise.yaml] [--tag perex_gkf5] [--boot 2000]
       [--exercises E01 ...]
Writes CSV tables and summary.md to <eval.results_dir>/<tag>/ (or --out; the group-probe tables are read from there).
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
sys.path.insert(0, str(REPO / "scripts" / "full" / "phase3"))
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase4"))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.data.labels import MQ_ITEMS, movement_quality  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_baselines, video_table  # noqa: E402
from cvdc.eval.metrics import add_sums, interval, subject_draws, vci_from_sums, vci_sums  # noqa: E402
from cvdc.train.trainer import fold_plan  # noqa: E402
from report_full import (BASELINES, REP_TARGETS, VIDEO_TARGETS, cell, d_cell, d_row, fci, fmt, noise_floors,  # noqa: E402
                         regression_tables, s_row, stratum_mask)
from report_phase4 import standardise_per_fold, view_stability  # noqa: E402

METHODS = ["b1", "b2_ktpformer", "proposed_j1", "proposed_lp"]
LABEL = {"b1": "B1 (2D xy, scratch)", "b2_ktpformer": "B2-KTPFormer (lifted xyz, scratch)",
         "proposed_j1": "Proposed (CVDC + fine-tune, λ = 1.0)", "proposed_lp": "Proposed alt. (frozen CVDC encoder)",
         "init_xy": "random-init encoder (2D, floor)", "base_mean": "predict-the-mean",
         "base_group": "group-only (within-group mean)", "base_duration": "duration-only",
         "base_repcount": "rep-count-only", "base_setfeat": "set-features"}
SHORT = {"b1": "B1", "b2_ktpformer": "B2-KTP", "proposed_j1": "Prop λ1", "proposed_lp": "Prop frozen",
         "init_xy": "random init", "base_mean": "mean", "base_group": "group-only", "base_duration": "duration",
         "base_repcount": "rep-count", "base_setfeat": "set-feat."}
POOLED_WITHIN = [("experts, all exercises", None, "expert"), ("non-experts, all exercises", None, "non-expert")]


def fkey(ex, f):
    return f"{ex}:{f}"


def load_runs(root, exercises, methods):
    vid, rep, done = [], [], {}
    for ex in exercises:
        for m in methods:
            for f in sorted((root / ex / m).glob("fold*/done.json")):
                d = json.loads(f.read_text(encoding="utf-8"))
                done[(ex, m, d["fold"])] = d
                for name, lst in (("test_predictions_video.csv", vid), ("test_predictions.csv", rep)):
                    p = pd.read_csv(f.parent / name)
                    p["fold"], p["method"] = fkey(ex, d["fold"]), m
                    lst.append(p)
    if not done:
        raise SystemExit(f"no finished runs in {root}")
    return pd.concat(vid, ignore_index=True), pd.concat(rep, ignore_index=True), done


def add_mq(pv, v, plans, active):
    """MQ_pred from the video head's items; items inactive in the video's group (T14) -> their training-fold mean."""
    pv = pv.copy()
    for k in MQ_ITEMS:
        pred = pv[f"{k}_pred"].to_numpy(float).copy()
        for f, idx in pv.groupby("fold").groups.items():
            p = plans[f]
            tr = v[v.subject.isin(p["train"]) & (v.exercise == p["exercise"])]
            rows = pv.loc[idx]
            for (ex, g), d in tr.groupby(["exercise", "group"]):
                if k in active[(ex, g)]:
                    continue
                sel = rows.index[(rows.exercise == ex) & (rows.group == g)]
                pred[pv.index.get_indexer(sel)] = d[k].mean()
        pv[f"{k}_pred_mq"] = pred
    pv["MQ_pred"] = 100.0 * pv[[f"{k}_pred_mq" for k in MQ_ITEMS]].mean(axis=1)
    return pv


def vci_tables(root, done, cfg, n_boot, seed, ci):
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
    for (ex, m, f) in sorted(done):
        for stage in ("init", "pretrained", "final"):
            fp = root / ex / m / f"fold{f}" / f"embeddings_{stage}.npz"
            if not fp.exists():
                continue
            with np.load(fp) as z:
                it, emb, center = z["item"], {"centred": z["z"], "raw": z["z"], "proj": z["proj"]}, z["z_train_mean"]
            mm = members.iloc[it]
            strata = [(f"{ex} {g}", ex, g) for g in sorted(mm.group.unique())] + \
                [(n, None, g) for n, _, g in POOLED_WITHIN if g in set(mm.group)]
            for sname, sex, grp in strata:
                sel = stratum_mask(mm, sex, grp)
                for var, e in emb.items():
                    s = vci_sums(e[sel], mm.set_id.to_numpy()[sel], mm.view.to_numpy()[sel], gcode[it][sel],
                                 np.array([spos[x] for x in mm.subject.to_numpy()[sel]]), len(subjects),
                                 center if var == "centred" else None)
                    k = (m, stage, sname, var)
                    sums[k] = add_sums(sums.get(k), s)  # pooled rows add within-exercise sums (no cross-exercise pairs)
                    fold_rows.append(dict(method=m, stage=stage, stratum=sname, variant=var, fold=fkey(ex, f),
                                          **vci_from_sums(s)))
    fv = pd.DataFrame(fold_rows)
    rows = []
    for (m, stage, sname, var), s in sums.items():
        present = np.flatnonzero((s["same_cnt"] > 0) | (s["other_cnt"].sum(1) > 0))
        Mf = np.zeros((n_boot, len(subjects)))
        Mf[:, present] = subject_draws(present, [sgroup[subjects[i]] for i in present], n_boot, seed)
        pt, b = vci_from_sums(s), vci_from_sums(s, Mf)
        lo, hi = interval(b["vci"], ci)
        rows.append(dict(method=m, stage=stage, stratum=sname, variant=var, vci=pt["vci"], vci_lo=lo, vci_hi=hi,
                         same=pt["same"], other=pt["other"], same_pairs=int(s["same_cnt"].sum() // 2)))
    return pd.DataFrame(rows), fv


def distance_stability(cfg, src, plans_by_ex, n_boot, seed, ci):
    """View stability of the standardised distance to the PT reference over synced expert performances (E22)."""
    path = src / "embedding_distance_reps.csv"
    if not path.exists():
        return None
    E = pd.read_csv(path)
    E = E[E.exercise.isin(plans_by_ex)]  # the reported exercises only (the file may hold all nine)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    index = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    E = E.merge(reps[["video_id", "rep", "rep_uid"]], on=["video_id", "rep"], how="left", validate="many_to_one")
    E = E.merge(index[["video_id", "subject", "group", "view", "group_id"]], on="video_id", how="left")
    sync = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    synced = set(sync.index[sync.cls == "synced"])
    rows = []
    for ex, de in E.groupby("exercise"):
        de = standardise_per_fold(de, plans_by_ex[ex], "d_cos")
        de = de[de.is_test & de.group_id.isin(synced) & (de.group == "expert")]
        icc_b = {}
        for m, d in de.groupby("method"):
            r, b, _ = view_stability(d.assign(perf=d.rep_uid), "d_cos_z", n_boot, seed, ci)
            icc_b[m] = b
            rows.append(dict(exercise=ex, method=m, **r))
        for row in rows:
            if row["exercise"] != ex or row["method"] == "b1" or "b1" not in icc_b:
                continue
            dd = icc_b[row["method"]] - icc_b["b1"]
            b1 = next(x for x in rows if x["exercise"] == ex and x["method"] == "b1")
            row["d_icc_vs_b1"] = row["icc"] - b1["icc"]
            row["d_icc_vs_b1_lo"], row["d_icc_vs_b1_hi"] = interval(dd, ci)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--tag", default="perex_gkf5")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--exercises", nargs="+")
    ap.add_argument("--out", help="output folder (default <results>/<tag>); e.g. the final-scope pools over E01-E04")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    ec, ia = cfg["eval"], cfg["heads"]["item_activity"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    low = ec.get("per_exercise_low_confidence_subjects", 15)
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    src = resolve(ec["results_dir"]) / a.tag  # embedding_distance_reps.csv (per rep, any exercise subset)
    out = resolve(a.out) if a.out else src
    out.mkdir(parents=True, exist_ok=True)
    smoke = a.tag.endswith("_smoke")

    v = video_table(cfg)
    v["MQ"] = movement_quality(v)
    index = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    index["MQ"] = movement_quality(index)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    act = item_activity(v, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}
    exercises = a.exercises or sorted(reps.exercise.unique())
    if a.exercises:  # pooled strata over the chosen exercises only (e.g. the final scope E01-E04, D18)
        global POOLED_WITHIN
        span = f"{exercises[0]}–{exercises[-1]}" if len(exercises) > 1 else exercises[0]
        POOLED_WITHIN = [(f"experts, {span}", None, "expert"), (f"non-experts, {span}", None, "non-expert")]
    pv, pr, done = load_runs(root, exercises, METHODS)
    methods = [m for m in METHODS if m in set(pv.method)]
    exercises = [e for e in exercises if any(k[0] == e for k in done)]
    plans_by_ex, plans = {}, {}
    for ex in exercises:
        pl = fold_plan(cfg, reps[reps.exercise == ex], smoke=smoke)
        pl = [{**p, "exercise": ex} for p in pl if any((ex, m, p["fold"]) in done for m in methods)]
        plans_by_ex[ex] = pl
        for p in pl:
            plans[fkey(ex, p["fold"])] = p
    for (ex, m, f), d in done.items():  # every run used exactly these folds and inner splits
        p = plans[fkey(ex, f)]
        for k in ("train", "inner_train", "inner_val", "test"):
            assert sorted(d["subjects"][k]) == sorted(p[k]), (ex, m, f, k)
    complete = [ex for ex in exercises if all((ex, m, p["fold"]) in done for m in methods for p in plans_by_ex[ex])]
    tv = v[v.exercise.isin(complete) & v.apply(lambda r: any(r.subject in p["test"] for p in plans_by_ex.get(
        r.exercise, [])), axis=1)]
    pv = pv[pv.exercise.isin(complete)]
    pr = pr[pr.exercise.isin(complete)]

    pv = add_mq(pv, v, plans, active)
    vp = pv[["method", "fold", "video_id", *[f"{t}_pred" for t in VIDEO_TARGETS]]].copy()
    vp["source"] = "video head"
    rp = pr.groupby(["method", "fold", "video_id"])[[f"{t}_pred" for t in REP_TARGETS]].mean().reset_index()
    rp["source"] = "rep head"
    bases = []
    for ex in complete:
        b = video_baselines(v[v.exercise == ex], plans_by_ex[ex], VIDEO_TARGETS)
        b["fold"] = [fkey(ex, f) for f in b.fold]
        bases.append(b[b.method.isin(BASELINES)])
    base = pd.concat(bases, ignore_index=True)
    base["source"] = "baseline"
    P = pd.concat([vp, rp, base], ignore_index=True).merge(
        v[["video_id", "subject", "group", "exercise", "view", "variation", *VIDEO_TARGETS]], on="video_id",
        how="left", validate="many_to_one")
    n = P.groupby(["method", "source"]).video_id.nunique()
    assert (n == len(tv)).all(), f"every method must predict every test video once: {n.to_dict()}"
    P.to_csv(out / "video_predictions.csv", index=False)
    floors = noise_floors(index, ["score", "MQ", "PF1", "PF2", "PF4", "CF1", "CF2", "CF3"])
    floors.to_csv(out / "noise_floor.csv")
    within = [(f"{e} {g}", e, g) for e, g in sorted({(e, g) for e, g in zip(tv.exercise, tv.group)})]
    strata = within + POOLED_WITHIN
    entries = [(m, "video head") for m in methods] + [(m, "rep head") for m in methods] + \
              [(b, "baseline") for b in BASELINES]
    S, D = regression_tables(P, strata, entries, active, floors, n_boot, seed, ci)
    S.to_csv(out / "score_metrics.csv", index=False)
    D.to_csv(out / "paired_deltas.csv", index=False)
    print(f"regression tables in {time.time() - t0:.0f} s", flush=True)
    V, FV = vci_tables(root, {k: d for k, d in done.items() if k[0] in complete}, cfg, n_boot, seed, ci)
    V.to_csv(out / "vci.csv", index=False)
    FV.to_csv(out / "vci_per_fold.csv", index=False)
    VS = distance_stability(cfg, src, plans_by_ex, n_boot, seed, ci)
    if VS is not None:
        VS.to_csv(out / "distance_view_stability.csv", index=False)
    rows = []
    for (ex, m, f), d in sorted(done.items()):
        for st in ("cvdc_stage", "supervised_stage"):
            if st in d:
                s_ = d[st]
                rows.append(dict(exercise=ex, method=m, fold=f, stage=st.split("_")[0], best_epoch=s_["best_epoch"],
                                 stopped_epoch=s_["stopped_epoch"], early_stopped=s_["early_stopped"],
                                 stage_s=s_["wall_s"], reused=bool(s_.get("reused_from"))))
        rows.append(dict(exercise=ex, method=m, fold=f, stage="total", stage_s=d["wall_s"]))
    T = pd.DataFrame(rows)
    T.to_csv(out / "stopping_runtime.csv", index=False)
    write_summary(out, S, D, V, VS, T, act, methods, complete, exercises, n_boot, smoke, tv, cfg, within, low)
    print(f"report written to {out} in {time.time() - t0:.0f} s")


def write_summary(out, S, D, V, VS, T, act, methods, complete, exercises, n_boot, smoke, tv, cfg, within, low):
    L = []
    w = L.append
    nsub = tv.groupby(["exercise", "group"]).subject.nunique()
    nex = tv.groupby("exercise").subject.nunique()
    flag = lambda s: " ⚠" if nsub.get(tuple(s.split(" ", 1)), 99) < low else ""  # noqa: E731
    wn = [s for s, _, _ in within]
    pw = [s for s, _, _ in POOLED_WITHIN]
    w("# Full scope, one model per exercise (T17): per-exercise results, GroupKFold-5"
      + (" (SMOKE RUN: plumbing check, not a result)" if smoke else ""))
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/full/phase3/report_perex.py`. Exercises reported: "
      f"{', '.join(complete)}" + (f" (still training: {', '.join(e for e in exercises if e not in complete)})"
                                  if set(exercises) - set(complete) else "") +
      f". Each exercise's models are trained and tested on its own videos only, with its own stratified GroupKFold-5 "
      f"(inner validation 20% per group, CVDC pretrained per fold on the inner training subjects); no exercise head. "
      f"Same settings as the pilot's step (a2) (T6, T7, T8a, T8b, T13, T14, T15). {len(tv)} test videos with ≥ 1 "
      f"rep. 95% CIs: subject bootstrap ({n_boot} draws, paired across methods; E17). **Bold** = a paired CI that "
      f"excludes 0. ⚠ = fewer than {low} subjects (low confidence): " +
      ", ".join(f"{e} {g} ({n})" for (e, g), n in nsub.items() if n < low) + "; exercises with fewer than "
      f"{low} subjects: " + (", ".join(f"{e} ({n})" for e, n in nex.items() if n < low) or "none") + ". "
      "Limitations: the expert / non-expert contrast is confounded, so within-group results are primary; E02, E04, "
      "E06, E07 have experts only; E07–E09 all male; scores skewed; PF3/PF4 view noise; GroupKFold-5, not LOSO.")
    w("")
    cols = methods + ["base_group", "base_setfeat", "base_repcount", "base_duration", "base_mean"]
    for t, title in (("score", "Total score"), ("MQ", "Movement-quality score (MQ, D13)")):
        w(f"## {'1' if t == 'score' else '2'}. {title} within groups, video level")
        w("")
        w("RMSE / Spearman ρ, video head (rep head and every baseline with CIs in `score_metrics.csv`). "
          "Noise floor: inter-view label noise of the expert groups (E3)." + (" MQ from the video head's items "
                                                                             "(non-varying items at their training mean)." if t == "MQ" else ""))
        w("")
        w("| Stratum (videos / subjects) | " + " | ".join(SHORT[c] for c in cols) + " | floor |")
        w("|---|" + "---|" * (len(cols) + 1))
        for s in wn:
            r0 = s_row(S, s, methods[0], "video head", t)
            if r0 is None:
                continue
            mark = "" if r0.varies_within_group else " ·"
            cells = [cell(S, s, c, "video head" if c in methods else "baseline", t, ci=False) for c in cols]
            w(f"| {s}{flag(s)}{mark} ({r0.n} / {r0.subjects}) | " + " | ".join(cells) + f" | {fmt(r0.noise_floor)} |")
        w("")
        w("Paired RMSE difference, video head − group-only [95% CI] (negative = better than the within-group mean):")
        w("")
        w("| Stratum | " + " | ".join(SHORT[m] for m in methods) + " |")
        w("|---|" + "---|" * len(methods))
        for s in wn + pw:
            w(f"| {s}{flag(s)} | " + " | ".join(d_cell(D, s, m, "video head", "base_group", t) for m in methods) + " |")
        w("")
        w("Paired RMSE difference, video head − set-features [95% CI]:")
        w("")
        w("| Stratum | " + " | ".join(SHORT[m] for m in methods) + " |")
        w("|---|" + "---|" * len(methods))
        for s in wn + pw:
            w(f"| {s}{flag(s)} | " + " | ".join(d_cell(D, s, m, "video head", "base_setfeat", t) for m in methods) + " |")
        w("")
        w(f"Pooled over the exercises within each group (each video by its own exercise's model): RMSE [95% CI] / ρ / "
          f"centred ρ (ranking within exercises):")
        w("")
        w("| Method | " + " | ".join(pw) + " |")
        w("|---|" + "---|" * len(pw))
        for c in cols:
            src = "video head" if c in methods else "baseline"
            w(f"| {LABEL[c]} | " + " | ".join(cell(S, s, c, src, t, centred=True) for s in pw) + " |")
        w("")
        w("Count of strata with a paired RMSE CI vs group-only below 0 / above 0 / including 0 (video head): " +
          "; ".join(f"{SHORT[m]} " + "/".join(str(x) for x in (
              sum(g.hi < 0 for g in gs), sum(g.lo > 0 for g in gs), sum(g.lo <= 0 <= g.hi for g in gs)))
              for m in methods for gs in [[x for x in (d_row(D, s, m, "video head", "base_group", t) for s in wn)
                                           if x is not None]]) + ".")
        w("")
    w("## 3. Within-group Viewpoint Consistency Index (centred 256-d)")
    w("")
    w("Same rep from another view − other (exercise, subject, variation) groups of the stratum; held-out subjects, "
      "pooled over the exercise's folds; the pooled rows add within-exercise sums (no cross-exercise pairs).")
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
    for s in wn + pw:
        d = V[(V.method == ref_init) & (V.stage == "init") & (V.stratum == s) & (V.variant == "centred")]
        npairs = int(d.same_pairs.iloc[0]) if len(d) else 0
        w(f"| {s}{flag(s)} ({npairs}) | " + " | ".join(vcell(m, st, s) for _, m, st in vcols) + " |")
    w("")
    w("## 4. View stability of the distance to the PT reference (synced expert performances, rep level)")
    w("")
    if VS is not None and len(VS):
        ms = [m for m in ["init_xy", *methods] if m in set(VS.method)]
        w("ICC(1) [95% CI] of the standardised centred-cosine distance across the F / L / R views of one rep "
          "(higher = more stable); ΔICC vs B1 with the paired CI below.")
        w("")
        w("| Exercise (performances) | " + " | ".join(SHORT[m] for m in ms) + " |")
        w("|---|" + "---|" * len(ms))
        for ex, d in VS.groupby("exercise"):
            cells = []
            for m in ms:
                r = d[d.method == m]
                if r.empty:
                    cells.append("–")
                    continue
                r = r.iloc[0]
                c = fci(r.icc, r.icc_lo, r.icc_hi)
                if m not in ("b1", "init_xy") and "d_icc_vs_b1" in r and np.isfinite(r.d_icc_vs_b1):
                    dd = fci(r.d_icc_vs_b1, r.d_icc_vs_b1_lo, r.d_icc_vs_b1_hi)
                    c += f"; Δ {'**' + dd + '**' if r.d_icc_vs_b1_lo > 0 or r.d_icc_vs_b1_hi < 0 else dd}"
                cells.append(c)
            w(f"| {ex} ({int(d.perfs.iloc[0])}) | " + " | ".join(cells) + " |")
    else:
        w("Not computed yet (`scripts/full/phase3/embedding_distance_perex.py`).")
    w("")
    w("## 5. Expert / non-expert group probe (E11)")
    w("")
    gp, sp = out / "group_probe_per_exercise.csv", out / "score_probe.csv"
    if gp.exists() and sp.exists():
        G, SPr = pd.read_csv(gp), pd.read_csv(sp)
        w("Linear probes on frozen video embeddings (each exercise's models; trained on the fold's training "
          "subjects, scored on its test subjects). Group probe: AUC [95% CI] / balanced accuracy, all videos and "
          "front-view FL (view-matched). Score probe: ridge on the total score within each group, Spearman / R² "
          "against the within-group training mean (R² > 0 = better than group-only).")
        w("")
        encs = [e for e in ["init_xy", *methods] if e in set(G.encoder)]
        w("| Exercise | probe | " + " | ".join(SHORT[e] for e in encs) + " |")
        w("|---|---|" + "---|" * len(encs))
        for ex in sorted(set(G.exercise) - {"pooled"}) + ["pooled"]:
            for kind in ("all", "view_matched"):
                cells = []
                for e in encs:
                    r = G[(G.encoder == e) & (G.exercise == ex) & (G.kind == kind)]
                    cells.append("–" if r.empty else f"{fci(r.auc.iloc[0], r.auc_lo.iloc[0], r.auc_hi.iloc[0])} / "
                                                     f"{fmt(r.bacc.iloc[0])}")
                w(f"| {ex} | group, {'all' if kind == 'all' else 'view-matched'} | " + " | ".join(cells) + " |")
        for (ex, g), d in SPr[SPr.exercise != "all"].groupby(["exercise", "group"]):
            cells = []
            for e in encs:
                r = d[d.encoder == e]
                cells.append("–" if r.empty else f"{fmt(r.rho.iloc[0])} / {fmt(r.r2.iloc[0])}")
            w(f"| {ex} | score, {g}{flag(f'{ex} {g}')} | " + " | ".join(cells) + " |")
        w("")
    else:
        w("Not run yet (`scripts/full/phase3/group_probe.py`).")
    w("")
    w("## 6. Training: kept / stopping epochs, wall time")
    w("")
    w("| Exercise | method | stage | kept epochs (folds) | stopped | early-stopped | wall (min) |")
    w("|---|---|---|---|---|---|---|")
    for (ex, m, st), d in T[T.stage != "total"].groupby(["exercise", "method", "stage"], sort=True):
        tt = T[(T.exercise == ex) & (T.method == m) & (T.stage == "total")]
        w(f"| {ex} | {SHORT[m]} | {st}{' (shared)' if d.reused.all() else ''} | "
          f"{', '.join(str(int(x)) for x in d.best_epoch)} | {', '.join(str(int(x)) for x in d.stopped_epoch)} | "
          f"{int(d.early_stopped.sum())}/{len(d)} | {tt.stage_s.sum() / 60:.1f} |")
    w("")
    w("Tables: `score_metrics.csv`, `paired_deltas.csv`, `video_predictions.csv`, `vci.csv`, `vci_per_fold.csv`, "
      "`distance_view_stability.csv`, `group_probe_*.csv`, `score_probe.csv`, `stopping_runtime.csv`, "
      "`noise_floor.csv`.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
