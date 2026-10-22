"""Cross-view protocol per exercise (spec §9, E24; T17, D18): report for the final scope E01-E04.

Every run of runs/full/<tag>/<EX>/heldout_<V>/<method>/fold<k>/ trained on two views of that exercise and predicted
every view of its test subjects, so each evaluated video is predicted once as a **held-out view** (each by a model that
saw neither its subject nor its camera) and twice as a **seen view**; the step-1 models of runs/full/perex_gkf5/ (all
three views, same folds) are the **all-view** reference. Within every exercise x group stratum first (E9), pooled over
the four exercises within each group second; video level; subject-bootstrap CIs, paired (E17):
  MQ / score  the movement-quality score (D13, from the video head's items) and the total score on the held-out view:
              RMSE, Spearman vs the group-only baseline (within-group mean of the run's training subjects' training-view
              videos); gaps held-out - seen and held-out - all-view; Proposed - B1; Proposed's gap - B1's gap
  VCI         centred 256-d, pairs with one member in the held-out view and one in a seen view vs seen <-> seen pairs
              of two different seen cameras (E24), per stratum; random init, after CVDC (Proposed), final; all-view
  distance    view stability (E22) of the standardised distance to the PT reference over synced expert performances:
              ICC(1) on held-out <-> seen, seen <-> seen and all three views; ΔICC vs B1 and vs the all-view models
Inputs: the cross-view runs, <results>/<tag>/embedding_distance_reps.csv (crossview_distance_perex.py), the step-1 runs
and <results>/perex_gkf5/embedding_distance_reps.csv (embedding_distance_perex.py).

usage: python scripts/full/phase3/report_crossview_perex.py [--config configs/per_exercise.yaml] [--tag perex_crossview]
       [--boot 2000] [--exercises E01 E02 E03 E04]
Writes CSVs and summary.md to <eval.results_dir>/<tag>/.
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
for d in ("full/phase3", "pilot/phase4", "pilot/phase5", "pilot/phase3"):
    sys.path.insert(0, str(REPO / "scripts" / d))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.data.labels import MQ_ITEMS, movement_quality  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_table  # noqa: E402
from cvdc.eval.consistency import pooled_vci  # noqa: E402
from cvdc.eval.cross_view import per_exercise_methods  # noqa: E402
from cvdc.eval.metrics import interval, item_weights, regression, subject_draws  # noqa: E402
from report_crossview import PAIRINGS, ci_cols, standardise, variants  # noqa: E402
from report_full import fci, fmt  # noqa: E402
from report_phase4 import view_stability  # noqa: E402

SHORT = {"b1": "B1", "b2_ktpformer": "B2-KTP", "proposed_j1": "Proposed", "base_group": "group-only"}
POOLED = [("experts, E01–E04", None, "expert"), ("non-experts, E01–E04", None, "non-expert")]


def smask(df, ex, grp):
    m = np.ones(len(df), bool)
    if ex:
        m &= (df.exercise == ex).to_numpy()
    if grp:
        m &= (df.group == grp).to_numpy()
    return m


def load(root, all_root, exercises, methods):
    """Video-head predictions of every cross-view and all-view run with role (held-out / seen / all-view)."""
    vid, done = [], {}
    for ex in exercises:
        for vdir in sorted((root / ex).glob("heldout_*")):
            held = vdir.name.split("_", 1)[1]
            for m in methods:
                for f in sorted((vdir / m).glob("fold*/done.json")):
                    d = json.loads(f.read_text(encoding="utf-8"))
                    done[(ex, held, m, d["fold"])] = d
                    p = pd.read_csv(f.parent / "test_predictions_video.csv")
                    p["heldout"], p["fold"], p["method"] = held, d["fold"], m
                    p["role"] = np.where(p.view == held, "held-out", "seen")
                    vid.append(p)
        for m in methods:
            for f in sorted((all_root / ex / m).glob("fold*/done.json")):
                d = json.loads(f.read_text(encoding="utf-8"))
                done[(ex, "all", m, d["fold"])] = d
                p = pd.read_csv(f.parent / "test_predictions_video.csv")
                p["heldout"], p["fold"], p["method"], p["role"] = "all", d["fold"], m, "all-view"
                vid.append(p)
    return pd.concat(vid, ignore_index=True), done


def training_videos(v, plan):
    sel = v.subject.isin(plan["train"]) & (v.exercise == plan["exercise"])
    if plan.get("views"):
        sel &= v.view.isin(plan["views"])
    return v[sel]


def add_mq_and_baseline(pv, v, done, active):
    pv = pv.copy()
    for k in MQ_ITEMS:
        pred = pv[f"{k}_pred"].to_numpy(float).copy()
        for (ex, held, m, f), idx in pv.groupby(["exercise", "heldout", "method", "fold"]).groups.items():
            tr = training_videos(v, done[(ex, held, m, f)]["subjects"])
            rows = pv.loc[idx]
            for (e, g), d in tr.groupby(["exercise", "group"]):
                if k in active[(e, g)]:
                    continue
                sel = rows.index[(rows.exercise == e) & (rows.group == g)]
                pred[pv.index.get_indexer(sel)] = d[k].mean()
        pv[f"{k}_pred_mq"] = pred
    pv["MQ_pred"] = 100.0 * pv[[f"{k}_pred_mq" for k in MQ_ITEMS]].mean(axis=1)
    base = []
    m0 = pv.method.iloc[0]
    for (ex, held, f), te in pv[pv.method == m0].groupby(["exercise", "heldout", "fold"]):
        tr = training_videos(v, done[(ex, held, m0, f)]["subjects"])
        gm = tr.groupby(["exercise", "group"])[["MQ", "score"]].mean()
        d = te[["video_id", "subject", "group", "exercise", "view", "heldout", "fold", "role"]].copy()
        d["method"] = "base_group"
        for t in ("MQ", "score"):
            d[f"{t}_pred"] = [gm.loc[(e, g), t] for e, g in zip(te.exercise, te.group)]
        base.append(d)
    return pv, pd.concat(base, ignore_index=True)


def regression_tables(P, v, strata, methods, n_boot, seed, ci):
    rows, deltas, rep = [], [], {}
    for sname, ex, grp in strata:
        sub = v[smask(v, ex, grp)].drop_duplicates("subject").sort_values("subject")
        subs = sub.subject.to_numpy()
        M = subject_draws(subs, sub.group.to_numpy(), n_boot, seed)
        d0 = P[smask(P, ex, grp)]
        for (m, role), d in d0.groupby(["method", "role"]):
            d = d.sort_values(["video_id", "heldout"])
            W = np.vstack([np.ones(len(d)), item_weights(d.subject.to_numpy(), subs, M)])
            for t in ("MQ", "score"):
                r = regression(d[t].to_numpy(float), d[f"{t}_pred"].to_numpy(float), W)
                rep[(sname, m, role, t)] = r
                row = dict(stratum=sname, method=m, role=role, target=t, videos=len(d), subjects=d.subject.nunique())
                for k, val in r.items():
                    row.update(ci_cols(k, val, ci))
                rows.append(row)
        for t in ("MQ", "score"):
            for m in methods:
                pairs = [("held-out", m, "held-out", "base_group", "vs group-only"),
                         ("held-out", m, "seen", m, "held-out − seen (gap)"),
                         ("held-out", m, "all-view", m, "held-out − all-view")]
                if m != "b1":
                    pairs.append(("held-out", m, "held-out", "b1", "vs B1 (held-out)"))
                for ra, ma, rb, mb, what in pairs:
                    a, b = rep.get((sname, ma, ra, t)), rep.get((sname, mb, rb, t))
                    if a is None or b is None:
                        continue
                    for k in ("rmse", "spearman"):
                        deltas.append(dict(stratum=sname, target=t, method=m, comparison=what, metric=k,
                                           **ci_cols("delta", a[k] - b[k], ci)))
                if m != "b1":
                    ga = [rep.get((sname, m, r, t)) for r in ("held-out", "seen")]
                    gb = [rep.get((sname, "b1", r, t)) for r in ("held-out", "seen")]
                    if all(x is not None for x in ga + gb):
                        for k in ("rmse", "spearman"):
                            dd = (ga[0][k] - ga[1][k]) - (gb[0][k] - gb[1][k])
                            deltas.append(dict(stratum=sname, target=t, method=m, comparison="gap − B1's gap",
                                               metric=k, **ci_cols("delta", dd, ci)))
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def vci_tables(root, all_root, exercises, methods, members, strata, n_boot, seed, ci, views):
    gcode = pd.factorize(members.group_id)[0]
    snaps = []
    for ex in exercises:
        for vdir in sorted((root / ex).glob("heldout_*")):
            held = vdir.name.split("_", 1)[1]
            for m in methods:
                for f in sorted((vdir / m).glob("fold*/done.json")):
                    for stage in ("init", "pretrained", "final"):
                        npz = f.parent / f"embeddings_{stage}.npz"
                        if npz.exists():
                            for pr in PAIRINGS:
                                snaps.append(((m, stage, pr), (ex, held, f.parent.name), npz, (held, pr)))
        for m in methods:
            for f in sorted((all_root / ex / m).glob("fold*/done.json")):
                for held in views:
                    for pr in PAIRINGS:
                        snaps.append(((m, "all-view", pr), (ex, held, f.parent.name),
                                      f.parent / "embeddings_final.npz", (held, pr)))

    def pair_filter(ctx, mm):
        held, pr = ctx
        h, vw = (mm.view == held).to_numpy(), mm.view.to_numpy()
        if pr == PAIRINGS[0]:
            return h[:, None] != h[None, :]
        return (~h)[:, None] & (~h)[None, :] & (vw[:, None] != vw[None, :])

    rows, _ = pooled_vci(snaps, members, gcode, strata, n_boot, seed, ci, pair_filter)
    V = pd.DataFrame([{**{k: r[k] for k in r if k not in ("key", "boot", "present")}, "method": r["key"][0],
                       "stage": r["key"][1], "pairing": r["key"][2]} for r in rows])
    boot = {(r["key"], r["stratum"], r["variant"]): (r["boot"], r["present"]) for r in rows}
    deltas = []
    for sname, _, _ in strata:
        for m in methods:
            for ka, kb, what in (((m, "final", PAIRINGS[0]), ("b1", "final", PAIRINGS[0]), "held-out↔seen vs B1"),
                                 ((m, "final", PAIRINGS[0]), (m, "final", PAIRINGS[1]), "held-out↔seen − seen↔seen"),
                                 ((m, "final", PAIRINGS[0]), (m, "all-view", PAIRINGS[0]), "cross-view − all-view")):
                if ka == kb:
                    continue
                a, b = boot.get((ka, sname, "centred")), boot.get((kb, sname, "centred"))
                if a is None or b is None or a[1] != b[1]:
                    continue
                pa = V[(V.method == ka[0]) & (V.stage == ka[1]) & (V.pairing == ka[2]) & (V.stratum == sname) &
                       (V.variant == "centred")].vci.iloc[0]
                pb = V[(V.method == kb[0]) & (V.stage == kb[1]) & (V.pairing == kb[2]) & (V.stratum == sname) &
                       (V.variant == "centred")].vci.iloc[0]
                lo, hi = interval(a[0] - b[0], ci)
                deltas.append(dict(stratum=sname, method=m, comparison=what, delta=pa - pb, delta_lo=lo, delta_hi=hi))
    return V, pd.DataFrame(deltas)


def distance_tables(cv_csv, all_csv, reps, synced, exercises, methods, views, n_boot, seed, ci):
    meta = reps[["video_id", "rep", "rep_uid", "group_id", "subject", "group", "view"]]
    C = pd.read_csv(cv_csv)
    C = C[C.method.isin(methods) & C.exercise.isin(exercises)].merge(meta, on=["video_id", "rep"], how="left")
    C = C.rename(columns={"train_view": "train"})
    C = standardise(C, ["exercise", "heldout", "method", "fold", "group"])
    A = pd.read_csv(all_csv)
    A = A[A.method.isin(methods) & A.exercise.isin(exercises)].merge(meta, on=["video_id", "rep"], how="left")
    A["train"] = ~A.is_test
    A = standardise(A, ["exercise", "method", "fold", "group"])
    sel = lambda E: E[E.is_test & E.group_id.isin(synced) & (E.group == "expert")]  # noqa: E731
    C, A = sel(C), sel(A)
    rows, icc = [], {}
    for ex in exercises:
        for setting, E0 in (("cross-view", C), ("all-view", A)):
            for m in methods:
                E = E0[(E0.method == m) & (E0.exercise == ex)]
                parts = {}
                for held in views:
                    Eh = E[E.heldout == held] if setting == "cross-view" else E
                    for name, d in variants(Eh, held, "rep", views).items():
                        if setting == "all-view" and name == "all three views" and held != views[0]:
                            continue
                        parts.setdefault(name, []).append(d)
                for name, lst in parts.items():
                    r, b, _ = view_stability(pd.concat(lst, ignore_index=True), "z", n_boot, seed, ci)
                    icc[(ex, setting, name, m)] = (b, r["perfs"])
                    rows.append(dict(exercise=ex, setting=setting, variant=name, method=m, **r))
    VS = pd.DataFrame(rows)
    deltas = []
    for (ex, setting, name, m), (b, n) in icc.items():
        r = VS[(VS.exercise == ex) & (VS.setting == setting) & (VS.variant == name) & (VS.method == m)].iloc[0]
        for other, what in ((("b1", setting), "vs B1"), ((m, "all-view"), "cross-view − all-view")):
            if other == (m, setting):
                continue
            o = icc.get((ex, other[1], name, other[0]))
            if o is None or o[1] != n:
                continue
            ro = VS[(VS.exercise == ex) & (VS.setting == other[1]) & (VS.variant == name) & (VS.method == other[0])].iloc[0]
            lo, hi = interval(b - o[0], ci)
            deltas.append(dict(exercise=ex, setting=setting, variant=name, method=m, comparison=what,
                               delta=r.icc - ro.icc, delta_lo=lo, delta_hi=hi))
    return VS, pd.DataFrame(deltas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--tag", default="perex_crossview")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--exercises", nargs="+", default=["E01", "E02", "E03", "E04"])
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    ec, ia = cfg["eval"], cfg["heads"]["item_activity"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    methods, views = per_exercise_methods(cfg), list(cfg["data"]["views"])
    runs = resolve(cfg["train"]["runs_dir"])
    root, all_root = runs / a.tag, runs / "perex_gkf5"
    out = resolve(ec["results_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    v = video_table(cfg)
    v["MQ"] = movement_quality(v)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    act = item_activity(v, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}
    pv, done = load(root, all_root, a.exercises, methods)
    for (ex, held, m, f), d in done.items():  # add the plan's exercise (all-view runs record it too)
        d["subjects"]["exercise"] = ex
    n_runs = pd.Series([k[2] for k in done if k[1] != "all"]).value_counts()
    print("cross-view runs per method:", n_runs.to_dict(), flush=True)
    pv, base = add_mq_and_baseline(pv, v, done, active)
    lab = v[["video_id", "MQ", "score"]]
    P = pd.concat([pv[["method", "role", "heldout", "fold", "video_id", "subject", "group", "exercise", "view",
                       "MQ_pred", "score_pred"]], base], ignore_index=True).merge(lab, on="video_id", how="left")
    P = P[P.exercise.isin(a.exercises)]
    P.to_csv(out / "video_predictions.csv", index=False)
    ve = v[v.exercise.isin(a.exercises) & v.video_id.isin(set(P.video_id))]
    within = [(f"{e} {g}", e, g) for e, g in sorted({(e, g) for e, g in zip(ve.exercise, ve.group)})]
    strata = within + POOLED
    S, D = regression_tables(P, ve, strata, methods, n_boot, seed, ci)
    S.to_csv(out / "mq_score_metrics.csv", index=False)
    D.to_csv(out / "mq_score_deltas.csv", index=False)
    print(f"regression in {time.time() - t0:.0f} s", flush=True)
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    VC, VD = vci_tables(root, all_root, a.exercises, methods, members, strata, n_boot, seed, ci, views)
    VC.to_csv(out / "vci.csv", index=False)
    VD.to_csv(out / "vci_deltas.csv", index=False)
    print(f"VCI in {time.time() - t0:.0f} s", flush=True)
    sync = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    synced = set(sync.index[sync.cls == "synced"])
    cv_csv = out / "embedding_distance_reps.csv"
    VS = VSD = None
    if cv_csv.exists():
        VS, VSD = distance_tables(cv_csv, resolve(ec["results_dir"]) / "perex_gkf5" / "embedding_distance_reps.csv",
                                  reps, synced, a.exercises, methods, views, n_boot, seed, ci)
        VS.to_csv(out / "distance_view_stability.csv", index=False)
        VSD.to_csv(out / "distance_view_stability_deltas.csv", index=False)
    write_summary(out, S, D, VC, VD, VS, VSD, within, methods, a.exercises, n_runs, n_boot)
    print(f"written to {out} in {time.time() - t0:.0f} s")


def dcell(D, stratum, method, comparison, metric="rmse", target="MQ"):
    g = D[(D.stratum == stratum) & (D.method == method) & (D.comparison == comparison) & (D.metric == metric) &
          (D.target == target)]
    if g.empty:
        return "–"
    g = g.iloc[0]
    s = fci(g.delta, g.delta_lo, g.delta_hi)
    return f"**{s}**" if g.delta_lo > 0 or g.delta_hi < 0 else s


def write_summary(out, S, D, VC, VD, VS, VSD, within, methods, exercises, n_runs, n_boot):
    L = []
    w = L.append
    ws = [s for s, _, _ in within] + [s for s, _, _ in POOLED]
    w("# Cross-view protocol per exercise (final scope E01–E04)")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/full/phase3/report_crossview_perex.py`. One model per "
      f"exercise (T17), GroupKFold-5 per exercise (the step-1 folds), each run trained on two views and tested on all "
      f"three; runs per method: {n_runs.to_dict()} (3 held-out views × 5 folds × {len(exercises)} exercises). "
      f"Held-out = the view the run never saw (each video once); seen = the two training views (each video twice); "
      f"all-view = the step-1 models. 95% CIs: subject bootstrap ({n_boot} draws, paired). **Bold** = a paired CI "
      "that excludes 0. Video level, within groups first.")
    w("")
    for t, title in (("MQ", "Movement-quality score (MQ)"), ("score", "Total score")):
        w(f"## {'1' if t == 'MQ' else '2'}. {title} on the held-out view")
        w("")
        w("RMSE / Spearman ρ; held-out view of every method, its seen views, the all-view model and the group-only "
          "baseline (training subjects' training-view videos).")
        w("")
        w("| Stratum | " + " | ".join(f"{SHORT[m]} held-out / seen / all-view" for m in methods) + " | group-only |")
        w("|---|" + "---|" * (len(methods) + 1))
        for s in ws:
            cells = []
            for m in methods:
                parts = []
                for role in ("held-out", "seen", "all-view"):
                    r = S[(S.stratum == s) & (S.method == m) & (S.role == role) & (S.target == t)]
                    parts.append("–" if r.empty else f"{fmt(r.rmse.iloc[0])}/{fmt(r.spearman.iloc[0])}")
                cells.append(" · ".join(parts))
            g = S[(S.stratum == s) & (S.method == "base_group") & (S.role == "held-out") & (S.target == t)]
            w(f"| {s} | " + " | ".join(cells) + f" | {'–' if g.empty else fmt(g.rmse.iloc[0])} |")
        w("")
        w("Paired RMSE differences (negative = lower error):")
        w("")
        comps = ["held-out − seen (gap)", "vs group-only", "vs B1 (held-out)", "gap − B1's gap"]
        w("| Stratum | method | " + " | ".join(comps) + " |")
        w("|---|---|" + "---|" * len(comps))
        for s in ws:
            for m in methods:
                w(f"| {s} | {SHORT[m]} | " + " | ".join(dcell(D, s, m, c, "rmse", t) if not (m == "b1" and c in (
                    "vs B1 (held-out)", "gap − B1's gap")) else "–" for c in comps) + " |")
        w("")
    w("## 3. VCI on the held-out view (centred 256-d)")
    w("")
    w("held-out↔seen = the same rep seen by the held-out camera and a seen camera, minus other groups with one member "
      "in each; seen↔seen = two different seen cameras.")
    w("")

    def vc(m, stage, pr, s):
        d = VC[(VC.method == m) & (VC.stage == stage) & (VC.pairing == pr) & (VC.stratum == s) & (VC.variant == "centred")]
        return "–" if d.empty or not np.isfinite(d.vci.iloc[0]) else fci(d.vci.iloc[0], d.vci_lo.iloc[0], d.vci_hi.iloc[0], 3)

    cols = [("random init", "b1", "init", PAIRINGS[0])] + \
        [("after CVDC (2 views)", "proposed_j1", "pretrained", PAIRINGS[0])] + \
        [(f"{SHORT[m]} final", m, "final", PAIRINGS[0]) for m in methods] + \
        [("Proposed seen↔seen", "proposed_j1", "final", PAIRINGS[1]),
         ("Proposed all-view model", "proposed_j1", "all-view", PAIRINGS[0])]
    w("| Stratum | " + " | ".join(c[0] for c in cols) + " |")
    w("|---|" + "---|" * len(cols))
    for s in ws:
        w(f"| {s} | " + " | ".join(vc(m, st, pr, s) for _, m, st, pr in cols) + " |")
    w("")
    w("Proposed − B1 (held-out↔seen): " + "; ".join(
        f"{s} {fci(*VD[(VD.stratum == s) & (VD.method == 'proposed_j1') & (VD.comparison == 'held-out↔seen vs B1')][['delta', 'delta_lo', 'delta_hi']].iloc[0].to_numpy(), 3)}"
        for s in ws if not VD[(VD.stratum == s) & (VD.method == "proposed_j1") & (VD.comparison == "held-out↔seen vs B1")].empty) + ".")
    w("")
    w("## 4. View stability of the distance to the PT reference (synced expert reps, ICC(1))")
    w("")
    if VS is not None:
        w("| Exercise | method | held-out↔seen | seen↔seen | all three | all-view model (held-out↔seen pairs) | Δ vs B1 (held-out↔seen) |")
        w("|---|---|---|---|---|---|---|")
        for ex in exercises:
            for m in methods:
                def ic(setting, var):
                    r = VS[(VS.exercise == ex) & (VS.setting == setting) & (VS.variant == var) & (VS.method == m)]
                    return "–" if r.empty else fci(r.icc.iloc[0], r.icc_lo.iloc[0], r.icc_hi.iloc[0])
                dd = VSD[(VSD.exercise == ex) & (VSD.setting == "cross-view") & (VSD.variant == PAIRINGS[0]) &
                         (VSD.method == m) & (VSD.comparison == "vs B1")]
                dtxt = "–" if dd.empty else fci(dd.delta.iloc[0], dd.delta_lo.iloc[0], dd.delta_hi.iloc[0])
                if not dd.empty and (dd.delta_lo.iloc[0] > 0 or dd.delta_hi.iloc[0] < 0):
                    dtxt = f"**{dtxt}**"
                w(f"| {ex} | {SHORT[m]} | {ic('cross-view', PAIRINGS[0])} | {ic('cross-view', PAIRINGS[1])} | "
                  f"{ic('cross-view', 'all three views')} | {ic('all-view', PAIRINGS[0])} | {dtxt} |")
    else:
        w("Not computed (`scripts/full/phase3/crossview_distance_perex.py`).")
    w("")
    w("Tables: `mq_score_metrics.csv`, `mq_score_deltas.csv`, `vci.csv`, `vci_deltas.csv`, "
      "`distance_view_stability*.csv`, `video_predictions.csv`.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
