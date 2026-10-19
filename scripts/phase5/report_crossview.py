"""Phase 5 A report: the cross-view protocol (spec §9, E24) on GroupKFold-5 for B1, B2-MediaPipe, B2-KTPFormer and
Proposed (joint loss, lambda 1.0).

Every run trained on two views and predicted every view of its test subjects (cvdc.eval.cross_view). So every
evaluated video is predicted once as a **held-out view** (in the rotation that held its view out; pooled over the 3
rotations and 5 folds, each video by a model that saw neither its subject nor its view) and twice as a **seen view**
(the other two rotations: the in-view reference of the same two-view models). The step (a2) models, trained on all
three views (``runs/phase3a2``), are a second reference ('all-view').

Within groups first (E01 expert, E02 expert, E01 non-expert; E9), video level (E15), 95% CIs from the subject
bootstrap (E17; the same draws for every row of a stratum, so differences are paired):
  exercise   accuracy and macro F1 of the exercise head (rep probabilities averaged per video, E18); pooled over
             both groups and exercises (headline) and per group
  MQ         the movement-quality score (D13) from the video head's items, 100 x mean(PF1, PF2, CF1, CF2, CF3); an item
             that does not vary in the video's group (T14) is not learned there and is replaced by its training mean
             in that group (E24). RMSE, MAE, Pearson, Spearman against the group-only baseline (within-group mean of the
             training subjects' training-view videos); the total score as a secondary target
  gaps       held-out - seen (paired, same models), Proposed - B1, and held-out - all-view
  VCI        centred 256-d (primary; raw and projection in the CSV) over pairs with one member in the held-out view
             and one in a seen view (same rep vs other groups), next to seen <-> seen pairs (two different seen
             cameras, for the other-group term too, so both pairings compare across cameras); random init, after
             CVDC (Proposed) and final; the all-view models on the same pairs
  distance   view stability (E22) of the embedding distance to the PT reference (E5) over synced expert
             performances: ICC(1) on held-out <-> seen pairs (a performance = one rep, or one set at video level, in
             the held-out view and one seen view), seen <-> seen and all three views; paired ΔICC vs B1 and vs the
             all-view models (Phase 4 distances) on the same pairs; between-view Spearman
Inputs: runs/<tag>/heldout_<V>/<method>/fold<k>/, outputs/phase5/crossview/embedding_distance_reps.csv
(scripts/phase5/crossview_distance.py), runs/phase3a2/<method>/fold<k>/ and outputs/phase4/embedding_distance_reps.csv.

usage: python scripts/phase5/report_crossview.py [--tag phase5_crossview] [--boot 2000] [--smoke]
Writes CSVs and summary.md to outputs/results/<tag>/.
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
sys.path.insert(0, str(REPO / "scripts" / "phase3"))
sys.path.insert(0, str(REPO / "scripts" / "phase4"))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.labels import MQ_ITEMS, movement_quality  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_table  # noqa: E402
from cvdc.eval.consistency import pooled_vci  # noqa: E402
from cvdc.eval.cross_view import main_methods  # noqa: E402
from cvdc.eval.metrics import classification, interval, item_weights, regression, subject_draws  # noqa: E402
from report_phase3a import fci, fmt  # noqa: E402
from report_phase4 import view_stability  # noqa: E402

STRATA = [("E01 expert", "E01", "expert"), ("E02 expert", "E02", "expert"), ("E01 non-expert", "E01", "non-expert")]
CLS_STRATA = [("pooled", None, None)] + STRATA
SHORT = {"b1": "B1", "b2_mediapipe": "B2-MediaPipe", "b2_ktpformer": "B2-KTPFormer", "proposed_j1": "Proposed",
         "base_group": "group-only", "base_mean": "predict-the-mean"}
PAIRINGS = ["held-out↔seen", "seen↔seen"]


def smask(df, ex, grp):
    m = np.ones(len(df), bool)
    if ex:
        m &= (df.exercise == ex).to_numpy()
    if grp:
        m &= (df.group == grp).to_numpy()
    return m


def ci_cols(prefix, rep, ci):
    lo, hi = interval(rep[1:], ci)
    return {prefix: float(rep[0]), f"{prefix}_lo": lo, f"{prefix}_hi": hi}


# ---------------------------------------------------------------- loading
def load_setting(root, methods, crossview):
    """Video-head and rep-head test predictions of every run, tagged with heldout ('all' for the all-view runs),
    fold, method and role (held-out / seen / all-view)."""
    vid, rep, done = [], [], {}
    dirs = [(d.name.split("_", 1)[1], d) for d in sorted(root.glob("heldout_*"))] if crossview else [("all", root)]
    for held, vdir in dirs:
        for m in methods:
            for f in sorted((vdir / m).glob("fold*/done.json")):
                d = json.loads(f.read_text(encoding="utf-8"))
                done[(held, m, d["fold"])] = d
                for name, lst in (("test_predictions_video.csv", vid), ("test_predictions.csv", rep)):
                    p = pd.read_csv(f.parent / name)
                    p["heldout"], p["fold"], p["method"] = held, d["fold"], m
                    p["role"] = np.where(p.view == held, "held-out", "seen") if crossview else "all-view"
                    lst.append(p)
    return pd.concat(vid, ignore_index=True), pd.concat(rep, ignore_index=True), done


def training_videos(v, plan):
    """The videos a run trained on (inner training + inner validation subjects, training views; E16)."""
    sel = v.subject.isin(plan["train"])
    if plan.get("views"):
        sel &= v.view.isin(plan["views"])
    return v[sel]


def add_mq(pv, v, done, active):
    """MQ_pred from the video head's items; items inactive in the video's group (T14) -> their training mean."""
    pv = pv.copy()
    for k in MQ_ITEMS:
        pred = pv[f"{k}_pred"].to_numpy(float).copy()
        for (held, m, f), idx in pv.groupby(["heldout", "method", "fold"]).groups.items():
            tr = training_videos(v, done[(held, m, f)]["subjects"])
            rows = pv.loc[idx]
            for (ex, g), d in tr.groupby(["exercise", "group"]):
                if k in active[(ex, g)]:
                    continue
                sel = rows.index[(rows.exercise == ex) & (rows.group == g)]
                pred[pv.index.get_indexer(sel)] = d[k].mean()
        pv[f"{k}_pred_mq"] = pred
    pv["MQ_pred"] = 100.0 * pv[[f"{k}_pred_mq" for k in MQ_ITEMS]].mean(axis=1)
    return pv


def baselines(pv, v, done, method):
    """Group-only and predict-the-mean of MQ and score per test video, fitted on the training videos of the run
    (same rotation and fold, ``method``'s plan: every method shares the folds and views)."""
    out = []
    keys = pv[pv.method == method][["heldout", "fold"]].drop_duplicates().itertuples(index=False)
    for held, f in keys:
        tr = training_videos(v, done[(held, method, f)]["subjects"])
        te = pv[(pv.method == method) & (pv.heldout == held) & (pv.fold == f)]
        gm = tr.groupby(["exercise", "group"])[["MQ", "score"]].mean()
        for name in ("base_group", "base_mean"):
            d = te[["video_id", "subject", "group", "exercise", "view", "heldout", "fold", "role"]].copy()
            d["method"] = name
            for t in ("MQ", "score"):
                d[f"{t}_pred"] = ([gm.loc[(e, g), t] for e, g in zip(te.exercise, te.group)] if name == "base_group"
                                  else tr[t].mean())
            out.append(d)
    return pd.concat(out, ignore_index=True)


def video_exercise(pr, codes):
    """Video-level exercise prediction: mean rep probabilities, argmax over the in-scope exercises (E18)."""
    cols = [f"p_ex_{c}" for c in codes]
    g = pr.groupby(["method", "role", "heldout", "fold", "video_id"], as_index=False)[cols].mean()
    g["ex_pred"] = np.asarray(codes)[g[cols].to_numpy().argmax(1)]
    return g


# ---------------------------------------------------------------- metrics
def stratum_draws(v, ex, grp, n_boot, seed):
    d = v[smask(v, ex, grp)].drop_duplicates("subject").sort_values("subject")
    return d.subject.to_numpy(), subject_draws(d.subject.to_numpy(), d.group.to_numpy(), n_boot, seed)


def weights(subj, subs, M):
    return np.vstack([np.ones(len(subj)), item_weights(subj, subs, M)])


def regression_table(P, v, n_boot, seed, ci, methods):
    """MQ (headline) and score metrics per stratum, method and role, plus paired differences."""
    rows, deltas, rep = [], [], {}
    for sname, ex, grp in STRATA:
        subs, M = stratum_draws(v, ex, grp, n_boot, seed)
        d0 = P[smask(P, ex, grp)]
        for (m, role), d in d0.groupby(["method", "role"]):
            W = weights(d.subject.to_numpy(), subs, M)
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
                         ("held-out", m, "all-view", m, "held-out − all-view"),
                         ("seen", m, "seen", "base_group", "seen vs group-only")]
                if m != "b1":
                    pairs += [("held-out", m, "held-out", "b1", "vs B1 (held-out)")]
                for ra, ma, rb, mb, what in pairs:
                    a, b = rep.get((sname, ma, ra, t)), rep.get((sname, mb, rb, t))
                    if a is None or b is None:
                        continue
                    for k in ("rmse", "spearman"):
                        dd = a[k] - b[k]
                        deltas.append(dict(stratum=sname, target=t, method=m, comparison=what, metric=k,
                                           **ci_cols("delta", dd, ci)))
                if m != "b1":  # does CVDC shrink the cross-view gap?
                    ga = [rep.get((sname, m, r, t)) for r in ("held-out", "seen")]
                    gb = [rep.get((sname, "b1", r, t)) for r in ("held-out", "seen")]
                    if all(x is not None for x in ga + gb):
                        for k in ("rmse", "spearman"):
                            dd = (ga[0][k] - ga[1][k]) - (gb[0][k] - gb[1][k])
                            deltas.append(dict(stratum=sname, target=t, method=m, comparison="gap − B1's gap",
                                               metric=k, **ci_cols("delta", dd, ci)))
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def classification_table(X, v, codes, n_boot, seed, ci, methods):
    rows, deltas, rep = [], [], {}
    for sname, ex, grp in CLS_STRATA:
        subs, M = stratum_draws(v, ex, grp, n_boot, seed)
        d0 = X[smask(X, ex, grp)]
        for (m, role), d in d0.groupby(["method", "role"]):
            W = weights(d.subject.to_numpy(), subs, M)
            t = d.exercise.map(codes.index).to_numpy()
            p = d.ex_pred.map(codes.index).to_numpy()
            r = classification(t, p, W, len(codes), labels=list(range(len(codes))) if ex is None else None)
            rep[(sname, m, role)] = r
            row = dict(stratum=sname, method=m, role=role, videos=len(d))
            for k in ("accuracy", "macro_f1"):
                row.update(ci_cols(k, r[k], ci))
            rows.append(row)
        for m in methods:
            for ra, ma, rb, mb, what in [("held-out", m, "seen", m, "held-out − seen (gap)"),
                                         ("held-out", m, "all-view", m, "held-out − all-view"),
                                         ("held-out", m, "held-out", "b1", "vs B1 (held-out)")]:
                if (ma, mb) == ("b1", "b1") and ra == rb:
                    continue
                a, b = rep.get((sname, ma, ra)), rep.get((sname, mb, rb))
                if a is None or b is None:
                    continue
                for k in ("accuracy", "macro_f1"):
                    deltas.append(dict(stratum=sname, method=m, comparison=what, metric=k,
                                       **ci_cols("delta", a[k] - b[k], ci)))
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def per_view_table(P, v, n_boot, seed, ci):
    """MQ metrics split by the video's view (held-out: the rotation that held this view out)."""
    rows = []
    for sname, ex, grp in STRATA:
        subs, M = stratum_draws(v, ex, grp, n_boot, seed)
        d0 = P[smask(P, ex, grp)]
        for (m, role, vw), d in d0.groupby(["method", "role", "view"]):
            if d.subject.nunique() < 3:
                continue
            W = weights(d.subject.to_numpy(), subs, M)
            r = regression(d.MQ.to_numpy(float), d.MQ_pred.to_numpy(float), W)
            row = dict(stratum=sname, method=m, role=role, view=vw, videos=len(d), subjects=d.subject.nunique())
            for k in ("rmse", "spearman"):
                row.update(ci_cols(k, r[k], ci))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- VCI
def vci_tables(root, a2root, methods, members, n_boot, seed, ci, views):
    gcode = pd.factorize(members.group_id)[0]
    snaps = []
    for vdir in sorted(root.glob("heldout_*")):
        held = vdir.name.split("_", 1)[1]
        for m in methods:
            for f in sorted((vdir / m).glob("fold*/done.json")):
                fold = json.loads(f.read_text(encoding="utf-8"))["fold"]
                for stage in ("init", "pretrained", "final"):
                    npz = f.parent / f"embeddings_{stage}.npz"
                    if npz.exists():
                        for pr in PAIRINGS:
                            snaps.append(((m, stage, pr), (held, fold), npz, (held, pr)))
    for m in methods:
        for f in sorted((a2root / m).glob("fold*/done.json")):
            fold = json.loads(f.read_text(encoding="utf-8"))["fold"]
            for held in views:
                for pr in PAIRINGS:
                    snaps.append(((m, "all-view", pr), (held, fold), f.parent / "embeddings_final.npz", (held, pr)))

    def pair_filter(ctx, mm):
        held, pr = ctx
        h, vw = (mm.view == held).to_numpy(), mm.view.to_numpy()
        if pr == PAIRINGS[0]:  # one member in the held-out view, one in a seen view
            return h[:, None] != h[None, :]
        # both in seen views and from different cameras, so the other-group term is cross-camera as above
        return (~h)[:, None] & (~h)[None, :] & (vw[:, None] != vw[None, :])

    rows, fv = pooled_vci(snaps, members, gcode, STRATA, n_boot, seed, ci, pair_filter)
    V = pd.DataFrame([{**{k: r[k] for k in r if k not in ("key", "boot", "present")}, "method": r["key"][0],
                       "stage": r["key"][1], "pairing": r["key"][2]} for r in rows])
    boot = {(r["key"], r["stratum"], r["variant"]): (r["boot"], r["present"]) for r in rows}
    deltas = []

    def diff(ka, kb, sname, var, what, method):
        a, b = boot.get((ka, sname, var)), boot.get((kb, sname, var))
        if a is None or b is None or a[1] != b[1]:
            return
        pa = V[(V.method == ka[0]) & (V.stage == ka[1]) & (V.pairing == ka[2]) & (V.stratum == sname) & (V.variant == var)]
        pb = V[(V.method == kb[0]) & (V.stage == kb[1]) & (V.pairing == kb[2]) & (V.stratum == sname) & (V.variant == var)]
        d0 = pa.vci.iloc[0] - pb.vci.iloc[0]
        lo, hi = interval(a[0] - b[0], ci)
        deltas.append(dict(stratum=sname, variant=var, method=method, comparison=what, delta=d0, delta_lo=lo,
                           delta_hi=hi))

    for sname, _, _ in STRATA:
        for var in ("centred", "raw", "proj"):
            for m in methods:
                diff((m, "final", PAIRINGS[0]), (m, "final", PAIRINGS[1]), sname, var, "held-out↔seen − seen↔seen (gap)", m)
                diff((m, "final", PAIRINGS[0]), (m, "all-view", PAIRINGS[0]), sname, var, "held-out↔seen: cross-view − all-view model", m)
                if m != "b1":
                    diff((m, "final", PAIRINGS[0]), ("b1", "final", PAIRINGS[0]), sname, var, "held-out↔seen vs B1", m)
    return V, pd.DataFrame(deltas), fv


# ---------------------------------------------------------------- distance view stability
def standardise(E, keys):
    """z-score the distance per ``keys`` with the rows marked train (training subjects in training views)."""
    E = E.copy()
    st = E[E.train].groupby(keys).d_cos.agg(["mean", "std"])
    j = E.join(st, on=keys)
    E["z"] = (j.d_cos - j["mean"]) / j["std"]
    return E


def variants(E, held, level, views):
    """Long frames per variant for one rotation ``held``: perf ids so that each performance holds the values to compare."""
    key = "rep_uid" if level == "rep" else "group_id"
    out = {}
    hs = []
    for sv in [x for x in views if x != held]:
        d = E[E.view.isin([held, sv])]
        d = d.assign(perf=held + "|" + d[key].astype(str) + "|" + sv)
        hs.append(d)
    out[PAIRINGS[0]] = pd.concat(hs, ignore_index=True)
    out[PAIRINGS[1]] = E[E.view != held].assign(perf=lambda x: held + "|" + x[key].astype(str))
    out["all three views"] = E.assign(perf=lambda x: held + "|" + x[key].astype(str))
    return out


def distance_tables(cv_csv, p4_csv, reps, synced, methods, views, n_boot, seed, ci):
    meta = reps[["video_id", "rep", "rep_uid", "group_id", "exercise", "subject", "group", "view"]]
    C = pd.read_csv(cv_csv).merge(meta, on=["video_id", "rep"], how="left", validate="many_to_one")
    C = C[C.method.isin(methods)].rename(columns={"train_view": "train"})
    C = standardise(C, ["heldout", "method", "fold", "exercise", "group"])
    A = pd.read_csv(p4_csv)
    A = A[A.method.isin(methods)].merge(meta, on=["video_id", "rep"], how="left", validate="many_to_one")
    A["train"] = ~A.is_test  # Phase 4 (E5): training subjects, every view
    A = standardise(A, ["method", "fold", "exercise", "group"])
    sel = lambda E: E[E.is_test & E.group_id.isin(synced) & (E.group == "expert")]  # noqa: E731
    C, A = sel(C), sel(A)
    rows, icc = [], {}
    for sname, ex, _ in STRATA[:2]:
        for level in ("rep", "video"):
            for setting, E0 in (("cross-view", C), ("all-view", A)):
                for m in methods:
                    E = E0[(E0.method == m) & (E0.exercise == ex)]
                    if level == "video":
                        g = ["heldout", "fold", "video_id", "group_id", "subject", "group", "view"] if setting == "cross-view" \
                            else ["fold", "video_id", "group_id", "subject", "group", "view"]
                        E = E.groupby(g, as_index=False).z.mean()
                    parts = {}
                    for held in views:
                        Eh = E[E.heldout == held] if setting == "cross-view" else E
                        for name, d in variants(Eh, held, level, views).items():
                            if setting == "all-view" and name == "all three views" and held != views[0]:
                                continue  # the all-view model has one copy of each performance
                            parts.setdefault(name, []).append(d)
                    for name, lst in parts.items():
                        d = pd.concat(lst, ignore_index=True)
                        r, b, _ = view_stability(d, "z", n_boot, seed, ci)
                        icc[(sname, level, setting, name, m)] = (b, r["perfs"])
                        rows.append(dict(stratum=sname, level=level, setting=setting, variant=name, method=m, **r))
    VS = pd.DataFrame(rows)
    deltas = []
    for (sname, level, setting, name, m), (b, n) in icc.items():
        r = VS[(VS.stratum == sname) & (VS.level == level) & (VS.setting == setting) & (VS.variant == name) &
               (VS.method == m)].iloc[0]
        for other, what in ((("b1", setting), "vs B1"), ((m, "all-view"), "cross-view − all-view model")):
            if (other[0], other[1]) == (m, setting):
                continue
            o = icc.get((sname, level, other[1], name, other[0]))
            if o is None or o[1] != n:
                continue
            ro = VS[(VS.stratum == sname) & (VS.level == level) & (VS.setting == other[1]) & (VS.variant == name) &
                    (VS.method == other[0])].iloc[0]
            lo, hi = interval(b - o[0], ci)
            deltas.append(dict(stratum=sname, level=level, setting=setting, variant=name, method=m, comparison=what,
                               delta=r.icc - ro.icc, delta_lo=lo, delta_hi=hi))
    return VS, pd.DataFrame(deltas)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="phase5_crossview")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--smoke", action="store_true", help="the smoke runs (<tag>_smoke, phase3a2_smoke): plumbing only")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    if a.smoke:
        a.tag += "_smoke"
    ec, ia = cfg["eval"], cfg["heads"]["item_activity"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    methods = main_methods(cfg)
    views = list(cfg["data"]["views"])
    codes = list(cfg["data"]["include_exercises"])
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    a2root = resolve(cfg["train"]["runs_dir"]) / ("phase3a2_smoke" if a.smoke else "phase3a2")
    out = resolve(ec["results_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)

    v = video_table(cfg)
    v["MQ"] = movement_quality(v)
    reps = pd.read_csv(resolve("data/reps.csv"))
    act = item_activity(v, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}

    cv_v, cv_r, done = load_setting(root, methods, True)
    av_v, av_r, done_a = load_setting(a2root, methods, False)
    done.update(done_a)
    n_runs = {m: sum(1 for k in done if k[1] == m and k[0] != "all") for m in methods}
    assert all(n == (3 if a.smoke else 15) for n in n_runs.values()), f"expected 3 views x 5 folds per method: {n_runs}"
    v_all = v  # training statistics (MQ item means, baselines) come from the full table
    if a.smoke:  # only the smoke test subjects were predicted
        v = v[v.subject.isin({s for d in done.values() for s in d["subjects"]["test"]})].reset_index(drop=True)
    # every evaluated video: once held-out, twice seen (cross-view), once all-view
    for m in methods:
        c = cv_v[cv_v.method == m]
        assert c[c.role == "held-out"].video_id.value_counts().eq(1).all()
        assert c[c.role == "seen"].video_id.value_counts().eq(2).all()
        assert set(c.video_id) == set(v.video_id) == set(av_v[av_v.method == m].video_id)
    pv = pd.concat([cv_v, av_v], ignore_index=True)
    pv = add_mq(pv, v_all, done, active)
    base = pd.concat([baselines(cv_v, v_all, done, methods[0]), baselines(av_v, v_all, done, methods[0])],
                     ignore_index=True)
    lab = v[["video_id", "variation", "MQ", "score"]]
    P = pd.concat([pv[["method", "role", "heldout", "fold", "video_id", "subject", "group", "exercise", "view",
                       "MQ_pred", "score_pred"]], base], ignore_index=True).merge(lab, on="video_id", how="left")
    P.to_csv(out / "video_predictions.csv", index=False)

    S, D = regression_table(P, v, n_boot, seed, ci, methods)
    S.to_csv(out / "mq_score_metrics.csv", index=False)
    D.to_csv(out / "mq_score_deltas.csv", index=False)
    PV = per_view_table(P[P.method.isin(methods + ["base_group"])], v, n_boot, seed, ci)
    PV.to_csv(out / "mq_by_view.csv", index=False)

    X = video_exercise(pd.concat([cv_r, av_r], ignore_index=True), codes).merge(
        v[["video_id", "subject", "group", "exercise", "view"]], on="video_id", how="left")
    CL, CD = classification_table(X, v, codes, n_boot, seed, ci, methods)
    CL.to_csv(out / "exercise_metrics.csv", index=False)
    CD.to_csv(out / "exercise_deltas.csv", index=False)

    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    VC, VD, VF = vci_tables(root, a2root, methods, members, n_boot, seed, ci, views)
    VC.to_csv(out / "vci.csv", index=False)
    VD.to_csv(out / "vci_deltas.csv", index=False)
    VF.to_csv(out / "vci_per_fold.csv", index=False)

    sync = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    synced = set(sync.index[sync.cls == "synced"])
    VS, VSD = distance_tables(resolve("outputs/phase5/crossview/embedding_distance_reps.csv"),
                              resolve("outputs/phase4/embedding_distance_reps.csv"), reps, synced, methods, views,
                              n_boot, seed, ci)
    VS.to_csv(out / "distance_view_stability.csv", index=False)
    VSD.to_csv(out / "distance_view_stability_deltas.csv", index=False)

    T = training_table(root, methods)
    T.to_csv(out / "stopping_runtime.csv", index=False)
    write_summary(out, cfg, methods, v, P, S, D, PV, CL, CD, VC, VD, VS, VSD, T, n_boot, act)
    print(f"report written to {out} in {time.time() - t0:.0f} s")


def training_table(root, methods):
    rows = []
    for vdir in sorted(root.glob("heldout_*")):
        held = vdir.name.split("_", 1)[1]
        for m in methods:
            for f in sorted((vdir / m).glob("fold*/done.json")):
                d = json.loads(f.read_text(encoding="utf-8"))
                row = dict(heldout=held, method=m, fold=d["fold"], wall_s=d["wall_s"], peak_gpu_gb=d.get("peak_gpu_gb"))
                for st in ("cvdc_stage", "supervised_stage"):
                    if st in d:
                        s = d[st]
                        row.update({f"{st[:4]}_best": s["best_epoch"], f"{st[:4]}_stopped": s["stopped_epoch"],
                                    f"{st[:4]}_s": s["wall_s"]})
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary
def write_summary(out, cfg, methods, v, P, S, D, PV, CL, CD, VC, VD, VS, VSD, T, n_boot, act):
    L = []
    w = L.append

    def s_cell(sname, m, role, t="MQ"):
        r = S[(S.stratum == sname) & (S.method == m) & (S.role == role) & (S.target == t)]
        if r.empty:
            return "–"
        r = r.iloc[0]
        return f"{fci(r.rmse, r.rmse_lo, r.rmse_hi)} / {fmt(r.spearman)}"

    def d_cell(df, **kw):
        r = df
        for k, val in kw.items():
            r = r[r[k] == val]
        if r.empty:
            return "–"
        r = r.iloc[0]
        return fci(r.delta, r.delta_lo, r.delta_hi)

    nv = v.view.value_counts()
    w("# Phase 5 A: cross-view protocol (train on two views, test on the held-out third), GroupKFold-5")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/phase5/report_crossview.py`. Same stratified "
      "GroupKFold-5 folds, inner validation splits, seeds, epoch budget, early stopping and schedule as Phase 3 step "
      "(a2) (T8b, T9, T13, T14); each run trains on two views only (supervised items, inner validation, target "
      "scalers, CVDC pretraining and joint-loss batches, embedding centre) and predicts every view of its test "
      "subjects. Rotations F / L / R held out → 3 × 5 runs per method; Proposed = CVDC pretraining on the two training "
      "views (fresh, not the step (a) encoders) + fine-tuning with + 1.0 · L_cvdc (T15). Every evaluated video "
      f"({len(v)}: F {nv.get('F', 0)} / L {nv.get('L', 0)} / R {nv.get('R', 0)}) is predicted **once as a held-out view** "
      "(by the rotation that held its view out) and twice as a seen view; the step (a2) models trained on all three "
      f"views are the 'all-view' reference. Video level; 95% CIs from the subject bootstrap ({n_boot} draws, the same "
      "for every row of a stratum, so differences are paired). All §0 limitations apply (2 shoulder exercises; E02 "
      "experts only; E01 non-experts mostly front-only, so their held-out L / R rows rest on few videos).")
    w("")
    w("## Exercise classification (secondary, E6), video level")
    w("")
    w("Accuracy [95% CI] / macro F1, pooled over both exercises and groups. Gap = held-out − seen (same models).")
    w("")
    w("| Method | held-out view | seen views | all-view model | gap (accuracy) | vs B1 (held-out) |")
    w("|---|---|---|---|---|---|")
    for m in methods:
        cells = []
        for role in ("held-out", "seen", "all-view"):
            r = CL[(CL.stratum == "pooled") & (CL.method == m) & (CL.role == role)]
            cells.append("–" if r.empty else f"{fci(r.accuracy.iloc[0], r.accuracy_lo.iloc[0], r.accuracy_hi.iloc[0], 3)} / {fmt(r.macro_f1.iloc[0], 3)}")
        g = CD[(CD.stratum == "pooled") & (CD.method == m) & (CD.comparison == "held-out − seen (gap)") & (CD.metric == "accuracy")]
        b = CD[(CD.stratum == "pooled") & (CD.method == m) & (CD.comparison == "vs B1 (held-out)") & (CD.metric == "accuracy")]
        cells += ["–" if g.empty else fci(g.delta.iloc[0], g.delta_lo.iloc[0], g.delta_hi.iloc[0], 3),
                  "–" if b.empty else fci(b.delta.iloc[0], b.delta_lo.iloc[0], b.delta_hi.iloc[0], 3)]
        w(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
    w("")
    w("Per group (held-out view accuracy): " + "; ".join(
        f"{SHORT[m]} " + ", ".join(f"{s} {fmt(CL[(CL.stratum == s) & (CL.method == m) & (CL.role == 'held-out')].accuracy.iloc[0], 3)}"
                                   for s, _, _ in STRATA) for m in methods) + ".")
    w("")
    w("## Movement-quality score (MQ, D13) within groups, video level")
    w("")
    w("RMSE [95% CI] / Spearman ρ. MQ from the video head's PF1, PF2, CF1–CF3; items that do not vary in a group (T14) "
      "are set to their training mean there. Group-only = the training subjects' within-group mean (training views).")
    w("")
    w("| Method | role | " + " | ".join(s for s, _, _ in STRATA) + " |")
    w("|---|---|" + "---|" * len(STRATA))
    for m in methods:
        for role in ("held-out", "seen", "all-view"):
            w(f"| {SHORT[m]} | {role} | " + " | ".join(s_cell(s, m, role) for s, _, _ in STRATA) + " |")
    for b in ("base_group", "base_mean"):
        w(f"| {SHORT[b]} | held-out | " + " | ".join(s_cell(s, b, "held-out") for s, _, _ in STRATA) + " |")
    w("")
    w("Paired differences (MQ RMSE; negative = lower error):")
    w("")
    w("| Method | comparison | " + " | ".join(s for s, _, _ in STRATA) + " |")
    w("|---|---|" + "---|" * len(STRATA))
    for m in methods:
        for comp in ("vs group-only", "held-out − seen (gap)", "held-out − all-view", "vs B1 (held-out)", "gap − B1's gap"):
            cells = [d_cell(D, stratum=s, method=m, comparison=comp, metric="rmse", target="MQ") for s, _, _ in STRATA]
            if all(c == "–" for c in cells):
                continue
            w(f"| {SHORT[m]} | {comp} | " + " | ".join(cells) + " |")
    w("")
    w("Spearman ρ differences (held-out − seen; vs B1) and the total score (secondary target, same layout) are in "
      "`mq_score_deltas.csv` / `mq_score_metrics.csv`; per view in `mq_by_view.csv`.")
    w("")
    w("Total score (secondary), held-out view, RMSE / ρ: " + "; ".join(
        f"{SHORT[m]} " + ", ".join(f"{s} {s_cell(s, m, 'held-out', 'score')}" for s, _, _ in STRATA) for m in methods + ["base_group"]) + ".")
    w("")
    w("## Viewpoint Consistency Index (E13), held-out subjects")
    w("")
    w("Centred 256-d VCI [95% CI]. **held-out↔seen**: same rep seen from the held-out view and a seen view, minus other "
      "groups (also one member in each); **seen↔seen**: both members in the two training views, from different cameras "
      "(other-group pairs too). Pooled over the 3 rotations and 5 folds, both pairings contain the same camera pairs "
      "(F–L, F–R, L–R), so for a model that saw every view the two agree by construction. 'all-view' = the step (a2) model of the method on the same pairs (it saw all views).")
    w("")
    w("| Method | stage | pairing | " + " | ".join(s for s, _, _ in STRATA) + " |")
    w("|---|---|---|" + "---|" * len(STRATA))
    for m in methods:
        for stage in ("init", "pretrained", "final", "all-view"):
            for pr in PAIRINGS:
                r = VC[(VC.method == m) & (VC.stage == stage) & (VC.pairing == pr) & (VC.variant == "centred")]
                if r.empty or (stage == "init" and m != "b1"):
                    continue
                cells = []
                for s, _, _ in STRATA:
                    x = r[r.stratum == s]
                    cells.append("–" if x.empty else f"{fci(x.vci.iloc[0], x.vci_lo.iloc[0], x.vci_hi.iloc[0], 3)} ({x.same_pairs.iloc[0]})")
                nm = "random init (B1 seed)" if stage == "init" else SHORT[m]
                w(f"| {nm} | {stage} | {pr} | " + " | ".join(cells) + " |")
    w("")
    w("(n) = same-rep view pairs. Paired differences (centred):")
    w("")
    w("| Method | comparison | " + " | ".join(s for s, _, _ in STRATA) + " |")
    w("|---|---|" + "---|" * len(STRATA))
    for m in methods:
        for comp in ("held-out↔seen vs B1", "held-out↔seen − seen↔seen (gap)", "held-out↔seen: cross-view − all-view model"):
            cells = [d_cell(VD, stratum=s, method=m, comparison=comp, variant="centred") for s, _, _ in STRATA]
            if all(c == "–" for c in cells):
                continue
            w(f"| {SHORT[m]} | {comp} | " + " | ".join(cells) + " |")
    w("")
    w("## View stability of the embedding distance to the PT reference (E5, E22)")
    w("")
    w("ICC(1) [95% CI] of the standardised distance (closeness to the PT demo; standardised per rotation, fold, "
      "exercise and group with the training subjects' training-view reps) over synced expert performances. "
      "held-out↔seen: a performance = one rep (video level: one set) in the held-out view and one seen view; "
      "seen↔seen: the training views; all three views. Higher = more stable. The PT demo is a front-view clip, so with "
      "F held out the reference itself comes from the unseen view.")
    w("")
    for level in ("rep", "video"):
        w(f"**{level} level**")
        w("")
        w("| Method | setting | " + " | ".join(f"{s} {var}" for s, _, _ in STRATA[:2] for var in (PAIRINGS[0], PAIRINGS[1], "all three views")) + " |")
        w("|---|---|" + "---|" * 6)
        for m in methods:
            for setting in ("cross-view", "all-view"):
                cells = []
                for s, _, _ in STRATA[:2]:
                    for var in (PAIRINGS[0], PAIRINGS[1], "all three views"):
                        r = VS[(VS.stratum == s) & (VS.level == level) & (VS.setting == setting) & (VS.variant == var) & (VS.method == m)]
                        cells.append("–" if r.empty else fci(r.icc.iloc[0], r.icc_lo.iloc[0], r.icc_hi.iloc[0]))
                w(f"| {SHORT[m]} | {setting} | " + " | ".join(cells) + " |")
        w("")
    w("Paired ΔICC (rep level, held-out↔seen):")
    w("")
    w("| Method | comparison | " + " | ".join(s for s, _, _ in STRATA[:2]) + " |")
    w("|---|---|---|---|")
    for m in methods:
        for comp in ("vs B1", "cross-view − all-view model"):
            cells = [d_cell(VSD, stratum=s, level="rep", setting="cross-view", variant=PAIRINGS[0], method=m, comparison=comp)
                     for s, _, _ in STRATA[:2]]
            if all(c == "–" for c in cells):
                continue
            w(f"| {SHORT[m]} | {comp} | " + " | ".join(cells) + " |")
    w("")
    w("Between-view Spearman and R_view are in `distance_view_stability.csv`; all ΔICC in `distance_view_stability_deltas.csv`.")
    w("")
    w("## Training")
    w("")
    for m in methods:
        t = T[T.method == m]
        line = f"- {SHORT[m]}: {t.wall_s.sum() / 3600:.2f} h for 15 runs"
        if "supe_best" in t:
            line += f"; kept fine-tuning epochs median {t.supe_best.median():.0f} (range {t.supe_best.min():.0f}–{t.supe_best.max():.0f})"
        if "cvdc_best" in t and t.cvdc_best.notna().any():
            line += f"; CVDC kept epochs median {t.cvdc_best.median():.0f} ({t.cvdc_best.min():.0f}–{t.cvdc_best.max():.0f}), {t.cvdc_s.sum() / 3600:.2f} h"
        w(line + ".")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
