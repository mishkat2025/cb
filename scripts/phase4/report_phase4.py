"""Phase 4 report: Module III (reference-based assessment) and the embedding-to-reference distance, E01 and E02.

Inputs (run first): python scripts/phase4/embedding_distance.py; python -m cvdc.assess.run;
python scripts/phase4/recording_noise.py.

Tables, within groups first (E01 expert, E02 expert, E01 non-expert; E9), video level (E15), test predictions of the
5 GroupKFold folds pooled (E1), 95% CIs from the subject-level bootstrap (E17, 2,000 draws, the same draws for every
row of a stratum):
  agreement      the Module III reference score (set score = mean rep score) vs the total score, the movement-quality
                 score (MQ, D13) and every EAAQ item; items that do not vary in a group (T14) are listed as such
  components     each Module III measure vs total / MQ / items (Spearman)
  regression     out-of-fold single-feature lines (total or MQ on the reference score / the distance, fitted per fold
                 and group on the training videos) vs the group-only baseline (paired RMSE difference)
  errors         rule firing rates per group, in the experts' >= 95 videos (a specificity reference) and their
                 relation to the scores
  embedding      (a) distance to the PT reference embedding vs total / MQ / items, per method; distances are
                 standardised per fold and exercise with the training subjects' videos before pooling
                 (b) view stability over synced expert performances (same rep seen from F / L / R): ICC(1) of the
                 standardised distance, between-view Spearman, and R_view = mean |change between views| / mean
                 |difference between performances|; Module III's score and end-range angle as comparison rows
  recording      item 6 (recording_noise.json)
  posture centre before / after (M14, post-hoc): the Phase 4 first run with every channel centred on the PT demo
                 (outputs/phase4/posture_pt, ``python -m cvdc.assess.run --posture-centre pt_demo --out ... --no-json``)
                 vs the posture checks centred on the high-scoring experts (default): agreement with paired
                 differences, rule rates overall and in the >= 95 videos, view stability of the score
  speed (exploratory, M15): the speed centred on the demo instead of the experts, as a component and as a score
                 variant; post-hoc, not the primary score
Agreement signs: "closeness" = -distance, so for every row a positive rho means agreement with the physiotherapists.

usage: python scripts/phase4/report_phase4.py [--boot 2000]
Writes outputs/phase4/*.csv, figures in outputs/phase4/figs/ and outputs/phase4/summary.md.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.assess.template import load_library  # noqa: E402
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_table  # noqa: E402
from cvdc.eval.metrics import (icc_weighted, interval, item_weights, mean_abs_diff_weighted, regression,  # noqa: E402
                               subject_draws)
from cvdc.train.trainer import fold_plan  # noqa: E402

STRATA = [("E01 expert", "E01", "expert"), ("E02 expert", "E02", "expert"), ("E01 non-expert", "E01", "non-expert")]
POOLED = [("E01 pooled", "E01", None)]
TARGETS = ["score", "MQ", *ALL_ITEMS]
METHODS = ["b1", "b2_mediapipe", "b2_ktpformer", "proposed_j1", "proposed_lp", "init_xy"]
LABEL = {"b1": "B1 (2D xy)", "b2_mediapipe": "B2-MediaPipe (world xyz)", "b2_ktpformer": "B2-KTPFormer (lifted xyz)",
         "proposed_j1": "**Proposed** (CVDC, joint λ = 1.0)", "proposed_lp": "Proposed alt. (frozen CVDC encoder)",
         "init_xy": "random-init encoder (xy; floor)", "module3": "Module III reference score",
         "module3_rom": "Module III end-range angle",
         "module3_spatial": "Module III score without speed (angle + ROM + symmetry)"}
COMPONENTS = {"X_angle": "angle-deviation distance (SD)", "X_rom": "ROM-shortfall distance (SD)",
              "X_speed": "speed distance (SD)", "X_symmetry": "symmetry distance (SD)", "rom": "end-range angle (deg)",
              "excursion": "excursion max-min (deg)", "log2_speed": "log2 speed ratio (demo / rep)",
              "omega_dev": "angular-velocity deviation (deg/s)", "duration_s": "rep duration (s)"}
SIGN = {"X_angle": -1, "X_rom": -1, "X_speed": -1, "X_symmetry": -1, "omega_dev": -1}  # distances: closer = better


def fmt(v, d=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def fci(v, lo, hi, d=2):
    return f"{fmt(v, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def mask(df, ex, grp):
    m = np.ones(len(df), bool)
    if ex:
        m &= (df.exercise == ex).to_numpy()
    if grp:
        m &= (df.group == grp).to_numpy()
    return m


def draws(subjects, groups, n, seed):
    subj = pd.DataFrame({"subject": subjects, "group": groups}).drop_duplicates("subject").sort_values("subject")
    return subj.subject.to_numpy(), subject_draws(subj.subject.to_numpy(), subj.group.to_numpy(), n, seed)


def corr_ci(y, x, W, ci):
    """Spearman and Pearson of x vs y with bootstrap rows W[1:] (row 0 = point)."""
    ok = np.isfinite(y) & np.isfinite(x)
    r = regression(y[ok], x[ok], W[:, ok])
    out = {}
    for k in ("spearman", "pearson"):
        lo, hi = interval(r[k][1:], ci)
        out[k], out[k + "_lo"], out[k + "_hi"] = float(r[k][0]), lo, hi
    out["n"] = int(ok.sum())
    return out


# ---------------------------------------------------------------- view stability
def view_stability(df, value, n_boot, seed, ci):
    """df: rows of one stratum with columns subject, group, perf (performance id), view, ``value``. Returns metrics with
    CIs and the bootstrap replicates of the ICC (for paired differences)."""
    d = df[np.isfinite(df[value])]
    d = d[d.groupby("perf").view.transform("nunique") >= 2].reset_index(drop=True)
    subs, M = draws(d.subject.to_numpy(), d.group.to_numpy(), n_boot, seed)
    M = np.vstack([np.ones(len(subs)), M])
    code, perfs = pd.factorize(d.perf)
    psub = d.drop_duplicates("perf").set_index("perf").subject.reindex(perfs).to_numpy()
    pos = {s: i for i, s in enumerate(subs)}
    Wset = M[:, [pos[s] for s in psub]]
    x = d[value].to_numpy(float)
    icc = icc_weighted(x, code, Wset)
    # R_view: mean |between-view change| of a performance / mean |difference between performances|
    within, wpair = [], []
    for _, g in d.groupby("perf"):
        v = g[value].to_numpy(float)
        within.append(np.mean([abs(a - b) for a, b in itertools.combinations(v, 2)]))
        wpair.append(pos[g.subject.iloc[0]])
    within = np.asarray(within)
    Wp = M[:, wpair]
    num = (Wp * within).sum(1) / Wp.sum(1)
    wi = item_weights(d.subject.to_numpy(), subs, M[1:])
    den = mean_abs_diff_weighted(x, np.vstack([np.ones(len(x)), wi]))
    rview = num / den
    out = {"perfs": len(perfs), "values": len(d), "subjects": len(subs), "icc": float(icc[0]),
           "icc_lo": interval(icc[1:], ci)[0], "icc_hi": interval(icc[1:], ci)[1], "r_view": float(rview[0]),
           "r_view_lo": interval(rview[1:], ci)[0], "r_view_hi": interval(rview[1:], ci)[1]}
    for va, vb in (("F", "L"), ("F", "R"), ("L", "R")):
        p = d.pivot_table(index="perf", columns="view", values=value, aggfunc="mean")
        if va in p and vb in p:
            p = p[[va, vb]].dropna()
            sub = d.drop_duplicates("perf").set_index("perf").subject.reindex(p.index).to_numpy()
            W = np.vstack([np.ones(len(p)), item_weights(sub, subs, M[1:])])
            r = regression(p[va].to_numpy(), p[vb].to_numpy(), W)["spearman"]
            out[f"rho_{va}{vb}"], (out[f"rho_{va}{vb}_lo"], out[f"rho_{va}{vb}_hi"]) = float(r[0]), interval(r[1:], ci)
            out[f"n_{va}{vb}"] = len(p)
    return out, icc[1:], rview[1:]


# ---------------------------------------------------------------- out-of-fold single-feature lines
def oof_line(v, feat, target, plans):
    """Within-group least-squares line target ~ feat, fitted per fold on the training videos (each stratum)."""
    pred = pd.Series(np.nan, index=v.index)
    base = pd.Series(np.nan, index=v.index)
    for p in plans:
        tr, te = v.subject.isin(p["train"]), v.subject.isin(p["test"])
        for (ex, g), d in v[tr].groupby(["exercise", "group"]):
            sel = te & (v.exercise == ex) & (v.group == g)
            ok = d[feat].notna() & d[target].notna()
            if ok.sum() < 3 or not sel.any():
                continue
            b = np.polyfit(d.loc[ok, feat], d.loc[ok, target], 1)
            x = v.loc[sel, feat].fillna(d.loc[ok, feat].median())
            pred[sel] = np.polyval(b, x)
            base[sel] = d[target].mean()
    return pred, base


def standardise_per_fold(E, plans, col):
    """Distance standardised per (method, fold, exercise, group) with the training subjects' values of the same
    method, fold, exercise and group (no test data), so test values of different fold models can be pooled."""
    E = E.copy()
    E[col + "_z"] = np.nan
    for p in plans:
        for (m, ex, g), d in E[E.fold == p["fold"]].groupby(["method", "exercise", "group"]):
            tr = d.subject.isin(p["train"])
            mu, sd = d.loc[tr, col].mean(), d.loc[tr, col].std()
            E.loc[d.index, col + "_z"] = (d[col] - mu) / sd
    return E


def rescore(R, sc, replace=None):
    """Rep score from the group distances X_g (M11), optionally with some groups replaced (dict g -> array)."""
    gs = ("angle", "rom", "speed", "symmetry")
    cols = {g: R[f"X_{g}"].to_numpy(float) for g in gs}
    cols.update(replace or {})
    P = np.stack([np.minimum(1.0, cols[g] / sc["d_zero"]) * sc["weights"][g] for g in gs], 1)
    Wt = np.stack([np.isfinite(cols[g]) * sc["weights"][g] for g in gs], 1)
    return 100 * (1 - np.nansum(P, 1) / Wt.sum(1))


def paired_rho(d, cols, targets, n_boot, seed, ci, active, ex, grp, signs=None):
    """Spearman of each column with each target on the same subject draws, plus paired differences vs cols[0]."""
    d = d.reset_index(drop=True)
    subs, M = draws(d.subject.to_numpy(), d.group.to_numpy(), n_boot, seed)
    W = np.vstack([np.ones(len(d)), item_weights(d.subject.to_numpy(), subs, M)])
    rows = []
    for t in targets:
        if t not in ("score", "MQ") and t not in active.get((ex, grp), set()):
            continue
        base = None
        for c in cols:
            x = (signs or {}).get(c, 1) * d[c].to_numpy(float)
            y = d[t].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            r = regression(y[ok], x[ok], W[:, ok])["spearman"]
            row = dict(target=t, column=c, rho=float(r[0]), lo=interval(r[1:], ci)[0], hi=interval(r[1:], ci)[1])
            if base is None:
                base = r
            else:
                dd = r - base
                row.update(delta=float(dd[0]), delta_lo=interval(dd[1:], ci)[0], delta_hi=interval(dd[1:], ci)[1])
            rows.append(row)
    return rows


def before_after(cfg, V, R, Vb, Rb, reps, synced, active, n_boot, seed, ci):
    """M14: posture checks centred on the PT demo (before) vs on the high-scoring experts (after)."""
    agree, rates, stab = [], [], []
    vb = Vb.set_index("video_id").ref_set_score
    Vx = V.assign(score_before=V.video_id.map(vb).to_numpy(), score_after=V.ref_set_score)
    for sname, ex, grp in STRATA:
        d = Vx[mask(Vx, ex, grp)]
        for r in paired_rho(d, ["score_before", "score_after"], TARGETS, n_boot, seed, ci, active, ex, grp):
            agree.append(dict(stratum=sname, **r))
    vscore = V.set_index("video_id").score
    errs = sorted({c for c in R.columns if c.startswith("err_")} | {c for c in Rb.columns if c.startswith("err_")})
    for nm, RR in (("before", Rb), ("after", R)):
        RR = RR.assign(vscore=RR.video_id.map(vscore))
        for sname, ex, grp in STRATA:
            d = RR[mask(RR, ex, grp)]
            top = d[d.vscore >= cfg["templates"]["min_score"]]
            for e in errs:
                if e not in d:
                    continue
                rates.append(dict(variant=nm, stratum=sname, error=e[4:], rate=d[e].mean(),
                                  rate_ge95=top[e].mean() if len(top) else np.nan,
                                  rate_lt80=d[d.vscore < 80][e].mean() if (d.vscore < 80).any() else np.nan))
        rr = RR.merge(reps[["video_id", "rep", "rep_uid", "group_id"]], on=["video_id", "rep"], how="left")
        rr = rr[rr.group_id.isin(synced) & (rr.group == "expert")]
        for sname, ex, _ in STRATA[:2]:
            x = rr[rr.exercise == ex]
            r, _, _ = view_stability(x.assign(perf=x.rep_uid), "ref_score", n_boot, seed, ci)
            stab.append(dict(variant=nm, stratum=sname, **r))
    return pd.DataFrame(agree), pd.DataFrame(rates), pd.DataFrame(stab), Vx


def speed_exploratory(cfg, V, R, active, n_boot, seed, ci):
    """M15 (post-hoc, exploratory): speed centred on the demo vs on the experts (primary)."""
    sc = cfg["assess"]["score"]
    R = R.copy()
    R["X_speed_demo"] = np.abs(R.z_speed_demo.to_numpy(float))
    R["score_speed_demo"] = rescore(R, sc, {"speed": R.X_speed_demo.to_numpy(float)})
    v = R.groupby("video_id")[["X_speed", "X_speed_demo", "score_speed_demo", "omega_dev"]].mean()
    d0 = V.join(v, on="video_id")
    rows = []
    for sname, ex, grp in STRATA:
        d = d0[mask(d0, ex, grp)]
        for r in paired_rho(d, ["X_speed", "X_speed_demo", "omega_dev"], TARGETS, n_boot, seed, ci, active, ex, grp,
                            signs={"X_speed": -1, "X_speed_demo": -1, "omega_dev": -1}):
            rows.append(dict(stratum=sname, part="component", **r))
        for r in paired_rho(d, ["ref_set_score", "score_speed_demo"], TARGETS, n_boot, seed, ci, active, ex, grp):
            rows.append(dict(stratum=sname, part="score", **r))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--boot", type=int)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    ec = cfg["eval"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    out = resolve("outputs/phase4")
    (out / "figs").mkdir(parents=True, exist_ok=True)
    ia = cfg["heads"]["item_activity"]
    reps = pd.read_csv(resolve("data/reps.csv"))
    plans = fold_plan(cfg, reps)
    V = pd.read_csv(out / "module3_videos.csv")
    R = pd.read_csv(out / "module3_reps.csv")
    act = item_activity(video_table(cfg), ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}

    # ---------------- (4) clinical agreement of the Module III reference score
    rows, comp_rows, view_rows, reg_rows = [], [], [], []
    for sname, ex, grp in STRATA + POOLED:
        d = V[mask(V, ex, grp)].reset_index(drop=True)
        subs, M = draws(d.subject.to_numpy(), d.group.to_numpy(), n_boot, seed)
        W = np.vstack([np.ones(len(d)), item_weights(d.subject.to_numpy(), subs, M)])
        for t in TARGETS:
            varies = grp is None or t in ("score", "MQ") or t in active.get((ex, grp), set())
            r = corr_ci(d[t].to_numpy(float), d.ref_set_score.to_numpy(float), W, ci) if varies else {}
            rows.append(dict(stratum=sname, target=t, varies=varies, videos=len(d), subjects=len(subs), **r))
        for c in COMPONENTS:
            col = f"mean_{c}"
            for t in TARGETS:
                varies = grp is None or t in ("score", "MQ") or t in active.get((ex, grp), set())
                if not varies:
                    continue
                x = SIGN.get(c, 1) * d[col].to_numpy(float)
                r = corr_ci(d[t].to_numpy(float), x, W, ci) if t in ("score", "MQ") else \
                    corr_ci(d[t].to_numpy(float), x, W[:1], ci)
                comp_rows.append(dict(stratum=sname, component=c, target=t, **r))
        if grp == "expert":
            for vw, dv in d.groupby("view"):
                dv = dv.reset_index(drop=True)
                s2, M2 = draws(dv.subject.to_numpy(), dv.group.to_numpy(), n_boot, seed)
                W2 = np.vstack([np.ones(len(dv)), item_weights(dv.subject.to_numpy(), s2, M2)])
                for t in ("score", "MQ"):
                    view_rows.append(dict(stratum=sname, view=vw, target=t, **corr_ci(
                        dv[t].to_numpy(float), dv.ref_set_score.to_numpy(float), W2, ci)))
    A = pd.DataFrame(rows)
    A.to_csv(out / "agreement.csv", index=False)
    C = pd.DataFrame(comp_rows)
    C.to_csv(out / "agreement_components.csv", index=False)
    VW = pd.DataFrame(view_rows)
    VW.to_csv(out / "agreement_by_view.csv", index=False)

    # between-group separation (context; pooled E01): AUC expert vs non-expert, all views and view-matched F + FL
    sep = []
    for name, sel in (("E01 all videos", V.exercise == "E01"),
                      ("E01 front view, FL (view-matched)", (V.exercise == "E01") & (V.view == "F") & (V.variation == "FL"))):
        d = V[sel]
        e, n = d[d.group == "expert"].ref_set_score.to_numpy(), d[d.group == "non-expert"].ref_set_score.to_numpy()
        auc = float(((e[:, None] > n[None, :]).mean() + 0.5 * (e[:, None] == n[None, :]).mean()))
        sep.append(dict(subset=name, expert_videos=len(e), non_expert_videos=len(n), expert_mean=float(e.mean()),
                        non_expert_mean=float(n.mean()), auc=auc))
    SEP = pd.DataFrame(sep)
    SEP.to_csv(out / "group_separation.csv", index=False)

    # ---------------- embedding distances: video level, standardised per fold
    E = pd.read_csv(out / "embedding_distance_reps.csv")
    E = E.merge(reps[["video_id", "rep", "rep_uid", "exercise", "subject", "group", "view", "group_id"]],
                on=["video_id", "rep"], how="left", validate="many_to_one")
    EV = E.groupby(["method", "fold", "video_id", "exercise", "subject", "group", "view", "group_id"], as_index=False) \
        .agg(d_cos=("d_cos", "mean"), d_l2=("d_l2", "mean"), is_test=("is_test", "first"))
    EV = standardise_per_fold(standardise_per_fold(EV, plans, "d_cos"), plans, "d_l2")
    EVt = EV[EV.is_test].merge(V[["video_id", "variation", *TARGETS, "ref_set_score"]], on="video_id")
    emb_rows, emb_reg = [], []
    methods = [m for m in METHODS if m in set(EVt.method)]
    for sname, ex, grp in STRATA:  # no pooled rows: the per-group standardisation removes the group offset
        b0 = EVt[EVt.method == methods[0]]
        base = b0[mask(b0, ex, grp)].sort_values("video_id")
        subs, M = draws(base.subject.to_numpy(), base.group.to_numpy(), n_boot, seed)
        W = np.vstack([np.ones(len(base)), item_weights(base.subject.to_numpy(), subs, M)])
        entries = [(m, "d_cos_z", -1) for m in methods] + [("module3", "ref_set_score", 1)] + \
                  [(m, "d_l2_z", -1) for m in methods]
        for m, col, sg in entries:
            dm = EVt[EVt.method == m]
            d = base if m == "module3" else dm[mask(dm, ex, grp)]
            d = d.sort_values("video_id")
            assert (d.video_id.to_numpy() == base.video_id.to_numpy()).all()
            for t in TARGETS:
                varies = grp is None or t in ("score", "MQ") or t in active.get((ex, grp), set())
                if not varies:
                    continue
                r = corr_ci(d[t].to_numpy(float), sg * d[col].to_numpy(float), W, ci)
                emb_rows.append(dict(stratum=sname, method=m, distance="l2" if col == "d_l2_z" else "cosine"
                                     if m != "module3" else "-", target=t, **r))
    EMB = pd.DataFrame(emb_rows)
    EMB.to_csv(out / "embedding_tracking.csv", index=False)

    # out-of-fold single-feature lines: total / MQ on the distance (per method) and on the Module III score
    for t in ("score", "MQ"):
        feats = {"module3": V.set_index("video_id").ref_set_score}
        for m in methods:
            feats[m] = EV[EV.method == m].set_index(["video_id", "fold"]).d_cos  # raw distance, refitted per fold
        for sname, ex, grp in STRATA:
            d0 = V[mask(V, ex, grp)].reset_index(drop=True)
            subs, M = draws(d0.subject.to_numpy(), d0.group.to_numpy(), n_boot, seed)
            W = np.vstack([np.ones(len(d0)), item_weights(d0.subject.to_numpy(), subs, M)])
            y = d0[t].to_numpy(float)
            res = {}
            for m in ["module3", *methods]:
                preds = np.full(len(d0), np.nan)
                basep = np.full(len(d0), np.nan)
                for p in plans:  # per fold: that fold's model distances for its training and test videos
                    if m == "module3":
                        x = d0.ref_set_score
                    else:
                        x = feats[m].xs(p["fold"], level="fold").reindex(d0.video_id).to_numpy()
                        x = pd.Series(x, index=d0.index)
                    dd = d0.assign(x=x)
                    pr, bs = oof_line(dd[dd.subject.isin(p["train"] + p["test"])], "x", t, [p])
                    preds[pr.dropna().index] = pr.dropna()
                    basep[bs.dropna().index] = bs.dropna()
                e2, b2 = (preds - y) ** 2, (basep - y) ** 2
                rm = np.sqrt((W * e2).sum(1) / W.sum(1))
                rb = np.sqrt((W * b2).sum(1) / W.sum(1))
                res[m] = (rm, rb)
                emb_reg.append(dict(target=t, stratum=sname, method=m, rmse=rm[0], rmse_lo=interval(rm[1:], ci)[0],
                                    rmse_hi=interval(rm[1:], ci)[1], group_only=rb[0], delta=rm[0] - rb[0],
                                    delta_lo=interval(rm[1:] - rb[1:], ci)[0], delta_hi=interval(rm[1:] - rb[1:], ci)[1]))
    REG = pd.DataFrame(emb_reg)
    REG.to_csv(out / "single_feature_regression.csv", index=False)

    # ---------------- (5b) view stability over synced expert performances
    sync = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    synced = set(sync.index[sync.cls == "synced"])
    # standardise rep distances with the training subjects' reps of the same method, fold and exercise
    Eall = standardise_per_fold(E, plans, "d_cos")
    Er = Eall[Eall.is_test & Eall.group_id.isin(synced) & (Eall.group == "expert")].copy()
    sc = cfg["assess"]["score"]  # Module III score without the speed group (synced views share rep timing, P6b)
    sp = ("angle", "rom", "symmetry")
    P_ = np.stack([np.minimum(1.0, R[f"X_{g}"].to_numpy(float) / sc["d_zero"]) * sc["weights"][g] for g in sp], 1)
    Wt = np.stack([np.isfinite(R[f"X_{g}"].to_numpy(float)) * sc["weights"][g] for g in sp], 1)
    R["ref_score_spatial"] = 100 * (1 - np.nansum(P_, 1) / Wt.sum(1))
    Rr = R.merge(reps[["video_id", "rep", "rep_uid", "group_id"]], on=["video_id", "rep"], how="left", validate="one_to_one")
    Rr = Rr[Rr.group_id.isin(synced) & (Rr.group == "expert")]
    vs_rows, icc_rep = [], {}
    for sname, ex, _ in STRATA[:2]:
        for level in ("rep", "video"):
            series = []
            for m in methods:
                d = Er[(Er.method == m) & (Er.exercise == ex)]
                if level == "video":
                    d = d.groupby(["video_id", "group_id", "subject", "group", "view"], as_index=False).d_cos_z.mean()
                    d["perf"] = d.group_id
                else:
                    d = d.assign(perf=d.rep_uid)
                series.append((m, d, "d_cos_z"))
            d = Rr[Rr.exercise == ex]
            if level == "video":
                for col, nm in (("ref_score", "module3"), ("ref_score_spatial", "module3_spatial"), ("rom", "module3_rom")):
                    dv = d.groupby(["video_id", "group_id", "subject", "group", "view"], as_index=False)[col].mean()
                    series.append((nm, dv.assign(perf=dv.group_id), col))
            else:
                series += [("module3", d.assign(perf=d.rep_uid), "ref_score"),
                           ("module3_spatial", d.assign(perf=d.rep_uid), "ref_score_spatial"),
                           ("module3_rom", d.assign(perf=d.rep_uid), "rom")]
            for m, d, col in series:
                r, icc_b, _ = view_stability(d, col, n_boot, seed, ci)
                icc_rep[(sname, level, m)] = icc_b
                vs_rows.append(dict(stratum=sname, level=level, method=m, **r))
    VS = pd.DataFrame(vs_rows)
    # paired ICC differences vs B1 (same subject draws: same strata and performances)
    VS["d_icc_vs_b1"] = np.nan
    VS["d_icc_vs_b1_lo"] = np.nan
    VS["d_icc_vs_b1_hi"] = np.nan
    for i, r in VS.iterrows():
        b = icc_rep.get((r.stratum, r.level, "b1"))
        mine = icc_rep[(r.stratum, r.level, r.method)]
        if b is not None and len(b) == len(mine) and r.method in methods and r.method != "b1":
            same = VS[(VS.stratum == r.stratum) & (VS.level == r.level) & (VS.method == "b1")].perfs.iloc[0] == r.perfs
            if same:
                dd = mine - b
                VS.loc[i, ["d_icc_vs_b1", "d_icc_vs_b1_lo", "d_icc_vs_b1_hi"]] = [r.icc - VS[(VS.stratum == r.stratum) & (
                    VS.level == r.level) & (VS.method == "b1")].icc.iloc[0], *interval(dd, ci)]
    VS.to_csv(out / "view_stability.csv", index=False)

    # ---------------- errors
    errs = sorted(c for c in R.columns if c.startswith("err_"))
    R2 = R.merge(V[["video_id", "score"]].rename(columns={"score": "vscore"}), on="video_id")
    er_rows = []
    for sname, ex, grp in STRATA:
        d = R2[mask(R2, ex, grp)]
        top = d[d.vscore >= cfg["templates"]["min_score"]]
        dv = V[mask(V, ex, grp)]
        for e in errs:
            if d[e].sum() == 0 and top[e].sum() == 0:
                continue
            col = f"mean_{e}"
            ok = dv[col].notna()
            er_rows.append(dict(stratum=sname, error=e[4:], rate=d[e].mean(), rate_videos_ge95=top[e].mean() if len(top) else np.nan,
                                rho_score=spearmanr(-dv.loc[ok, col], dv.loc[ok, "score"])[0] if dv.loc[ok, col].std() > 0 else np.nan,
                                rho_MQ=spearmanr(-dv.loc[ok, col], dv.loc[ok, "MQ"])[0] if dv.loc[ok, col].std() > 0 else np.nan))
        er_rows.append(dict(stratum=sname, error="too_few_reps (set)", rate=dv.too_few_reps.mean(), rate_videos_ge95=np.nan,
                            rho_score=spearmanr(-dv.too_few_reps, dv.score)[0], rho_MQ=spearmanr(-dv.too_few_reps, dv.MQ)[0]))
    ER = pd.DataFrame(er_rows)
    ER.to_csv(out / "errors_summary.csv", index=False)

    assert np.allclose(rescore(R, cfg["assess"]["score"]), R.ref_score, equal_nan=True)  # M11 reproduced
    bdir = out / "posture_pt"
    BA = None
    if (bdir / "module3_videos.csv").exists():
        BA = before_after(cfg, V, R, pd.read_csv(bdir / "module3_videos.csv"), pd.read_csv(bdir / "module3_reps.csv"),
                          reps, synced, active, n_boot, seed, ci)
        for k, t in zip(("agreement", "error_rates", "view_stability"), BA[:3]):
            t.to_csv(out / f"posture_before_after_{k}.csv", index=False)
    SPX = speed_exploratory(cfg, V, R, active, n_boot, seed, ci)
    SPX.to_csv(out / "speed_exploratory.csv", index=False)
    rn = json.loads((out / "recording_noise.json").read_text(encoding="utf-8"))
    figures(cfg, out, R, V)
    write_summary(out, cfg, A, C, VW, SEP, EMB, REG, VS, ER, rn, V, R, methods, n_boot, act, BA, SPX)
    print(f"report written to {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def figures(cfg, out, R, V):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lib_dir = resolve(cfg["templates"]["library_dir"])
    for ex in cfg["data"]["include_exercises"]:
        ref, lib, meta = load_library(lib_dir / ex)
        chans = list(ref["curves"])
        fig, axes = plt.subplots(len(chans), 3, figsize=(12, 2.3 * len(chans)), sharex=True, squeeze=False)
        t = np.arange(ref["n"]) / cfg["assess"]["fps"]
        k = cfg["assess"]["band_k"]
        for i, c in enumerate(chans):
            for j, v in enumerate(cfg["data"]["views"]):
                ax = axes[i, j]
                tgt = ref["curves"][c] + lib[v]["bias"][c]
                ax.fill_between(t, tgt - k * lib[v]["sigma"][c], tgt + k * lib[v]["sigma"][c], color="#9ecae1", alpha=0.6,
                                lw=0, label=f"band ±{k:g} SD (experts ≥ {meta['min_score']})")
                ax.plot(t, ref["curves"][c], color="#555", lw=1, ls="--", label="PT demo (front)")
                ax.plot(t, tgt, color="#08519c", lw=1.6, label=f"target in view {v}")
                ax.plot(t, lib[v]["mean"][c], color="#e6550d", lw=1, label="expert mean")
                if i == 0:
                    ax.set_title(f"view {v} ({lib[v]['n_reps']} expert reps)", fontsize=9)
                if j == 0:
                    ax.set_ylabel(c.replace("_", " ") + " (deg)", fontsize=8)
                ax.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=6, loc="best")
        for ax in axes[-1]:
            ax.set_xlabel("template time (s)", fontsize=8)
        fig.suptitle(f"{ex}: reference template ({meta['file']}) and per-view tolerance bands (all subjects)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "figs" / f"template_{ex}.png", dpi=110)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (sname, ex, grp) in zip(axes, STRATA):
        d = V[mask(V, ex, grp)]
        for vw, col in zip("FLR", ("#3182bd", "#e6550d", "#31a354")):
            dv = d[d.view == vw]
            ax.scatter(dv.ref_set_score, dv.score, s=8, alpha=0.6, color=col, label=f"view {vw}")
        ax.set_xlabel("Module III reference score (set)", fontsize=8)
        ax.set_ylabel("physiotherapist total score", fontsize=8)
        ax.set_title(sname, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figs" / "agreement_scatter.png", dpi=110)
    plt.close(fig)


def write_summary(out, cfg, A, C, VW, SEP, EMB, REG, VS, ER, rn, V, R, methods, n_boot, act, BA=None, SPX=None):
    L = []
    w = L.append
    w("# Phase 4: Module III (reference-based assessment) and the embedding-to-reference distance\n")
    w(f"E01 and E02, GroupKFold-5 (the Phase 3 folds), video level, {int(V.shape[0])} videos / {int(R.shape[0])} reps; "
      f"95% CIs: subject bootstrap ({n_boot} draws). Within groups first (E9). Libraries (view bias, bands, ROM and speed "
      "tolerances) are fitted on each fold's training subjects only; the PT demo is the reference curve. Posture checks "
      f"centred on: **{cfg['assess'].get('posture_centre', 'pt_demo')}** (M14; before / after at the end).\n")
    w("## Reference score vs physiotherapist scores (within groups)\n")
    w("Spearman ρ [95% CI] of the set score (mean rep score) with each label; '·' = the item does not vary in the "
      "group (T14).\n")
    hdr = ["target"] + [s for s, _, _ in STRATA] + ["E01 pooled (secondary)"]
    w("| " + " | ".join(hdr) + " |")
    w("|" + "---|" * len(hdr))
    for t in TARGETS:
        cells = []
        for s in [x for x, _, _ in STRATA] + ["E01 pooled"]:
            r = A[(A.stratum == s) & (A.target == t)].iloc[0]
            cells.append(fci(r.spearman, r.spearman_lo, r.spearman_hi) if r.varies and np.isfinite(r.get("spearman", np.nan)) else "·")
        w(f"| {t} | " + " | ".join(cells) + " |")
    w("")
    w("Pearson r for total / MQ: " + "; ".join(
        f"{s}: {fmt(A[(A.stratum == s) & (A.target == 'score')].pearson.iloc[0])} / "
        f"{fmt(A[(A.stratum == s) & (A.target == 'MQ')].pearson.iloc[0])}" for s, _, _ in STRATA) + "\n")
    w("Score level per group (mean ± SD of the set score): " + "; ".join(
        f"{s}: {V[mask(V, e, g)].ref_set_score.mean():.1f} ± {V[mask(V, e, g)].ref_set_score.std():.1f}"
        for s, e, g in STRATA) + "\n")
    w("Between-group separation (context, not a within-group result): " + "; ".join(
        f"{r.subset}: AUC {r.auc:.2f} (expert {r.expert_mean:.1f}, non-expert {r.non_expert_mean:.1f}; "
        f"{r.expert_videos} / {r.non_expert_videos} videos)" for r in SEP.itertuples()) + "\n")
    w("### By view (expert groups): ρ with total / MQ\n")
    w("| stratum | F | L | R |")
    w("|---|---|---|---|")
    for s in ("E01 expert", "E02 expert"):
        cells = []
        for vw in "FLR":
            a_ = VW[(VW.stratum == s) & (VW.view == vw) & (VW.target == "score")].iloc[0]
            b_ = VW[(VW.stratum == s) & (VW.view == vw) & (VW.target == "MQ")].iloc[0]
            cells.append(f"{fmt(a_.spearman)} [{fmt(a_.spearman_lo)}, {fmt(a_.spearman_hi)}] / "
                         f"{fmt(b_.spearman)} [{fmt(b_.spearman_lo)}, {fmt(b_.spearman_hi)}] (n {a_.n})")
        w(f"| {s} | " + " | ".join(cells) + " |")
    w("")
    w("### Components (Spearman with total / MQ; distances sign-flipped so + = agreement)\n")
    w("| component | " + " | ".join(s for s, _, _ in STRATA) + " |")
    w("|---|" + "---|" * len(STRATA))
    for c, lab in COMPONENTS.items():
        cells = []
        for s, _, _ in STRATA:
            a_ = C[(C.stratum == s) & (C.component == c) & (C.target == "score")]
            b_ = C[(C.stratum == s) & (C.component == c) & (C.target == "MQ")]
            if len(a_):
                a_, b_ = a_.iloc[0], b_.iloc[0]
                cells.append(f"{fmt(a_.spearman)} [{fmt(a_.spearman_lo)}, {fmt(a_.spearman_hi)}] / "
                             f"{fmt(b_.spearman)} [{fmt(b_.spearman_lo)}, {fmt(b_.spearman_hi)}]")
            else:
                cells.append("–")
        w(f"| {lab} | " + " | ".join(cells) + " |")
    w("\nComponent × item ρ (point estimates) are in `agreement_components.csv`.\n")
    w("### Out-of-fold single-feature line vs the group-only baseline (RMSE, total score)\n")
    w("| feature | " + " | ".join(s for s, _, _ in STRATA) + " |")
    w("|---|" + "---|" * len(STRATA))
    for m in ["module3", *methods]:
        cells = []
        for s, _, _ in STRATA:
            r = REG[(REG.target == "score") & (REG.stratum == s) & (REG.method == m)].iloc[0]
            cells.append(f"{fmt(r.rmse)} (Δ {fmt(r.delta)} [{fmt(r.delta_lo)}, {fmt(r.delta_hi)}])")
        w(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    w("| group-only | " + " | ".join(fmt(REG[(REG.target == 'score') & (REG.stratum == s) & (REG.method == 'module3')]
                                          .group_only.iloc[0]) for s, _, _ in STRATA) + " |")
    w("\nSame for MQ in `single_feature_regression.csv`.\n")
    w("## Errors (rule-based, M12)\n")
    w("Share of reps with each error; in the reps of expert videos scoring ≥ 95 (out of fold; a specificity "
      "reference); Spearman of the video's error rate (sign-flipped) with total / MQ.\n")
    w("| stratum | error | rate | rate in ≥ 95 videos | ρ total / MQ |")
    w("|---|---|---|---|---|")
    for r in ER.itertuples():
        w(f"| {r.stratum} | {r.error} | {r.rate:.2f} | {fmt(r.rate_videos_ge95)} | {fmt(r.rho_score)} / {fmt(r.rho_MQ)} |")
    w("")
    w("## Embedding distance to the PT reference (\"vector difference\")\n")
    w("### (a) Tracking the physiotherapist scores (closeness = −distance; centred cosine, standardised per fold)\n")
    w("| method | " + " | ".join(f"{s}: total / MQ" for s, _, _ in STRATA) + " |")
    w("|---|" + "---|" * len(STRATA))
    for m in [*methods, "module3"]:
        cells = []
        for s, _, _ in STRATA:
            e_ = EMB[(EMB.stratum == s) & (EMB.method == m) & (EMB.distance != "l2")]
            a_, b_ = e_[e_.target == "score"].iloc[0], e_[e_.target == "MQ"].iloc[0]
            cells.append(f"{fmt(a_.spearman)} [{fmt(a_.spearman_lo)}, {fmt(a_.spearman_hi)}] / "
                         f"{fmt(b_.spearman)} [{fmt(b_.spearman_lo)}, {fmt(b_.spearman_hi)}]")
        w(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    w("\nItems and the L2 distance in `embedding_tracking.csv`.\n")
    w("### (b) View stability: the same performance filmed from F / L / R (synced expert groups, test subjects)\n")
    w("ICC(1) [95% CI] of the standardised distance across views (higher = more stable); ΔICC vs B1 [CI]; R_view = mean "
      "|change between views| / mean |difference between performances| (lower = more stable).\n")
    for level in ("rep", "video"):
        w(f"**{level} level** (performance = {'one rep seen from 2-3 views (shared rep_uid)' if level == 'rep' else 'one set (group) seen from 2-3 views'})\n")
        w("| method | " + " | ".join(f"{s}: ICC | ΔICC vs B1 | R_view" for s, _, _ in STRATA[:2]) + " |")
        w("|---|" + "---|---|---|" * 2)
        for m in [*methods, "module3", "module3_spatial", "module3_rom"]:
            cells = []
            for s, _, _ in STRATA[:2]:
                r = VS[(VS.stratum == s) & (VS.level == level) & (VS.method == m)].iloc[0]
                cells += [fci(r.icc, r.icc_lo, r.icc_hi), fci(r.d_icc_vs_b1, r.d_icc_vs_b1_lo, r.d_icc_vs_b1_hi)
                          if np.isfinite(r.d_icc_vs_b1) else "–", fci(r.r_view, r.r_view_lo, r.r_view_hi)]
            w(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
        n = VS[(VS.level == level) & (VS.method == "b1")]
        w("\n" + "; ".join(f"{r.stratum}: {r.perfs} performances, {r.values} values, {r.subjects} subjects"
                           for r in n.itertuples()) + "\n")
    w("Between-view Spearman of the distance (F–L / F–R / L–R) in `view_stability.csv`.\n")
    w("## Recording-noise test (item 6)\n")
    for ex, r in rn.items():
        w(f"- **{ex}**: {r['groups']} synced expert performances ({r['views']} views); PF4 differs between views in "
          f"{r['groups_PF4_differs']}, PF3 in {r['groups_PF3_differs']}, the number of fully recorded reps in "
          f"{r['groups_count_differs']}. Over the {r['pairs_PF4_differs']} view pairs with different PF4, the view with "
          f"the higher PF4 holds more / the same / fewer fully recorded reps in {r['pairs_contained_higher']:.0%} / "
          f"{r['pairs_contained_equal']:.0%} / {r['pairs_contained_lower']:.0%} (permutation null for 'more': "
          f"{r['perm_contained_null']:.0%}, p = {r['perm_contained_p']:.3f}). Within-performance Spearman of PF4 with "
          f"the contained count {r['spearman_PF4_contained_within']:.2f}; the count explains "
          f"{r['r2_score_on_contained_within']:.0%} of the between-view variance of the total score. PF3 = 1 in "
          f"{r['PF3_1_given_count_ge10']:.0%} of views with ≥ 10 recorded reps vs {r['PF3_1_given_count_lt10']:.0%} "
          f"with fewer.")
    w("")
    if BA is not None:
        agree, rates, stab, _ = BA
        w("## Posture checks centred on the experts: before / after (M14, post-hoc change)\n")
        w("Before = every channel centred on the PT demo (+ view bias); after = the posture channels "
          f"({', '.join(sorted(set(sum((cfg['exercises'][e]['assess'].get('posture_channels', []) for e in cfg['data']['include_exercises']), []))))}) "
          "centred on the ≥ 95 experts' mean curve per view (training subjects of each fold). The main movement curve, "
          "ROM, speed and the bands are unchanged.\n")
        w("**Agreement** (Spearman of the set score; Δ = after − before, paired CI):\n")
        w("| stratum | label | before | after | Δ |")
        w("|---|---|---|---|---|")
        for s_, _, _ in STRATA:
            for t in ("score", "MQ", "PF2", "CF1", "CF2", "CF3"):
                d = agree[(agree.stratum == s_) & (agree.target == t)]
                if len(d) < 2:
                    continue
                b_, a_ = d[d.column == "score_before"].iloc[0], d[d.column == "score_after"].iloc[0]
                w(f"| {s_} | {t} | {fci(b_.rho, b_.lo, b_.hi)} | {fci(a_.rho, a_.lo, a_.hi)} | "
                  f"{fci(a_.delta, a_.delta_lo, a_.delta_hi)} |")
        w("\n**Rule rates** (share of reps; overall / in videos scoring ≥ 95 / in videos < 80):\n")
        w("| stratum | error | before | after |")
        w("|---|---|---|---|")
        for s_, _, _ in STRATA:
            for e in sorted(set(rates[rates.stratum == s_].error)):
                cells = []
                for nm in ("before", "after"):
                    r = rates[(rates.variant == nm) & (rates.stratum == s_) & (rates.error == e)]
                    cells.append("–" if not len(r) else f"{r.rate.iloc[0]:.2f} / {fmt(r.rate_ge95.iloc[0])} / "
                                 f"{fmt(r.rate_lt80.iloc[0])}")
                if rates[(rates.stratum == s_) & (rates.error == e)].rate.fillna(0).max() == 0:
                    continue  # a rule of the other exercise
                w(f"| {s_} | {e} | {cells[0]} | {cells[1]} |")
        w("\n**View stability of the score** (rep level, synced expert performances): " + "; ".join(
            f"{r.stratum} {r.variant}: ICC {fci(r.icc, r.icc_lo, r.icc_hi)}" for r in stab.itertuples()) + "\n")
    if SPX is not None:
        w("## Speed centred on the demo (M15, post-hoc exploratory; not the primary score)\n")
        w("The primary speed component stays centred on the experts (M5a). Here the same speed distance is centred on "
          "the demo instead (|log2(demo / rep duration)| / expert SD), next to the angular-velocity deviation from the "
          "demo, and as a score variant with that speed group. Post-hoc: chosen after seeing the Phase 4 components, so "
          "these intervals are not confirmatory.\n")
        w("| stratum | label | speed, expert-centred (primary) | speed, demo-centred | angular-velocity dev. from demo | "
          "score (primary) | score with demo-centred speed (Δ) |")
        w("|---|---|---|---|---|---|---|")
        for s_, _, _ in STRATA:
            for t in ("score", "MQ", "PF2"):
                d = SPX[(SPX.stratum == s_) & (SPX.target == t)]
                if not len(d):
                    continue
                g = {r.column: r for r in d.itertuples()}
                cell = lambda r: fci(r.rho, r.lo, r.hi)  # noqa: E731
                sd_ = g["score_speed_demo"]
                w(f"| {s_} | {t} | {cell(g['X_speed'])} | {cell(g['X_speed_demo'])} | {cell(g['omega_dev'])} | "
                  f"{cell(g['ref_set_score'])} | {cell(sd_)} (Δ {fci(sd_.delta, sd_.delta_lo, sd_.delta_hi)}) |")
        w("")
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
