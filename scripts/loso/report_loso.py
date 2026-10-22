"""LOSO per exercise, final scope E01-E04 (E30): the report, in the format of the GroupKFold-5 step-1 tables, and the
LOSO vs GroupKFold-5 comparison.

Reads runs/full/<tag>/<EX>/<method>/fold<k>/ (cvdc.train.run_loso: k = the subject's index, folds.csv),
<results>/<tag>/embedding_distance_reps.csv (scripts/phase3_full/embedding_distance_perex.py --loso) and, for the
comparison, the GroupKFold-5 step-1 outputs (<results>/final_E01_E04/step1/: video_predictions.csv, vci.csv,
distance_view_stability.csv; runs/full/perex_gkf5/ for the same-subject VCI). No GroupKFold-5 file is rewritten.

Same metrics, strata, baselines and bootstrap as scripts/phase3_full/report_perex.py (E9, E10, E16, E17, E20): video
level, within every exercise x group first, pooled over E01-E04 within each group second; total score and the
movement-quality score MQ (D13) from the video head; predict-the-mean, group-only, duration-only, rep-count-only and
set-features fitted on each LOSO fold's training videos; 95% CIs from the subject bootstrap (2,000 draws, paired).
Added for LOSO:
  paired differences also vs B2-KTPFormer, duration-only and predict-the-mean
  subject-level paired tests: per subject, the RMSE over its videos; Wilcoxon signed-rank (two-sided) on the
      per-subject differences, Holm over the six within-exercise strata of each comparison and target; effect sizes:
      matched-pairs rank-biserial r and Cohen's d_z (negative = the first method has the lower error)
  VCI: a LOSO fold holds one subject, so 'other groups' are that subject's other variations (same exercise); the
      GroupKFold-5 VCI restricted the same way (other-group pairs within a subject) is computed from its embeddings
  LOSO vs GroupKFold-5: RMSE / rho of every method and baseline under both splits on the same videos and bootstrap
      draws, with paired CIs of LOSO - GroupKFold-5; the GroupKFold-5 numbers are recomputed from its saved
      predictions with this code and checked against its published table
  the encoder check (C7) of every CVDC run, read back from done.json and the encoder's run_info.json

usage: python scripts/loso/report_loso.py [--config configs/per_exercise_loso.yaml] [--tag perex_loso] [--boot 2000]
Writes CSV tables and summary.md to <eval.results_dir>/<tag>/ and (not for smoke runs) the committed copy
reports/final_E01_E04_loso_tables.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "phase3_full"))
sys.path.insert(0, str(REPO / "scripts" / "phase4"))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.data.labels import movement_quality  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_baselines, video_table  # noqa: E402
from cvdc.eval.metrics import (add_sums, interval, item_weights, regression, subject_draws, vci_from_sums,  # noqa: E402
                               vci_sums)
from cvdc.train.trainer import loso_plans  # noqa: E402
import report_perex as RP  # noqa: E402
from report_full import (BASELINES, REP_TARGETS, VIDEO_TARGETS, VS_DEFAULT, draws_for, noise_floors,  # noqa: E402
                         regression_tables, stratum_mask)

EXS = ["E01", "E02", "E03", "E04"]
METHODS = ["b1", "b2_ktpformer", "proposed_j1", "proposed_lp"]
SHORT, LABEL = RP.SHORT, RP.LABEL
POOLED = [("experts, E01–E04", None, "expert"), ("non-experts, E01–E04", None, "non-expert")]
VS = VS_DEFAULT + [("b2_ktpformer", None), ("base_duration", "baseline"), ("base_mean", "baseline")]
TESTS = [("proposed_j1", "b1"), ("proposed_j1", "b2_ktpformer"), ("proposed_lp", "b1"), ("proposed_lp", "b2_ktpformer"),
         ("b2_ktpformer", "b1")] + [(m, "base_group") for m in METHODS]


def f2(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace("-", "−")


def fci(x, lo, hi, d=2, bold=False):
    s = f"{f2(x, d)} [{f2(lo, d)}, {f2(hi, d)}]"
    return f"**{s}**" if bold and np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0) else s


def fp(p):
    return "–" if p is None or not np.isfinite(p) else (f"{p:.3f}" if p >= 0.001 else "<0.001")


# ---------------------------------------------------------------- runs and plans
def loso_runs(cfg, root, reps):
    pv, pr, done = RP.load_runs(root, EXS, METHODS)
    plans_by_ex, plans = {}, {}
    for ex in EXS:
        pl = [{**p, "exercise": ex} for p in loso_plans(cfg, reps[reps.exercise == ex])]
        pl = [p for p in pl if any((ex, m, p["fold"]) in done for m in METHODS)]
        plans_by_ex[ex] = pl
        for p in pl:
            plans[RP.fkey(ex, p["fold"])] = p
    checks = []
    for (ex, m, f), d in sorted(done.items()):
        p = plans[RP.fkey(ex, f)]
        for k in ("train", "inner_train", "inner_val", "test"):  # every run used exactly the recomputed LOSO fold
            assert sorted(d["subjects"][k]) == sorted(p[k]), (ex, m, f, k)
        if d.get("cvdc"):
            c = d["cvdc_stage"]
            enc = json.loads((Path(c["reused_from"]) / "run_info.json").read_text(encoding="utf-8"))
            seen = set(enc["inner_train"]) | set(enc["inner_val"])
            checks.append(dict(exercise=ex, method=m, fold=f, subject=p["subject"],
                               encoder_fold=c["shortcut_check"]["encoder_fold"], encoder=c["reused_from"],
                               held_out_seen_by_encoder=bool(set(p["test"]) & seen),
                               held_out_in_encoder_test=set(p["test"]) <= set(enc["test"]),
                               encoder_subjects_in_loso_train=seen <= set(p["train"])))
    complete = [ex for ex in EXS if plans_by_ex[ex] and
                all((ex, m, p["fold"]) in done for m in METHODS for p in plans_by_ex[ex])]
    return pv, pr, done, plans_by_ex, plans, pd.DataFrame(checks), complete


def assemble(pv, pr, v, plans_by_ex, plans, active, complete):
    pv, pr = pv[pv.exercise.isin(complete)], pr[pr.exercise.isin(complete)]
    pv = RP.add_mq(pv, v, plans, active)
    vp = pv[["method", "fold", "video_id", *[f"{t}_pred" for t in VIDEO_TARGETS]]].copy()
    vp["source"] = "video head"
    rp = pr.groupby(["method", "fold", "video_id"])[[f"{t}_pred" for t in REP_TARGETS]].mean().reset_index()
    rp["source"] = "rep head"
    bases = []
    for ex in complete:
        b = video_baselines(v[v.exercise == ex], plans_by_ex[ex], VIDEO_TARGETS)
        b["fold"] = [RP.fkey(ex, f) for f in b.fold]
        bases.append(b[b.method.isin(BASELINES)])
    base = pd.concat(bases, ignore_index=True)
    base["source"] = "baseline"
    P = pd.concat([vp, rp, base], ignore_index=True).merge(
        v[["video_id", "subject", "group", "exercise", "view", "variation", *VIDEO_TARGETS]], on="video_id",
        how="left", validate="many_to_one")
    test_subj = {(ex, s) for ex in complete for p in plans_by_ex[ex] for s in p["test"]}
    tv = v[[(e, s) in test_subj for e, s in zip(v.exercise, v.subject)]]
    n = P.groupby(["method", "source"]).video_id.nunique()
    assert (n == len(tv)).all(), f"every method must predict every test video once: {n.to_dict()}"
    return P, tv


# ---------------------------------------------------------------- subject-level paired tests
def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(order) - r) * p[i])
        out[i] = min(1.0, run)
    return out


def subject_rmse(d, m, t):
    src = "baseline" if m.startswith("base_") else "video head"
    dm = d[(d.method == m) & (d.source == src)]
    return np.sqrt(((dm[f"{t}_pred"] - dm[t]) ** 2).groupby(dm.subject).mean())


def subject_tests(P, strata, pairs, targets=("score", "MQ")):
    rows = []
    for sname, ex, grp in strata:
        d = P[stratum_mask(P, ex, grp)]
        for t in targets:
            for a, b in pairs:
                ra, rb = subject_rmse(d, a, t).align(subject_rmse(d, b, t), join="inner")
                x = (ra - rb).to_numpy()
                nz = x[x != 0]
                p = r_rb = np.nan
                if len(nz):
                    p = float(stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)
                    rk = stats.rankdata(np.abs(nz))
                    r_rb = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())
                sd = x.std(ddof=1) if len(x) > 1 else np.nan
                rows.append(dict(stratum=sname, target=t, method=a, vs=b, subjects=len(x), mean_delta=x.mean(),
                                 median_delta=float(np.median(x)), lower=int((x < 0).sum()), higher=int((x > 0).sum()),
                                 wilcoxon_p=p, rank_biserial=r_rb, cohen_dz=x.mean() / sd if sd and sd > 0 else np.nan))
    T = pd.DataFrame(rows)
    T["p_holm"] = np.nan
    within = T.stratum.str.match(r"E\d\d ")
    for _, g in T[within].groupby(["method", "vs", "target"]):
        T.loc[g.index, "p_holm"] = holm(g.wilcoxon_p.to_numpy())
    return T


# ---------------------------------------------------------------- VCI (with an optional same-subject restriction)
def vci_tables(root, done, cfg, n_boot, seed, ci, same_subject):
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
    sums = {}
    for (ex, m, f) in sorted(done):
        if ex not in EXS:
            continue
        for stage in ("init", "pretrained", "final"):
            fp_ = root / ex / m / f"fold{f}" / f"embeddings_{stage}.npz"
            if not fp_.exists():
                continue
            with np.load(fp_) as z:
                it, emb, centre = z["item"], z["z"], z["z_train_mean"]
            mm = members.iloc[it]
            sp = np.array([spos[x] for x in mm.subject.to_numpy()])
            strata = [(f"{ex} {g}", ex, g) for g in sorted(mm.group.unique())] + \
                [(n, None, g) for n, _, g in POOLED if g in set(mm.group)]
            for sname, sex, grp in strata:
                sel = stratum_mask(mm, sex, grp)
                ok = (sp[sel][:, None] == sp[sel][None, :]) if same_subject else None
                s = vci_sums(emb[sel], mm.set_id.to_numpy()[sel], mm.view.to_numpy()[sel], gcode[it][sel], sp[sel],
                             len(subjects), centre, ok)
                k = (m, stage, sname)
                sums[k] = add_sums(sums.get(k), s)
    rows = []
    for (m, stage, sname), s in sums.items():
        present = np.flatnonzero((s["same_cnt"] > 0) | (s["other_cnt"].sum(1) > 0))
        Mf = np.zeros((n_boot, len(subjects)))
        Mf[:, present] = subject_draws(present, [sgroup[subjects[i]] for i in present], n_boot, seed)
        pt, b = vci_from_sums(s), vci_from_sums(s, Mf)
        lo, hi = interval(b["vci"], ci)
        rows.append(dict(method=m, stage=stage, stratum=sname, vci=pt["vci"], vci_lo=lo, vci_hi=hi, same=pt["same"],
                         other=pt["other"], same_pairs=int(s["same_cnt"].sum() // 2),
                         other_pairs=int(s["other_cnt"].sum() // 2)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- LOSO vs GroupKFold-5
def split_comparison(PL, PG, strata, n_boot, seed, ci):
    entries = [(m, "video head") for m in METHODS] + [(b, "baseline") for b in BASELINES]
    rows = []
    for sname, ex, grp in strata:
        dl, dg = PL[stratum_mask(PL, ex, grp)], PG[stratum_mask(PG, ex, grp)]
        ref = dl[(dl.method == "b1") & (dl.source == "video head")].sort_values("video_id")
        subs, M = draws_for(ref, n_boot, seed)
        W = np.vstack([np.ones(len(ref)), item_weights(ref.subject.to_numpy(), subs, M)])
        for m, src in entries:
            a = dl[(dl.method == m) & (dl.source == src)].sort_values("video_id")
            b = dg[(dg.method == m) & (dg.source == src)].sort_values("video_id")
            assert (a.video_id.to_numpy() == ref.video_id.to_numpy()).all() and \
                (b.video_id.to_numpy() == ref.video_id.to_numpy()).all(), (sname, m)
            for t in ("score", "MQ"):
                ra = regression(a[t].to_numpy(), a[f"{t}_pred"].to_numpy(), W)
                rb = regression(b[t].to_numpy(), b[f"{t}_pred"].to_numpy(), W)
                for k in ("rmse", "spearman"):
                    if k == "spearman" and (ex is None or m.startswith("base_")):
                        continue
                    dv = ra[k] - rb[k]
                    lo, hi = interval(dv[1:], ci)
                    rows.append(dict(stratum=sname, method=m, target=t, metric=k, loso=ra[k][0], gkf=rb[k][0],
                                     delta=dv[0], lo=lo, hi=hi, subjects=len(subs), videos=len(ref)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise_loso.yaml")
    ap.add_argument("--tag", default="perex_loso")
    ap.add_argument("--gkf-tag", default="perex_gkf5")
    ap.add_argument("--gkf-results", default="final_E01_E04/step1")
    ap.add_argument("--boot", type=int)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    cfg = load_config(REPO / a.config)
    ec, ia = cfg["eval"], cfg["heads"]["item_activity"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out = resolve(ec["results_dir"]) / a.tag
    gres = resolve(ec["results_dir"]) / a.gkf_results
    out.mkdir(parents=True, exist_ok=True)
    smoke = a.tag.endswith("_smoke")
    RP.POOLED_WITHIN = POOLED

    v = video_table(cfg)
    v["MQ"] = movement_quality(v)
    index = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    index["MQ"] = movement_quality(index)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    act = item_activity(v, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}

    pv, pr, done, plans_by_ex, plans, checks, complete = loso_runs(cfg, root, reps)
    if not complete:
        raise SystemExit(f"no exercise has all its LOSO runs in {root}")
    checks.to_csv(out / "encoder_checks.csv", index=False)
    P, tv = assemble(pv, pr, v, plans_by_ex, plans, active, complete)
    P.to_csv(out / "video_predictions.csv", index=False)
    floors = noise_floors(index, ["score", "MQ"])
    within = [(f"{e} {g}", e, g) for e, g in sorted({(e, g) for e, g in zip(tv.exercise, tv.group)})]
    strata = within + [s for s in POOLED if s[2] in set(tv.group)]
    entries = [(m, "video head") for m in METHODS] + [(m, "rep head") for m in METHODS] + \
              [(b, "baseline") for b in BASELINES]
    S, D = regression_tables(P, strata, entries, active, floors, n_boot, seed, ci, VS)
    S.to_csv(out / "score_metrics.csv", index=False)
    D.to_csv(out / "paired_deltas.csv", index=False)
    T = subject_tests(P, strata, TESTS)
    T.to_csv(out / "subject_tests.csv", index=False)
    print(f"score tables in {time.time() - t0:.0f} s", flush=True)

    # GroupKFold-5 reference, recomputed from its saved predictions on the same videos (nothing rewritten there)
    PG = pd.read_csv(gres / "video_predictions.csv")
    PG = PG[PG.video_id.isin(set(tv.video_id))]
    SG, DG = regression_tables(PG, strata, entries, active, floors, n_boot, seed, ci, VS)
    pub = pd.read_csv(gres / "score_metrics.csv")
    if not smoke:  # the recomputation must reproduce the published GroupKFold-5 table
        m = SG.merge(pub, on=["stratum", "method", "source", "target"], suffixes=("", "_pub"))
        gap = np.nanmax(np.abs(m.rmse - m.rmse_pub)) if len(m) else np.nan
        assert len(m) and gap < 1e-6, f"GroupKFold-5 recomputation differs from the published table by {gap}"
    SG.to_csv(out / "gkf_reference_score_metrics.csv", index=False)
    DG.to_csv(out / "gkf_reference_paired_deltas.csv", index=False)
    TG = subject_tests(PG, strata, TESTS)
    TG.to_csv(out / "gkf_reference_subject_tests.csv", index=False)
    C = split_comparison(P, PG, strata, n_boot, seed, ci)
    C.to_csv(out / "loso_vs_gkf.csv", index=False)
    print(f"comparison in {time.time() - t0:.0f} s", flush=True)

    done_c = {k: d for k, d in done.items() if k[0] in complete}
    V = vci_tables(root, done_c, cfg, n_boot, seed, ci, same_subject=False)
    V.to_csv(out / "vci.csv", index=False)
    groot = resolve(cfg["train"]["runs_dir"]) / a.gkf_tag
    _, _, gdone = RP.load_runs(groot, complete, METHODS)
    VG = vci_tables(groot, gdone, cfg, n_boot, seed, ci, same_subject=True)
    VG.to_csv(out / "gkf_reference_vci_same_subject.csv", index=False)
    VGp = pd.read_csv(gres / "vci.csv")
    VGp = VGp[VGp.variant == "centred"]
    print(f"VCI in {time.time() - t0:.0f} s", flush=True)
    VS_ = RP.distance_stability(cfg, out, {ex: plans_by_ex[ex] for ex in complete}, n_boot, seed, ci)
    if VS_ is not None:
        VS_.to_csv(out / "distance_view_stability.csv", index=False)
    VSG = pd.read_csv(gres / "distance_view_stability.csv") if (gres / "distance_view_stability.csv").exists() else None

    rows = []
    for (ex, m, f), d in sorted(done_c.items()):
        s_ = d["supervised_stage"]
        rows.append(dict(exercise=ex, method=m, fold=f, best_epoch=s_["best_epoch"], stopped_epoch=s_["stopped_epoch"],
                         early_stopped=s_["early_stopped"], sup_s=s_["wall_s"], total_s=d["wall_s"],
                         peak_gpu_gb=d.get("peak_gpu_gb"), finished=d.get("finished")))
    TR = pd.DataFrame(rows)
    TR.to_csv(out / "stopping_runtime.csv", index=False)
    write_summary(out, S, D, T, SG, DG, TG, C, V, VG, VGp, VS_, VSG, TR, checks, tv, within, complete, n_boot, smoke,
                  floors)
    print(f"report written to {out} in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- summary
def write_summary(out, S, D, T, SG, DG, TG, C, V, VG, VGp, VS_, VSG, TR, checks, tv, within, complete, n_boot, smoke,
                  floors):
    L = []
    w = L.append
    wn = [s for s, _, _ in within]
    pw = [s for s, _, _ in POOLED if s in set(S.stratum)]
    nsub = tv.groupby(["exercise", "group"]).subject.nunique()

    def table(head, rows):
        w("| " + " | ".join(head) + " |")
        w("|" + "---|" * len(head))
        for r in rows:
            w("| " + " | ".join(str(x) for x in r) + " |")
        w("")

    def srow(SS, st, m, src, t):
        r = SS[(SS.stratum == st) & (SS.method == m) & (SS.source == src) & (SS.target == t)]
        return None if r.empty else r.iloc[0]

    def drow(DD, st, m, vs, t, k="rmse"):
        r = DD[(DD.stratum == st) & (DD.method == m) & (DD.vs == vs) & (DD.target == t) & (DD.metric == k) &
               (DD.source == "video head")]
        return None if r.empty else r.iloc[0]

    def dcell(DD, st, m, vs, t, k="rmse"):
        r = drow(DD, st, m, vs, t, k)
        return "–" if r is None else fci(r.delta, r.lo, r.hi, bold=True)

    w("# LOSO per exercise, final scope E01–E04" + (" (SMOKE RUN: plumbing check, not a result)" if smoke else ""))
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/loso/report_loso.py`. Exercises: {', '.join(complete)}. "
      "Leave-one-subject-out within each exercise (one model per exercise, T17): every subject with ≥ 1 rep is the test "
      "set once; early stopping on 20% of each group's training subjects (T8); every other setting as the GroupKFold-5 "
      "runs (video head T13, per-video weights T6, targets standardised on the inner training subjects T7, T8a / T8b, "
      "no exercise head, E03 P02 kept in score regression D8). Proposed (λ 1.0) and the frozen Proposed start from the "
      "CVDC encoder of the GroupKFold-5 fold that held the subject out (C7). Video level, within groups first (E9). "
      f"{len(tv)} test videos. 95% CIs: subject bootstrap ({n_boot} draws, paired across methods and splits). "
      "**Bold** = a paired CI that excludes 0.")
    w("")
    w("## 0. Folds and the encoder check (C7)")
    w("")
    nf = tv.groupby("exercise").subject.nunique()
    w("LOSO folds: " + ", ".join(f"{e} {nf[e]} ({', '.join(f'{g} {n}' for (e2, g), n in nsub.items() if e2 == e)})"
                                 for e in complete) + f" = {int(nf.sum())} folds × 4 methods = {len(TR)} runs.")
    if len(checks):
        bad = checks[checks.held_out_seen_by_encoder | ~checks.held_out_in_encoder_test |
                     ~checks.encoder_subjects_in_loso_train]
        w(f"Encoder check of the {len(checks)} CVDC runs (Proposed λ 1.0 and frozen, read back from each run's done.json "
          f"and the encoder's run_info.json): held-out subject among the encoder's inner training / validation "
          f"subjects in {int(checks.held_out_seen_by_encoder.sum())}; held-out subject in the encoder's test fold in "
          f"{int(checks.held_out_in_encoder_test.sum())}; every subject the encoder saw is a training subject of the "
          f"LOSO fold in {int(checks.encoder_subjects_in_loso_train.sum())}. "
          + ("**No leakage.**" if bad.empty else f"**{len(bad)} VIOLATIONS** (`encoder_checks.csv`)."))
    w("")
    for t, title, num in (("score", "Total score", "1"), ("MQ", "Movement-quality score (MQ, D13)", "2")):
        cols = METHODS + ["base_group", "base_setfeat", "base_repcount", "base_duration", "base_mean"]
        w(f"## {num}. {title} within groups, video level (LOSO)")
        w("")
        w("RMSE / Spearman ρ, video head; baselines fitted on each LOSO fold's training videos; pooled rows: centred ρ "
          "(ranking within exercises). Noise floor: inter-view label noise of the expert groups (E3).")
        w("")
        rows = []
        for st in wn + pw:
            r0 = srow(S, st, "b1", "video head", t)
            if r0 is None:
                continue
            rho = "spearman_centred" if st in pw else "spearman"
            cells = []
            for c in cols:
                r = srow(S, st, c, "video head" if c in METHODS else "baseline", t)
                cells.append("–" if r is None else (f2(r.rmse) if c in ("base_group", "base_mean") else
                                                    f"{f2(r.rmse)} / {f2(r[rho])}"))
            ex = st.split()[0]
            fl = floors.loc[ex, t] if ex in floors.index and st.endswith(" expert") else np.nan
            rows.append([f"{st} ({r0.n} / {r0.subjects})", *cells, f2(fl)])
        table(["Stratum (videos / subjects)"] + [SHORT[c] for c in cols] + ["floor"], rows)
        w("Paired RMSE difference [95% CI] (negative = lower error than the comparison), video head:")
        w("")
        rows = []
        for st in wn + pw:
            for m in METHODS:
                rows.append([st, SHORT[m]] + [dcell(D, st, m, vs, t) if vs != m else "–"
                                              for vs in ("base_group", "base_setfeat", "b1", "b2_ktpformer")])
        table(["Stratum", "method", "vs group-only", "vs set-features", "vs B1", "vs B2-KTP"], rows)
        cnt = []
        for m in METHODS:
            ds = [drow(D, st, m, "base_group", t) for st in wn]
            ds = [x for x in ds if x is not None]
            cnt.append(f"{SHORT[m]} {sum(x.hi < 0 for x in ds)}/{sum(x.lo > 0 for x in ds)}/"
                       f"{sum(x.lo <= 0 <= x.hi for x in ds)}")
        w(f"Strata with a paired RMSE CI vs group-only below 0 / above 0 / including 0 ({len(wn)} within-exercise "
          f"strata): " + "; ".join(cnt) + ".")
        w("")
        w(f"Subject-level paired tests, {'total score' if t == 'score' else 'MQ'} (per-subject RMSE over its videos; "
          "Δ = first − second; "
          "Wilcoxon signed-rank, two-sided, Holm over the within-exercise strata; r = rank-biserial, d_z = Cohen's; "
          "lower / higher = subjects with a lower / higher error under the first method):")
        w("")
        rows = []
        for st in wn + pw:
            for a_, b_ in TESTS[:5]:
                r = T[(T.stratum == st) & (T.target == t) & (T.method == a_) & (T.vs == b_)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([st, f"{SHORT[a_]} vs {SHORT[b_]}", r.subjects, f2(r.mean_delta), f2(r.median_delta),
                             f"{r.lower} / {r.higher}", fp(r.wilcoxon_p), fp(r.p_holm), f2(r.rank_biserial),
                             f2(r.cohen_dz)])
        table(["Stratum", "comparison", "subjects", "mean Δ", "median Δ", "lower / higher", "p", "p (Holm)", "r",
               "d_z"], rows)
    w("## 3. Within-group Viewpoint Consistency Index (centred 256-d, LOSO)")
    w("")
    w("Same rep from another view − other (exercise, subject, variation) groups. **Under LOSO each fold's test set is "
      "one subject, so the 'other groups' are that subject's other variations** (the only embeddings a fold's model "
      "produces for held-out data); this is a stricter contrast than the GroupKFold-5 VCI, whose other groups are "
      "mostly other subjects. The GroupKFold-5 VCI restricted the same way (other-group pairs within a subject; from "
      "the GroupKFold-5 embeddings) is given for a like-for-like comparison, and the published GroupKFold-5 value "
      "after it.")
    w("")

    def vc(VV, m, stage, st):
        r = VV[(VV.method == m) & (VV.stage == stage) & (VV.stratum == st)]
        return "–" if r.empty or not np.isfinite(r.vci.iloc[0]) else fci(r.vci.iloc[0], r.vci_lo.iloc[0],
                                                                          r.vci_hi.iloc[0])
    rows = []
    for st in wn + pw:
        r = V[(V.method == "proposed_j1") & (V.stage == "final") & (V.stratum == st)]
        npairs = int(r.same_pairs.iloc[0]) if len(r) else 0
        rows.append([f"{st} ({npairs})", vc(V, "proposed_j1", "init", st), vc(V, "proposed_j1", "pretrained", st),
                     vc(V, "b1", "final", st), vc(V, "b2_ktpformer", "final", st), vc(V, "proposed_j1", "final", st),
                     " / ".join(_pub(VG, m, s_, st) for m, s_ in (("proposed_j1", "init"), ("proposed_j1", "pretrained"),
                                                                  ("b1", "final"), ("proposed_j1", "final"))),
                     f"{_pub(VGp, 'proposed_j1', 'final', st)} / {_pub(VGp, 'b1', 'final', st)}"])
    table(["Stratum (same-rep pairs)", "random init", "after CVDC", "B1", "B2-KTP", "Prop λ1",
           "GKF-5 same-subject: random / after CVDC / B1 / Prop λ1", "GKF-5 published: Prop λ1 / B1"], rows)
    w("## 4. View stability of the distance to the PT reference (synced expert performances, rep level, LOSO)")
    w("")
    if VS_ is not None and len(VS_):
        ms = [m for m in ["init_xy", *METHODS] if m in set(VS_.method)]
        w("ICC(1) [95% CI] of the standardised centred-cosine distance across the F / L / R views of one rep "
          "(standardised per fold with its training subjects); Δ vs B1 with the paired CI; last column: GroupKFold-5 "
          "Prop λ1 / B1 (published).")
        w("")
        rows = []
        for ex, d in VS_.groupby("exercise"):
            cells = []
            for m in ms:
                r = d[d.method == m]
                if r.empty:
                    cells.append("–")
                    continue
                r = r.iloc[0]
                c = fci(r.icc, r.icc_lo, r.icc_hi)
                if m not in ("b1", "init_xy") and "d_icc_vs_b1" in r and np.isfinite(r.d_icc_vs_b1):
                    c += f"; Δ {fci(r.d_icc_vs_b1, r.d_icc_vs_b1_lo, r.d_icc_vs_b1_hi, bold=True)}"
                cells.append(c)
            g = "–"
            if VSG is not None:
                gp, gb = VSG[(VSG.exercise == ex) & (VSG.method == "proposed_j1")], VSG[(VSG.exercise == ex) &
                                                                                       (VSG.method == "b1")]
                if len(gp) and len(gb):
                    g = f"{f2(gp.icc.iloc[0])} / {f2(gb.icc.iloc[0])}"
            rows.append([f"{ex} ({int(d.perfs.iloc[0])})", *cells, g])
        table(["Exercise (performances)"] + [SHORT[m] for m in ms] + ["GKF-5 Prop λ1 / B1"], rows)
    else:
        w("Not computed (`scripts/phase3_full/embedding_distance_perex.py --loso`).")
        w("")
    w("## 5. LOSO vs GroupKFold-5")
    w("")
    w("Same videos, same subject-bootstrap draws; GroupKFold-5 recomputed from its saved predictions with this code"
      + (" (identical to its published table)" if not smoke else "") + ". Δ = LOSO − GroupKFold-5 [95% CI]; "
      "negative RMSE Δ = LOSO more accurate.")
    w("")
    for t, title in (("score", "Total score"), ("MQ", "MQ")):
        w(f"**{title}, RMSE: GroupKFold-5 → LOSO (Δ)**")
        w("")
        rows = []
        for st in wn + pw:
            cells = []
            for m in METHODS + ["base_group", "base_setfeat"]:
                r = C[(C.stratum == st) & (C.method == m) & (C.target == t) & (C.metric == "rmse")]
                cells.append("–" if r.empty else f"{f2(r.gkf.iloc[0])} → {f2(r.loso.iloc[0])} "
                             f"({fci(r.delta.iloc[0], r.lo.iloc[0], r.hi.iloc[0], bold=True)})")
            rows.append([st, *cells])
        table(["Stratum"] + [SHORT[m] for m in METHODS + ["base_group", "base_setfeat"]], rows)
    w("**Conclusions side by side** (total score unless stated; video head; paired CIs):")
    w("")
    rows = []

    def both(fn):
        return fn(DG, SG, TG), fn(D, S, T)

    def n_beat(DD, SS, TT, t="score"):
        out = []
        for m in METHODS:
            ds = [x for x in (drow(DD, st, m, "base_group", t) for st in wn) if x is not None]
            out.append(f"{SHORT[m]} {sum(x.hi < 0 for x in ds)}/{sum(x.lo > 0 for x in ds)}")
        return "; ".join(out)
    g_, l_ = both(lambda DD, SS, TT: n_beat(DD, SS, TT))
    rows.append([f"strata (of {len(wn)}) where a network beats / loses to group-only (total score)", g_, l_])
    g_, l_ = both(lambda DD, SS, TT: n_beat(DD, SS, TT, "MQ"))
    rows.append([f"strata (of {len(wn)}) where a network beats / loses to group-only (MQ)", g_, l_])
    for st in pw:
        for m, vs in (("proposed_j1", "base_group"), ("b1", "base_group"), ("proposed_j1", "b1"),
                      ("proposed_j1", "b2_ktpformer"), ("proposed_lp", "b1")):
            for t in ("score", "MQ"):
                g_, l_ = both(lambda DD, SS, TT: dcell(DD, st, m, vs, t))
                rows.append([f"{st}: {SHORT[m]} − {SHORT[vs]} RMSE ({t})", g_, l_])
        g_, l_ = both(lambda DD, SS, TT: " / ".join(
            f"{SHORT[m]} {f2(srow(SS, st, m, 'video head', 'score').spearman_centred)}" for m in METHODS
            if srow(SS, st, m, "video head", "score") is not None))
        rows.append([f"{st}: within-exercise ranking (centred ρ)", g_, l_])
    for st in wn:
        for vs in ("b1", "b2_ktpformer"):
            g_, l_ = both(lambda DD, SS, TT: dcell(DD, st, "proposed_j1", vs, "score"))
            rows.append([f"{st}: Prop λ1 − {SHORT[vs]} RMSE", g_, l_])
    table(["Conclusion", "GroupKFold-5", "LOSO"], rows)
    w("## 6. Training: kept / stopping epochs, wall time (LOSO)")
    w("")
    rows = []
    for (ex, m), d in TR.groupby(["exercise", "method"]):
        rows.append([ex, SHORT[m], len(d), f"{d.best_epoch.mean():.1f} ({int(d.best_epoch.min())}–{int(d.best_epoch.max())})",
                     f"{d.stopped_epoch.mean():.1f} ({int(d.stopped_epoch.min())}–{int(d.stopped_epoch.max())})",
                     f"{int(d.early_stopped.sum())}/{len(d)}", f"{d.total_s.sum() / 3600:.2f}"])
    table(["Exercise", "method", "folds", "kept epoch mean (range)", "stopped mean (range)", "early-stopped",
           "wall h"], rows)
    w(f"Total wall time of the runs: {TR.total_s.sum() / 3600:.1f} h (first run finished {TR.finished.min()}, last "
      f"{TR.finished.max()}).")
    w("")
    w("Tables: `score_metrics.csv`, `paired_deltas.csv`, `subject_tests.csv`, `video_predictions.csv`, `vci.csv`, "
      "`distance_view_stability.csv`, `encoder_checks.csv`, `stopping_runtime.csv`, `loso_vs_gkf.csv`, and the "
      "GroupKFold-5 reference recomputed on the same draws (`gkf_reference_*.csv`).")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    if not smoke:  # committed copy next to the GroupKFold-5 tables (reports/final_E01_E04_tables.md)
        (REPO / "reports" / "final_E01_E04_loso_tables.md").write_text(text, encoding="utf-8")


def _pub(VGp, m, stage, st):
    r = VGp[(VGp.method == m) & (VGp.stage == stage) & (VGp.stratum == st)]
    return "–" if r.empty else f2(r.vci.iloc[0])


if __name__ == "__main__":
    main()
