"""Proposal completion (2026-10-22), quick measures 1-2, from the saved predictions only (nothing is trained):

1. Classification metrics (proposal D7.6, "Algorithmic accuracy: precision, recall, F1, accuracy"): the derived
   quality class (D6: score >= 80 correct, 50-79 partially correct, < 50 incorrect; our derivation, not a dataset
   label) of every test video, from
     - the quality head: the mean of its class probabilities over the video's reps (E15, E18), and
     - the video head's predicted total score, binned with the same cuts;
   next to the group-only and set-features baselines binned the same way. Accuracy, macro precision / recall / F1
   (over the classes present in the stratum's truth or predictions) and unweighted Cohen's kappa.
2. Results per recording condition (proposal D7.6 "Geometric robustness ... across viewpoints and recording
   conditions"; the dataset's variations FL, ML, LL, LJ, HJ, O, LR): video-head total score RMSE and Spearman per
   variation, the RMSE difference to the same method's full-light (FL) videos (paired subject draws), and Module III's
   agreement (Spearman of its set score with the physiotherapists' total score) per variation.

Both for LOSO (the final table, E30) and GroupKFold-5 (step 1); strata = exercise x group, then pooled over E01-E04
within each group (pooled Spearman on values centred on the group-only prediction, so exercise differences do not
count as ranking). 95% CIs: subject bootstrap (eval.bootstrap, paired across methods and variations).

usage: python scripts/full/completion/classification_conditions.py [--boot 2000]
Writes <results>/completion/quick/{classification,conditions,module3_conditions}.csv and summary.md.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "full" / "phase3"))
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase4"))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.eval.metrics import classification, interval, item_weights, regression  # noqa: E402
import report_perex as RP  # noqa: E402
from report_full import draws_for, stratum_mask  # noqa: E402

EXS = ["E01", "E02", "E03", "E04"]
METHODS = ["b1", "b2_ktpformer", "proposed_j1", "proposed_lp"]
SHORT = {**RP.SHORT, "base_group": "group-only", "base_setfeat": "set-features"}
VARS = ["FL", "ML", "LL", "LJ", "HJ", "O", "LR"]
VAR_NAME = {"FL": "full light", "ML": "medium light", "LL": "low light", "LJ": "low jitter", "HJ": "high jitter",
            "O": "occlusion", "LR": "low resolution"}
SCHEMES = {"LOSO": ("perex_loso", "perex_loso"), "GroupKFold-5": ("perex_gkf5", "final_E01_E04/step1")}
POOLED = [("experts, E01–E04", None, "expert"), ("non-experts, E01–E04", None, "non-expert")]


def qclass(score, bins):
    return np.digitize(np.asarray(score, float), bins)


def fmt(x, lo=None, hi=None, d=2):
    f = lambda v: "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}".replace("-", "−")  # noqa: E731
    return f(x) if lo is None else f"{f(x)} [{f(lo)}, {f(hi)}]"


def strata_of(P):
    within = [(f"{e} {g}", e, g) for e, g in sorted({(e, g) for e, g in zip(P.exercise, P.group)})]
    return within + [s for s in POOLED if s[2] in set(P.group)]


# ---------------------------------------------------------------- 1. classification
def class_predictions(scheme_root, P, bins):
    """Video-level predicted classes: quality head (mean rep probabilities) and binned scores (video head and
    baselines). Returns long DataFrame (method, source, video_id, pred)."""
    _, pr, _ = RP.load_runs(scheme_root, EXS, METHODS)
    pcols = [c for c in pr.columns if c.startswith("p_quality_")]
    q = pr.groupby(["method", "video_id"])[pcols].mean()
    rows = [pd.DataFrame({"method": q.index.get_level_values(0), "source": "quality head",
                          "video_id": q.index.get_level_values(1), "pred": q.to_numpy().argmax(1)})]
    vh = P[(P.source == "video head") & P.method.isin(METHODS)]
    rows.append(pd.DataFrame({"method": vh.method, "source": "binned score", "video_id": vh.video_id,
                              "pred": qclass(vh.score_pred, bins)}))
    bl = P[(P.source == "baseline") & P.method.isin(["base_group", "base_setfeat"])]
    rows.append(pd.DataFrame({"method": bl.method, "source": "binned score", "video_id": bl.video_id,
                              "pred": qclass(bl.score_pred, bins)}))
    # the rep table's own class label must be the binned video score (the quality head's target)
    chk = pr.drop_duplicates("video_id").set_index("video_id")
    truth = qclass(P.drop_duplicates("video_id").set_index("video_id").score.reindex(chk.index), bins)
    assert (chk.quality.to_numpy() == truth).all(), "quality labels differ from the binned video score"
    return pd.concat(rows, ignore_index=True)


def classification_tables(name, C, V, n_boot, seed, ci, bins):
    rows = []
    entries = [(m, "quality head") for m in METHODS] + [(m, "binned score") for m in METHODS] + \
              [("base_group", "binned score"), ("base_setfeat", "binned score")]
    for sname, ex, grp in strata_of(V):
        v = V[stratum_mask(V, ex, grp)].sort_values("video_id")
        subs, M = draws_for(v, n_boot, seed)
        W = np.vstack([np.ones(len(v)), item_weights(v.subject.to_numpy(), subs, M)])
        t = qclass(v.score, bins)
        bal = np.bincount(t, minlength=3)
        for m, src in entries:
            c = C[(C.method == m) & (C.source == src)].set_index("video_id").pred.reindex(v.video_id)
            assert c.notna().all(), (name, sname, m, src)
            p = c.to_numpy().astype(int)
            labels = sorted(set(t) | set(p))
            r = classification(t, p, W, 3, labels)
            row = dict(scheme=name, stratum=sname, method=m, source=src, videos=len(v), subjects=len(subs),
                       n_incorrect=int(bal[0]), n_partial=int(bal[1]), n_correct=int(bal[2]))
            for k, x in r.items():
                lo, hi = interval(x[1:], ci)
                row.update({k: float(x[0]), f"{k}_lo": lo, f"{k}_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. recording conditions
def condition_tables(name, P, n_boot, seed, ci):
    rows = []
    entries = [(m, "video head") for m in METHODS] + [("base_group", "baseline"), ("base_setfeat", "baseline")]
    grp_pred = P[P.method == "base_group"].set_index("video_id").score_pred
    for sname, ex, grp in strata_of(P):
        d = P[stratum_mask(P, ex, grp)]
        ref = d[(d.method == "b1") & (d.source == "video head")].sort_values("video_id")
        subs, M = draws_for(ref, n_boot, seed)  # one set of subject draws for every variation and method (paired)
        pooled = ex is None
        res = {}
        for var in VARS:
            r = ref[ref.variation == var]
            if r.subject.nunique() < 3:
                continue
            W = np.vstack([np.ones(len(r)), item_weights(r.subject.to_numpy(), subs, M)])
            for m, src in entries:
                a = d[(d.method == m) & (d.source == src) & (d.variation == var)].set_index("video_id").reindex(r.video_id)
                y, p = a.score.to_numpy(), a.score_pred.to_numpy()
                if pooled:  # rank within exercises: centre on the group-only prediction of the same video
                    g = grp_pred.reindex(r.video_id).to_numpy()
                    y, p = y - g, p - g
                res[(var, m)] = regression(a.score.to_numpy(), a.score_pred.to_numpy(), W)["rmse"]
                rho = regression(y, p, W)["spearman"]
                lo, hi = interval(res[(var, m)][1:], ci)
                rlo, rhi = interval(rho[1:], ci)
                rows.append(dict(scheme=name, stratum=sname, variation=var, method=m, videos=len(r),
                                 subjects=r.subject.nunique(), rmse=float(res[(var, m)][0]), rmse_lo=lo, rmse_hi=hi,
                                 spearman=float(rho[0]) if not m.startswith("base_group") else np.nan,
                                 spearman_lo=rlo if m != "base_group" else np.nan,
                                 spearman_hi=rhi if m != "base_group" else np.nan))
        for row in rows:  # paired difference to the same method's FL videos (same subject draws)
            if row["scheme"] != name or row["stratum"] != sname or (row["variation"], row["method"]) not in res \
                    or ("FL", row["method"]) not in res:
                continue
            dv = res[(row["variation"], row["method"])] - res[("FL", row["method"])]
            lo, hi = interval(dv[1:], ci)
            row.update(d_rmse_vs_fl=float(dv[0]), d_rmse_vs_fl_lo=lo, d_rmse_vs_fl_hi=hi)
    return pd.DataFrame(rows)


def module3_conditions(m3, n_boot, seed, ci):
    """Module III set score vs the physiotherapists' total score per variation (GroupKFold-5 libraries, M13)."""
    rows = []
    for sname, ex, grp in strata_of(m3):
        d = m3[stratum_mask(m3, ex, grp)].dropna(subset=["ref_set_score"])
        subs, M = draws_for(d, n_boot, seed)
        for var in VARS:
            r = d[d.variation == var]
            if r.subject.nunique() < 3:
                continue
            W = np.vstack([np.ones(len(r)), item_weights(r.subject.to_numpy(), subs, M)])
            y, p = r.score.to_numpy(), r.ref_set_score.to_numpy()
            if ex is None:  # centre within exercise x group
                y = y - r.groupby("exercise").score.transform("mean").to_numpy()
                p = p - r.groupby("exercise").ref_set_score.transform("mean").to_numpy()
            rho = regression(y, p, W)["spearman"]
            lo, hi = interval(rho[1:], ci)
            rows.append(dict(stratum=sname, variation=var, videos=len(r), subjects=r.subject.nunique(),
                             spearman=float(rho[0]), spearman_lo=lo, spearman_hi=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary
def write_summary(out, CL, CO, M3, det, n_boot):
    L = []
    w = L.append
    w("# Proposal completion: classification metrics and results per recording condition (E01–E04)\n")
    w(f"Generated by `scripts/full/completion/classification_conditions.py` from the saved predictions (nothing "
      f"trained). 95% CIs: subject bootstrap ({n_boot} draws). Classes are derived from the score (D6: ≥ 80 correct, "
      "50–79 partially correct, < 50 incorrect); they are not dataset labels.\n")
    for scheme in ("LOSO", "GroupKFold-5"):
        c = CL[CL.scheme == scheme]
        w(f"\n## 1. Quality class, {scheme}\n")
        w("Accuracy / macro F1 [95% CI] / Cohen's κ. Quality head = mean class probabilities over the video's reps; "
          "binned = the predicted total score cut at 50 / 80.\n")
        strata = list(dict.fromkeys(c.stratum))
        head = ["Method"] + strata
        w("| " + " | ".join(head) + " |")
        w("|" + "---|" * len(head))
        bal = c.drop_duplicates("stratum").set_index("stratum")
        w("| class balance (incorrect / partial / correct) | " + " | ".join(
            f"{int(bal.loc[s, 'n_incorrect'])} / {int(bal.loc[s, 'n_partial'])} / {int(bal.loc[s, 'n_correct'])}"
            for s in strata) + " |")
        for (m, src), g in c.groupby(["method", "source"], sort=False):
            g = g.set_index("stratum")
            lab = f"{SHORT.get(m, m)} ({src})"
            w(f"| {lab} | " + " | ".join(
                f"{fmt(g.loc[s, 'accuracy'])} / {fmt(g.loc[s, 'macro_f1'], g.loc[s, 'macro_f1_lo'], g.loc[s, 'macro_f1_hi'])}"
                f" / {fmt(g.loc[s, 'kappa'])}" for s in strata) + " |")
        w("\nMacro precision / recall per stratum are in `classification.csv`.")
    for scheme in ("LOSO", "GroupKFold-5"):
        c = CO[(CO.scheme == scheme) & CO.stratum.isin([s for s, _, _ in POOLED])]
        w(f"\n## 2. Total score per recording condition, {scheme} (pooled over E01–E04 within group)\n")
        w("RMSE [95% CI] (ΔRMSE vs the same method's full-light videos [95% CI]); per exercise in `conditions.csv`.\n")
        for sname in dict.fromkeys(c.stratum):
            g = c[c.stratum == sname]
            vars_ = [v for v in VARS if v in set(g.variation)]
            w(f"\n**{sname}** (videos per condition: " + ", ".join(
                f"{v} {int(g[g.variation == v].videos.iloc[0])}" for v in vars_) + ")\n")
            w("| Method | " + " | ".join(f"{v} ({VAR_NAME[v]})" for v in vars_) + " |")
            w("|" + "---|" * (len(vars_) + 1))
            for m in METHODS + ["base_group", "base_setfeat"]:
                gm = g[g.method == m].set_index("variation")
                cells = []
                for v in vars_:
                    r = gm.loc[v]
                    s = fmt(r.rmse, r.rmse_lo, r.rmse_hi)
                    if v != "FL" and np.isfinite(r.get("d_rmse_vs_fl", np.nan)):
                        bold = r.d_rmse_vs_fl_lo > 0 or r.d_rmse_vs_fl_hi < 0
                        dd = fmt(r.d_rmse_vs_fl, r.d_rmse_vs_fl_lo, r.d_rmse_vs_fl_hi)
                        s += f" ({'**' + dd + '**' if bold else dd})"
                    cells.append(s)
                w(f"| {SHORT.get(m, m)} | " + " | ".join(cells) + " |")
    w("\n## 3. Module III agreement per recording condition (GroupKFold-5 libraries)\n")
    w("Spearman ρ of the Module III set score with the physiotherapists' total score [95% CI] (pooled rows: centred "
      "within exercise).\n")
    strata = list(dict.fromkeys(M3.stratum))
    vars_ = [v for v in VARS if v in set(M3.variation)]
    w("| Stratum | " + " | ".join(vars_) + " |")
    w("|" + "---|" * (len(vars_) + 1))
    for s in strata:
        g = M3[M3.stratum == s].set_index("variation")
        w(f"| {s} | " + " | ".join(fmt(g.loc[v, 'spearman'], g.loc[v, 'spearman_lo'], g.loc[v, 'spearman_hi'])
                                    if v in g.index else "–" for v in vars_) + " |")
    if det is not None:
        w("\n## 4. Pose detection per recording condition (Phase 1, context)\n")
        w("Mean share of frames with a detected pose, E01–E04 (`outputs/full/phase1/detection_by_variation.csv`).\n")
        w(det)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def detection_context(cfg):
    p = resolve(cfg["outputs"]["phase1"]) / "detection_by_variation.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.exercise.isin(EXS)]
    d["w"] = d.det_rate_mean * d.videos
    t = d.groupby(["variation", "view"]).agg(w=("w", "sum"), n=("videos", "sum"))
    t = (t.w / t.n).unstack("view")
    vars_ = [v for v in VARS if v in t.index]
    rows = ["| Condition | " + " | ".join(t.columns) + " |", "|" + "---|" * (len(t.columns) + 1)]
    rows += [f"| {v} ({VAR_NAME[v]}) | " + " | ".join(f"{t.loc[v, c]:.3f}" if np.isfinite(t.loc[v, c]) else "–"
                                                     for c in t.columns) + " |" for v in vars_]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--boot", type=int)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    cfg = load_config(REPO / a.config)
    ec = cfg["eval"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    bins = cfg["labels"]["quality_bins"]
    res = resolve(ec["results_dir"])
    runs = resolve(cfg["train"]["runs_dir"])
    out = res / "completion" / "quick"
    out.mkdir(parents=True, exist_ok=True)
    CL, CO = [], []
    for name, (run_tag, res_tag) in SCHEMES.items():
        P = pd.read_csv(res / res_tag / "video_predictions.csv")
        P = P[P.exercise.isin(EXS)]
        V = P[(P.method == "b1") & (P.source == "video head")][["video_id", "subject", "group", "exercise", "view",
                                                                 "variation", "score"]]
        C = class_predictions(runs / run_tag, P, bins)
        CL.append(classification_tables(name, C, V, n_boot, seed, ci, bins))
        CO.append(condition_tables(name, P, n_boot, seed, ci))
        print(f"{name}: done ({time.time() - t0:.0f} s)", flush=True)
    CL, CO = pd.concat(CL, ignore_index=True), pd.concat(CO, ignore_index=True)
    CL.to_csv(out / "classification.csv", index=False)
    CO.to_csv(out / "conditions.csv", index=False)
    m3 = pd.read_csv(resolve(cfg["assess"]["json_dir"]).parent / "module3_videos.csv")
    M3 = module3_conditions(m3[m3.exercise.isin(EXS)], n_boot, seed, ci)
    M3.to_csv(out / "module3_conditions.csv", index=False)
    write_summary(out, CL, CO, M3, detection_context(cfg), n_boot)
    print(f"written to {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
