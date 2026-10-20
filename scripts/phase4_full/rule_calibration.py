"""M18 (post-hoc, user 2026-10-20): Module III rule calibration for E03 (range rule) and E08 (knee rule), before / after,
and the E08 non-expert investigation.

Before = the targets on the PT demo (``python -m cvdc.assess.run --config configs/per_exercise.yaml --exercises E03 E08
--set E03:rom_centre=pt_demo --set 'E08:channel_centre={knee_flexion: pt_demo}' --out
outputs/full/phase4/calibration_pt_demo --no-json``); after = the default run (outputs/full/phase4/): E03's range
target = the >= 95 experts' mean end range per view, E08's knee target = the >= 95 experts' mean curve per view (both
fitted per fold on the training subjects). Same folds, reps and bootstrap draws, so the differences are paired.
  rules        firing rate of the calibrated rule (share of reps): experts' videos >= 95 / < 80, all experts,
               non-experts; the exercise's other rules for reference
  agreement    Spearman of the set score with the total score and MQ per stratum, before / after / paired Δ [CI]
  score        mean set score; expert vs non-expert AUC (context); view stability ICC(1) of the rep score over synced
               expert performances
E08 non-experts (the set score correlated negatively with the physiotherapists): video-level Spearman of the
total score / MQ with every score group distance, channel deviation and measure, the per-video rule rates, and the
score recomputed without each group (which group carries the negative sign), plus a subject-level table.

usage: python scripts/phase4_full/rule_calibration.py [--config configs/per_exercise.yaml] [--boot 2000]
Writes outputs/full/phase4/calibration/*.csv and summary.md.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "phase4"))
sys.path.insert(0, str(REPO / "scripts" / "phase3_full"))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.eval.metrics import interval, item_weights, regression  # noqa: E402
from group_probe import weighted_auc  # noqa: E402
from report_phase4 import draws, fci, fmt, rescore, view_stability  # noqa: E402

RULES = {"E03": "lateral_rotation_range_insufficient", "E08": "knee_flexion_deviation"}
GROUPS = ("angle", "rom", "speed", "symmetry")


def load(path, ex):
    R = pd.read_csv(path / "module3_reps.csv")
    V = pd.read_csv(path / "module3_videos.csv")
    return R[R.exercise == ex].reset_index(drop=True), V[V.exercise == ex].sort_values("video_id").reset_index(drop=True)


def W_for(d, n_boot, seed):
    subs, M = draws(d.subject.to_numpy(), d.group.to_numpy(), n_boot, seed)
    return np.vstack([np.ones(len(d)), item_weights(d.subject.to_numpy(), subs, M)])


def rho(y, x, W):
    ok = np.isfinite(y) & np.isfinite(x)
    return regression(np.asarray(y, float)[ok], np.asarray(x, float)[ok], W[:, ok])["spearman"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--boot", type=int)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    ec = cfg["eval"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    base = resolve(cfg["assess"]["json_dir"]).parent
    out = base / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    sync = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    synced = set(sync.index[sync.cls == "synced"])
    sc = cfg["assess"]["score"]
    rule_rows, ag_rows, sc_rows = [], [], []
    for ex, rule in RULES.items():
        runs = {"before (PT demo)": load(base / "calibration_pt_demo", ex), "after (experts)": load(base, ex)}
        vb, va = runs["before (PT demo)"][1], runs["after (experts)"][1]
        assert (vb.video_id.to_numpy() == va.video_id.to_numpy()).all()
        for name, (R, V) in runs.items():
            R2 = R.merge(V[["video_id", "score"]].rename(columns={"score": "vscore"}), on="video_id")
            for c in sorted(x for x in R.columns if x.startswith("err_")):
                e, n = R2[R2.group == "expert"], R2[R2.group == "non-expert"]
                rule_rows.append(dict(exercise=ex, version=name, rule=c[4:], calibrated=c[4:] == rule,
                                      experts_ge95=e[e.vscore >= 95][c].mean(), experts_lt80=e[e.vscore < 80][c].mean(),
                                      experts_all=e[c].mean(), non_experts=n[c].mean() if len(n) else np.nan,
                                      reps_ge95=int((e.vscore >= 95).sum()), reps_lt80=int((e.vscore < 80).sum()),
                                      videos_lt80=int(e[e.vscore < 80].video_id.nunique())))
        for g, dv in va.groupby("group"):
            sel = (va.group == g).to_numpy()
            W = W_for(va[sel], n_boot, seed)
            for t in ("score", "MQ"):
                rb, ra = rho(vb[sel][t].to_numpy(), vb[sel].ref_set_score.to_numpy(), W), \
                    rho(va[sel][t].to_numpy(), va[sel].ref_set_score.to_numpy(), W)
                d = ra - rb
                row = dict(exercise=ex, group=g, target=t, subjects=dv.subject.nunique(), videos=len(dv),
                           before=rb[0], after=ra[0], delta=d[0])
                row["before_lo"], row["before_hi"] = interval(rb[1:], ci)
                row["after_lo"], row["after_hi"] = interval(ra[1:], ci)
                row["delta_lo"], row["delta_hi"] = interval(d[1:], ci)
                ag_rows.append(row)
        for name, (R, V) in runs.items():
            row = dict(exercise=ex, version=name)
            for g, dv in V.groupby("group"):
                row[f"mean_set_score_{g}"] = dv.ref_set_score.mean()
            if V.group.nunique() == 2:
                W = W_for(V, n_boot, seed)
                auc = weighted_auc((V.group == "expert").to_numpy(), V.ref_set_score.to_numpy(float), W)
                row["auc"], (row["auc_lo"], row["auc_hi"]) = auc[0], interval(auc[1:], ci)
            Rr = R.merge(reps[["video_id", "rep", "rep_uid", "group_id"]], on=["video_id", "rep"], how="left")
            Rr = Rr[Rr.group_id.isin(synced) & (Rr.group == "expert")]
            Rr = Rr.assign(perf=Rr.rep_uid, ref_score_spatial=rescore(Rr, sc, {"speed": np.full(len(Rr), np.nan)}))
            for col, nm in (("ref_score", "icc_score"), ("ref_score_spatial", "icc_score_no_speed")):
                r, _, _ = view_stability(Rr, col, n_boot, seed, ci)
                row[nm], row[nm + "_lo"], row[nm + "_hi"] = r["icc"], r["icc_lo"], r["icc_hi"]
            sc_rows.append(row)
    RR, AG, SC = pd.DataFrame(rule_rows), pd.DataFrame(ag_rows), pd.DataFrame(sc_rows)
    RR.to_csv(out / "rule_rates.csv", index=False)
    AG.to_csv(out / "agreement.csv", index=False)
    SC.to_csv(out / "scores.csv", index=False)
    INV, SUBJ, LOO = e08_non_experts(base, cfg, n_boot, seed, ci)
    INV.to_csv(out / "e08_non_expert_components.csv", index=False)
    SUBJ.to_csv(out / "e08_non_expert_subjects.csv", index=False)
    LOO.to_csv(out / "e08_non_expert_without_group.csv", index=False)
    write_summary(out, RR, AG, SC, INV, SUBJ, LOO, n_boot)
    print(f"written to {out} in {time.time() - t0:.0f} s")


def e08_non_experts(base, cfg, n_boot, seed, ci):
    R, V = load(base, "E08")
    R, V = R[R.group == "non-expert"].reset_index(drop=True), V[V.group == "non-expert"].reset_index(drop=True)
    sc = cfg["assess"]["score"]
    num = [c for c in R.columns if c.startswith(("X_", "D_", "dev_abs_", "dev_mean_")) or
           c in ("rom", "excursion", "duration_s", "log2_speed", "omega_dev", "z_speed", "z_rom", "ref_score")]
    errs = [c for c in R.columns if c.startswith("err_")]
    for g in GROUPS:  # the score without one group
        R[f"score_without_{g}"] = rescore(R, sc, {g: np.full(len(R), np.nan)})
    agg = R.groupby("video_id")[num + errs + [f"score_without_{g}" for g in GROUPS]].mean().reset_index()
    agg["n_reps"] = R.groupby("video_id").size().to_numpy()
    D = V[["video_id", "subject", "group", "score", "MQ", "ref_set_score", "PF1", "PF2", "CF1", "CF2", "CF3"]].merge(
        agg, on="video_id").sort_values("video_id").reset_index(drop=True)
    W = W_for(D, n_boot, seed)
    rows = []
    for c in num + errs + ["n_reps"]:
        if D[c].nunique() < 2:
            continue
        for t in ("score", "MQ"):
            r = rho(D[t].to_numpy(), D[c].to_numpy(float), W)
            lo, hi = interval(r[1:], ci)
            rows.append(dict(measure=c, target=t, rho=r[0], lo=lo, hi=hi, mean=D[c].mean()))
    INV = pd.DataFrame(rows)
    loo = []
    for c in ["ref_set_score"] + [f"score_without_{g}" for g in GROUPS]:
        col = "ref_score" if c == "ref_set_score" else c
        x = D["ref_set_score"] if c == "ref_set_score" else D[col]
        for t in ("score", "MQ"):
            r = rho(D[t].to_numpy(), x.to_numpy(float), W)
            lo, hi = interval(r[1:], ci)
            loo.append(dict(score=c, target=t, rho=r[0], lo=lo, hi=hi))
    SUBJ = D.groupby("subject").agg(videos=("video_id", "size"), physio_score=("score", "mean"), MQ=("MQ", "mean"),
                                    PF1=("PF1", "mean"), set_score=("ref_set_score", "mean"),
                                    X_angle=("X_angle", "mean"), X_rom=("X_rom", "mean"), X_speed=("X_speed", "mean"),
                                    X_symmetry=("X_symmetry", "mean"), rom=("rom", "mean"),
                                    duration_s=("duration_s", "mean"), n_reps=("n_reps", "mean"),
                                    too_fast=("err_too_fast", "mean")).reset_index().sort_values("physio_score")
    return INV, SUBJ, pd.DataFrame(loo)


def write_summary(out, RR, AG, SC, INV, SUBJ, LOO, n_boot):
    L = []
    w = L.append
    w("# M18: Module III rule calibration (post-hoc) and the E08 non-expert investigation")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/phase4_full/rule_calibration.py`. Before = the target "
      "on the PT demo; after = on the ≥ 95 experts per view, fitted per fold on the training subjects (E03: the range "
      "target; E08: the knee curve, which M14 had already centred on the experts, so for E08 'after' is the existing "
      f"default and 'before' the demo-centred version). Same reps, folds and bootstrap draws ({n_boot}); post-hoc "
      "(chosen after seeing the rule rates), reported next to the original as M14 was.")
    w("")
    w("## 1. The calibrated rule: firing rate (share of reps)")
    w("")
    w("| Exercise | rule | version | experts ≥ 95 | experts < 80 | all experts | non-experts | reps ≥ 95 / < 80 (videos < 80) |")
    w("|---|---|---|---|---|---|---|---|")
    for r in RR[RR.calibrated].itertuples():
        w(f"| {r.exercise} | {r.rule} | {r.version} | {100 * r.experts_ge95:.0f}% | {100 * r.experts_lt80:.0f}% | "
          f"{100 * r.experts_all:.0f}% | {100 * r.non_experts:.0f}% | {r.reps_ge95} / {r.reps_lt80} ({r.videos_lt80}) |")
    w("")
    w("Other rules of the two exercises are unchanged by construction except where a rule's channel is the calibrated "
      "one (`rule_rates.csv`).")
    w("")
    w("## 2. Agreement of the set score with the physiotherapists, before → after (Spearman, paired Δ [95% CI])")
    w("")
    w("| Stratum (subjects) | target | before | after | Δ |")
    w("|---|---|---|---|---|")
    for r in AG.itertuples():
        d = fci(r.delta, r.delta_lo, r.delta_hi)
        w(f"| {r.exercise} {r.group} ({r.subjects}) | {r.target} | {fci(r.before, r.before_lo, r.before_hi)} | "
          f"{fci(r.after, r.after_lo, r.after_hi)} | {'**' + d + '**' if r.delta_lo > 0 or r.delta_hi < 0 else d} |")
    w("")
    w("## 3. Set score, separation and view stability, before → after")
    w("")
    w("| Exercise | version | mean set score experts / non-experts | AUC expert vs non-expert | ICC rep score | ICC without speed |")
    w("|---|---|---|---|---|---|")
    for r in SC.itertuples():
        auc = fci(r.auc, r.auc_lo, r.auc_hi) if "auc" in SC and np.isfinite(getattr(r, "auc", np.nan)) else "–"
        w(f"| {r.exercise} | {r.version} | {fmt(r.mean_set_score_expert, 1)} / "
          f"{fmt(getattr(r, 'mean_set_score_non_expert', np.nan) if hasattr(r, 'mean_set_score_non_expert') else np.nan, 1)} | "
          f"{auc} | {fci(r.icc_score, r.icc_score_lo, r.icc_score_hi)} | "
          f"{fci(r.icc_score_no_speed, r.icc_score_no_speed_lo, r.icc_score_no_speed_hi)} |")
    w("")
    w("## 4. E08 non-experts: what carries the negative correlation")
    w("")
    w("Video level (non-experts, after calibration). Distances (X_*, D_*, |dev|) are 'farther = worse', so a "
      "*positive* ρ with the physiotherapists' score means the measure moves the wrong way.")
    w("")
    w("**Score without one group** (Spearman with the total score / MQ):")
    w("")
    w("| Score | total | MQ |")
    w("|---|---|---|")
    for s_, d in LOO.groupby("score", sort=False):
        w(f"| {s_} | " + " | ".join(fci(*d[d.target == t][["rho", "lo", "hi"]].iloc[0].to_numpy()) for t in
                                     ("score", "MQ")) + " |")
    w("")
    w("**Components and rules** (Spearman with the total score, ordered by |ρ|; MQ in the CSV):")
    w("")
    w("| Measure | mean | ρ with total [95% CI] |")
    w("|---|---|---|")
    T = INV[INV.target == "score"].assign(a=lambda d: d.rho.abs()).sort_values("a", ascending=False)
    for r in T.head(18).itertuples():
        w(f"| {r.measure} | {fmt(r.mean, 3)} | {fci(r.rho, r.lo, r.hi)} |")
    w("")
    w("**Subjects** (sorted by the physiotherapists' score; subject means):")
    w("")
    w("| Subject | videos | physio score | PF1 | set score | X_angle | X_rom | X_speed | X_symmetry | end range (deg) | rep s | reps | too_fast |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in SUBJ.itertuples():
        w(f"| {r.subject} | {r.videos} | {fmt(r.physio_score, 1)} | {fmt(r.PF1)} | {fmt(r.set_score, 1)} | "
          f"{fmt(r.X_angle)} | {fmt(r.X_rom)} | {fmt(r.X_speed)} | {fmt(r.X_symmetry)} | {fmt(r.rom, 1)} | "
          f"{fmt(r.duration_s)} | {fmt(r.n_reps, 1)} | {fmt(r.too_fast)} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
