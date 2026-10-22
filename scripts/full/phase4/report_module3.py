"""Module III on the full scope (E01-E09; twelfth review, CPU work during the per-exercise training): summary.

Input: ``python -m cvdc.assess.run --config configs/per_exercise.yaml`` (outputs/full/phase4/: module3_reps.csv,
module3_videos.csv; libraries fitted per exercise and GroupKFold-5 fold on the training subjects, the folds of the
per-exercise models, T17). Video level (set score = mean rep score), within every exercise x group stratum (E9),
95% CIs from the subject bootstrap (E17; 2,000 draws, the same draws for every row of a stratum):
  agreement   Spearman of the set score with the total score, the movement-quality score (MQ, D13) and every EAAQ
              item that varies in the stratum (T14)
  predictor   out-of-fold within-group line total ~ set score (and MQ ~ set score), fitted per fold on the training
              videos: RMSE vs the group-only baseline (paired difference; spec §8 "correlation and RMSE")
  separation  AUC of the set score for expert vs non-expert (exercises with both groups), all videos and the
              view-matched front-view FL subset
  rules       firing rates per rule (share of reps): experts' videos >= 95, experts' videos < 80, non-experts
  view        view stability over synced expert performances (the same rep seen from F / L / R, test subjects): ICC(1)
              of the rep score, of the score without its speed group (synced views share rep timing, P6b) and of the
              end-range angle
Strata with fewer than ``eval.per_exercise_low_confidence_subjects`` subjects are flagged (T17).

usage: python scripts/full/phase4/report_module3.py [--config configs/per_exercise.yaml] [--boot 2000]
Writes agreement.csv, predictor.csv, separation.csv, rule_rates.csv, view_stability.csv and summary.md to
outputs/full/phase4/.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase4"))
sys.path.insert(0, str(REPO / "scripts" / "full" / "phase3"))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.data.labels import movement_quality  # noqa: E402
from cvdc.eval.baselines import ALL_ITEMS, item_activity, video_table  # noqa: E402
from cvdc.eval.metrics import interval  # noqa: E402
from cvdc.assess.module3 import assess_config  # noqa: E402
from cvdc.train.trainer import fold_plan, loso_plans  # noqa: E402
from group_probe import weighted_auc  # noqa: E402
from report_phase4 import corr_ci, draws, fci, fmt, oof_line, rescore, view_stability  # noqa: E402


def weights(d, n_boot, seed):
    subs, M = draws(d.subject.to_numpy(), d.group.to_numpy(), n_boot, seed)
    from cvdc.eval.metrics import item_weights
    return np.vstack([np.ones(len(d)), item_weights(d.subject.to_numpy(), subs, M)]), len(subs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--boot", type=int)
    ap.add_argument("--dir", help="a cvdc.assess.run output folder (default: the folder above assess.json_dir)")
    ap.add_argument("--loso", action="store_true", help="the run used LOSO folds (cvdc.assess.run --loso)")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    ec = cfg["eval"]
    n_boot, ci, seed = a.boot or ec["bootstrap"]["n"], ec["bootstrap"]["ci"], ec["bootstrap"]["seed"]
    low = ec.get("per_exercise_low_confidence_subjects", 15)
    out = resolve(a.dir) if a.dir else resolve(cfg["assess"]["json_dir"]).parent
    R = pd.read_csv(out / "module3_reps.csv")
    V = pd.read_csv(out / "module3_videos.csv")
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    vt = video_table(cfg)
    ia = cfg["heads"]["item_activity"]
    act = item_activity(vt, ALL_ITEMS, ia["min_subjects_off_mode"], ia["min_sd"])
    active = {(e, g): set(d[d.varies].item) for (e, g), d in act.groupby(["exercise", "group"])}
    strata = sorted({(e, g) for e, g in zip(V.exercise, V.group)})

    # ---- agreement and out-of-fold predictor
    ag, pr = [], []
    for ex, g in strata:
        d = V[(V.exercise == ex) & (V.group == g)].sort_values("video_id").reset_index(drop=True)
        W, ns = weights(d, n_boot, seed)
        for t in ["score", "MQ", *[k for k in ALL_ITEMS if k in active[(ex, g)]]]:
            r = corr_ci(d[t].to_numpy(float), d.ref_set_score.to_numpy(float), W, ci)
            ag.append(dict(exercise=ex, group=g, subjects=ns, videos=len(d), target=t, low_confidence=ns < low, **r))
    for ex in sorted(V.exercise.unique()):
        plans = (loso_plans if a.loso else fold_plan)(cfg, reps[reps.exercise == ex])
        v = V[V.exercise == ex].reset_index(drop=True)
        for t in ("score", "MQ"):
            p, b = oof_line(v, "ref_set_score", t, plans)
            v[f"pred_{t}"], v[f"base_{t}"] = p, b
        for g, d in v.groupby("group"):
            d = d.sort_values("video_id").reset_index(drop=True)
            W, ns = weights(d, n_boot, seed)
            for t in ("score", "MQ"):
                ok = d[f"pred_{t}"].notna().to_numpy()
                y = d[t].to_numpy(float)
                e1 = (d[f"pred_{t}"].to_numpy(float) - y) ** 2
                e0 = (d[f"base_{t}"].to_numpy(float) - y) ** 2
                r1 = np.sqrt((W[:, ok] * e1[ok]).sum(1) / W[:, ok].sum(1))
                r0 = np.sqrt((W[:, ok] * e0[ok]).sum(1) / W[:, ok].sum(1))
                lo, hi = interval((r1 - r0)[1:], ci)
                pr.append(dict(exercise=ex, group=g, subjects=ns, target=t, rmse_line=r1[0], rmse_group_mean=r0[0],
                               delta=r1[0] - r0[0], delta_lo=lo, delta_hi=hi, low_confidence=ns < low))
    AG, PR = pd.DataFrame(ag), pd.DataFrame(pr)
    AG.to_csv(out / "agreement.csv", index=False)
    PR.to_csv(out / "predictor.csv", index=False)

    # ---- separation (expert vs non-expert)
    sep = []
    for ex in sorted(V.exercise.unique()):
        for name, extra in (("all videos", None), ("front view, FL (view-matched)", ("F", "FL"))):
            d = V[V.exercise == ex]
            if extra:
                d = d[(d.view == extra[0]) & (d.variation == extra[1])]
            d = d[d.ref_set_score.notna()].sort_values("video_id")
            if d.group.nunique() < 2:
                continue
            W, ns = weights(d, n_boot, seed)
            y = (d.group == "expert").to_numpy()
            auc = weighted_auc(y, d.ref_set_score.to_numpy(float), W)
            lo, hi = interval(auc[1:], ci)
            sep.append(dict(exercise=ex, subset=name, videos=len(d), subjects=ns, auc=auc[0], auc_lo=lo, auc_hi=hi,
                            score_expert=d[y].ref_set_score.mean(), score_non_expert=d[~y].ref_set_score.mean()))
    SEP = pd.DataFrame(sep)
    SEP.to_csv(out / "separation.csv", index=False)

    # ---- rule rates
    errs = sorted(c for c in R.columns if c.startswith("err_"))
    R2 = R.merge(V[["video_id", "score"]].rename(columns={"score": "vscore"}), on="video_id", how="left")
    rr = []
    for ex, d in R2.groupby("exercise"):
        for c in errs:
            rules = [x["type"] for x in assess_config(cfg, ex)["rules"]]  # incl. the rules M19 / M20 add
            if c[4:] not in rules:
                continue
            e = d[d.group == "expert"]
            n = d[d.group == "non-expert"]
            rr.append(dict(exercise=ex, rule=c[4:], experts_ge95=e[e.vscore >= 95][c].mean(),
                           experts_lt80=e[e.vscore < 80][c].mean(), experts_all=e[c].mean(),
                           non_experts=n[c].mean() if len(n) else np.nan, reps_ge95=int((e.vscore >= 95).sum()),
                           reps_lt80=int((e.vscore < 80).sum())))
    RR = pd.DataFrame(rr)
    RR.to_csv(out / "rule_rates.csv", index=False)

    # ---- view stability over synced expert performances
    sync = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    synced = set(sync.index[sync.cls == "synced"])
    sc = cfg["assess"]["score"]
    R["ref_score_spatial"] = rescore(R, sc, {"speed": np.full(len(R), np.nan)})
    Rr = R.merge(reps[["video_id", "rep", "rep_uid", "group_id"]], on=["video_id", "rep"], how="left",
                 validate="one_to_one")
    Rr = Rr[Rr.group_id.isin(synced) & (Rr.group == "expert")]
    vs = []
    for ex, d in Rr.groupby("exercise"):
        d = d.assign(perf=d.rep_uid)
        for col, nm in (("ref_score", "Module III score"), ("ref_score_spatial", "score without speed"),
                        ("rom", "end-range angle")):
            r, _, _ = view_stability(d, col, n_boot, seed, ci)
            vs.append(dict(exercise=ex, measure=nm, **r))
    VS = pd.DataFrame(vs)
    VS.to_csv(out / "view_stability.csv", index=False)
    write_summary(out, AG, PR, SEP, RR, VS, V, R, cfg, n_boot, low)
    print(f"written to {out} in {time.time() - t0:.0f} s")


def write_summary(out, AG, PR, SEP, RR, VS, V, R, cfg, n_boot, low):
    L = []
    w = L.append
    w("# Module III, full scope (E01–E09): reference-based assessment")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/full/phase4/report_module3.py` from "
      f"`python -m cvdc.assess.run --config configs/per_exercise.yaml`: {len(V)} videos, {len(R)} reps. Template = "
      "the exercise's PT demo (M1; E03 both arms over the 90/90 working phase, M3), per-view bias and bands from the "
      "≥ 95 experts fitted per exercise and GroupKFold-5 fold on the training subjects (the per-exercise models' "
      "folds, T17), posture checks centred on those experts (M14), speed from expert data (M5a); E06's wrist angle "
      "from MediaPipe Hands in the side views only, centred on the side-view experts (P1c, M17); channels, alignment "
      "segments and rules per exercise in `configs/exercises/EXX.yaml` (M16, M17; rules TODO: verify against the "
      "EAAQs). Video level, within groups; 95% CIs from the subject bootstrap "
      f"({n_boot} draws). ⚠ = fewer than {low} subjects (low confidence, T17).")
    w("")
    w("## 1. Agreement of the set score with the physiotherapists (Spearman ρ [95% CI])")
    w("")
    w("| Stratum | subjects / videos | mean set score | total score | MQ | items that vary (T14) |")
    w("|---|---|---|---|---|---|")
    for (ex, g), d in AG.groupby(["exercise", "group"], sort=True):
        r0 = d.iloc[0]
        tot, mq = d[d.target == "score"].iloc[0], d[d.target == "MQ"].iloc[0]
        items = d[~d.target.isin(["score", "MQ"])]
        ms = V[(V.exercise == ex) & (V.group == g)].ref_set_score.mean()
        w(f"| {ex} {g}{' ⚠' if r0.low_confidence else ''} | {r0.subjects} / {r0.videos} | {fmt(ms, 1)} | "
          f"{fci(tot.spearman, tot.spearman_lo, tot.spearman_hi)} | {fci(mq.spearman, mq.spearman_lo, mq.spearman_hi)} | "
          + ("; ".join(f"{x.target} {fmt(x.spearman)}" for x in items.itertuples()) or "–") + " |")
    w("")
    w("## 2. As a predictor: out-of-fold within-group line vs the group-only baseline")
    w("")
    w("RMSE of the line (total ~ set score, MQ ~ set score) − RMSE of the within-group training mean [95% CI]; "
      "negative = the Module III score adds information.")
    w("")
    w("| Stratum | total: line / group mean / Δ | MQ: line / group mean / Δ |")
    w("|---|---|---|")
    for (ex, g), d in PR.groupby(["exercise", "group"], sort=True):
        cells = []
        for t in ("score", "MQ"):
            r = d[d.target == t].iloc[0]
            s = fci(r.delta, r.delta_lo, r.delta_hi)
            cells.append(f"{fmt(r.rmse_line)} / {fmt(r.rmse_group_mean)} / {'**' + s + '**' if r.delta_hi < 0 or r.delta_lo > 0 else s}")
        w(f"| {ex} {g}{' ⚠' if d.low_confidence.iloc[0] else ''} | " + " | ".join(cells) + " |")
    w("")
    w("## 3. Expert vs non-expert separation (context, not a within-group result)")
    w("")
    w("| Exercise | subset | videos | AUC [95% CI] | mean set score experts / non-experts |")
    w("|---|---|---|---|---|")
    for r in SEP.itertuples():
        w(f"| {r.exercise} | {r.subset} | {r.videos} | {fci(r.auc, r.auc_lo, r.auc_hi)} | "
          f"{fmt(r.score_expert, 1)} / {fmt(r.score_non_expert, 1)} |")
    w("")
    w("## 4. Rule firing rates (share of reps)")
    w("")
    w("Specificity check: a useful rule fires more often in the experts' lower-scoring videos (< 80) than in their "
      "best (≥ 95), and more often in non-experts.")
    w("")
    w("| Exercise | rule | experts ≥ 95 | experts < 80 | experts all | non-experts |")
    w("|---|---|---|---|---|---|")
    for r in RR.itertuples():
        w(f"| {r.exercise} | {r.rule} | {fmt(100 * r.experts_ge95, 0)}% | {fmt(100 * r.experts_lt80, 0)}% | "
          f"{fmt(100 * r.experts_all, 0)}% | {fmt(100 * r.non_experts, 0) + '%' if np.isfinite(r.non_experts) else '–'} |")
    w("")
    w("## 5. View stability over synced expert performances (rep level, ICC(1) [95% CI])")
    w("")
    w("| Exercise | performances | Module III score | score without speed | end-range angle |")
    w("|---|---|---|---|---|")
    for ex, d in VS.groupby("exercise"):
        cells = [fci(*d[d.measure == m][["icc", "icc_lo", "icc_hi"]].iloc[0].to_numpy()) for m in
                 ("Module III score", "score without speed", "end-range angle")]
        w(f"| {ex} | {int(d.perfs.iloc[0])} | " + " | ".join(cells) + " |")
    w("")
    w("Synced views share their rep boundaries (P6b), so the speed group is identical across views by construction; "
      "the score without speed is the fair comparison.")
    w("")
    w("Tables: `agreement.csv`, `predictor.csv`, `separation.csv`, `rule_rates.csv`, `view_stability.csv`; per rep "
      "`module3_reps.csv`, per video `module3_videos.csv` and `json/`.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
