"""Runtime estimate for the per-exercise plan (T17; nothing is trained): (1-2) GroupKFold-5 per exercise for B1,
B2-KTPFormer, Proposed (CVDC pretraining + lambda 1.0) and the frozen Proposed; (6) the per-exercise cross-view
protocol (3 rotations x 5 folds) for B1, B2-KTPFormer and Proposed; (7) LOSO per exercise (not run).

Measured inputs (one GPU job at a time, RTX 4060 Ti, bf16):
  supervised   seconds per epoch per inner-training rep (validation included), per method, from the pilot's step
               (a2) runs (runs/phase3a2/<method>/fold*/finetune/history.csv, run_info.json); B2-KTPFormer and the
               frozen variant measured the same way; epochs run until early stopping (T8b) from the same runs
  CVDC         seconds per epoch per inner-training rep set from the pilot's step (a) pretraining
               (runs/phase3a/proposed/fold*/cvdc), epochs run (T8a) from the same runs
  overhead     per run: done.json wall time minus its stages (embedding snapshots, predictions)
  check        the aborted shared-model fold (runs/full/aborted_shared_model/full_gkf10/b1/fold0) gives the
               full-scope B1 cost per rep for comparison
Counts are exact: every exercise's GroupKFold-5 plans (inner training reps and rep sets), cross-view plans (two
training views) and LOSO plans (the T8 inner split of n - 1 subjects). Epochs: the pilot's mean (central), its
minimum (low) and the cap (high: 60 supervised, 100 CVDC).

usage: python scripts/phase3_full/runtime_estimate_perex.py [--config configs/per_exercise.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.eval.splits import loso  # noqa: E402
from cvdc.train.trainer import fold_plan, inner_split  # noqa: E402

PILOT = {"b1": "b1", "b2_ktpformer": "b2_ktpformer", "proposed_j1": "proposed_j1", "proposed_lp": "proposed_lp"}


def sup_costs(root):
    rows = []
    for m, pm in PILOT.items():
        for f in sorted((root / "phase3a2" / pm).glob("fold*/done.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            ri = json.loads((f.parent / "finetune" / "run_info.json").read_text(encoding="utf-8"))
            h = pd.read_csv(f.parent / "finetune" / "history.csv")
            stage = d["supervised_stage"]["wall_s"] + (d.get("cvdc_stage", {}).get("wall_s", 0)
                                                       if not d.get("cvdc_stage", {}).get("reused_from") else 0)
            rows.append(dict(method=m, fold=d["fold"], s_per_rep_epoch=h.epoch_s.median() / ri["n_train_items"],
                             epochs=len(h), overhead_s=d["wall_s"] - stage))
    return pd.DataFrame(rows)


def cvdc_costs(root):
    rows = []
    for f in sorted((root / "phase3a" / "proposed").glob("fold*/cvdc/history.csv")):
        h = pd.read_csv(f)
        ri = json.loads((f.parent / "run_info.json").read_text(encoding="utf-8"))
        rows.append(dict(fold=ri["fold"], s_per_set_epoch=h.epoch_s.dropna().median() / ri["batcher_train"]["sets"],
                         epochs=len(h)))
    return pd.DataFrame(rows)


def counts(cfg, reps, members):
    """Per exercise: inner-training reps and rep sets summed over the folds of each protocol."""
    set_of = members.set_index("item").set_id
    reps = reps.assign(item=np.arange(len(reps)), set_id=set_of.reindex(np.arange(len(reps))).to_numpy())
    es = cfg["train"]["early_stopping"]
    out = []
    for ex, r in reps.groupby("exercise"):
        plans = fold_plan(cfg, r)
        n_sub = r.subject.nunique()
        gk_reps = sum(r.subject.isin(p["inner_train"]).sum() for p in plans)
        gk_sets = sum(r[r.subject.isin(p["inner_train"])].set_id.nunique() for p in plans)
        cv_reps = cv_sets = 0
        for held in "FLR":
            keep = r.view != held
            for p in plans:
                sel = r.subject.isin(p["inner_train"]) & keep
                cv_reps += sel.sum()
                cv_sets += r[sel].set_id.nunique()
        lo_reps = lo_sets = 0
        for f, (tr, te) in enumerate(loso(r)):
            itr, _ = inner_split(r, tr, es["inner_frac"], cfg["eval"]["seed"] + es["seed_offset"] + f)
            lo_reps += r.subject.isin(itr).sum()
            lo_sets += r[r.subject.isin(itr)].set_id.nunique()
        out.append(dict(exercise=ex, subjects=n_sub, experts=r[r.group == "expert"].subject.nunique(),
                        non_experts=r[r.group != "expert"].subject.nunique(), reps=len(r),
                        gkf_inner_train_reps=int(gk_reps), gkf_inner_train_sets=int(gk_sets),
                        cv_inner_train_reps=int(cv_reps), cv_inner_train_sets=int(cv_sets),
                        loso_folds=n_sub, loso_inner_train_reps=int(lo_reps), loso_inner_train_sets=int(lo_sets)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    runs = resolve("runs")
    S, Cv = sup_costs(runs), cvdc_costs(runs)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    N = counts(cfg, reps, members)
    ab = runs / "full" / "aborted_shared_model" / "full_gkf10" / "b1" / "fold0" / "finetune"
    if ab.exists():
        h = pd.read_csv(ab / "history.csv")
        ri = json.loads((ab / "run_info.json").read_text(encoding="utf-8"))
        print(f"check: full-scope B1 (aborted fold 0) {h.epoch_s.median() / ri['n_train_items'] * 1e3:.3f} ms per rep "
              f"and epoch vs pilot B1 {S[S.method == 'b1'].s_per_rep_epoch.mean() * 1e3:.3f} ms")
    cost = S.groupby("method").agg(s_per_rep_epoch=("s_per_rep_epoch", "mean"), ep_mean=("epochs", "mean"),
                                   ep_min=("epochs", "min"), overhead_s=("overhead_s", "mean"))
    cc = dict(s=Cv.s_per_set_epoch.mean(), ep_mean=Cv.epochs.mean(), ep_min=Cv.epochs.min())
    print("measured supervised costs (pilot step a2):")
    print((cost.assign(ms_per_rep_epoch=cost.s_per_rep_epoch * 1e3).drop(columns="s_per_rep_epoch")).round(2).to_string())
    print(f"measured CVDC: {cc['s'] * 1e3:.2f} ms per rep set and epoch, epochs run mean {cc['ep_mean']:.0f} "
          f"(min {cc['ep_min']})")
    print(N.to_string(index=False))

    def sup_h(m, reps_epoch, runs_n, level):
        ep = {"central": cost.loc[m, "ep_mean"], "low": cost.loc[m, "ep_min"], "high": 60}[level]
        return (reps_epoch * cost.loc[m, "s_per_rep_epoch"] * ep + runs_n * cost.loc[m, "overhead_s"]) / 3600

    def cvdc_h(sets_epoch, level):
        ep = {"central": cc["ep_mean"], "low": cc["ep_min"], "high": 100}[level]
        return sets_epoch * cc["s"] * ep / 3600

    rows = []
    for _, r in N.iterrows():
        for level in ("central", "low", "high"):
            d = dict(exercise=r.exercise, level=level)
            k = 5
            d["gkf_b1"] = sup_h("b1", r.gkf_inner_train_reps, k, level)
            d["gkf_b2_ktpformer"] = sup_h("b2_ktpformer", r.gkf_inner_train_reps, k, level)
            d["gkf_cvdc"] = cvdc_h(r.gkf_inner_train_sets, level)
            d["gkf_proposed_j1"] = sup_h("proposed_j1", r.gkf_inner_train_reps, k, level)
            d["gkf_proposed_lp"] = sup_h("proposed_lp", r.gkf_inner_train_reps, k, level)
            d["cv_b1"] = sup_h("b1", r.cv_inner_train_reps, 3 * k, level)
            d["cv_b2_ktpformer"] = sup_h("b2_ktpformer", r.cv_inner_train_reps, 3 * k, level)
            d["cv_cvdc"] = cvdc_h(r.cv_inner_train_sets, level)
            d["cv_proposed_j1"] = sup_h("proposed_j1", r.cv_inner_train_reps, 3 * k, level)
            n = r.loso_folds
            d["loso_b1"] = sup_h("b1", r.loso_inner_train_reps, n, level)
            d["loso_b2_ktpformer"] = sup_h("b2_ktpformer", r.loso_inner_train_reps, n, level)
            d["loso_proposed_j1"] = sup_h("proposed_j1", r.loso_inner_train_reps, n, level)
            d["loso_proposed_lp"] = sup_h("proposed_lp", r.loso_inner_train_reps, n, level)
            d["loso_cvdc_without_shortcut"] = cvdc_h(r.loso_inner_train_sets, level)
            rows.append(d)
    E = pd.DataFrame(rows)
    E["gkf_total"] = E.filter(like="gkf_").sum(1)
    E["cv_total"] = E[[c for c in E if c.startswith("cv_") and c != "cv_total"]].sum(1)
    E["loso_total_shortcut"] = E[["loso_b1", "loso_b2_ktpformer", "loso_proposed_j1", "loso_proposed_lp"]].sum(1)
    E["loso_total_no_shortcut"] = E.loso_total_shortcut + E.loso_cvdc_without_shortcut
    out = resolve(cfg["eval"]["results_dir"]) / "perex_runtime_estimate"
    out.mkdir(parents=True, exist_ok=True)
    E.to_csv(out / "runtime_estimate.csv", index=False)
    N.to_csv(out / "counts.csv", index=False)
    cost.to_csv(out / "measured_costs.csv")
    tot = E.groupby("level")[[c for c in E if c not in ("exercise", "level")]].sum().T.round(2)
    print("\nhours (sum over the nine exercises):")
    print(tot[["low", "central", "high"]].to_string())
    print("\ncentral hours per exercise:")
    print(E[E.level == "central"].set_index("exercise")[["gkf_total", "cv_total", "loso_total_shortcut",
                                                          "loso_total_no_shortcut"]].round(2).to_string())


if __name__ == "__main__":
    main()
