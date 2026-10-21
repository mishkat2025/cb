"""Runtime estimate of the LOSO job (before launching; E30), from the LOSO smoke test and the GroupKFold-5 runs.

Per exercise and method:
  s / epoch    measured in the LOSO smoke runs (full LOSO training sets, 2 epochs): the second epoch's epoch_s (the
               first carries start-up costs), median over the smoke folds
  overhead     per run, smoke wall time minus its epochs (random-init / encoder snapshots, predictions, set-up)
  epochs       epochs run until early stopping in the GroupKFold-5 runs of the same exercise and method (T8b):
               central = mean, low = minimum, high = the 60-epoch cap
  per run      overhead + epochs x s / epoch;  total = LOSO folds x per run
Cross-check: the GroupKFold-5 s / epoch scaled by the ratio of inner-training reps (LOSO / GroupKFold).

usage: python scripts/loso/runtime_estimate.py [--smoke-tag perex_loso_smoke] [--gkf-tag perex_gkf5]
Writes <eval.results_dir>/perex_loso_estimate/runtime_estimate.csv and summary.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402


def runs(root, ex, m):
    out = []
    for d in sorted((root / ex / m).glob("fold*/done.json")):
        dj = json.loads(d.read_text(encoding="utf-8"))
        h = pd.read_csv(d.parent / "finetune" / "history.csv")
        ri = json.loads((d.parent / "finetune" / "run_info.json").read_text(encoding="utf-8"))
        out.append(dict(wall=dj["wall_s"], epoch_s=h.epoch_s.to_numpy(float), stopped=dj["supervised_stage"]["stopped_epoch"],
                        n_train=ri["n_train_items"], cap=ri["max_epochs"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise_loso.yaml")
    ap.add_argument("--smoke-tag", default="perex_loso_smoke")
    ap.add_argument("--gkf-tag", default="perex_gkf5")
    ap.add_argument("--loso-tag", default="perex_loso")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(REPO / a.config)
    lc = cfg["eval"]["loso"]
    rr = resolve(cfg["train"]["runs_dir"])
    folds = pd.read_csv(rr / a.loso_tag / "folds.csv")  # written by `python -m cvdc.train.run_loso --dry-run`
    out = resolve(cfg["eval"]["results_dir"]) / "perex_loso_estimate"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ex in lc["exercises"]:
        nf = int((folds.exercise == ex).sum())
        for m in lc["methods"]:
            sm, gk = runs(rr / a.smoke_tag, ex, m), runs(rr / a.gkf_tag, ex, m)
            s_ep = float(np.median([r["epoch_s"][-1] for r in sm]))
            over = float(np.median([r["wall"] - r["epoch_s"].sum() for r in sm]))
            ep = np.array([r["stopped"] for r in gk], float)
            cap = gk[0]["cap"]
            gk_rate = np.median([np.median(r["epoch_s"][1:]) / r["n_train"] for r in gk])  # s / epoch / inner-train rep
            n_loso = float(np.median([r["n_train"] for r in sm]))
            per = over + ep.mean() * s_ep
            rows.append(dict(exercise=ex, method=m, folds=nf, s_per_epoch=s_ep, s_per_epoch_from_gkf=gk_rate * n_loso,
                             loso_inner_train_reps=n_loso,
                             gkf_inner_train_reps=float(np.median([r["n_train"] for r in gk])), overhead_s=over,
                             epochs_mean=ep.mean(), epochs_min=ep.min(), epochs_cap=cap, per_run_s=per,
                             total_h=nf * per / 3600, low_h=nf * (over + ep.min() * s_ep) / 3600,
                             high_h=nf * (over + cap * s_ep) / 3600,
                             gkf_wall_per_fold_s=float(np.mean([r["wall"] for r in gk]))))
    E = pd.DataFrame(rows)
    E.to_csv(out / "runtime_estimate.csv", index=False)
    L = ["# LOSO runtime estimate (E01–E04, 4 methods, per-fold-encoder shortcut C7)", "",
         "From the LOSO smoke test (full LOSO training sets, 2 epochs: s / epoch = 2nd epoch, overhead = wall − epochs) "
         "and the GroupKFold-5 runs (epochs until early stopping: mean / minimum / 60-epoch cap). One GPU job at a time.",
         "", "| Exercise | method | folds | s / epoch (smoke; from GKF) | overhead s | epochs mean (min) | per run min | "
         "total h (low–high) |", "|---|---|---|---|---|---|---|---|"]
    for r in E.itertuples():
        L.append(f"| {r.exercise} | {r.method} | {r.folds} | {r.s_per_epoch:.2f}; {r.s_per_epoch_from_gkf:.2f} | "
                 f"{r.overhead_s:.0f} | {r.epochs_mean:.1f} ({r.epochs_min:.0f}) | {r.per_run_s / 60:.1f} | "
                 f"{r.total_h:.2f} ({r.low_h:.2f}–{r.high_h:.2f}) |")
    L += ["", "| Exercise | total h (low–high) |", "|---|---|"]
    for ex, d in E.groupby("exercise"):
        L.append(f"| {ex} | {d.total_h.sum():.2f} ({d.low_h.sum():.2f}–{d.high_h.sum():.2f}) |")
    L.append(f"| **all** | **{E.total_h.sum():.2f}** ({E.low_h.sum():.2f}–{E.high_h.sum():.2f}) |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
