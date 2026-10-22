"""LOSO runtime estimate for the 4 main methods (E2, E14 step c) with early stopping and the per-fold-encoder shortcut
(C7). Nothing is trained.

Measured inputs (RTX 4060 Ti, bf16, one process at a time; Phase 5 found two concurrent GPU processes slower than
running them in sequence):
  - per-epoch time of the supervised stage per inner-training rep, from the step (a2) GroupKFold-5 runs of each method
    (runs/phase3a2/<method>/fold<k>/finetune/history.csv: median epoch_s / n_train_items), which already include the
    per-epoch validation pass and, for Proposed (joint loss, lambda 1.0), one CVDC batch per supervised step;
  - epochs run until early stopping (T8b: min 10, patience 15, max 60) in the same runs: mean (central), min and
    the 60-epoch cap (upper bound);
  - per-fold overhead outside the training loop (embedding snapshots, test predictions): done.json wall_s minus the
    stage times;
  - CVDC pretraining (only for the full-LOSO reference): epoch time per inner-training rep set and epochs run in the
    step (a) pretraining (runs/phase3a/proposed/fold<k>/cvdc).
LOSO folds (E1): the 50 subjects with >= 1 rep; per fold the inner split follows the same rule as GroupKFold (T8: 20%
of each group's training subjects, at least 1) and its inner-training rep count scales the epoch time.
Per-fold-encoder shortcut (C7): Proposed fine-tunes from the step (a) CVDC encoder of the GroupKFold fold in which
the LOSO subject was held out, so no CVDC pretraining is run for LOSO (the 5 encoders exist).

usage: python scripts/pilot/phase5/loso_estimate.py
Writes outputs/results/phase5_loso_estimate/{per_fold.csv, estimate.csv, estimate.md}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.eval.splits import fold_of, group_kfold, loso  # noqa: E402
from cvdc.train.trainer import inner_split  # noqa: E402

METHODS = {"b1": "B1", "b2_mediapipe": "B2-MediaPipe", "b2_ktpformer": "B2-KTPFormer", "proposed_j1": "Proposed (λ 1.0)"}


def main():
    cfg = load_config()
    es = cfg["train"]["early_stopping"]
    runs = resolve(cfg["train"]["runs_dir"])
    out = resolve(cfg["eval"]["results_dir"]) / "phase5_loso_estimate"
    out.mkdir(parents=True, exist_ok=True)
    reps = pd.read_csv(resolve("data/reps.csv"))
    meta = reps[["subject", "group"]]
    n_reps = reps.subject.value_counts()

    # measured per method on the step (a2) GroupKFold-5 runs
    meas = {}
    for m in METHODS:
        per_rep, stopped, overhead = [], [], []
        for f in range(cfg["eval"]["k"]):
            d = runs / "phase3a2" / m / f"fold{f}"
            h = pd.read_csv(d / "finetune" / "history.csv")
            info = json.loads((d / "finetune" / "run_info.json").read_text(encoding="utf-8"))
            done = json.loads((d / "done.json").read_text(encoding="utf-8"))
            per_rep.append(h.epoch_s.median() / info["n_train_items"])
            stopped.append(done["supervised_stage"]["stopped_epoch"])
            overhead.append(done["wall_s"] - done["supervised_stage"]["wall_s"] - done.get("cvdc_stage", {}).get("wall_s", 0)
                            * (not done.get("cvdc_stage", {}).get("reused_from")))
        meas[m] = dict(s_per_rep_epoch=float(np.median(per_rep)), epochs_mean=float(np.mean(stopped)),
                       epochs_min=int(min(stopped)), epochs_cap=int(cfg["train"]["max_epochs_sup"]),
                       overhead_s=float(np.median(overhead)), stopped=stopped)
    cv = []  # CVDC pretraining (full-LOSO reference only)
    for f in range(cfg["eval"]["k"]):
        d = runs / "phase3a" / "proposed" / f"fold{f}" / "cvdc"
        h = pd.read_csv(d / "history.csv")
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        cv.append((h.epoch_s.median() / info["batcher_train"]["sets"], s["stopped_epoch"]))
    cvdc_s_per_set, cvdc_epochs = float(np.median([x for x, _ in cv])), float(np.mean([e for _, e in cv]))

    # LOSO folds: the 50 subjects with >= 1 rep; inner split as for GroupKFold (T8)
    gk = group_kfold(meta, cfg["eval"]["k"], cfg["eval"]["seed"])
    enc_fold = fold_of(gk)
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    sets_per_subject = members.groupby("subject").set_id.nunique()
    rows = []
    for i, (tr, te) in enumerate(loso(meta)):
        itr, iva = inner_split(meta, tr, es["inner_frac"], cfg["eval"]["seed"] + es["seed_offset"] + i)
        n_itr = int(n_reps.reindex(itr).sum())
        row = dict(fold=i, test_subject=te[0], encoder_from_groupkfold_fold=enc_fold[te[0]], inner_train_subjects=len(itr),
                   inner_val_subjects=len(iva), inner_train_reps=n_itr,
                   inner_train_sets=int(sets_per_subject.reindex(itr).sum()))
        for m, x in meas.items():
            t_ep = x["s_per_rep_epoch"] * n_itr
            row[f"{m}_epoch_s"] = t_ep
            for k in ("epochs_mean", "epochs_min", "epochs_cap"):
                row[f"{m}_{k[7:]}_s"] = t_ep * x[k] + x["overhead_s"]
        row["cvdc_pretrain_s"] = cvdc_s_per_set * row["inner_train_sets"] * cvdc_epochs
        rows.append(row)
    F = pd.DataFrame(rows)
    F.to_csv(out / "per_fold.csv", index=False)

    est = []
    for m, lab in METHODS.items():
        x = meas[m]
        est.append(dict(method=lab, folds=len(F), s_per_epoch_median=F[f"{m}_epoch_s"].median(),
                        epochs_measured=", ".join(map(str, x["stopped"])), epochs_mean=x["epochs_mean"],
                        hours_central=F[f"{m}_mean_s"].sum() / 3600, hours_low=F[f"{m}_min_s"].sum() / 3600,
                        hours_cap=F[f"{m}_cap_s"].sum() / 3600, cvdc_pretraining_h=0.0))
    E = pd.DataFrame(est)
    tot = dict(method="**4 main methods (sequential)**", folds=len(F), s_per_epoch_median=np.nan, epochs_measured="",
               epochs_mean=np.nan, hours_central=E.hours_central.sum(), hours_low=E.hours_low.sum(),
               hours_cap=E.hours_cap.sum(), cvdc_pretraining_h=0.0)
    full = dict(method="reference: Proposed without the shortcut (CVDC pretraining per LOSO fold)", folds=len(F),
                s_per_epoch_median=np.nan, epochs_measured=f"CVDC {cvdc_epochs:.0f} (mean, step a)", epochs_mean=np.nan,
                hours_central=(F.proposed_j1_mean_s.sum() + F.cvdc_pretrain_s.sum()) / 3600, hours_low=np.nan,
                hours_cap=np.nan, cvdc_pretraining_h=F.cvdc_pretrain_s.sum() / 3600)
    E = pd.concat([E, pd.DataFrame([tot, full])], ignore_index=True)
    E.to_csv(out / "estimate.csv", index=False)
    L = ["# LOSO runtime estimate (4 main methods, early stopping, per-fold-encoder shortcut)", "",
         f"{len(F)} LOSO folds (subjects with ≥ 1 rep), inner split as T8 (median {F.inner_train_subjects.median():.0f} "
         f"inner-training / {F.inner_val_subjects.median():.0f} inner-validation subjects, median "
         f"{F.inner_train_reps.median():.0f} inner-training reps per fold vs 4,185–4,835 in the GroupKFold runs). "
         "Epoch time per inner-training rep and the epochs run until early stopping are measured on the step (a2) "
         "GroupKFold-5 runs; one process at a time.", "",
         "| Method | s / epoch (median fold) | epochs run in step (a2) | central (mean epochs) | low (fewest) | cap (60 epochs) |",
         "|---|---|---|---|---|---|"]
    for r in E.itertuples():
        if r.method.startswith("reference"):
            continue
        L.append(f"| {r.method} | {'' if np.isnan(r.s_per_epoch_median) else f'{r.s_per_epoch_median:.1f}'} | "
                 f"{r.epochs_measured} | {r.hours_central:.1f} h | {r.hours_low:.1f} h | {r.hours_cap:.1f} h |")
    L += ["", f"Without the shortcut, Proposed would also pretrain CVDC in every LOSO fold: +{full['cvdc_pretraining_h']:.1f} h "
          f"(→ {full['hours_central']:.1f} h for Proposed alone)."]
    (out / "estimate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
