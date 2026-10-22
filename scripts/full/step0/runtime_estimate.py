"""Runtime estimates for the full scope from the pilot's measured timings (nothing is trained or extracted).

Scaling rules (all one job at a time, RTX 4060 Ti, bf16, Ryzen 7 7700 with 12 MediaPipe processes):
  * Phase 1 extraction: frames / 315 fps (pilot: 712,585 frames in 37.6 min, 12 CPU processes); E06 adds a
    MediaPipe Hands pass, assumed to cost as much as the pose pass per frame (not measured: no hand model yet).
    KTPFormer lifting (B2): frames / 850 fps on the GPU (A10).
  * Training: wall time proportional to the number of training reps with the pilot's epochs to early stopping
    (epoch cost is linear in reps: one pass over the inner-training reps, ~5 s per epoch per 5,568 reps for B1).
    Pilot reps = 7,095 (E01 + E02). New reps per exercise = expert videos x the sample's median rep count of the
    best rep signal x 0.9 (the pilot lost ~10% edge reps to identical rep sets, P6b) + non-expert videos x the
    sample's median non-expert count.
  * Main GroupKFold-5 run = the 4 main methods + CVDC pretraining as in step (a2) (pilot 2.10 h).
    Cross-view protocol = pilot 3.9 h x the rep factor. LOSO = pilot per-rep epoch cost x epochs (E26) x folds
    (subjects with >= 1 rep) x inner-training reps (80% of the other subjects' reps), with the per-fold-encoder
    shortcut (C7).

usage: python -I scripts/full/step0/runtime_estimate.py --out outputs/step0_full [--exercises E01 E02 E03 ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = dict(reps=7095, extract_fps=712585 / (37.6 * 60), lift_fps=850.0,
             gkf_h={"B1": 16.8 / 60, "B2-MediaPipe": 15.2 / 60, "B2-KTPFormer": 15.1 / 60, "Proposed (lambda 1.0)": 33.0 / 60,
                    "CVDC pretraining (5 folds)": 46.0 / 60},
             crossview_h=3.9,
             # LOSO (E26): hours_central over 50 folds of median 5,568 inner-training reps
             loso_h={"B1": 3.389, "B2-MediaPipe": 3.078, "B2-KTPFormer": 3.102, "Proposed (lambda 1.0)": 6.634},
             loso_low_h=11.353, loso_cap_h=22.271, loso_folds=50, loso_inner_reps=5568)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/step0_full")
    ap.add_argument("--exercises", nargs="*", default=[f"E0{i}" for i in range(1, 10)])
    ap.add_argument("--hands-cost", type=float, default=1.0, help="hand pass cost relative to the pose pass")
    a = ap.parse_args()
    out = Path(a.out)
    meta = pd.read_csv(out / "video_meta.csv")
    inv = pd.read_csv(out / "inventory.csv")
    rc = pd.read_csv(out / "rep_counts_sample.csv")
    best = pd.read_csv(out / "rep_signal_agreement.csv").groupby("exercise").head(1).set_index("exercise").signal
    meta["frames_est"] = meta.duration_s * meta.fps
    rows = []
    for ex in a.exercises:
        m = meta[meta.exercise == ex]
        g = inv[inv.exercise == ex]
        n_exp, n_non = int((g.group == "expert").sum()), int((g.group == "non-expert").sum())
        if ex in ("E01", "E02"):
            reps = {"E01": 3578, "E02": 3517}[ex]   # pilot training reps (data/reps.csv): measured, not estimated
            per_exp = np.nan
        else:
            r = rc[(rc.exercise == ex) & (rc.signal == best.get(ex))]
            per_exp = float(r[r.group == "expert"].n.median())
            per_non = float(r[r.group == "non-expert"].n.median()) if n_non else 0.0
            reps = int(round(n_exp * per_exp * 0.9 + n_non * per_non))
        rows.append(dict(exercise=ex, videos=len(g), frames=int(m.frames_est.sum()), reps_est=reps,
                         reps_per_expert_video=per_exp, subjects=g.subject.nunique()))
    T = pd.DataFrame(rows)
    new = T[~T.exercise.isin(["E01", "E02"])]
    frames_new = int(new.frames.sum())
    e06 = int(T.loc[T.exercise == "E06", "frames"].sum())
    reps_all = int(T.reps_est.sum())
    k = reps_all / PILOT["reps"]
    subjects = inv[inv.exercise.isin(a.exercises)].subject.nunique()
    inner = 0.8 * reps_all * (1 - 1 / subjects)
    k_loso = (subjects * inner) / (PILOT["loso_folds"] * PILOT["loso_inner_reps"])
    est = [
        ("Phase 1 pose extraction, new exercises only (E01/E02 cached)", frames_new / PILOT["extract_fps"] / 3600),
        ("  + MediaPipe Hands pass for E06 (assumed = pose cost)", a.hands_cost * e06 / PILOT["extract_fps"] / 3600),
        ("  KTPFormer lifting of the new frames (GPU, B2)", frames_new / PILOT["lift_fps"] / 3600),
        ("Main GroupKFold-5 run (4 methods + CVDC pretraining)", sum(PILOT["gkf_h"].values()) * k),
        ("Cross-view protocol (3 rotations x 5 folds x 4 methods)", PILOT["crossview_h"] * k),
        ("LOSO, 4 methods, per-fold-encoder shortcut (central)", sum(PILOT["loso_h"].values()) * k_loso),
        ("  LOSO range (fastest pilot folds .. 60-epoch cap)", (PILOT["loso_low_h"] * k_loso, PILOT["loso_cap_h"] * k_loso)),
    ]
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    print(f"\nnew frames {frames_new:,}; all frames {int(T.frames.sum()):,}; estimated training reps {reps_all:,} "
          f"(pilot {PILOT['reps']:,}; factor {k:.2f}); LOSO folds = {subjects} subjects, inner-training reps "
          f"~{inner:,.0f} per fold (factor {k_loso:.2f} vs pilot LOSO)")
    print("\nper-method GroupKFold-5 (h):", {m: round(h * k, 1) for m, h in PILOT["gkf_h"].items()})
    print("per-method LOSO (h):", {m: round(h * k_loso, 1) for m, h in PILOT["loso_h"].items()})
    for name, h in est:
        print(f"{name:62s} " + (f"{h[0]:.1f} - {h[1]:.1f} h" if isinstance(h, tuple) else f"{h:.1f} h"))
    T.to_csv(out / "runtime_inputs.csv", index=False)
    pd.DataFrame([dict(item=n, hours=(f"{h[0]:.1f}-{h[1]:.1f}" if isinstance(h, tuple) else round(h, 2)))
                  for n, h in est]).to_csv(out / "runtime_estimate.csv", index=False)


if __name__ == "__main__":
    main()
