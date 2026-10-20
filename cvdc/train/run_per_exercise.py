"""One model per exercise (supervisor decision, T17): for every exercise E01-E09 separately, train and test on that
exercise's videos only, with GroupKFold-5 over its subjects (stratified by expert / non-expert where both exist),
the inner validation split and the CVDC pretraining inside each fold on its training subjects only.

Methods (``eval.per_exercise_methods``): B1, B2-KTPFormer, Proposed (CVDC pretraining on the exercise's multi-view
rep sets + full fine-tuning with L_sup + 1.0 L_cvdc) and the frozen-encoder Proposed (the reported alternative,
sharing the fold's CVDC encoder). No exercise head and no exercise embedding (``configs/per_exercise.yaml``);
every other setting as the pilot's step (a2) (T6, T7, T8a, T8b, T9-T15). Each (exercise, method, fold) is
``cvdc.train.run.run_method_fold`` on the full packs with the item selection scoped to the exercise
(``PackData.set_scope``), so item ids, CVDC pairs and saved embeddings stay global.

Runs: runs/full/<tag>/<EX>/<method>/fold<k>/ (as cvdc.train.run); finished folds (done.json) are skipped unless
--force. One process, one GPU job at a time.

usage: python -m cvdc.train.run_per_exercise [--config configs/per_exercise.yaml] [--exercises E01 ...]
       [--methods b1 b2_ktpformer proposed_j1 proposed_lp] [--folds 0 1 ...] [--tag perex_gkf5] [--smoke] [--force]
"""
import argparse
import time

from ..config import load_config, resolve
from ..data.dataset import PackData
from .run import run_method_fold
from .trainer import fold_plan, get_device


def exercise_plans(cfg, data, exercise, smoke=False):
    """GroupKFold plans over the subjects with reps in ``exercise`` (stratified by group, T17), with the plan's
    exercise recorded."""
    meta = data.meta[data.meta.exercise == exercise]
    plans = fold_plan(cfg, meta, smoke=smoke)
    return [{**p, "exercise": exercise} for p in plans]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--model-config")
    ap.add_argument("--exercises", nargs="+")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--tag", default="perex_gkf5")
    ap.add_argument("--smoke", action="store_true", help="fold 0 per exercise, 3 training subjects, 2 epochs")
    ap.add_argument("--force", action="store_true", help="rerun folds that have done.json")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    methods = a.methods or cfg["eval"]["per_exercise_methods"]
    unknown = set(methods) - set(cfg["methods"])
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}")
    if "exercise" in cfg["heads"]["targets_rep"]:
        raise SystemExit("per-exercise models have no exercise head (T17): use configs/per_exercise.yaml")
    lift = sorted({cfg["methods"][m]["input"].split("_", 1)[1] for m in methods if "_" in cfg["methods"][m]["input"]})
    if len(lift) > 1:
        raise SystemExit(f"one lifter per run: {lift}")
    dev = get_device(cfg)
    data = PackData(cfg, cfg["train"]["unit"], dev, lift=lift[0] if lift else None)
    exercises = a.exercises or sorted(data.meta.exercise.unique())
    tag = a.tag + ("_smoke" if a.smoke else "")
    root = resolve(cfg["train"]["runs_dir"]) / tag
    epochs = cfg["train"]["smoke"]["epochs"] if a.smoke else None
    print(f"device {dev}, {len(data)} reps, exercises {exercises}, methods {methods}, runs in {root}", flush=True)
    t0 = time.time()
    for ex in exercises:
        data.set_scope([ex])
        plans = exercise_plans(cfg, data, ex, smoke=a.smoke)
        if a.folds is not None and not a.smoke:
            plans = [p for p in plans if p["fold"] in a.folds]
        for method in methods:
            for p in plans:
                out = root / ex / method / f"fold{p['fold']}"
                if (out / "done.json").exists() and not a.force:
                    print(f"skip {ex} {method} fold {p['fold']} (done)", flush=True)
                    continue
                print(f"\n=== {ex} {method} fold {p['fold']}: {len(p['inner_train'])} inner-train, "
                      f"{len(p['inner_val'])} inner-val, {len(p['test'])} test subjects ===", flush=True)
                d = run_method_fold(cfg, data, method, p, out, None, epochs, log=lambda *x: print(*x, flush=True))
                print(f"=== {ex} {method} fold {p['fold']} done in {d['wall_s'] / 60:.1f} min "
                      f"(peak GPU {d['peak_gpu_gb'] or float('nan'):.2f} GB; {(time.time() - t0) / 3600:.2f} h "
                      "since start) ===", flush=True)
        print(f"=== exercise {ex} finished ({(time.time() - t0) / 3600:.2f} h since start) ===", flush=True)
    data.set_scope(None)


if __name__ == "__main__":
    main()
