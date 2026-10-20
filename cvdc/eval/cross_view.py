"""Cross-view protocol (spec §9, E24): train on two views, test on the held-out third, rotating F / L / R, on the
GroupKFold-5 folds of Phase 3 (same subjects, inner validation splits, seeds, epoch budget and schedule).

For every held-out view V, method and fold, ``cvdc.train.run.run_method_fold`` runs with the fold's plan restricted
to the two training views (``trainer.with_views``): supervised training and its inner validation, the target
scalers, the video head's count statistics, CVDC pretraining and its validation, the joint loss's CVDC batches and
the embedding centre use items of the training views only. The test subjects keep all three views, so every run
predicts both the held-out view and the seen views of its test subjects. CVDC methods always pretrain afresh on the
two training views: ``train.cvdc_init_from`` is switched off, because the step (a) encoders saw all three views
(``reuse_cvdc`` also refuses an encoder trained on other views).

Runs: runs/<tag>/heldout_<V>/<method>/fold<k>/ (same files as cvdc.train.run; finished folds are skipped unless
--force). Then: scripts/phase5/crossview_distance.py (embedding distance to the PT reference) and
scripts/phase5/report_crossview.py (tables).

One model per exercise (T17; twelfth review item 6): with ``configs/per_exercise.yaml`` (``eval.per_exercise_methods``
set) the protocol runs per exercise, on that exercise's videos and its own GroupKFold-5 folds
(``PackData.set_scope``), for B1, B2-KTPFormer and Proposed (lambda 1.0); runs in
runs/full/<tag>/<EX>/heldout_<V>/<method>/fold<k>/.

usage: python -m cvdc.eval.cross_view [--config C] [--methods b1 b2_mediapipe b2_ktpformer proposed_j1] [--heldout F L R]
       [--folds 0 1 2 3 4] [--exercises E01 ...] [--tag phase5_crossview | perex_crossview] [--smoke] [--force]
"""
import argparse

from ..config import load_config, resolve
from ..data.dataset import LIFTED, PackData
from ..train.run import run_method_fold
from ..train.trainer import fold_plan, get_device, items_of, with_views


def main_methods(cfg):
    """The 4 main methods (E14): B1, B2-MediaPipe, B2-KTPFormer and Proposed (``eval.proposed_default``)."""
    return ["b1", "b2_mediapipe", "b2_ktpformer", cfg["eval"]["proposed_default"]]


def per_exercise_methods(cfg):
    """T17 cross-view methods: B1, B2-KTPFormer and Proposed (twelfth review item 6)."""
    return ["b1", "b2_ktpformer", cfg["eval"]["proposed_default"]]


def crossview_plans(cfg, meta, heldout, smoke=False):
    """GroupKFold plans (Phase 3 folds and inner splits) restricted to the views other than ``heldout``."""
    views = [v for v in cfg["data"]["views"] if v != heldout]
    if len(views) == len(cfg["data"]["views"]):
        raise ValueError(f"held-out view {heldout!r} is not one of data.views {cfg['data']['views']}")
    return [with_views(p, views) for p in fold_plan(cfg, meta, smoke=smoke)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--heldout", nargs="+", help="held-out views (default: data.views)")
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--exercises", nargs="+", help="per-exercise mode only")
    ap.add_argument("--tag", help="default phase5_crossview (pilot) or perex_crossview (per exercise)")
    ap.add_argument("--smoke", action="store_true", help="fold 0 with the smoke subjects, train.smoke.epochs epochs")
    ap.add_argument("--force", action="store_true", help="rerun folds that have done.json")
    ap.add_argument("--device", help="override train.device (e.g. cpu for a smoke test while the GPU trains)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    if a.device:
        cfg["train"]["device"] = a.device
    cfg["train"]["cvdc_init_from"] = None  # E24: never reuse an encoder that saw the held-out view
    per_ex = bool(cfg["eval"].get("per_exercise_methods"))  # T17
    methods = a.methods or (per_exercise_methods(cfg) if per_ex else main_methods(cfg))
    unknown = set(methods) - set(cfg["methods"])
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}; configured: {sorted(cfg['methods'])}")
    lift = sorted({LIFTED[cfg["methods"][m]["input"]] for m in methods if cfg["methods"][m]["input"] in LIFTED})
    if len(lift) > 1:
        raise SystemExit(f"one lifter per run: {lift}")
    dev = get_device(cfg)
    data = PackData(cfg, cfg["train"]["unit"], dev, lift=lift[0] if lift else None)
    tag = (a.tag or ("perex_crossview" if per_ex else "phase5_crossview")) + ("_smoke" if a.smoke else "")
    root = resolve(cfg["train"]["runs_dir"]) / tag
    epochs = cfg["train"]["smoke"]["epochs"] if a.smoke else None
    heldout = a.heldout or cfg["data"]["views"]
    print(f"device {dev}, unit {data.unit}, {len(data)} items, methods {methods}, held-out views {heldout}, "
          f"runs in {root}", flush=True)
    exercises = (a.exercises or sorted(data.meta.exercise.unique())) if per_ex else [None]
    for ex in exercises:
        if ex is not None:
            data.set_scope([ex])
        meta = data.meta if ex is None else data.meta[data.meta.exercise == ex]
        for v in heldout:
            run_rotation(cfg, data, meta, v, ex, methods, root, epochs, a)
    data.set_scope(None)


def run_rotation(cfg, data, meta, v, ex, methods, root, epochs, a):
    """Every method and fold of one held-out view (and exercise in the per-exercise mode)."""
    plans = crossview_plans(cfg, meta, v, smoke=a.smoke)
    if a.folds is not None and not a.smoke:
        plans = [p for p in plans if p["fold"] in a.folds]
    base = root if ex is None else root / ex
    name = f"held-out {v}" if ex is None else f"{ex} held-out {v}"
    for method in methods:
        for p in plans:
            if ex is not None:
                p = {**p, "exercise": ex}
            out = base / f"heldout_{v}" / method / f"fold{p['fold']}"
            if (out / "done.json").exists() and not a.force:
                print(f"skip {name} {method} fold {p['fold']} (done)", flush=True)
                continue
            n = {k: len(items_of(data, p[k], p["views"])) for k in ("inner_train", "inner_val")}
            n_te = len(items_of(data, p["test"], [v]))
            if not n["inner_train"] or not n["inner_val"] or not n_te:
                raise RuntimeError(f"{name} fold {p['fold']}: empty item set ({n}, test {n_te})")
            print(f"\n=== {name} | {method} fold {p['fold']}: training views {p['views']}; "
                  f"{n['inner_train']} inner-train / {n['inner_val']} inner-val reps; {n_te} held-out-view test "
                  f"reps ===", flush=True)
            d = run_method_fold(cfg, data, method, p, out, None, epochs, log=lambda *x: print(*x, flush=True))
            print(f"=== {name} | {method} fold {p['fold']} done in {d['wall_s'] / 60:.1f} min "
                  f"(peak GPU {d['peak_gpu_gb'] or float('nan'):.2f} GB) ===", flush=True)


if __name__ == "__main__":
    main()
