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

usage: python -m cvdc.eval.cross_view [--methods b1 b2_mediapipe b2_ktpformer proposed_j1] [--heldout F L R]
       [--folds 0 1 2 3 4] [--tag phase5_crossview] [--smoke] [--force]
"""
import argparse

from ..config import load_config, resolve
from ..data.dataset import LIFTED, PackData
from ..train.run import run_method_fold
from ..train.trainer import fold_plan, get_device, items_of, with_views


def main_methods(cfg):
    """The 4 main methods (E14): B1, B2-MediaPipe, B2-KTPFormer and Proposed (``eval.proposed_default``)."""
    return ["b1", "b2_mediapipe", "b2_ktpformer", cfg["eval"]["proposed_default"]]


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
    ap.add_argument("--tag", default="phase5_crossview")
    ap.add_argument("--smoke", action="store_true", help="fold 0 with the smoke subjects, train.smoke.epochs epochs")
    ap.add_argument("--force", action="store_true", help="rerun folds that have done.json")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    cfg["train"]["cvdc_init_from"] = None  # E24: never reuse an encoder that saw the held-out view
    methods = a.methods or main_methods(cfg)
    unknown = set(methods) - set(cfg["methods"])
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}; configured: {sorted(cfg['methods'])}")
    lift = sorted({LIFTED[cfg["methods"][m]["input"]] for m in methods if cfg["methods"][m]["input"] in LIFTED})
    if len(lift) > 1:
        raise SystemExit(f"one lifter per run: {lift}")
    dev = get_device(cfg)
    data = PackData(cfg, cfg["train"]["unit"], dev, lift=lift[0] if lift else None)
    tag = a.tag + ("_smoke" if a.smoke else "")
    root = resolve(cfg["train"]["runs_dir"]) / tag
    epochs = cfg["train"]["smoke"]["epochs"] if a.smoke else None
    heldout = a.heldout or cfg["data"]["views"]
    print(f"device {dev}, unit {data.unit}, {len(data)} items, methods {methods}, held-out views {heldout}, "
          f"runs in {root}", flush=True)
    for v in heldout:
        plans = crossview_plans(cfg, data.meta, v, smoke=a.smoke)
        if a.folds is not None and not a.smoke:
            plans = [p for p in plans if p["fold"] in a.folds]
        for method in methods:
            for p in plans:
                out = root / f"heldout_{v}" / method / f"fold{p['fold']}"
                if (out / "done.json").exists() and not a.force:
                    print(f"skip held-out {v} {method} fold {p['fold']} (done)", flush=True)
                    continue
                n = {k: len(items_of(data, p[k], p["views"])) for k in ("inner_train", "inner_val")}
                n_te = len(items_of(data, p["test"], [v]))
                if not n["inner_train"] or not n["inner_val"] or not n_te:
                    raise RuntimeError(f"held-out {v} fold {p['fold']}: empty item set ({n}, test {n_te})")
                print(f"\n=== held-out {v} | {method} fold {p['fold']}: training views {p['views']}; "
                      f"{n['inner_train']} inner-train / {n['inner_val']} inner-val reps; {n_te} held-out-view test "
                      f"reps ===", flush=True)
                d = run_method_fold(cfg, data, method, p, out, None, epochs, log=lambda *x: print(*x, flush=True))
                print(f"=== held-out {v} | {method} fold {p['fold']} done in {d['wall_s'] / 60:.1f} min "
                      f"(peak GPU {d['peak_gpu_gb'] or float('nan'):.2f} GB) ===", flush=True)


if __name__ == "__main__":
    main()
