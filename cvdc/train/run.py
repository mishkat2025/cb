"""Phase 3 runner (E14 step a): train and test the main methods on every GroupKFold-5 fold, with identical folds,
inner validation splits, preprocessing, epoch budget, schedule and encoder size.

Per method and fold (seed = train.seed + fold for every method, so B1 and Proposed start from the same random
encoder):
  1. embeddings_init.npz: the random-init encoder's embeddings of the test subjects (VCI reference, E13)
  2. Proposed only: CVDC pretraining on the inner training subjects (cvdc/), early-stopped on the inner validation
     subjects; embeddings_pretrained.npz
  3. supervised training (finetune/): from scratch (B1, B2) or from the CVDC encoder (Proposed)
  4. test_predictions.csv (rep level; the report aggregates to video level), embeddings_final.npz, done.json
     (stopping epochs, wall times, peak GPU memory)
Everything lives under runs/<tag>/<method>/fold<k>/ (gitignored). Finished folds (done.json) are skipped unless
--force. Evaluation and tables: scripts/phase3/report_phase3a.py.

usage: python -m cvdc.train.run [--config C] [--model-config M] [--methods b1 b2_mediapipe b2_ktpformer proposed]
       [--folds 0 1 2 3 4] [--tag phase3a] [--smoke] [--linear-probe] [--force]
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from ..config import load_config, resolve
from ..data.dataset import PackData
from ..models.model import CVDCModel
from .pretrain_cvdc import pretrain_cvdc
from .train_supervised import predict, train_supervised
from .trainer import embedding_snapshot, fold_plan, get_device, items_of, seed_everything


def run_method_fold(cfg, data, method, plan, out, linear_probe=None, epochs=None, log=print):
    mc = cfg["methods"][method]
    c = copy.deepcopy(cfg)
    c["model"]["input"] = mc["input"]
    seed = cfg["train"]["seed"] + plan["fold"]
    dev = data.device
    out.mkdir(parents=True, exist_ok=True)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    seed_everything(seed)
    np.savez_compressed(out / "embeddings_init.npz", **embedding_snapshot(CVDCModel(c).to(dev), data, plan, c))
    done = {"method": method, "fold": plan["fold"], "input": mc["input"], "cvdc": mc["cvdc"], "seed": seed,
            "subjects": plan}
    init = None
    if mc["cvdc"]:
        model, s = pretrain_cvdc(c, data, plan, out / "cvdc", seed, epochs, log)
        done["cvdc_stage"] = s
        np.savez_compressed(out / "embeddings_pretrained.npz", **embedding_snapshot(model, data, plan, c))
        init = {"encoder": model.encoder.state_dict(), "projection": model.projection.state_dict()}
        del model
    model, scaler, s = train_supervised(c, data, plan, out / "finetune", seed, init, linear_probe, epochs, log)
    done["supervised_stage"] = s
    t_pred = time.time()
    predict(model, data, items_of(data, plan["test"]), c, scaler).to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "embeddings_final.npz", **embedding_snapshot(model, data, plan, c))
    done.update(wall_s=time.time() - t0, eval_s=time.time() - t_pred,
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None,
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "done.json").write_text(json.dumps(done, indent=1, default=float), encoding="utf-8")
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--methods", nargs="+", default=["b1", "b2_mediapipe", "b2_ktpformer", "proposed"])
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--tag", default="phase3a")
    ap.add_argument("--smoke", action="store_true", help="fold 0, 3 training subjects, train.smoke.epochs epochs")
    ap.add_argument("--linear-probe", action="store_true", help="frozen encoder, heads only (default: train.fine_tune)")
    ap.add_argument("--force", action="store_true", help="rerun folds that have done.json")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    unknown = set(a.methods) - set(cfg["methods"])
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}; configured: {sorted(cfg['methods'])}")
    lift = sorted({cfg["methods"][m]["input"].split("_", 1)[1] for m in a.methods if "_" in cfg["methods"][m]["input"]})
    if len(lift) > 1:
        raise SystemExit(f"one lifter per run: {lift}")
    dev = get_device(cfg)
    data = PackData(cfg, cfg["train"]["unit"], dev, lift=lift[0] if lift else None)
    plans = fold_plan(cfg, data.meta, smoke=a.smoke)
    if a.folds is not None and not a.smoke:
        plans = [p for p in plans if p["fold"] in a.folds]
    tag = a.tag + ("_smoke" if a.smoke else "")
    root = resolve(cfg["train"]["runs_dir"]) / tag
    epochs = cfg["train"]["smoke"]["epochs"] if a.smoke else None
    print(f"device {dev}, unit {data.unit}, {len(data)} items, methods {a.methods}, folds {[p['fold'] for p in plans]}, "
          f"runs in {root}", flush=True)
    for method in a.methods:
        for p in plans:
            out = root / method / f"fold{p['fold']}"
            if (out / "done.json").exists() and not a.force:
                print(f"skip {method} fold {p['fold']} (done)", flush=True)
                continue
            print(f"\n=== {method} fold {p['fold']}: {len(p['inner_train'])} inner-train, {len(p['inner_val'])} "
                  f"inner-val, {len(p['test'])} test subjects ===", flush=True)
            d = run_method_fold(cfg, data, method, p, out, True if a.linear_probe else None, epochs,
                                log=lambda *x: print(*x, flush=True))
            print(f"=== {method} fold {p['fold']} done in {d['wall_s'] / 60:.1f} min "
                  f"(peak GPU {d['peak_gpu_gb'] or float('nan'):.2f} GB) ===", flush=True)


if __name__ == "__main__":
    main()
