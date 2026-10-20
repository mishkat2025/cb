"""Phase 3 runner (E14 step a, step a2): train and test the methods on every GroupKFold-5 fold, with identical folds,
inner validation splits, preprocessing, epoch budget, schedule and encoder size.

Per method and fold (seed = train.seed + fold for every method, so B1 and Proposed start from the same random
encoder):
  1. embeddings_init.npz: the random-init encoder's embeddings of the test subjects (VCI reference, E13)
  2. CVDC methods: CVDC pretraining on the inner training subjects (cvdc/), early-stopped on the inner validation
     subjects, or (``train.cvdc_init_from``, step a2) the encoder of that earlier run's fold, after checking that it
     was trained on exactly this fold's inner training subjects; embeddings_pretrained.npz
  3. supervised training (finetune/): from scratch (B1, B2) or from the CVDC encoder (Proposed variants: full
     fine-tuning, linear probe, or fine-tuning with + lambda * L_cvdc; per-method keys ``fine_tune``, ``cvdc_lambda``)
  4. test_predictions.csv (rep level; the report aggregates to video level), test_predictions_video.csv (video head,
     T13), embeddings_final.npz, done.json (stopping epochs, wall times, peak GPU memory)
Everything lives under runs/<tag>/<method>/fold<k>/ (gitignored). Finished folds (done.json) are skipped unless
--force. Evaluation and tables: scripts/phase3/report_phase3a2.py (step a: report_phase3a.py).

usage: python -m cvdc.train.run [--config C] [--model-config M] [--methods b1 proposed proposed_lp ...]
       [--folds 0 1 2 3 4] [--tag phase3a2] [--smoke] [--linear-probe] [--force]
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
from .train_supervised import predict, predict_videos, train_supervised
from .trainer import embedding_snapshot, fold_plan, get_device, items_of, seed_everything


def method_config(cfg, method):
    """The config of one method: its input mode and its optional ``fine_tune`` / ``cvdc_lambda`` overrides."""
    mc = cfg["methods"][method]
    c = copy.deepcopy(cfg)
    c["model"]["input"] = mc["input"]
    for k in ("fine_tune", "cvdc_lambda"):
        if k in mc:
            c["train"][k] = mc[k]
    return c


def reuse_cvdc(cfg, data, plan, log=print):
    """Load the CVDC encoder of runs/<train.cvdc_init_from>/proposed/fold<k>/cvdc/best.pt if it exists and was
    trained on exactly this fold's inner training / validation subjects. Returns (model, summary) or None."""
    tag = cfg["train"].get("cvdc_init_from")
    if not tag:
        return None
    return load_cvdc(cfg, data, plan, resolve(cfg["train"]["runs_dir"]) / tag / "proposed" / f"fold{plan['fold']}" /
                     "cvdc", log)


def load_cvdc(cfg, data, plan, d, log=print):
    """Load the CVDC encoder of run directory ``d`` (<...>/fold<k>/cvdc) if its pretraining finished (best.pt and
    summary.json) on exactly this fold's inner training / validation and test subjects and training views.
    Returns (model, summary) or None."""
    if not (d / "best.pt").exists() or not (d / "summary.json").exists():
        log(f"  cvdc: no finished pretraining in {d}")
        return None
    info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    for k in ("inner_train", "inner_val", "test"):
        if sorted(info[k]) != sorted(plan[k]):
            log(f"  cvdc: {d} used other {k} subjects; not reused")
            return None
    if info.get("views") != plan.get("views"):  # E24: an all-view encoder has seen the held-out view
        log(f"  cvdc: {d} used training views {info.get('views') or 'all'}, not {plan.get('views') or 'all'}; not "
            "reused")
        return None
    ck = torch.load(d / "best.pt", map_location=data.device, weights_only=True)
    if ck["input"] != cfg["model"]["input"]:
        raise RuntimeError(f"{d}: CVDC input {ck['input']} vs {cfg['model']['input']}")
    model = CVDCModel(cfg).to(data.device)
    sd = ck["model"]
    model.encoder.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")})
    model.projection.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("projection.")})
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    log(f"  cvdc: reusing {d} (best epoch {s['best_epoch']}, stopped {s['stopped_epoch']})")
    return model, {**s, "reused_from": str(d)}


def run_method_fold(cfg, data, method, plan, out, linear_probe=None, epochs=None, log=print):
    mc = cfg["methods"][method]
    c = method_config(cfg, method)
    seed = cfg["train"]["seed"] + plan["fold"]
    dev = data.device
    out.mkdir(parents=True, exist_ok=True)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    seed_everything(seed)
    np.savez_compressed(out / "embeddings_init.npz", **embedding_snapshot(CVDCModel(c).to(dev), data, plan, c))
    done = {"method": method, "fold": plan["fold"], "input": mc["input"], "cvdc": mc["cvdc"], "seed": seed,
            "fine_tune": c["train"]["fine_tune"] if linear_probe is None else ("linear_probe" if linear_probe else "full"),
            "cvdc_lambda": c["train"].get("cvdc_lambda", 0.0), "subjects": plan}
    init = None
    if mc["cvdc"]:
        if mc.get("cvdc_from"):  # T15: share the CVDC encoder that another method of this run pretrained on the fold
            d = out.parent.parent / mc["cvdc_from"] / f"fold{plan['fold']}" / "cvdc"
            r = load_cvdc(c, data, plan, d, log)
            if r is None:
                raise RuntimeError(f"{method} fold {plan['fold']}: no usable CVDC encoder in {d}; run "
                                   f"{mc['cvdc_from']} first")
        else:
            r = None if epochs else reuse_cvdc(c, data, plan, log)
        model, s = r if r else pretrain_cvdc(c, data, plan, out / "cvdc", seed, epochs, log)
        done["cvdc_stage"] = s
        np.savez_compressed(out / "embeddings_pretrained.npz", **embedding_snapshot(model, data, plan, c))
        init = {"encoder": model.encoder.state_dict(), "projection": model.projection.state_dict()}
        del model
    model, scalers, s = train_supervised(c, data, plan, out / "finetune", seed, init, linear_probe, epochs, log)
    done["supervised_stage"] = s
    t_pred = time.time()
    te = items_of(data, plan["test"])
    predict(model, data, te, c, scalers["rep"]).to_csv(out / "test_predictions.csv", index=False)
    if model.video_head is not None:
        predict_videos(model, data, te, c, scalers["video"]).to_csv(out / "test_predictions_video.csv", index=False)
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
    ap.add_argument("--methods", nargs="+", default=["b1", "proposed", "proposed_lp", "proposed_j01", "proposed_j1",
                                                    "b2_mediapipe", "b2_ktpformer"])
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--tag", default="phase3a2")
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
