"""Supervised training (spec §7 steps 2-3; T1, T2, T7-T11): B1 / B2 from scratch, Proposed fine-tuned from its CVDC
encoder. Same heads, loss, budget and schedule for every method.

Rep level by default (T1): every rep carries its video's score and CF1-CF3 (T2) as a weak label, plus the derived
quality class and the exercise. Score and items are standardised with the inner training subjects' mean and std
(T7; the test fold is never used); predictions are inverted to 0-100 / 0-1 for reporting. Training uses the CVDC
augmentations with per-item temporal crops (T11). After every epoch the validation metric is the standardised
supervised loss (the training objective: weighted score + items + quality + exercise terms) on the inner
validation subjects, without augmentation (T8); the best epoch's weights are kept. Full fine-tuning by default;
``linear_probe`` freezes the encoder (and its BatchNorm statistics) and trains the heads only (T10).

usage: python -m cvdc.train.train_supervised [--config C] [--model-config M] --method b1|b2_mediapipe|b2_ktpformer|proposed
       --fold K [--init runs/.../cvdc/best.pt] [--linear-probe] [--smoke] [--tag phase3a]
Writes runs/<tag>/<method>/fold<K>/finetune/ (tb/, config.yaml, run_info.json, history.csv, best.pt, summary.json)
and test_predictions.csv next to it. ``python -m cvdc.train.run`` runs all methods and folds (pretraining included).
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch

from ..config import load_config, resolve
from ..data.augment import augment
from ..data.dataset import PackData
from ..losses.supervised import SupervisedLoss, TargetScaler
from ..models.model import CVDCModel
from .trainer import (EarlyStopping, Run, autocast, fold_plan, forward_all, get_device, items_of, seed_everything,
                      warmup_cosine)


def coord_channels(mode):
    return 3 if mode.startswith("xyz") else 2


def train_supervised(cfg, data, plan, run_dir, seed, init=None, linear_probe=None, max_epochs=None, log=print):
    """Train on plan['inner_train'] with early stopping on plan['inner_val']. init: dict with 'encoder' (and
    optionally 'projection') state dicts to start from (Proposed), or None (from scratch). Returns (model with the
    best epoch's weights, fitted TargetScaler, summary dict)."""
    tc, ec = cfg["train"], cfg["train"]["early_stopping"]
    linear_probe = (tc["fine_tune"] == "linear_probe") if linear_probe is None else linear_probe
    max_epochs = int(max_epochs or tc["max_epochs_sup"])
    mode, dev = cfg["model"]["input"], data.device
    seed_everything(seed)
    model = CVDCModel(cfg, data.unit).to(dev)
    if init is not None:
        model.encoder.load_state_dict(init["encoder"])
        if "projection" in init:
            model.projection.load_state_dict(init["projection"])
    if linear_probe:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    params = (list(model.heads.parameters()) if linear_probe
              else list(model.encoder.parameters()) + list(model.heads.parameters()))
    items_list = model.heads.items
    tr, va = items_of(data, plan["inner_train"]), items_of(data, plan["inner_val"])
    scaler = TargetScaler().fit(data.targets(tr, items_list))
    sup = SupervisedLoss(items_list, cfg["supervised"]["weights"], scaler)
    bs = tc["batch_size"] if data.unit == "rep" else tc["batch_size_video"]
    n_batches = int(np.ceil(len(tr) / bs))
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = warmup_cosine(opt, max_epochs * n_batches, tc["warmup_frac"])
    ac = {**cfg["cvdc"]["augment"], "share_temporal_crop": False}
    ac["enabled"] = bool(tc["augment_supervised"] and ac.get("enabled", True))
    g = torch.Generator(device=dev).manual_seed(seed)
    es = EarlyStopping(ec["patience"], ec["min_delta"])
    run = Run(run_dir, cfg, {"stage": "supervised", "seed": seed, "max_epochs": max_epochs, "unit": data.unit,
                             "linear_probe": linear_probe, "init": "cvdc" if init is not None else "scratch",
                             "n_train_items": len(tr), "n_val_items": len(va), **plan})
    y_va = data.targets(va, items_list)
    log(f"  supervised ({'linear probe' if linear_probe else 'full fine-tuning'}, init "
        f"{'cvdc' if init is not None else 'scratch'}, input {mode}): {len(tr)} training / {len(va)} validation "
        f"{data.unit}s, {n_batches} batches")
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        te = time.time()
        model.train()
        if linear_probe:
            model.encoder.eval()
        perm = tr[torch.randperm(len(tr), device=dev, generator=g)]
        parts = []
        for i in range(0, len(perm), bs):
            it = perm[i:i + bs]
            x = augment(data.inputs(it, mode), ac, coord_channels(mode), generator=g).permute(0, 3, 1, 2).contiguous()
            y = data.targets(it, items_list)
            with autocast(cfg, dev):
                o = model(x, exercise=y["exercise"])
            loss, p = sup(o, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            parts.append(dict(total=loss.item(), **p))
        trm = pd.DataFrame(parts).mean().to_dict()
        if not np.isfinite(trm["total"]):
            raise RuntimeError(f"non-finite supervised loss at epoch {ep}")
        o = forward_all(model, data, va, cfg)
        val, vparts = sup(o, y_va)
        val = val.item()
        inv = scaler.inverse(o)
        val_rmse = torch.sqrt(((inv["score"] - y_va["score"]) ** 2).mean()).item()
        stop = es.step(ep, val, model)
        run.log_epoch(ep, {**{f"train/{k}": v for k, v in trm.items()}, "val/total": val,
                           **{f"val/{k}": v for k, v in vparts.items()}, "val/score_rmse": val_rmse,
                           "lr": sched.get_last_lr()[0], "epoch_s": time.time() - te})
        log(f"  sup epoch {ep}/{max_epochs}: train {trm['total']:.4f} (score {trm.get('score', float('nan')):.3f}) | "
            f"val {val:.4f} (score {vparts.get('score', float('nan')):.3f}, RMSE {val_rmse:.2f}) | "
            f"best {es.best_value:.4f} @ {es.best_epoch} | {time.time() - te:.1f} s")
        if stop:
            break
    es.restore(model)
    summary = {**es.summary(max_epochs), "wall_s": time.time() - t0, "seed": seed, "linear_probe": linear_probe}
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(), "input": mode, "unit": data.unit,
                "items": items_list, **summary}, run.dir / "best.pt")
    run.finish(summary)
    log(f"  supervised: best epoch {es.best_epoch} (val {es.best_value:.4f}), stopped at {es.last_epoch}, "
        f"{summary['wall_s']:.0f} s")
    return model, scaler, summary


@torch.no_grad()
def predict(model, data, items, cfg, scaler):
    """Per-item predictions on the 0-100 (score) and 0-1 (items) scales, quality and exercise probabilities."""
    o = forward_all(model, data, items, cfg)
    inv = scaler.inverse(o)
    it = items.cpu().numpy()
    m = data.meta.iloc[it]
    df = pd.DataFrame({"item": it, "video_id": m.video_id.to_numpy(), "subject": m.subject.to_numpy(),
                       "group": m.group.to_numpy(), "exercise": m.exercise.to_numpy(), "view": m.view.to_numpy(),
                       "variation": m.variation.to_numpy(), "score": m.score.to_numpy(),
                       "score_pred": inv["score"].cpu().numpy()})
    for j, k in enumerate(model.heads.items):
        df[k] = m[k].to_numpy()
        df[f"{k}_pred"] = inv["items"][:, j].cpu().numpy()
    df["quality"] = data.quality[items].cpu().numpy()
    pq = torch.softmax(o["quality"], -1).cpu().numpy()
    for j in range(pq.shape[1]):
        df[f"p_quality_{j}"] = pq[:, j]
    df["quality_pred"] = pq.argmax(1)
    pe = torch.softmax(o["exercise"], -1).cpu().numpy()
    codes = model.exercises
    df["exercise_pred"] = [codes[j] for j in pe.argmax(1)]
    for c in sorted(data.meta.exercise.unique()):
        df[f"p_ex_{c}"] = pe[:, codes.index(c)]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--method", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--init", help="CVDC checkpoint (best.pt of pretrain_cvdc); default for proposed: the fold's run")
    ap.add_argument("--linear-probe", action="store_true")
    ap.add_argument("--tag", default="phase3a")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    mc = cfg["methods"][a.method]
    cfg = copy.deepcopy(cfg)
    cfg["model"]["input"] = mc["input"]
    data = PackData(cfg, cfg["train"]["unit"], get_device(cfg))
    plans = fold_plan(cfg, data.meta, smoke=a.smoke)
    plan = plans[0] if a.smoke else plans[a.fold]
    base = resolve(cfg["train"]["runs_dir"]) / (a.tag + ("_smoke" if a.smoke else "")) / a.method / f"fold{plan['fold']}"
    init = None
    if mc["cvdc"]:
        ck = torch.load(a.init or base / "cvdc" / "best.pt", map_location=data.device, weights_only=True)
        sd = ck["model"]
        init = {"encoder": {k[8:]: v for k, v in sd.items() if k.startswith("encoder.")},
                "projection": {k[11:]: v for k, v in sd.items() if k.startswith("projection.")}}
    model, scaler, _ = train_supervised(cfg, data, plan, base / "finetune", cfg["train"]["seed"] + plan["fold"], init,
                                        a.linear_probe or None, cfg["train"]["smoke"]["epochs"] if a.smoke else None)
    predict(model, data, items_of(data, plan["test"]), cfg, scaler).to_csv(base / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
