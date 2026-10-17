"""CVDC pretraining (spec §7 step 1; C1-C7, A2-A8, T8, T9): the weighted multi-positive InfoNCE
(cvdc/losses/cvdc.py) on rep-set batches (cvdc/data/cvdc_batch.py) of one fold's training subjects only.

The encoder and the projection head train on the inner training subjects' rep sets (all enabled sources:
``cvdc.sources``, default sync, audio, rep, synthetic). After every epoch the validation metric is the CVDC loss on
the inner validation subjects' rep sets (T8): the same sources and synthetic views, no augmentation, and a fixed
seed, so every epoch sees the same validation batches. The best epoch's weights are kept (early stopping,
patience ``train.early_stopping.patience``, at most ``train.max_epochs_cvdc`` epochs).

usage: python -m cvdc.train.pretrain_cvdc [--config C] [--model-config M] --fold K [--smoke] [--tag phase3a]
Writes runs/<tag>/proposed/fold<K>/cvdc/ (tb/, config.yaml, run_info.json, history.csv, best.pt, summary.json).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from ..config import load_config, resolve
from ..data.cvdc_batch import CVDCBatcher
from ..data.dataset import PackData
from ..losses.cvdc import cvdc_loss, cvdc_stats
from ..models.model import CVDCModel
from .trainer import EarlyStopping, Run, autocast, fold_plan, get_device, seed_everything, warmup_cosine


def cvdc_epoch(model, batcher, cfg, opt=None, sched=None):
    """One pass over the batcher. With an optimiser: training (augmented views); without: validation (eval mode,
    no augmentation, no gradients). Returns the anchor-weighted means of the loss and the monitoring stats."""
    lc, train = cfg["cvdc"]["loss"], opt is not None
    model.train(train)
    rows = []
    for b in batcher.iterate(augment_views=train):
        with torch.set_grad_enabled(train):
            with autocast(cfg, batcher.data.device):
                o = model(b["x"], project=True, heads=False)
            loss = cvdc_loss(o["proj"], b["set_ids"], b["W"], b["group_ids"], b["exercise"], lc["temperature"],
                             lc["negatives"], lc["same_group_other_rep"])
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        st = cvdc_stats(o["proj"].detach(), b["set_ids"], b["W"], b["group_ids"], b["exercise"], lc["negatives"],
                        lc["same_group_other_rep"], z=o["z"].detach())
        rows.append(dict(loss=loss.item(), items=len(b["x"]), **st))
    d = pd.DataFrame(rows)
    w = d.n_anchors.clip(lower=1).to_numpy()
    out = {k: float(np.average(d[k], weights=w)) for k in d.columns if k not in ("n_pos_pairs", "n_anchors", "items")}
    out.update(batches=len(d), items=int(d["items"].sum()), anchors=int(d.n_anchors.sum()))
    return out


def pretrain_cvdc(cfg, data, plan, run_dir, seed, max_epochs=None, log=print):
    """CVDC pretraining on plan['inner_train'] with early stopping on plan['inner_val']. Returns (model with the
    best epoch's weights, summary dict)."""
    tc, ec = cfg["train"], cfg["train"]["early_stopping"]
    max_epochs = int(max_epochs or tc["max_epochs_cvdc"])
    seed_everything(seed)
    model = CVDCModel(cfg).to(data.device)
    tr_b = CVDCBatcher(data, cfg, allowed=data.subject_mask(plan["inner_train"]), seed=seed)
    va_b = CVDCBatcher(data, cfg, allowed=data.subject_mask(plan["inner_val"]), seed=seed + 1)
    params = list(model.encoder.parameters()) + list(model.projection.parameters())
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = warmup_cosine(opt, max_epochs * len(tr_b), tc["warmup_frac"])
    es = EarlyStopping(ec["patience"], ec["min_delta"])
    run = Run(run_dir, cfg, {"stage": "cvdc", "seed": seed, "max_epochs": max_epochs, **plan,
                             "batcher_train": tr_b.summary(), "batcher_val": va_b.summary()})
    log(f"  cvdc: {tr_b.summary()['sets']} training rep sets ({len(tr_b)} batches), "
        f"{va_b.summary()['sets']} validation rep sets")
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        te = time.time()
        tr = cvdc_epoch(model, tr_b, cfg, opt, sched)
        va_b.reseed(seed + 1)
        va = cvdc_epoch(model, va_b, cfg)
        if not np.isfinite(tr["loss"]):
            raise RuntimeError(f"non-finite CVDC loss at epoch {ep}")
        stop = es.step(ep, va["loss"], model)
        run.log_epoch(ep, {**{f"train/{k}": v for k, v in tr.items()}, **{f"val/{k}": v for k, v in va.items()},
                           "lr": sched.get_last_lr()[0], "epoch_s": time.time() - te})
        log(f"  cvdc epoch {ep}/{max_epochs}: train {tr['loss']:.4f} (pos {tr['pos_sim']:.3f}, neg {tr['neg_sim']:.3f}, "
            f"proj_std {tr['proj_std']:.4f}) | val {va['loss']:.4f} | best {es.best_value:.4f} @ {es.best_epoch} "
            f"| {time.time() - te:.1f} s")
        if stop:
            break
    es.restore(model)
    summary = {**es.summary(max_epochs), "wall_s": time.time() - t0, "seed": seed}
    torch.save({"model": model.state_dict(), "input": cfg["model"]["input"], **summary}, run.dir / "best.pt")
    run.finish(summary)
    log(f"  cvdc: best epoch {es.best_epoch} (val {es.best_value:.4f}), stopped at {es.last_epoch}, "
        f"{summary['wall_s']:.0f} s")
    return model, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--tag", default="phase3a")
    ap.add_argument("--smoke", action="store_true", help="fold 0, 3 training subjects, train.smoke.epochs epochs")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    cfg["model"]["input"] = cfg["methods"]["proposed"]["input"]
    data = PackData(cfg, "rep", get_device(cfg))
    plans = fold_plan(cfg, data.meta, smoke=a.smoke)
    plan = plans[0] if a.smoke else plans[a.fold]
    tag = a.tag + ("_smoke" if a.smoke else "")
    out = resolve(cfg["train"]["runs_dir"]) / tag / "proposed" / f"fold{plan['fold']}" / "cvdc"
    pretrain_cvdc(cfg, data, plan, out, cfg["train"]["seed"] + plan["fold"],
                  cfg["train"]["smoke"]["epochs"] if a.smoke else None)


if __name__ == "__main__":
    main()
