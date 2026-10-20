"""Supervised training (spec §7 steps 2-3; T1, T2, T6-T14): B1 / B2 from scratch, Proposed fine-tuned from its CVDC
encoder. Same heads, loss, budget and schedule for every method.

Rep level by default (T1): every rep carries its video's score and CF1-CF3 (T2) as a weak label, plus the derived
quality class and the exercise. Step (a2) adds:
  video batches (T13)  ``train.sampler: video``: each batch holds whole videos (all their reps, >= train.batch_size
                       reps), so the set-level video head can pool them
  video head (T13)     mean + attention pooling of the video's rep embeddings plus the set rep count -> the video's
                       score and items (PF1, PF2, PF4, CF1-CF3); L_sup = L_rep + heads.video_head.weight * L_video
  T6                   ``train.rep_weighting: per_video``: rep weight = 1 / n_reps of its video, so every video counts
                       once in L_rep (L_video is a mean over videos)
  T14                  an item enters the loss only for samples of a stratum (exercise x group) where it varies
  T8b                  early stopping on the EMA-smoothed validation objective, nothing kept before min_epochs
  joint objective      ``train.cvdc_lambda`` > 0: + lambda * L_cvdc on a CVDC batch of the inner training subjects'
                       rep sets at every step (spec §7; Proposed variants), projection head trained too
Score and items are standardised with the inner training subjects' mean and std (T7; rep targets over reps, video
targets over videos); predictions are inverted to 0-100 / 0-1 for reporting. Training uses the CVDC augmentations
with per-item temporal crops (T11). The validation metric is the supervised objective (rep + video terms, without
the CVDC term) on the inner validation subjects, without augmentation (T8). Full fine-tuning by default;
``linear_probe`` freezes the encoder (and its BatchNorm statistics) and trains the heads only (T10).

usage: python -m cvdc.train.train_supervised [--config C] [--model-config M] --method b1|...|proposed --fold K
       [--init runs/.../cvdc/best.pt] [--linear-probe] [--smoke] [--tag phase3a2]
Writes runs/<tag>/<method>/fold<K>/finetune/ (tb/, config.yaml, run_info.json, history.csv, best.pt, summary.json)
and test_predictions.csv (+ test_predictions_video.csv) next to it. ``python -m cvdc.train.run`` runs all methods
and folds.
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch

from ..config import load_config, resolve
from ..data.augment import augment
from ..data.cvdc_batch import CVDCBatcher
from ..data.dataset import PackData
from ..losses.cvdc import cvdc_loss
from ..losses.supervised import SupervisedLoss, TargetScaler
from ..models.model import CVDCModel
from .trainer import (EarlyStopping, Run, autocast, fold_plan, forward_all, get_device, items_of, seed_everything,
                      warmup_cosine)


def coord_channels(mode):
    return 3 if mode.startswith("xyz") else 2


def make_batches(data, items, bs, gen, sampler):
    """One epoch's batches: list of (rep items, video codes or None, segment index or None). ``rep``: shuffled reps
    in chunks of bs; ``video``: shuffled videos, consecutive videos joined until a batch holds >= bs reps."""
    if sampler == "rep":
        perm = items[torch.randperm(len(items), device=items.device, generator=gen)]
        return [(perm[i:i + bs], None, None) for i in range(0, len(perm), bs)]
    vids = torch.unique(data.video_of[items])
    vids = vids[torch.randperm(len(vids), device=vids.device, generator=gen)]
    n = data.v_nreps[vids].cpu().numpy()
    out, start, acc = [], 0, 0
    for j in range(len(vids)):
        acc += n[j]
        if acc >= bs or j == len(vids) - 1:
            v = vids[start:j + 1]
            it, seg = data.video_items(v)
            out.append((it, v, seg))
            start, acc = j + 1, 0
    return out


class Objective:
    """The supervised objective of one batch (training) or of a whole subject set (validation): rep heads (T2, T6,
    T14) + video head (T13). Returns (total, parts) with parts prefixed rep/ and video/."""

    def __init__(self, cfg, data, model, scaler, vscaler, rep_mask, vid_mask):
        tc = cfg["train"]
        self.data, self.model = data, model
        self.rep = SupervisedLoss(model.heads.items, cfg["supervised"]["weights"], scaler)
        self.vh = model.video_head
        if self.vh is not None:
            self.vid = SupervisedLoss(self.vh.items, {"score": 1.0, "items": 1.0}, vscaler)
            self.w_video = float(cfg["heads"]["video_head"]["weight"])
        self.per_video = tc["rep_weighting"] == "per_video"
        if tc["rep_weighting"] not in ("per_video", "none"):
            raise ValueError(f"train.rep_weighting {tc['rep_weighting']!r}: per_video | none")
        self.rep_mask, self.vid_mask = rep_mask, vid_mask

    def __call__(self, o, items, vids=None, seg=None, vo=None):
        d = self.data
        y = d.targets(items, self.model.heads.items)
        y["item_mask"] = self.rep_mask[d.stratum[items]]
        sw = (1.0 / d.v_nreps[d.video_of[items]].float()) if self.per_video else None
        total, parts = self.rep(o, y, sw)
        parts = {f"rep/{k}": v for k, v in parts.items()}
        if self.vh is not None and vo is not None:
            yv = d.video_targets(vids, self.vh.items)
            yv["item_mask"] = self.vid_mask[d.v_stratum[vids]]
            lv, pv = self.vid(vo, yv)
            total = total + self.w_video * lv
            parts.update({f"video/{k}": v for k, v in pv.items()})
        return total, parts


def video_outputs(model, data, z, items):
    """Video head on rep embeddings z of ``items`` (grouped by video). Returns (video codes, outputs)."""
    vids, seg = data.videos_of(items)
    return vids, model.video(z, seg, len(vids), data.v_count[vids] if data.v_count is not None else None,
                             data.v_exercise[vids])


def train_supervised(cfg, data, plan, run_dir, seed, init=None, linear_probe=None, max_epochs=None, log=print):
    """Train on plan['inner_train'] with early stopping on plan['inner_val']. init: dict with 'encoder' (and
    optionally 'projection') state dicts to start from (Proposed), or None (from scratch). Returns (model with the
    kept epoch's weights, {'rep': TargetScaler, 'video': TargetScaler or None}, summary dict)."""
    tc = cfg["train"]
    linear_probe = (tc["fine_tune"] == "linear_probe") if linear_probe is None else linear_probe
    max_epochs = int(max_epochs or tc["max_epochs_sup"])
    lam = float(tc.get("cvdc_lambda") or 0.0)
    if lam and linear_probe:
        raise ValueError("cvdc_lambda > 0 needs a trainable encoder (not a linear probe)")
    mode, dev = cfg["model"]["input"], data.device
    sampler = tc.get("sampler", "rep") if data.unit == "rep" else "rep"
    seed_everything(seed)
    model = CVDCModel(cfg, data.unit).to(dev)
    if model.video_head is not None and sampler != "video":
        raise ValueError("heads.video_head needs train.sampler: video (whole videos per batch)")
    if model.video_head is not None and model.video_head.count_feature and data.v_count is None:
        raise RuntimeError("heads.video_head.count_feature: no set rep counts (heads.video_head.count_csv)")
    if init is not None:
        model.encoder.load_state_dict(init["encoder"])
        if "projection" in init:
            model.projection.load_state_dict(init["projection"])
    if linear_probe:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    heads = list(model.heads.parameters()) + (list(model.video_head.parameters()) if model.video_head else [])
    params = heads if linear_probe else list(model.encoder.parameters()) + heads
    if lam:
        params += list(model.projection.parameters())
    items_list = model.heads.items
    views = plan.get("views")  # cross-view protocol (E24): training and validation items of these views only
    tr, va = items_of(data, plan["inner_train"], views), items_of(data, plan["inner_val"], views)
    scaler = TargetScaler().fit(data.targets(tr, items_list))
    vscaler = None
    ia = cfg["heads"].get("item_activity") or {}
    rep_mask = data.item_masks(items_list, ia.get("min_subjects_off_mode", 0), ia.get("min_sd", 0.0)) if ia else \
        torch.ones(len(data.strata), len(items_list), dtype=torch.bool, device=dev)
    vid_mask = None
    if model.video_head is not None:
        vtr = torch.unique(data.video_of[tr])
        vscaler = TargetScaler().fit(data.video_targets(vtr, model.video_head.items))
        if data.v_count is not None:
            model.video_head.set_count_stats(data.v_count[vtr])
        vid_mask = data.item_masks(model.video_head.items, ia.get("min_subjects_off_mode", 0), ia.get("min_sd", 0.0))
    obj = Objective(cfg, data, model, scaler, vscaler, rep_mask, vid_mask)
    bs = tc["batch_size"] if data.unit == "rep" else tc["batch_size_video"]
    g = torch.Generator(device=dev).manual_seed(seed)
    n_batches = len(make_batches(data, tr, bs, torch.Generator(device=dev).manual_seed(seed), sampler))
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = warmup_cosine(opt, max_epochs * n_batches, tc["warmup_frac"])
    ac = {**cfg["cvdc"]["augment"], "share_temporal_crop": False}
    ac["enabled"] = bool(tc["augment_supervised"] and ac.get("enabled", True))
    es = EarlyStopping.from_config({**tc["early_stopping"],  # min_epochs never above the budget (smoke: 2 epochs)
                                    "min_epochs": min(tc["early_stopping"].get("min_epochs", 1), max_epochs)})
    cb, cb_iter = None, None
    if lam:
        cb = CVDCBatcher(data, cfg, allowed=data.subject_mask(plan["inner_train"], views), seed=seed)
        cb_iter, lc = cb.iterate(augment_views=True), cfg["cvdc"]["loss"]
    masks = {"rep": {f"{e} {gr}": [k for k, a in zip(items_list, row) if a] for (e, gr), row in
                     zip(data.strata, rep_mask.tolist())}}
    if vid_mask is not None:
        masks["video"] = {f"{e} {gr}": [k for k, a in zip(model.video_head.items, row) if a] for (e, gr), row in
                          zip(data.strata, vid_mask.tolist())}
    run = Run(run_dir, cfg, {"stage": "supervised", "seed": seed, "max_epochs": max_epochs, "unit": data.unit,
                             "sampler": sampler, "linear_probe": linear_probe, "cvdc_lambda": lam,
                             "init": "cvdc" if init is not None else "scratch", "n_train_items": len(tr),
                             "n_val_items": len(va), "active_items": masks, **plan})
    log(f"  supervised ({'linear probe' if linear_probe else 'full fine-tuning'}, init "
        f"{'cvdc' if init is not None else 'scratch'}, input {mode}, sampler {sampler}, rep weighting "
        f"{tc['rep_weighting']}, video head {'on' if model.video_head is not None else 'off'}, cvdc lambda {lam}): "
        f"{len(tr)} training / {len(va)} validation {data.unit}s, {n_batches} batches; active items {masks}")
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        te = time.time()
        model.train()
        if linear_probe:
            model.encoder.eval()
        parts = []
        for it, vids, seg in make_batches(data, tr, bs, g, sampler):
            x = augment(data.inputs(it, mode), ac, coord_channels(mode), generator=g).permute(0, 3, 1, 2).contiguous()
            with autocast(cfg, dev):
                o = model(x, exercise=data.exercise[it])
                vo = (model.video(o["z"], seg, len(vids), data.v_count[vids] if data.v_count is not None else None,
                                  data.v_exercise[vids]) if model.video_head is not None else None)
            loss, p = obj(o, it, vids, seg, vo)
            if lam:
                b = next(cb_iter, None)
                if b is None:  # the CVDC batches cycle independently of the supervised epoch
                    cb_iter = cb.iterate(augment_views=True)
                    b = next(cb_iter)
                with autocast(cfg, dev):
                    oc = model(b["x"], project=True, heads=False)
                lcv = cvdc_loss(oc["proj"], b["set_ids"], b["W"], b["group_ids"], b["exercise"], lc["temperature"],
                                lc["negatives"], lc["same_group_other_rep"])
                p["cvdc"] = lcv.item()
                p["sup"] = loss.item()
                loss = loss + lam * lcv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            parts.append(dict(total=loss.item(), **p))
        trm = pd.DataFrame(parts).mean().to_dict()
        if not np.isfinite(trm["total"]):
            raise RuntimeError(f"non-finite supervised loss at epoch {ep}")
        o = forward_all(model, data, va, cfg)
        vids, vo = (None, None)
        if model.video_head is not None:
            model.eval()
            with torch.no_grad(), autocast(cfg, dev):
                vids, vo = video_outputs(model, data, o["z"], va)
        with torch.no_grad():
            val, vparts = obj(o, va, vids, None, vo)
        val = val.item()
        inv = scaler.inverse(o)
        rmse = {"val/rep_score_rmse": torch.sqrt(((inv["score"] - data.score[va]) ** 2).mean()).item()}
        pv = pd.DataFrame({"v": data.video_of[va].cpu().numpy(), "p": inv["score"].cpu().numpy()}).groupby("v").p.mean()
        yv = data.v_score[torch.tensor(pv.index.to_numpy(), device=dev)].cpu().numpy()
        rmse["val/rep_head_video_rmse"] = float(np.sqrt(((pv.to_numpy() - yv) ** 2).mean()))
        if vo is not None:
            sv = vscaler.inverse(vo)["score"]
            rmse["val/video_head_rmse"] = torch.sqrt(((sv - data.v_score[vids]) ** 2).mean()).item()
        stop = es.step(ep, val, model)
        run.log_epoch(ep, {**{f"train/{k}": v for k, v in trm.items()}, "val/total": val, "val/total_ema": es.smoothed,
                           **{f"val/{k}": v for k, v in vparts.items()}, **rmse,
                           "lr": sched.get_last_lr()[0], "epoch_s": time.time() - te})
        log(f"  sup epoch {ep}/{max_epochs}: train {trm['total']:.4f} (rep score {trm.get('rep/score', float('nan')):.3f}"
            f"{', cvdc %.3f' % trm['cvdc'] if 'cvdc' in trm else ''}) | val {val:.4f} (ema {es.smoothed or float('nan'):.4f}; RMSE rep "
            f"{rmse['val/rep_score_rmse']:.2f}, rep->video {rmse['val/rep_head_video_rmse']:.2f}"
            f"{', video head %.2f' % rmse['val/video_head_rmse'] if 'val/video_head_rmse' in rmse else ''}) | best "
            f"{es.best_value:.4f} @ {es.best_epoch} | {time.time() - te:.1f} s")
        if stop:
            break
    es.restore(model)
    summary = {**es.summary(max_epochs), "wall_s": time.time() - t0, "seed": seed, "linear_probe": linear_probe,
               "cvdc_lambda": lam, "sampler": sampler, "rep_weighting": tc["rep_weighting"]}
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                "video_scaler": vscaler.state_dict() if vscaler is not None else None, "input": mode, "unit": data.unit,
                "items": items_list, "video_items": model.video_head.items if model.video_head is not None else None,
                **summary}, run.dir / "best.pt")
    run.finish(summary)
    log(f"  supervised: best epoch {es.best_epoch} (smoothed val {es.best_value:.4f}), stopped at {es.last_epoch}, "
        f"{summary['wall_s']:.0f} s")
    return model, {"rep": scaler, "video": vscaler}, summary


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
    if "exercise" in o:  # no exercise head in the per-exercise models (T17)
        pe = torch.softmax(o["exercise"], -1).cpu().numpy()
        codes = model.exercises
        df["exercise_pred"] = [codes[j] for j in pe.argmax(1)]
        for c in sorted(data.meta.exercise.unique()):
            df[f"p_ex_{c}"] = pe[:, codes.index(c)]
    return df


@torch.no_grad()
def predict_videos(model, data, items, cfg, vscaler):
    """Video-head predictions (T13) for the videos of ``items``: one row per video, score 0-100 and items 0-1."""
    o = forward_all(model, data, items, cfg, heads=False)
    model.eval()
    with autocast(cfg, data.device):
        vids, vo = video_outputs(model, data, o["z"], items)
    inv = vscaler.inverse(vo)
    m = data.vmeta.iloc[vids.cpu().numpy()]
    df = pd.DataFrame({"video_id": m.video_id.to_numpy(), "subject": m.subject.to_numpy(), "group": m.group.to_numpy(),
                       "exercise": m.exercise.to_numpy(), "view": m.view.to_numpy(),
                       "variation": m.variation.to_numpy(), "n_reps": data.v_nreps[vids].cpu().numpy(),
                       "count": data.v_count[vids].cpu().numpy() if data.v_count is not None else np.nan,
                       "score": m.score.to_numpy(), "score_pred": inv["score"].cpu().numpy()})
    for j, k in enumerate(model.video_head.items):
        df[k] = m[k].to_numpy()
        df[f"{k}_pred"] = inv["items"][:, j].cpu().numpy()
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--method", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--init", help="CVDC checkpoint (best.pt of pretrain_cvdc); default for proposed: the fold's run")
    ap.add_argument("--linear-probe", action="store_true")
    ap.add_argument("--tag", default="phase3a2")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    from .run import method_config

    cfg = method_config(cfg, a.method)
    data = PackData(cfg, cfg["train"]["unit"], get_device(cfg))
    plans = fold_plan(cfg, data.meta, smoke=a.smoke)
    plan = plans[0] if a.smoke else plans[a.fold]
    base = resolve(cfg["train"]["runs_dir"]) / (a.tag + ("_smoke" if a.smoke else "")) / a.method / f"fold{plan['fold']}"
    init = None
    if cfg["methods"][a.method]["cvdc"]:
        ck = torch.load(a.init or base / "cvdc" / "best.pt", map_location=data.device, weights_only=True)
        sd = ck["model"]
        init = {"encoder": {k[8:]: v for k, v in sd.items() if k.startswith("encoder.")},
                "projection": {k[11:]: v for k, v in sd.items() if k.startswith("projection.")}}
    model, scalers, _ = train_supervised(cfg, data, plan, base / "finetune", cfg["train"]["seed"] + plan["fold"], init,
                                         a.linear_probe or None, cfg["train"]["smoke"]["epochs"] if a.smoke else None)
    te = items_of(data, plan["test"])
    predict(model, data, te, cfg, scalers["rep"]).to_csv(base / "test_predictions.csv", index=False)
    if model.video_head is not None:
        predict_videos(model, data, te, cfg, scalers["video"]).to_csv(base / "test_predictions_video.csv", index=False)


if __name__ == "__main__":
    main()
