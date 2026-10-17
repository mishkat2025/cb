"""Phase 2 smoke run: checks that the models, the CVDC loss, the pair builder, synthetic views and the data path
work together on the real Phase 1 caches (E01 + E02). Not a result: one subject-held-out split, few epochs.

Steps
 1. pair counts per source (from cvdc.data.pairs)
 2. synthetic views on real data: yaw-0 projection of the world landmarks vs the real 2D keypoints of the same
    rep (torso units, per view); skeleton-only figure of one rep in its real and synthetic views
 3. CVDC pretraining for a few epochs on the training subjects (all sources), logging loss, positive/negative
    similarity, projection std (collapse), alignment/uniformity, step time and GPU memory
 4. viewpoint consistency on held-out subjects before and after pretraining (embedding z and projection),
    and cosine similarity of held-out positive pairs per source
 5. supervised fine-tuning for a few epochs (score, CF1-CF3, quality, exercise), held-out RMSE next to the
    predict-the-mean baseline; B2 (xyz input) and xyv input for a few steps; negatives=different_exercise_only

usage: python scripts/phase2/smoke.py [--cvdc-epochs 10] [--sup-epochs 4] [--holdout 5] [--seed 0]
Writes outputs/phase2/smoke_*.csv|json|txt and outputs/checks/phase2_*.png (skeleton-only).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.augment import augment  # noqa: E402
from cvdc.data.cvdc_batch import CVDCBatcher  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.data.skeleton import JOINT_SETS  # noqa: E402
from cvdc.data.synthetic import normalize_2d, synthetic_views  # noqa: E402
from cvdc.eval.consistency import viewpoint_consistency  # noqa: E402
from cvdc.losses.cvdc import cvdc_loss, cvdc_stats  # noqa: E402
from cvdc.losses.supervised import SupervisedLoss  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402
from cvdc.viz.skeleton import draw_skeleton  # noqa: E402

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def autocast(cfg, dev):
    amp = cfg["train"]["amp"]
    if dev != "cuda" or amp == "off":
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


@torch.no_grad()
def encode(model, x, cfg, dev, bs=2048):
    model.eval()
    zs, ps = [], []
    for i in range(0, len(x), bs):
        with autocast(cfg, dev):
            o = model(x[i:i + bs].permute(0, 3, 1, 2).contiguous(), project=True, heads=False)
        zs.append(o["z"].float())
        ps.append(o["proj"].float())
    return torch.cat(zs), torch.cat(ps)


def heldout_consistency(model, data, cfg, dev, items, members, pairs):
    """VCI on held-out real reps (plain packs, no augmentation) + mean cosine per pair source."""
    x = data.inputs(items, cfg["model"]["input"])
    z, p = encode(model, x, cfg, dev)
    m = members.iloc[items.cpu().numpy()]
    key = torch.tensor(m.set_id.to_numpy(), device=dev)
    view = torch.tensor(m.view.map("FLR".index).to_numpy(), device=dev)
    grp = data.group_code[items]
    out = {}
    for nm, e in (("z", z), ("proj", p)):
        v = viewpoint_consistency(e, key, view, grp)
        out.update({f"{nm}_{k}": v[k] for k in ("vci", "same", "other")})
    out["n_same_view_pairs"] = v["n_same_pairs"]
    pos = {int(it): k for k, it in enumerate(items.cpu().numpy())}
    zn = torch.nn.functional.normalize(z, dim=-1)
    pp = pairs[pairs.item_a.isin(pos) & pairs.item_b.isin(pos)]
    for s, g in pp.groupby("pair_source"):
        a = torch.tensor([pos[i] for i in g.item_a], device=dev)
        b = torch.tensor([pos[i] for i in g.item_b], device=dev)
        out[f"z_cos_{s}"] = (zn[a] * zn[b]).sum(-1).mean().item()
        out[f"n_{s}"] = len(g)
    return out


def synthetic_check(data, cfg, dev, members, out_png):
    """Yaw-0 synthetic projection vs the real 2D of the same rep: per-joint distance in torso units."""
    sc = {**cfg["cvdc"]["synthetic"], "distance_jitter": 0.0}
    rows = []
    real = normalize_2d(data.xy.float())  # re-normalised per rep, like the synthetic views
    for view in "FLR":
        it = torch.tensor(np.flatnonzero(members.view.to_numpy() == view), device=dev)
        sv = synthetic_views(data.world[it], sc, yaw_deg=0.0)["xy"]
        d = torch.linalg.norm(sv - real[it], dim=-1)  # (n, T, V)
        vis = data.vis[it] >= 0.5
        rows.append(dict(view=view, reps=len(it), mean_dist_torso=d.mean().item(),
                         median_dist_torso=d.median().item(), mean_dist_visible_joints=d[vis].mean().item(),
                         mean_dist_arms=d[..., 5:11].mean().item()))
    # figure: one 3-view rep set of E01, real views and synthetic yaws from the front view's world landmarks
    s = members[(members.n_members == 3) & (members.exercise == "E01")].set_id.iloc[0]
    m = members[members.set_id == s].sort_values("view", key=lambda v: v.map("FLR".index))
    r = data.meta.iloc[int(m.item.iloc[0])]
    t = int(round((r.peak - r.start) / max(1, r.end - r.start) * (data.T - 1)))  # the peak on the T-frame grid
    yaws = [0, 45, -45, 90, -90]
    f_item = int(m.item.iloc[0])
    syn = [synthetic_views(data.world[[f_item]], sc, yaw_deg=y)["xy"][0, t].cpu().numpy() for y in yaws]
    panels = [(f"real {v}", real[int(i), t].cpu().numpy()) for v, i in zip(m.view, m.item)]
    panels += [(f"synthetic yaw {y:+d}° (from F)", s_) for y, s_ in zip(yaws, syn)]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels), 4.2))
    coco = JOINT_SETS["coco17"]
    for ax, (title, xy) in zip(axes, panels):
        full = np.full((33, 2), np.nan)
        full[coco] = xy
        draw_skeleton(ax, full)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(2.4, -2.6)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"Rep set {s} ({m.group_id.iloc[0]}), frame at the peak; skeleton only", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return pd.DataFrame(rows)


def split_subjects(data, n_hold, seed):
    rng = np.random.default_rng(seed)
    subj = data.meta.drop_duplicates("subject")[["subject", "group"]]
    hold = []
    for g, d in subj.groupby("group"):
        hold += list(rng.choice(sorted(d.subject), size=min(n_hold, len(d) - 1), replace=False))
    return sorted(set(subj.subject) - set(map(str, hold))), sorted(map(str, hold))


def pretrain(model, batcher, cfg, dev, epochs):
    lc = cfg["cvdc"]["loss"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    steps = epochs * len(batcher)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["train"]["lr"], total_steps=steps, pct_start=0.1)
    rows, t_step = [], []
    for ep in range(epochs):
        model.train()
        acc = []
        for b in batcher:
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with autocast(cfg, dev):
                o = model(b["x"], project=True, heads=False)
            loss = cvdc_loss(o["proj"], b["set_ids"], b["W"], b["group_ids"], b["exercise"],
                             lc["temperature"], lc["negatives"], lc["same_group_other_rep"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            t_step.append(time.time() - t0)
            st = cvdc_stats(o["proj"].detach(), b["set_ids"], b["W"], b["group_ids"], b["exercise"],
                            lc["negatives"], lc["same_group_other_rep"], z=o["z"].detach())
            acc.append(dict(loss=loss.item(), items=len(b["x"]), **st))
        a = pd.DataFrame(acc).mean(numeric_only=True)
        rows.append(dict(epoch=ep + 1, **a.round(4).to_dict()))
        log(f"  cvdc epoch {ep + 1}/{epochs}: loss {a.loss:.3f}  pos_sim {a.pos_sim:.3f}  neg_sim {a.neg_sim:.3f}  "
            f"proj_std {a.proj_std:.4f}  z_std {a.z_std:.3f}  align {a['align']:.3f}  uniform {a['uniform']:.3f}")
        if not np.isfinite(a.loss):
            raise RuntimeError("non-finite CVDC loss")
    return pd.DataFrame(rows), float(np.median(t_step[5:] if len(t_step) > 10 else t_step))


def supervised(model, data, cfg, dev, train_items, test_items, epochs, items_list, tag, bs=256):
    sup = SupervisedLoss(items_list, cfg["supervised"]["weights"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    mode = cfg["model"]["input"]
    cc = 3 if mode == "xyz" else 2
    ac = {**cfg["cvdc"]["augment"], "share_temporal_crop": False}
    g = torch.Generator(device=dev).manual_seed(cfg["train"]["seed"])
    rows, t_step = [], []
    for ep in range(epochs):
        model.train()
        perm = train_items[torch.randperm(len(train_items), device=dev, generator=g)]
        parts = []
        for i in range(0, len(perm), bs):
            it = perm[i:i + bs]
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            x = augment(data.inputs(it, mode), ac, cc, generator=g).permute(0, 3, 1, 2).contiguous()
            y = data.targets(it, items_list)
            with autocast(cfg, dev):
                o = model(x, exercise=y["exercise"])
            loss, p = sup(o, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            t_step.append(time.time() - t0)
            parts.append(dict(total=loss.item(), **p))
        tr = pd.DataFrame(parts).mean()
        ev = evaluate(model, data, cfg, dev, test_items, items_list, train_items)
        rows.append(dict(model=tag, epoch=ep + 1, **{f"train_{k}": round(v, 4) for k, v in tr.items()}, **ev))
        log(f"  {tag} epoch {ep + 1}/{epochs}: train loss {tr.total:.4f} ({', '.join(f'{k} {v:.4f}' for k, v in tr.items() if k != 'total')}) | "
            f"held-out RMSE {ev['rmse']:.2f} (mean baseline {ev['rmse_mean_baseline']:.2f}), CF MAE {ev['cf_mae']:.3f}, "
            f"exercise acc {ev['exercise_acc']:.3f}")
        if not np.isfinite(tr.total):
            raise RuntimeError(f"non-finite supervised loss ({tag})")
    return pd.DataFrame(rows), float(np.median(t_step))


@torch.no_grad()
def evaluate(model, data, cfg, dev, items, items_list, train_items):
    model.eval()
    mode = cfg["model"]["input"]
    preds = {"score": [], "items": [], "exercise": []}
    for i in range(0, len(items), 2048):
        it = items[i:i + 2048]
        with autocast(cfg, dev):
            o = model(data.inputs(it, mode).permute(0, 3, 1, 2).contiguous(), exercise=data.exercise[it])
        preds["score"].append(o["score"].float() * 100)
        preds["items"].append(o["items"].float())
        preds["exercise"].append(o["exercise"].argmax(-1))
    s = torch.cat(preds["score"])
    y = data.targets(items, items_list)
    mean_train = data.score[train_items].mean()
    cf = torch.cat(preds["items"])
    return {"rmse": torch.sqrt(((s - y["score"]) ** 2).mean()).item(),
            "rmse_mean_baseline": torch.sqrt(((mean_train - y["score"]) ** 2).mean()).item(),
            "cf_mae": (cf - y["items"]).abs().nanmean().item(),
            "exercise_acc": (torch.cat(preds["exercise"]) == y["exercise"]).float().mean().item(),
            "pred_min": s.min().item(), "pred_max": s.max().item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--cvdc-epochs", type=int, default=10)
    ap.add_argument("--sup-epochs", type=int, default=4)
    ap.add_argument("--holdout", type=int, default=5, help="held-out subjects per group (expert / non-expert)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    cfg["train"]["seed"] = a.seed
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if (cfg["train"]["device"] in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    out = resolve(cfg["outputs"].get("phase2", "outputs/phase2"))
    out.mkdir(parents=True, exist_ok=True)
    checks = resolve(cfg["outputs"]["checks"])
    t_all = time.time()
    log(f"device {dev} ({torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'}), torch {torch.__version__}, "
        f"amp {cfg['train']['amp']}, input {cfg['model']['input']}, sources {cfg['cvdc']['sources']}")

    data = PackData(cfg, "rep", dev)
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    pairs = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["pairs_csv"]))
    log("\n# 1. pair counts (rep level, outputs/phase2/table_pair_counts.csv)")
    log(pd.read_csv(resolve(cfg["cvdc"]["pairs"]["summary_csv"])).to_string(index=False))

    log("\n# 2. synthetic views vs real 2D (yaw 0, distance 5 torso lengths; per-joint distance in torso units)")
    syn = synthetic_check(data, cfg, dev, members, checks / "phase2_synthetic_views.png")
    syn.to_csv(out / "smoke_synthetic_check.csv", index=False)
    log(syn.round(3).to_string(index=False))

    train_s, hold_s = split_subjects(data, a.holdout, a.seed)
    tr_mask = data.subject_mask(train_s)
    tr_items = torch.nonzero(tr_mask).squeeze(1)
    te_items = torch.nonzero(~tr_mask).squeeze(1)
    log(f"\nsplit (seed {a.seed}): {len(train_s)} training subjects ({len(tr_items)} reps), held out {hold_s} "
        f"({len(te_items)} reps)")

    log("\n# 3. CVDC pretraining (training subjects only)")
    batcher = CVDCBatcher(data, cfg, allowed=tr_mask, seed=a.seed)
    log("  batcher:", json.dumps(batcher.summary()))
    model = CVDCModel(cfg).to(dev)
    before = heldout_consistency(model, data, cfg, dev, te_items, members, pairs)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    hist, t_cvdc = pretrain(model, batcher, cfg, dev, a.cvdc_epochs)
    t_pre = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
    hist.to_csv(out / "smoke_cvdc_epochs.csv", index=False)
    log(f"  {a.cvdc_epochs} epochs x {len(batcher)} batches in {t_pre:.0f} s; median step {1000 * t_cvdc:.0f} ms "
        f"(~{hist['items'].mean():.0f} items per batch); peak GPU memory {mem:.2f} GB")
    after = heldout_consistency(model, data, cfg, dev, te_items, members, pairs)
    tr_after = heldout_consistency(model, data, cfg, dev, tr_items, members, pairs)

    log("\n# 4. viewpoint consistency on held-out subjects (cosine; vci = same rep other view - other groups)")
    vc = pd.DataFrame([dict(state="random init (held out)", **before), dict(state="after CVDC (held out)", **after),
                       dict(state="after CVDC (training subjects)", **tr_after)])
    vc.to_csv(out / "smoke_viewpoint_consistency.csv", index=False)
    log(vc.round(3).T.to_string(header=False))

    log("\n# 5. supervised fine-tuning from the CVDC encoder (rep level: score, CF1-CF3, quality, exercise)")
    items_rep = model.heads.items
    sup_hist, t_sup = supervised(model, data, cfg, dev, tr_items, te_items, a.sup_epochs, items_rep, "proposed_xy")
    log("\n  B1-shaped run (xy, no CVDC), B2 (xyz world landmarks, no CVDC) and xyv, same epochs:")
    runs = [sup_hist]
    for mode, tag in (("xy", "b1_xy_scratch"), ("xyz", "b2_xyz_scratch"), ("xyv", "xyv_scratch")):
        c2 = copy.deepcopy(cfg)
        c2["model"]["input"] = mode
        torch.manual_seed(a.seed)
        m2 = CVDCModel(c2).to(dev)
        h, _ = supervised(m2, data, c2, dev, tr_items, te_items, a.sup_epochs, items_rep, tag)
        runs.append(h)
    pd.concat(runs).to_csv(out / "smoke_supervised_epochs.csv", index=False)

    log("\n  negatives=different_exercise_only, sources=[synthetic] and [sync]: one epoch each")
    for change in ({"loss": {"negatives": "different_exercise_only"}}, {"sources": ["synthetic"]}, {"sources": ["sync"]}):
        c2 = copy.deepcopy(cfg)
        for k, v in change.items():
            c2["cvdc"][k] = {**c2["cvdc"][k], **v} if isinstance(v, dict) else v
        b2 = CVDCBatcher(data, c2, allowed=tr_mask, seed=a.seed)
        log(f"  {json.dumps(change)}: {json.dumps(b2.summary())}")
        pretrain(CVDCModel(c2).to(dev), b2, c2, dev, 1)

    # training curves (no video pixels)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    ax[0].plot(hist.epoch, hist.loss, "o-")
    ax[0].set_title("CVDC loss")
    ax[1].plot(hist.epoch, hist.pos_sim, "o-", label="positives")
    ax[1].plot(hist.epoch, hist.neg_sim, "o-", label="negatives")
    ax[1].set_title("cosine similarity (projection)")
    ax[1].legend()
    ax[2].plot(hist.epoch, hist.proj_std, "o-")
    ax[2].axhline(1 / np.sqrt(cfg["model"]["projection"]["out"]), ls="--", c="gray", label="1/sqrt(128)")
    ax[2].set_title("projection std (collapse -> 0)")
    ax[2].legend()
    for x in ax:
        x.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(checks / "phase2_cvdc_smoke_curves.png", dpi=110)
    plt.close(fig)

    summary = {"device": dev, "seed": a.seed, "train_subjects": train_s, "heldout_subjects": hold_s,
               "batcher": batcher.summary(), "cvdc_epochs": a.cvdc_epochs, "cvdc_step_ms": 1000 * t_cvdc,
               "cvdc_epoch_s": t_pre / a.cvdc_epochs, "sup_step_ms": 1000 * t_sup, "peak_gpu_gb": mem,
               "vc_before": before, "vc_after": after, "runtime_s": time.time() - t_all}
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    log(f"\nsmoke run done in {time.time() - t_all:.0f} s")
    (out / "smoke_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
