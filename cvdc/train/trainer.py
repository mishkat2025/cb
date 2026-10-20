"""Trainer pieces shared by CVDC pretraining (pretrain_cvdc.py) and supervised training (train_supervised.py).

  device, bf16 autocast (A9) and seeding
  fold_plan      the GroupKFold-5 folds (E1, stratified by expert / non-expert) and, inside each fold, the inner
                 validation split (T8): ~20% of the fold's training subjects per group, seeded per fold. Early
                 stopping and checkpoint selection use only the inner validation subjects; the model trains on
                 the remaining (inner training) subjects; the test fold is never used for either.
  EarlyStopping  keeps the best state by the (optionally EMA-smoothed) validation metric, only from ``min_epochs``
                 on, and stops after ``patience`` eligible epochs without improvement (T8, T8b)
  warmup_cosine  linear warm-up, then cosine decay over the max-epoch budget (T9)
  Run            a run directory: TensorBoard, config snapshot, run info (git commit, subjects), history.csv,
                 best.pt checkpoint, summary.json
  forward_all    forward-only passes in batches (embeddings, predictions, validation)
"""
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..config import REPO
from ..eval.splits import check_no_leakage, group_kfold, subject_table


def get_device(cfg):
    want = cfg["train"]["device"]
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if want == "cuda":
        raise RuntimeError("train.device is cuda, but CUDA is not available")
    return "cpu"


def autocast(cfg, dev):
    amp = cfg["train"]["amp"]
    if amp not in ("bf16", "off"):
        raise ValueError(f"train.amp {amp!r}: use bf16 or off (fp16 would need a GradScaler)")
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda" and amp == "bf16")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inner_split(meta, train_subjects, frac=0.2, seed=0):
    """Hold out round(frac * n) of the training subjects of each group (expert / non-expert), at least 1 and at
    most n - 1, as the inner validation set (T8). Returns (inner_train, inner_val): sorted, disjoint, and their
    union is ``train_subjects``."""
    train_subjects = sorted(map(str, train_subjects))
    subj = subject_table(meta)
    subj = subj[subj.subject.isin(set(train_subjects))]
    rng = np.random.default_rng(seed)
    val = []
    for _, d in subj.groupby("group"):
        s = sorted(d.subject)
        n = min(max(int(round(frac * len(s))), 1), len(s) - 1) if len(s) > 1 else 0
        val += [str(x) for x in rng.permutation(s)[:n]]
    val = sorted(val)
    return sorted(set(train_subjects) - set(val)), val


def fold_plan(cfg, meta, smoke=False):
    """Per fold: dict(fold, train, inner_train, inner_val, test), lists of subject keys. ``train`` is the whole
    training fold (inner_train + inner_val). Smoke mode (spec §7): fold 0 only, cut to ``train.smoke``'s numbers
    of (expert, non-expert) subjects per role."""
    ec, es = cfg["eval"], cfg["train"]["early_stopping"]
    if ec["split"] != "group_kfold":
        raise NotImplementedError(f"eval.split {ec['split']!r} (Phase 3 step (a) runs GroupKFold)")
    splits = group_kfold(meta, ec["k"], ec["seed"])
    check_no_leakage(splits, meta.subject.unique())
    plans = []
    for f, (tr, te) in enumerate(splits):
        itr, iva = inner_split(meta, tr, es["inner_frac"], ec["seed"] + es["seed_offset"] + f)
        plans.append(dict(fold=f, train=sorted(tr), inner_train=itr, inner_val=iva, test=sorted(te)))
    if smoke:
        sm, p = cfg["train"]["smoke"], plans[0]
        grp = subject_table(meta).set_index("subject").group

        def pick(subs, counts):
            out = []
            for g, n in zip(("expert", "non-expert"), counts):
                out += [x for x in subs if grp[x] == g][:n]
            return sorted(out)

        p = dict(fold=0, inner_train=pick(p["inner_train"], sm["train_subjects"]),
                 inner_val=pick(p["inner_val"], sm["val_subjects"]), test=pick(p["test"], sm["test_subjects"]))
        p["train"] = sorted(p["inner_train"] + p["inner_val"])
        plans = [p]
    for p in plans:
        check_plan(p)
    return plans


def check_plan(p):
    """Raise if the inner split touches the test fold or overlaps the inner training subjects, or if the optional
    training-view restriction (``views``, cross-view protocol E24) is empty."""
    tr, itr, iva, te = (set(p[k]) for k in ("train", "inner_train", "inner_val", "test"))
    if tr & te or (itr | iva) & te:
        raise AssertionError(f"fold {p['fold']}: training and test subjects overlap")
    if itr & iva or itr | iva != tr or not iva:
        raise AssertionError(f"fold {p['fold']}: bad inner split")
    if p.get("views") is not None and not p["views"]:
        raise AssertionError(f"fold {p['fold']}: empty training-view list")
    return True


def with_views(plan, views):
    """The plan restricted to training views ``views`` (cross-view protocol, E24): every training input (supervised
    items, inner validation, CVDC rep sets, the embedding centre) comes from these views only; the test subjects
    keep all their views, so a held-out view can be evaluated. views None = all views."""
    p = {**plan, "views": None if views is None else sorted(views)}
    check_plan(p)
    return p


class EarlyStopping:
    """Tracks a validation metric (lower is better): keeps a copy of the model state at the best epoch and says
    stop after ``patience`` epochs in a row without an improvement larger than ``min_delta`` (T8).

    T8b (step a2): with ``ema`` > 0 the decision uses the smoothed metric s_t = ema * s_(t-1) + (1 - ema) * v_t
    (s_1 = v_1); the kept weights are those of the epoch where s_t was lowest. Epochs before ``min_epochs`` update
    s_t but can be neither kept nor counted towards ``patience``. ema 0, min_epochs 1 = the step (a) rule."""

    def __init__(self, patience=10, min_delta=0.0, min_epochs=1, ema=0.0):
        self.patience, self.min_delta = int(patience), float(min_delta)
        self.min_epochs, self.ema = int(min_epochs), float(ema)
        if not 0.0 <= self.ema < 1.0:
            raise ValueError(f"ema {ema}: use 0 <= ema < 1")
        self.best_value, self.best_epoch, self.best_state = math.inf, None, None
        self.last_epoch, self.bad, self.smoothed = 0, 0, None

    def step(self, epoch, value, model):
        self.last_epoch = epoch
        if np.isfinite(value):
            self.smoothed = value if self.smoothed is None or self.ema == 0 else (
                self.ema * self.smoothed + (1 - self.ema) * value)
        if epoch < self.min_epochs:
            return False
        s = self.smoothed if self.smoothed is not None else math.inf
        if np.isfinite(value) and s < self.best_value - self.min_delta:
            self.best_value, self.best_epoch, self.bad = float(s), epoch, 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, model):
        if self.best_state is None:
            raise RuntimeError("no finite validation metric was seen")
        model.load_state_dict(self.best_state)
        return model

    def summary(self, max_epochs):
        return {"best_epoch": self.best_epoch, "best_val": self.best_value, "stopped_epoch": self.last_epoch,
                "max_epochs": int(max_epochs), "early_stopped": self.last_epoch < max_epochs,
                "min_epochs": self.min_epochs, "ema": self.ema, "patience": self.patience}

    @classmethod
    def from_config(cls, c):
        return cls(c["patience"], c.get("min_delta", 0.0), c.get("min_epochs", 1), c.get("ema", 0.0))


def warmup_cosine(opt, total_steps, warmup_frac):
    """Linear warm-up over ``warmup_frac`` of the steps, then cosine decay to 0 at ``total_steps`` (T9)."""
    w = max(1, int(round(warmup_frac * total_steps)))

    def f(step):
        if step < w:
            return (step + 1) / w
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - w) / max(1, total_steps - w))))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": head, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


class Run:
    """A run directory: tb/ (TensorBoard), config.yaml (the merged config), run_info.json, history.csv (one row per
    epoch), best.pt and summary.json."""

    def __init__(self, root, cfg, info):
        from torch.utils.tensorboard import SummaryWriter

        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        snap = {k: v for k, v in cfg.items() if k != "exercises"}
        inc = cfg["data"]["include_exercises"]  # [] = every exercise (full scope, D0)
        snap["exercises"] = {k: v for k, v in cfg["exercises"].items() if not inc or k in inc}
        (self.dir / "config.yaml").write_text(yaml.safe_dump(snap, sort_keys=False), encoding="utf-8")
        info = {**info, "git": git_state(), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.dir / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        self.tb = SummaryWriter(str(self.dir / "tb"))
        self.rows = []

    def log_epoch(self, epoch, row):
        for k, v in row.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                self.tb.add_scalar(k, v, epoch)
        self.rows.append({"epoch": epoch, **row})
        pd.DataFrame(self.rows).to_csv(self.dir / "history.csv", index=False)

    def finish(self, summary):
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
        self.tb.flush()
        self.tb.close()


def items_of(data, subjects, views=None):
    """Items of ``subjects``, optionally only those filmed from ``views`` (None = every view)."""
    return torch.nonzero(data.subject_mask(subjects, views)).squeeze(1)


@torch.no_grad()
def forward_all(model, data, items, cfg, project=False, heads=True):
    """Model outputs for ``items`` (plain packs, no augmentation), eval mode, in batches of train.eval_batch."""
    model.eval()
    mode, bs, dev = cfg["model"]["input"], cfg["train"]["eval_batch"], data.device
    outs = {}
    for i in range(0, len(items), bs):
        it = items[i:i + bs]
        with autocast(cfg, dev):
            o = model(data.inputs(it, mode).permute(0, 3, 1, 2).contiguous(), exercise=data.exercise[it],
                      project=project, heads=heads)
        for k, v in o.items():
            outs.setdefault(k, []).append(v.float())
    return {k: torch.cat(v) for k, v in outs.items()}


def embedding_snapshot(model, data, plan, cfg):
    """Embeddings for the VCI (E13): z and the projection of the test items (every view), and the mean z of the
    whole training fold in its training views (the centre of vci_centred; all views unless ``plan['views']``)."""
    te, tr = items_of(data, plan["test"]), items_of(data, plan["train"], plan.get("views"))
    o = forward_all(model, data, te, cfg, project=True, heads=False)
    z_tr = forward_all(model, data, tr, cfg, project=False, heads=False)["z"]
    return {"item": te.cpu().numpy(), "z": o["z"].cpu().numpy().astype(np.float32),
            "proj": o["proj"].cpu().numpy().astype(np.float32),
            "z_train_mean": z_tr.mean(0).cpu().numpy().astype(np.float32)}
