"""Supervised losses for the task heads (A3, T7): MSE on standardised regression targets (score and EAAQ items),
cross-entropy on the quality class and the exercise.

Regression targets are standardised with the **training fold's** mean and std only (``TargetScaler.fit`` on the
training items), so the heads predict z-scores; ``TargetScaler.inverse`` maps predictions back to the score
(0-100) and item (0-1) scales for reporting. A target with ~zero variance in the training fold is skipped (T4).
Missing targets (NaN, or -1 for classes) are masked out. Optional per-sample weights are the hook for the
skew-aware score weighting (T6, set in Phase 3).
"""
import torch
import torch.nn.functional as F


def _wmean(x, w):
    return (x * w).sum() / w.sum().clamp(min=1e-8)


class TargetScaler:
    """Per-fold standardisation of the score (N,) and items (N, k): fit on training items, then transform /
    inverse. ``active`` marks items with variance in the training fold (T4)."""

    def __init__(self, min_std=1e-6):
        self.min_std = min_std
        self.stats = None

    def fit(self, y):
        s, it = y["score"], y["items"]
        self.stats = {"score_mean": torch.nanmean(s), "score_std": _nanstd(s)}
        if it is not None:
            self.stats["items_mean"] = torch.nanmean(it, 0)
            self.stats["items_std"] = _nanstd(it, 0)
        return self

    @property
    def active(self):
        return self.stats["items_std"] > self.min_std if "items_std" in self.stats else None

    def transform(self, y):
        st = self.stats
        out = dict(y)
        out["score"] = (y["score"] - st["score_mean"]) / st["score_std"].clamp(min=self.min_std)
        if y.get("items") is not None:
            out["items"] = (y["items"] - st["items_mean"]) / st["items_std"].clamp(min=self.min_std)
        return out

    def inverse(self, out):
        """Model outputs (z-scores) -> dict(score on 0-100, items on 0-1), inactive items set to the training mean."""
        st = self.stats
        res = {}
        if "score" in out:
            res["score"] = out["score"].float() * st["score_std"] + st["score_mean"]
        if "items" in out:
            items = out["items"].float() * st["items_std"] + st["items_mean"]
            res["items"] = torch.where(self.active, items, st["items_mean"].expand_as(items))
        return res

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.stats.items()}

    def load_state_dict(self, sd, device="cpu"):
        self.stats = {k: v.to(device) for k, v in sd.items()}
        return self


def _nanstd(x, dim=None):
    m = torch.nanmean(x, dim=dim, keepdim=dim is not None)
    v = torch.nanmean((x - m) ** 2, dim=dim)
    return v.sqrt()


class SupervisedLoss:
    def __init__(self, items, weights, scaler=None):
        self.items = list(items)
        self.w = dict(weights)
        self.scaler = scaler

    def __call__(self, out, batch, sample_weight=None):
        """out: model outputs (z-scores for score / items when a scaler is set); batch: raw targets (score 0-100,
        items 0-1, quality, exercise). Returns (total, {part: value})."""
        dev = next(iter(out.values())).device
        sw = torch.ones(len(batch["score"]), device=dev) if sample_weight is None else sample_weight
        y = self.scaler.transform(batch) if self.scaler is not None else {**batch, "score": batch["score"] / 100.0}
        parts = {}
        if "score" in out:
            m = torch.isfinite(y["score"])
            if m.any():
                parts["score"] = _wmean(F.mse_loss(out["score"][m].float(), y["score"][m], reduction="none"), sw[m])
        if "items" in out and self.items:
            t = y["items"]
            m = torch.isfinite(t)
            if self.scaler is not None and self.scaler.active is not None:
                m = m & self.scaler.active[None]
            if m.any():
                err = F.mse_loss(out["items"].float(), torch.nan_to_num(t), reduction="none") * m
                parts["items"] = _wmean(err.sum(1) / m.sum(1).clamp(min=1), sw * (m.sum(1) > 0))
        for k in ("quality", "exercise"):
            if k in out:
                t = batch[k]
                m = t >= 0
                if m.any():
                    parts[k] = _wmean(F.cross_entropy(out[k][m].float(), t[m], reduction="none"), sw[m])
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: v.item() for k, v in parts.items()}
