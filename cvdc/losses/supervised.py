"""Supervised losses for the task heads: MSE on score / 100 and on the EAAQ items (0-1), cross-entropy on the
quality class and the exercise. Missing targets (NaN, or -1 for classes) are masked out. Optional per-sample
weights are the hook for the skew-aware score weighting (T6, set in Phase 3)."""
import torch
import torch.nn.functional as F


def _wmean(x, w):
    return (x * w).sum() / w.sum().clamp(min=1e-8)


class SupervisedLoss:
    def __init__(self, items, weights):
        self.items = list(items)
        self.w = dict(weights)

    def __call__(self, out, batch, sample_weight=None):
        """out: model outputs; batch: score (N,) 0-100, items (N, n_items) 0-1, quality (N,), exercise (N,).
        Returns (total, {part: value})."""
        dev = next(iter(out.values())).device
        sw = torch.ones(len(batch["score"]), device=dev) if sample_weight is None else sample_weight
        parts = {}
        if "score" in out:
            y = batch["score"] / 100.0
            m = torch.isfinite(y)
            if m.any():
                parts["score"] = _wmean(F.mse_loss(out["score"][m].float(), y[m], reduction="none"), sw[m])
        if "items" in out and self.items:
            y = batch["items"]
            m = torch.isfinite(y)
            if m.any():
                err = F.mse_loss(out["items"].float(), torch.nan_to_num(y), reduction="none") * m
                parts["items"] = _wmean(err.sum(1) / m.sum(1).clamp(min=1), sw * (m.sum(1) > 0))
        for k in ("quality", "exercise"):
            if k in out:
                y = batch[k]
                m = y >= 0
                if m.any():
                    parts[k] = _wmean(F.cross_entropy(out[k][m].float(), y[m], reduction="none"), sw[m])
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: v.item() for k, v in parts.items()}
