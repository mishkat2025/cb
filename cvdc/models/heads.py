"""Projection head (CVDC) and task heads (supervised).

Projection: 2-layer MLP (Linear-ReLU-Linear) -> 128-d, L2-normalised (C6). No BatchNorm, so a projection does
not depend on the batch mix of real and synthetic views.

Task heads (A3, T2, T5), linear by default (``heads.hidden: 0``), so a frozen encoder + heads is a linear probe:
  exercise  logits over all configured exercises (9), from the embedding only
  score     total score / 100
  items     EAAQ items (0-1), e.g. CF1-CF3 at rep level, plus PF1, PF2, PF4 at set level
  quality   3 logits (incorrect / partially correct / correct, derived from the score, D6)
The score, item and quality heads also get an exercise embedding (one shared model for all exercises, T5).
The exercise head never sees it, since it would give away the label; when the exercise is unknown (live
demo) the predicted exercise is used.

Video head (T13, step a2): pools the rep embeddings of each video (their mean and attention pooling, Ilse et al.
2018), adds the standardised set rep count and an exercise embedding, and predicts the video's score and EAAQ
items with linear layers. The rep-level task heads stay.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NON_ITEM = ("score", "quality", "exercise")


def mlp(d_in, d_out, hidden=0, dropout=0.0):
    if not hidden:
        return nn.Linear(d_in, d_out)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, d_out))


class ProjectionHead(nn.Module):
    def __init__(self, d_in=256, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out))

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class TaskHeads(nn.Module):
    def __init__(self, d_in, n_exercises, targets, ex_dim=16, hidden=0, dropout=0.0, n_quality=3):
        super().__init__()
        self.targets = list(targets)
        self.items = [t for t in self.targets if t not in NON_ITEM]
        self.ex_emb = nn.Embedding(n_exercises, ex_dim) if ex_dim else None
        d_cond = d_in + (ex_dim or 0)
        self.heads = nn.ModuleDict()
        if "exercise" in self.targets:
            self.heads["exercise"] = mlp(d_in, n_exercises, hidden, dropout)
        if "score" in self.targets:
            self.heads["score"] = mlp(d_cond, 1, hidden, dropout)
        if self.items:
            self.heads["item"] = mlp(d_cond, len(self.items), hidden, dropout)
        if "quality" in self.targets:
            self.heads["quality"] = mlp(d_cond, n_quality, hidden, dropout)

    def forward(self, z, exercise=None):
        out = {}
        if "exercise" in self.heads:
            out["exercise"] = self.heads["exercise"](z)
        zc = z
        if self.ex_emb is not None:
            if exercise is None:
                if "exercise" not in out:
                    raise ValueError("exercise ids are needed when there is no exercise head")
                exercise = out["exercise"].argmax(-1)
            zc = torch.cat([z, self.ex_emb(exercise)], dim=-1)
        if "score" in self.heads:
            out["score"] = self.heads["score"](zc).squeeze(-1)
        if "item" in self.heads:
            out["items"] = self.heads["item"](zc)
        if "quality" in self.heads:
            out["quality"] = self.heads["quality"](zc)
        return out


def segment_sum(x, seg, n):
    """Sum of the rows of x per segment id in [0, n)."""
    return x.new_zeros((n, *x.shape[1:])).index_add_(0, seg, x)


def segment_softmax(s, seg, n):
    """Softmax of the scores s (N,) within each segment (numerically stable)."""
    mx = s.new_full((n,), float("-inf")).scatter_reduce(0, seg, s, reduce="amax", include_self=True)
    e = torch.exp(s - mx[seg])
    return e / segment_sum(e, seg, n)[seg]


class VideoHead(nn.Module):
    def __init__(self, d_in, n_exercises, targets, pooling=("mean", "attention"), attn_hidden=128,
                 count_feature=True, ex_dim=16):
        super().__init__()
        self.targets = list(targets)
        self.items = [t for t in self.targets if t != "score"]
        self.pooling = list(pooling)
        if not self.pooling or set(self.pooling) - {"mean", "attention"}:
            raise ValueError(f"video_head.pooling {self.pooling!r}: use mean and/or attention")
        self.attn = (nn.Sequential(nn.Linear(d_in, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1))
                     if "attention" in self.pooling else None)
        self.count_feature = bool(count_feature)
        self.register_buffer("count_mean", torch.zeros(()))
        self.register_buffer("count_std", torch.ones(()))
        self.ex_emb = nn.Embedding(n_exercises, ex_dim) if ex_dim else None
        d = d_in * len(self.pooling) + int(self.count_feature) + (ex_dim or 0)
        self.score = nn.Linear(d, 1) if "score" in self.targets else None
        self.item = nn.Linear(d, len(self.items)) if self.items else None

    def set_count_stats(self, counts):
        """Standardise the set rep count with the training videos' mean and std."""
        c = counts.float()
        self.count_mean.fill_(c.mean().item())
        self.count_std.fill_(max(c.std(unbiased=False).item(), 1e-6))

    def forward(self, z, seg, n, count=None, exercise=None):
        """z: (N, D) rep embeddings; seg: (N,) video index in [0, n); count: (n,) set rep counts; exercise: (n,)."""
        z = z.float()
        out, parts = {}, []
        if "mean" in self.pooling:
            parts.append(segment_sum(z, seg, n) / segment_sum(torch.ones_like(z[:, 0]), seg, n)[:, None])
        if self.attn is not None:
            a = segment_softmax(self.attn(z).squeeze(-1).float(), seg, n)
            parts.append(segment_sum(a[:, None] * z, seg, n))
            out["attention"] = a
        if self.count_feature:
            parts.append(((count.float() - self.count_mean) / self.count_std)[:, None])
        if self.ex_emb is not None:
            parts.append(self.ex_emb(exercise).float())
        h = torch.cat(parts, -1)
        if self.score is not None:
            out["score"] = self.score(h).squeeze(-1)
        if self.item is not None:
            out["items"] = self.item(h)
        return out
