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
