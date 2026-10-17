"""Viewpoint Consistency Index (VCI, spec Phase 5): mean cosine similarity between embeddings of the same
performance seen from different views, minus the mean similarity to other groups. ``key`` defines "same
performance": the rep set (rep level, default) or the (exercise, subject, variation) group."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def viewpoint_consistency(z, key, view, group):
    """z: (N, D) embeddings; key: (N,) same-performance ids; view: (N,) view ids; group: (N,) group ids.
    Returns dict(vci, same, other, n_same_pairs)."""
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.T
    same = (key[:, None] == key[None, :]) & (view[:, None] != view[None, :])
    other = group[:, None] != group[None, :]
    s, o = sim[same].mean().item(), sim[other].mean().item()
    return {"vci": s - o, "same": s, "other": o, "n_same_pairs": int(same.sum().item()) // 2}
