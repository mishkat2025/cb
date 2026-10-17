"""Viewpoint Consistency Index (VCI, E13): mean cosine similarity between embeddings of the same performance seen
from different views, minus the mean cosine similarity to other groups' embeddings.

``key`` defines "same performance": the rep set at rep level (default) or the (exercise, subject, variation)
group. Three variants are reported (supervisor, 2026-10-17):
  vci_centred  PRIMARY: the 256-d embedding z minus the training fold's mean embedding (what the task heads and
               the embedding-to-reference distance use; centring removes the common offset of the non-negative
               pooled ST-GCN features, which otherwise makes every cosine high)
  vci_raw      the 256-d embedding as is
  vci_proj     the 128-d L2-normalised projection (what the CVDC loss optimises)
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def viewpoint_consistency(z, key, view, group, center=None):
    """z: (N, D) embeddings; key: (N,) same-performance ids; view: (N,) view ids; group: (N,) group ids;
    center: (D,) vector subtracted first (the training fold's mean embedding) or None.
    Returns dict(vci, same, other, n_same_pairs)."""
    z = z.float()
    if center is not None:
        z = z - center.float()
    z = F.normalize(z, dim=-1)
    sim = z @ z.T
    same = (key[:, None] == key[None, :]) & (view[:, None] != view[None, :])
    other = group[:, None] != group[None, :]
    s, o = sim[same].mean().item(), sim[other].mean().item()
    return {"vci": s - o, "same": s, "other": o, "n_same_pairs": int(same.sum().item()) // 2}


@torch.no_grad()
def vci_report(z, proj, key, view, group, z_train_mean):
    """The three VCI variants (E13). Returns a flat dict: vci_centred (primary), vci_raw, vci_proj, plus the
    same / other similarities of each."""
    out = {}
    for name, e, c in (("centred", z, z_train_mean), ("raw", z, None), ("proj", proj, None)):
        v = viewpoint_consistency(e, key, view, group, c)
        out.update({f"vci_{name}": v["vci"], f"same_{name}": v["same"], f"other_{name}": v["other"]})
    out["n_same_view_pairs"] = v["n_same_pairs"]
    return out
