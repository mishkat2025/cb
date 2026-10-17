"""Skeleton augmentations for CVDC (A2), in torch on (N, T, V, C) tensors.

Coordinates are channels 0:2 (x, y) or 0:3 (x, y, z for the xyz input); a third channel of the xyv input is
visibility. Per item: in-plane rotation, uniform scale and shear of (x, y) (z gets the same scale), Gaussian
noise, joint dropout (joint moved to the root and visibility 0) and a temporal crop of ``crop_min``-100 % of the
sequence resampled back to T (which also jitters the speed). Each view is augmented independently, except that
the temporal crop is shared by all items of a rep set when ``share_temporal_crop`` is true, so augmented
positives stay time-aligned.
"""
import math

import torch


def temporal_crop(x, ratio, start):
    """x: (N, T, V, C); ratio, start: (N,) fractions of the sequence -> (N, T, V, C) linear resampling of
    [start, start + ratio] back to T steps."""
    N, T = x.shape[:2]
    u = torch.linspace(0, 1, T, device=x.device)
    pos = ((start[:, None] + ratio[:, None] * u[None]) * (T - 1)).clamp(0, T - 1)
    i0 = pos.floor().long()
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (pos - i0).view(N, T, 1, 1)
    take = lambda i: torch.gather(x, 1, i.view(N, T, 1, 1).expand(-1, -1, *x.shape[2:]))
    return (1 - w) * take(i0) + w * take(i1)


def augment(x, ac, coord_channels=2, set_ids=None, generator=None):
    """x: (N, T, V, C) float tensor; ac: cfg['cvdc']['augment']; coord_channels: 2 (xy, xyv) or 3 (xyz);
    set_ids: (N,) to share the temporal crop within a set. Returns a new tensor."""
    if not ac.get("enabled", True):
        return x
    N, dev = x.shape[0], x.device
    x = x.clone()

    def U(lo, hi, n=N):
        return lo + (hi - lo) * torch.rand(n, device=dev, generator=generator)

    if ac["crop_min"] < 1:
        if set_ids is not None and ac.get("share_temporal_crop", True):
            uniq, inv = torch.unique(set_ids, return_inverse=True)
            ratio = U(ac["crop_min"], 1.0, len(uniq))
            start = U(0.0, 1.0, len(uniq)) * (1 - ratio)
            ratio, start = ratio[inv], start[inv]
        else:
            ratio = U(ac["crop_min"], 1.0)
            start = U(0.0, 1.0) * (1 - ratio)
        x = temporal_crop(x, ratio, start)

    th = U(-ac["rotate_deg"], ac["rotate_deg"]) * math.pi / 180
    s = 1 + U(-ac["scale"], ac["scale"])
    sh = U(-ac["shear"], ac["shear"])
    c, si = torch.cos(th), torch.sin(th)
    A = torch.stack([torch.stack([c, -si], -1), torch.stack([si, c], -1)], -2)          # rotation
    A = A @ torch.stack([torch.stack([torch.ones_like(sh), sh], -1),
                         torch.stack([torch.zeros_like(sh), torch.ones_like(sh)], -1)], -2)  # shear
    A = A * s[:, None, None]
    x[..., 0:2] = torch.einsum("nij,ntvj->ntvi", A, x[..., 0:2])
    if coord_channels == 3:
        x[..., 2] = x[..., 2] * s[:, None, None]
    if ac["noise"]:
        x[..., :coord_channels] += ac["noise"] * torch.randn(x[..., :coord_channels].shape, device=dev, generator=generator)
    if ac["joint_dropout"]:
        drop = torch.rand(N, 1, x.shape[2], device=dev, generator=generator) < ac["joint_dropout"]
        x = x.masked_fill(drop[..., None], 0.0)  # coordinates -> root (0), visibility -> 0
    return x
