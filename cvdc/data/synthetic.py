"""Synthetic camera views (C4): lift -> rotate about the vertical axis -> pinhole projection.

The 3D source is the MediaPipe world landmarks (root-centred on the mid-hip, divided by the median 3D torso
length, axes x right, y down, z away from the camera; P5). A view is made by orbiting the camera about the
person's vertical axis by a yaw from ``cvdc.synthetic.yaw_deg`` plus uniform jitter, then projecting with a
pinhole camera on the z axis at ``distance`` torso lengths.

The world landmarks are camera-aligned, and the body axis (mid-ankle -> mid-shoulder) leans ~15 deg (front),
~9 deg (left) and ~13 deg (right views) in depth (median over the reps, Phase 2 check): the camera's pitch.
Rotating about the camera's y axis would tip the person over in side views. With ``level: true`` (A8) the
rep's median body-axis pitch phi is removed first and put back after the yaw, R = Rx(-phi) Rx(jitter) Ry(yaw)
Rx(phi): the virtual camera keeps the real camera's pitch, yaw 0 reproduces the source, and the person stays
upright at every yaw. Lateral lean (visible in 2D) is not touched. The 2D result is
normalised like the real 2D keypoints: root-centred on the mid-hip per frame and divided by the median 2D torso
length (focal length cancels). All in torch so batches are built on the GPU.
"""
import math

import torch

from .skeleton import COCO17_CENTER, JOINT_SETS, MP33

COCO17 = [MP33[i] for i in JOINT_SETS["coco17"]]
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = (COCO17.index(n) for n in (
    "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"))
assert COCO17_CENTER == COCO17.index("nose")


def rotation(yaw, pitch):
    """yaw, pitch: (N,) radians -> (N, 3, 3): rotation about y (vertical) by yaw, then about x by pitch."""
    cy, sy, cp, sp = torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)
    o, z = torch.ones_like(yaw), torch.zeros_like(yaw)
    Ry = torch.stack([torch.stack([cy, z, sy], -1), torch.stack([z, o, z], -1), torch.stack([-sy, z, cy], -1)], -2)
    Rx = torch.stack([torch.stack([o, z, z], -1), torch.stack([z, cp, -sp], -1), torch.stack([z, sp, cp], -1)], -2)
    return Rx @ Ry


def body_pitch(world):
    """(N, T, V, 3) -> (N,) radians: median over time of the mid-ankle -> mid-shoulder direction's tilt in the
    y-z plane (0 = upright along -y; > 0 = top toward +z, away from the camera)."""
    up = (world[..., L_SH, :] + world[..., R_SH, :]) / 2 - (world[..., L_ANK, :] + world[..., R_ANK, :]) / 2
    return torch.atan2(up[..., 2], -up[..., 1]).median(dim=1).values


def normalize_2d(xy):
    """(N, T, V, 2) -> root-centred on the mid-hip per frame, divided by the median torso length per item."""
    root = (xy[..., L_HIP, :] + xy[..., R_HIP, :]) / 2
    xy = xy - root[..., None, :]
    torso = torch.linalg.norm((xy[..., L_SH, :] + xy[..., R_SH, :]) / 2, dim=-1)  # mid-shoulder to the root
    return xy / torso.median(dim=1).values.clamp(min=1e-6)[:, None, None, None]


def project(world, R, distance):
    """world: (N, T, V, 3); R: (N, 3, 3); distance: (N,) -> (rotated 3D (N, T, V, 3), image xy (N, T, V, 2))."""
    P = torch.einsum("nij,ntvj->ntvi", R, world)
    depth = (P[..., 2] + distance[:, None, None]).clamp(min=0.1)
    return P, P[..., :2] / depth[..., None]


def synthetic_views(world, sc, generator=None, yaw_deg=None):
    """world: (N, T, V, 3) coco17 world landmarks; sc: cfg['cvdc']['synthetic'].
    Returns dict xy (N, T, V, 2) normalised 2D, world (N, T, V, 3) rotated 3D, yaw_deg (N,)."""
    n, dev = world.shape[0], world.device

    def U(lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=dev, generator=generator)

    if yaw_deg is None:
        choices = torch.tensor(sc["yaw_deg"], dtype=torch.float32, device=dev)
        yaw_deg = choices[torch.randint(len(choices), (n,), device=dev, generator=generator)]
        yaw_deg = yaw_deg + U(-sc["yaw_jitter_deg"], sc["yaw_jitter_deg"])
    else:
        yaw_deg = torch.as_tensor(yaw_deg, dtype=torch.float32, device=dev).expand(n)
    pitch = U(-sc["pitch_jitter_deg"], sc["pitch_jitter_deg"]) if sc.get("pitch_jitter_deg") else torch.zeros(n, device=dev)
    dist = sc["distance"] + U(-sc["distance_jitter"], sc["distance_jitter"])
    R = rotation(yaw_deg * math.pi / 180, pitch * math.pi / 180)
    if sc.get("level", True):
        phi = body_pitch(world.float())
        zero = torch.zeros_like(phi)
        R = rotation(zero, -phi) @ R @ rotation(zero, phi)
    P, xy = project(world.float(), R, dist)
    return {"xy": normalize_2d(xy), "world": P, "yaw_deg": yaw_deg}
