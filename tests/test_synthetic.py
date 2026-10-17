import math

import pytest
import torch

from cvdc.data.synthetic import L_HIP, L_SH, R_HIP, R_SH, normalize_2d, project, rotation, synthetic_views

SC = {"yaw_deg": [0, 45, -45, 90, -90], "yaw_jitter_deg": 10, "pitch_jitter_deg": 0, "distance": 5.0,
      "distance_jitter": 1.0}


def skeleton(n=2, T=8):
    """Upright coco17 skeleton (x right, y down, z away from the camera), torso length 1, arms out to the side."""
    P = torch.zeros(n, T, 17, 3)
    P[..., L_HIP, :] = torch.tensor([-0.2, 0, 0])
    P[..., R_HIP, :] = torch.tensor([0.2, 0, 0])
    P[..., L_SH, :] = torch.tensor([-0.3, -1, 0])
    P[..., R_SH, :] = torch.tensor([0.3, -1, 0])
    P[..., 7, :] = torch.tensor([-0.8, -1, 0.1])   # left elbow
    P[..., 8, :] = torch.tensor([0.8, -1, -0.1])   # right elbow
    P[..., 0, :] = torch.tensor([0, -1.4, -0.1])   # nose
    return P


def test_rotation_is_orthonormal_and_keeps_the_vertical_axis():
    R = rotation(torch.tensor([0.3, -1.2]), torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(R @ R.transpose(1, 2), torch.eye(3).expand(2, 3, 3))
    torch.testing.assert_close(R[:, :, 1], torch.tensor([[0.0, 1, 0], [0, 1, 0]]))  # y (vertical) unchanged


def test_quarter_turn_moves_width_into_depth():
    P = skeleton(1, 1)
    R = rotation(torch.tensor([math.pi / 2]), torch.tensor([0.0]))
    rot, xy = project(P, R, torch.tensor([5.0]))
    shoulder_w = (rot[0, 0, R_SH] - rot[0, 0, L_SH]).abs()
    assert shoulder_w[0] < 1e-6 and shoulder_w[2] == pytest.approx(0.6, abs=1e-6)  # shoulders now along depth
    bone = lambda Q: torch.linalg.norm(Q[..., 8, :] - Q[..., R_SH, :], dim=-1)
    torch.testing.assert_close(bone(rot), bone(P))  # bone lengths preserved


def test_yaw_zero_reproduces_the_frontal_shape():
    P = skeleton()
    out = synthetic_views(P, {**SC, "distance_jitter": 0.0}, yaw_deg=0.0)
    xy = out["xy"]
    mid_hip = (xy[..., L_HIP, :] + xy[..., R_HIP, :]) / 2
    torch.testing.assert_close(mid_hip, torch.zeros_like(mid_hip))
    torso = torch.linalg.norm((xy[..., L_SH, :] + xy[..., R_SH, :]) / 2, dim=-1)
    torch.testing.assert_close(torso, torch.ones_like(torso))
    ref = normalize_2d(P[..., :2])  # orthographic frontal view
    assert (xy - ref).abs().max() < 0.05  # perspective at 5 torso lengths changes little


def test_random_views_are_finite_and_vary():
    P = skeleton(n=64)
    out = synthetic_views(P, SC, torch.Generator().manual_seed(0))
    assert torch.isfinite(out["xy"]).all()
    assert out["xy"].shape == (64, 8, 17, 2)
    yaw = out["yaw_deg"]
    assert yaw.abs().max() <= 100 and yaw.std() > 30
    width = (out["xy"][:, 0, R_SH, 0] - out["xy"][:, 0, L_SH, 0]).abs()
    near_side = yaw.abs() > 80
    assert width[near_side].max() < width[yaw.abs() < 10].min()  # side views narrow the shoulders


def test_levelling_keeps_the_person_upright_in_side_views():
    """A skeleton leaning 15 deg in depth (the camera's pitch) seen from yaw 90: with levelling the body axis is
    vertical in the image; without it the side view shows the whole body tilted. Yaw 0 is the source view."""
    P = skeleton(1, 4)
    P[..., 15, :] = torch.tensor([-0.15, 1.6, 0])  # ankles
    P[..., 16, :] = torch.tensor([0.15, 1.6, 0])
    tilt = rotation(torch.zeros(1), torch.tensor([math.radians(-15.0)]))
    P = torch.einsum("nij,ntvj->ntvi", tilt, P)
    sc = {**SC, "distance_jitter": 0.0, "yaw_jitter_deg": 0}

    def lean_deg(xy):  # image angle of the mid-ankle -> mid-shoulder line from vertical
        d = (xy[0, 0, L_SH] + xy[0, 0, R_SH]) / 2 - (xy[0, 0, 15] + xy[0, 0, 16]) / 2
        return math.degrees(math.atan2(d[0].item(), -d[1].item()))

    side = synthetic_views(P, {**sc, "level": True}, yaw_deg=90.0)["xy"]
    tipped = synthetic_views(P, {**sc, "level": False}, yaw_deg=90.0)["xy"]
    assert abs(lean_deg(side)) < 1.0
    assert abs(lean_deg(tipped)) > 10.0
    front_l = synthetic_views(P, {**sc, "level": True}, yaw_deg=0.0)["xy"]
    front = synthetic_views(P, {**sc, "level": False}, yaw_deg=0.0)["xy"]
    torch.testing.assert_close(front_l, front, atol=1e-5, rtol=1e-5)
