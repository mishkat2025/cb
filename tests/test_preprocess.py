import numpy as np
import pytest

from cvdc.data.preprocess import preprocess_arrays, resample_fixed, resample_uniform
from cvdc.data.skeleton import J

PC = dict(fps=30, visibility_threshold=0.5, max_gap_s=0.5, smooth=None, root="mid_hip", scale="torso",
          scale_3d="torso")


def test_resample_uniform_irregular_timestamps_and_gap():
    t = np.array([0.0, 0.031, 0.07, 0.1, 0.2, 0.5, 0.52])
    X = (2 * t)[:, None, None].repeat(2, 1)              # linear in time -> exact after interpolation
    valid = np.ones((len(t), 2), bool)
    valid[4:6, 1] = False                                  # joint 1 missing at 0.2 and 0.5 s
    tu, Xu, gap = resample_uniform(t, X, valid, 30)
    assert tu[1] - tu[0] == pytest.approx(1 / 30) and tu[-1] <= 0.52
    assert np.allclose(Xu[:, 0, 0], 2 * tu)
    assert np.allclose(Xu[:, 1, 0], 2 * tu)                # bridged linearly across the gap
    assert gap[0, 1] == 0 and gap[np.argmin(np.abs(tu - 0.3)), 1] == pytest.approx(0.2, abs=0.02)


def test_resample_fixed_length_and_endpoints():
    X = np.arange(10, dtype=float)[:, None]
    Y = resample_fixed(X, 4)
    assert Y.shape == (4, 1) and Y[0, 0] == 0 and Y[-1, 0] == 9 and np.allclose(np.diff(Y[:, 0]), 3)
    assert np.allclose(resample_fixed(X, 19)[::2, 0], X[:, 0])


def test_root_centre_and_torso_scale():
    n = 40
    lm = np.zeros((n, 33, 5), np.float32)
    lm[..., 3:] = 1.0
    hw = (1000, 500)                                        # portrait frame: pixels, not normalised units
    for name, (x, y) in {"left_shoulder": (0.6, 0.3), "right_shoulder": (0.4, 0.3),
                         "left_hip": (0.55, 0.5), "right_hip": (0.45, 0.5)}.items():
        lm[:, J[name], 0], lm[:, J[name], 1] = x, y
    lm[:, J["left_hip"], 0] += np.linspace(0, 0.1, n)       # body drifts sideways
    lm[:, J["right_hip"], 0] += np.linspace(0, 0.1, n)
    lm[:, J["left_shoulder"], 0] += np.linspace(0, 0.1, n)
    lm[:, J["right_shoulder"], 0] += np.linspace(0, 0.1, n)
    world = np.zeros((n, 33, 3), np.float32)
    world[:, J["left_shoulder"], 1] = world[:, J["right_shoulder"], 1] = -0.5
    t = np.arange(n) / 30
    d = preprocess_arrays(lm, world, t, np.ones(n, bool), hw, PC)
    mid_hip = (d["xy"][:, J["left_hip"]] + d["xy"][:, J["right_hip"]]) / 2
    mid_sh = (d["xy"][:, J["left_shoulder"]] + d["xy"][:, J["right_shoulder"]]) / 2
    assert np.allclose(mid_hip, 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(mid_sh - mid_hip, axis=1), 1.0, atol=1e-6)
    assert d["torso_px"] == pytest.approx(0.2 * 1000)
    # shoulder width 0.2 * 500 px = 100 px = 0.5 torso lengths: aspect handled in pixel space
    assert np.allclose(d["xy"][:, J["left_shoulder"], 0] - d["xy"][:, J["right_shoulder"], 0], 0.5, atol=1e-6)


def test_low_visibility_joint_is_interpolated():
    n = 30
    lm = np.zeros((n, 33, 5), np.float32)
    lm[..., 3:] = 1.0
    lm[:, J["left_hip"], 1] = lm[:, J["right_hip"], 1] = 0.5
    lm[:, J["right_wrist"], 0] = np.linspace(0, 0.29, n)
    lm[10:15, J["right_wrist"], 0] = 5.0                    # garbage while occluded ...
    lm[10:15, J["right_wrist"], 3] = 0.1                    # ... flagged by low visibility
    world = np.zeros((n, 33, 3), np.float32)
    world[:, J["left_shoulder"], 1] = world[:, J["right_shoulder"], 1] = -0.5
    d = preprocess_arrays(lm, world, np.arange(n) / 30, np.ones(n, bool), (100, 100), PC)
    x = d["xy"][:, J["right_wrist"], 0]
    assert np.all(np.abs(np.diff(x)) < 0.2) and (d["gap_s"][10:15, J["right_wrist"]] > 0).all()


def test_long_low_visibility_run_keeps_estimate():
    n = 90                                                   # 3 s; joint low-visibility for 2 s (self-occlusion)
    lm = np.zeros((n, 33, 5), np.float32)
    lm[..., 3:] = 1.0
    lm[:, J["left_hip"], 1] = lm[:, J["right_hip"], 1] = 0.5
    lm[:, J["right_wrist"], 0] = 0.5 + 0.2 * np.sin(np.linspace(0, 6, n))
    lm[20:80, J["right_wrist"], 3] = 0.2
    world = np.zeros((n, 33, 3), np.float32)
    world[:, J["left_shoulder"], 1] = world[:, J["right_shoulder"], 1] = -0.5
    d = preprocess_arrays(lm, world, np.arange(n) / 30, np.ones(n, bool), (100, 100), PC)
    raw = (lm[:, J["right_wrist"], 0] - 0.0) * 100           # mid-hip x = 0, torso: shoulders at y=0
    assert np.allclose(d["xy"][:, J["right_wrist"], 0] * d["torso_px"], raw, atol=1e-3)
    assert (d["gap_s"][:, J["right_wrist"]] == 0).all() and (d["vis"][20:80, J["right_wrist"]] < 0.5).all()
