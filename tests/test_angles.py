import numpy as np
import pytest

from cvdc.assess.angles import angle_between, joint_angle, shoulder_elevation
from cvdc.data.skeleton import J


def body(arm_dir, side="right"):
    """33-landmark 3D pose: upright trunk (y down), the given unit direction for the arm of `side`."""
    P = np.zeros((33, 3))
    P[J["left_shoulder"]], P[J["right_shoulder"]] = [0.2, -0.5, 0], [-0.2, -0.5, 0]
    P[J["left_hip"]], P[J["right_hip"]] = [0.1, 0.0, 0], [-0.1, 0.0, 0]
    sh = P[J[f"{side}_shoulder"]]
    P[J[f"{side}_elbow"]] = sh + 0.3 * np.asarray(arm_dir, float)
    P[J[f"{side}_wrist"]] = sh + 0.6 * np.asarray(arm_dir, float)
    return P


@pytest.mark.parametrize("arm, deg", [([0, 1, 0], 0), ([-1, 0, 0], 90), ([0, 0, -1], 90), ([0, -1, 0], 180),
                                      ([-1, -1, 0], 135)])
def test_shoulder_elevation_known_geometry(arm, deg):
    v = np.asarray(arm, float) / np.linalg.norm(arm)
    assert shoulder_elevation(body(v)) == pytest.approx(deg, abs=1e-6)


def test_elbow_angle_and_batch_shape():
    P = np.stack([body([0, 1, 0]), body([-1, 0, 0])])
    assert np.allclose(joint_angle(P, "right_shoulder", "right_elbow", "right_wrist"), 180)
    P[1, J["right_wrist"]] = P[1, J["right_elbow"]] + [0, 0.3, 0]  # forearm down from a horizontal upper arm
    assert joint_angle(P, "right_shoulder", "right_elbow", "right_wrist")[1] == pytest.approx(90)
    assert shoulder_elevation(P).shape == (2,)


def test_angle_between_nan_on_zero_vector():
    assert np.isnan(angle_between([0, 0, 0], [1, 0, 0]))


@pytest.mark.parametrize("side, arm, deg", [("right", [0, 1, 0], 0), ("right", [-1, 0, 0], 90),
                                            ("right", [1, 1, 0], -45), ("left", [1, 0, 0], 90),
                                            ("left", [-1, 1, 0], -45), ("right", [0, 1, -1], 0)])
def test_arm_frontal_angle_signed(side, arm, deg):
    """Right shoulder at -x here, so outward for the right arm is -x; forward motion does not count."""
    from cvdc.assess.angles import arm_frontal_angle
    v = np.asarray(arm, float) / np.linalg.norm(arm)
    assert arm_frontal_angle(body(v, side), side) == pytest.approx(deg, abs=1e-6)


def test_wrist_path_pc1_positive_toward_midline():
    """Wrist sweeps from the side to across the body: excursions toward the midline are positive."""
    from cvdc.assess.angles import wrist_path_pc1
    T = 120
    phase = (1 - np.cos(np.linspace(0, 6 * np.pi, T))) / 2          # 0 = rest at the side, 1 = across
    P = np.stack([body([0, 1, 0]) for _ in range(T)])
    for side, outward in (("right", -1.0), ("left", 1.0)):
        Q = P.copy()
        sh = Q[:, J[f"{side}_shoulder"]]
        Q[:, J[f"{side}_wrist"]] = sh + np.stack([outward * (0.1 - 0.5 * phase), 0.6 - 0.1 * phase, -0.3 * phase], 1)
        s = wrist_path_pc1(Q, side)
        assert np.corrcoef(s, phase)[0, 1] > 0.99


@pytest.mark.parametrize("side, arm, deg", [
    ("right", [-1, 0, 0], 0), ("right", [0, 0, -1], 90), ("right", [1, 0, -1], 135), ("right", [0, 0, 1], -90),
    ("left", [1, 0, 0], 0), ("left", [0, 0, -1], 90), ("left", [-1, 0, -1], 135), ("right", [-1, 1, -1], 45)])
def test_horizontal_adduction_known_geometry(side, arm, deg):
    """body(): y down, right shoulder at -x, the trunk faces -z (anterior). Elevation does not change the angle."""
    from cvdc.assess.angles import horizontal_adduction
    v = np.asarray(arm, float) / np.linalg.norm(arm)
    assert horizontal_adduction(body(v, side), side) == pytest.approx(deg, abs=1e-6)


def test_horizontal_adduction_masks_vertical_arm():
    from cvdc.assess.angles import horizontal_adduction
    v = np.array([0.05, 1.0, -0.05]) / np.linalg.norm([0.05, 1.0, -0.05])   # arm nearly hanging
    assert np.isfinite(horizontal_adduction(body(v), "right"))
    assert np.isnan(horizontal_adduction(body(v), "right", min_proj=0.25))
