"""Module III on the full scope (M17): the E03-E09 channels on known geometry, the alignment segments, the end range
of a decreasing main angle (E03), the E03 working phase, experts-centred main channel (E06) and the configs."""
import numpy as np
import pytest

from cvdc.assess.angles import ASSESS_ANGLES, segment_direction
from cvdc.assess.module3 import compare
from cvdc.assess.run import working_phase
from cvdc.assess.template import align_feature, arm_direction, build_reference, end_range, rep_signals
from cvdc.config import load_config
from cvdc.data.skeleton import J

PEREX = load_config("configs/per_exercise.yaml")


def lying(n=1, hip_r=0.0, knee_r=0.0, hip_l=0.0, trunk_raise=0.0):
    """(n, 33, 3) supine / prone body along +x (head at +x), y up. The right thigh is flexed by ``hip_r`` degrees and
    the right knee by ``knee_r``; the left hip by ``hip_l``; the trunk raised by ``trunk_raise`` (about the hips)."""
    P = np.zeros((n, 33, 3))
    for s, z in (("left", -0.15), ("right", 0.15)):
        P[:, J[f"{s}_hip"]] = (0, 0, z)
        t = np.radians(trunk_raise)
        P[:, J[f"{s}_shoulder"]] = (0.5 * np.cos(t), 0.5 * np.sin(t), z)
        P[:, J[f"{s}_elbow"]] = (0.5 * np.cos(t) + 0.3, 0.5 * np.sin(t), z)
        P[:, J[f"{s}_wrist"]] = (0.5 * np.cos(t) + 0.6, 0.5 * np.sin(t), z)
        h = np.radians({"right": hip_r, "left": hip_l}[s])
        k = np.radians(knee_r if s == "right" else 0.0)
        knee = np.array([-0.45 * np.cos(h), 0.45 * np.sin(h), z])
        P[:, J[f"{s}_knee"]] = knee
        P[:, J[f"{s}_ankle"]] = knee + 0.45 * np.array([-np.cos(h + k), np.sin(h + k), 0])
    return P


def test_lower_body_channels_on_known_geometry():
    P = lying(hip_r=90, knee_r=60, hip_l=10)
    a = {c: float(ASSESS_ANGLES[c](P, "right")[0]) for c in ("hip_flexion", "knee_flexion", "contralateral_hip_flexion",
                                                            "contralateral_knee_flexion", "hip_flexion_asymmetry")}
    assert a["hip_flexion"] == pytest.approx(90, abs=1e-6) and a["knee_flexion"] == pytest.approx(60, abs=1e-6)
    assert a["contralateral_hip_flexion"] == pytest.approx(10, abs=1e-6)
    assert a["contralateral_knee_flexion"] == pytest.approx(0, abs=1e-6)
    assert a["hip_flexion_asymmetry"] == pytest.approx(80, abs=1e-6)
    assert float(ASSESS_ANGLES["knee_flexion_both"](P)[0]) == pytest.approx(30, abs=1e-6)
    assert float(ASSESS_ANGLES["trunk_thigh_angle"](lying(trunk_raise=30))[0]) == pytest.approx(30, abs=1e-6)


def test_arm_segment_equals_the_pilot_arm_direction():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 33, 3))
    for side in ("left", "right"):
        assert np.allclose(segment_direction(P, side, "shoulder", "elbow"), arm_direction(P, side))
        assert np.allclose(align_feature(P, side, {"align": "arm"}, {}), arm_direction(P, side))
    f = align_feature(P, "right", {"align": "thighs"}, {})
    assert f.shape == (20, 6) and np.allclose(np.linalg.norm(f.reshape(20, 2, 3), axis=-1), 1)
    m = align_feature(P, "right", {"align": "main", "angle": "x"}, {"x": np.r_[np.nan, np.arange(19.0)]})
    assert m.shape == (20, 1) and np.isfinite(m).all()


def test_end_range_of_a_decreasing_angle():
    a = np.r_[np.linspace(130, 27, 30), np.linspace(27, 130, 30)]
    assert end_range(a, direction="min") == pytest.approx(27, abs=4)
    assert end_range(a) == pytest.approx(130, abs=1e-6)


def test_rom_shortfall_sign_follows_the_direction():
    ref = {"main": "m", "channels": ["m"], "curves": {"m": np.full(10, 50.0)}, "rom": 20.0, "n": 10}
    lib = {"bias": {"m": np.zeros(10)}, "sigma": {"m": np.full(10, 5.0)}, "mean": {"m": np.full(10, 50.0)},
           "rom_bias": 0.0, "rom_sd": 5.0, "rom_mu": 25.0, "speed_mu": 0.0, "speed_sd": 1.0}
    al = {"curves": {"m": np.full(10, 50.0)}, "omega": np.zeros(10), "log2_speed": 0.0}
    ac = {"min_defined_frac": 0.3, "fps": 30, "rom_direction": "min"}
    # min direction: reaching only 30 deg when the target is 20 deg is a 10 deg shortfall
    assert compare({"rom": 30.0}, al, ref, lib, ac)["measures"]["rom_shortfall"] == pytest.approx(10)
    assert compare({"rom": 30.0}, al, ref, lib, {**ac, "rom_direction": "max"})["measures"]["rom_shortfall"] == \
        pytest.approx(-10)
    # experts-centred main channel (E06, M17): the ROM target is the view's expert mean
    m = compare({"rom": 30.0}, al, ref, lib, {**ac, "rom_direction": "max", "main_centre": "experts"})["measures"]
    assert m["rom_target"] == pytest.approx(25) and m["rom_shortfall"] == pytest.approx(-5)


def test_working_phase_keeps_the_longest_run_above_the_threshold():
    n = 60
    P = np.zeros((n, 33, 3))
    P[:, J["left_hip"]], P[:, J["right_hip"]] = (-0.15, 0, 0), (0.15, 0, 0)
    P[:, J["left_shoulder"]], P[:, J["right_shoulder"]] = (-0.2, 0.5, 0), (0.2, 0.5, 0)
    elev = np.r_[np.full(10, 20.0), np.full(40, 90.0), np.full(10, 20.0)]
    r = np.radians(elev)
    P[:, J["right_elbow"]] = np.stack([0.2 + 0.3 * np.sin(r), 0.5 - 0.3 * np.cos(r), np.zeros(n)], 1)
    assert working_phase(P, 0, n - 1, "right", {"channel": "shoulder_elevation", "min_deg": 70}) == (10, 49)
    assert working_phase(P, 0, 9, "right", {"channel": "shoulder_elevation", "min_deg": 70}) is None


def test_every_exercise_config_builds_a_template():
    """Every exercise's assess block runs end to end on a synthetic rep (channels, alignment, template)."""
    rng = np.random.default_rng(1)
    for ex in sorted(PEREX["exercises"]):
        from cvdc.assess.module3 import assess_config
        ac = assess_config(PEREX, ex)
        W = rng.normal(size=(80, 33, 3)) * 0.05 + lying(80, hip_r=40, knee_r=30)
        extra = {ac["angle"]: np.linspace(10, 60, 80)} if ac.get("angle_source") == "hands" else None
        sigs = [rep_signals(W, 5, 60, "right", ac, 30, extra), rep_signals(W, 10, 70, "right", ac, 30, extra)]
        ref = build_reference(sigs, ac, 30)
        assert set(ref["curves"]) == set(ac["channels"]) and ref["feat"].shape[0] == ref["n"]
        assert all(r["type"] for r in ac["rules"])
