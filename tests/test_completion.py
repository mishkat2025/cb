"""Proposal completion (2026-10-22): M20 angular-velocity channel, M19 Module III options, A11 DepthAnything sampling."""
import copy

import numpy as np
import torch

from cvdc.assess.module3 import assess_config, groups, score
from cvdc.assess.template import aligned_rep, rep_signals
from cvdc.config import load_config
from cvdc.data.dataset import LIFTED
from cvdc.models.depth import DepthAnything
from cvdc.models.model import INPUT_CHANNELS


def _cfg():
    return load_config("configs/completion.yaml")


def test_completion_config_adds_rules_and_centres_without_touching_the_exercise_config():
    cfg = _cfg()
    before = copy.deepcopy(cfg["exercises"]["E01"]["assess"])
    ac = assess_config(cfg, "E01")
    types = [r["type"] for r in ac["rules"]]
    assert "angular_velocity_deviation" in types and "movement_pattern_deviation" in types
    rv = next(r for r in ac["rules"] if r["type"] == "angular_velocity_deviation")
    rom = next(r for r in ac["rules"] if r.get("measure") == "rom_shortfall")
    assert rv["joint"] == rom["joint"] and rv["channel"] == "angvel"
    assert ac["channel_centre"]["angvel"] == "experts" and ac["channel_centre"]["emb_dist"] == "experts"
    assert cfg["exercises"]["E01"]["assess"] == before  # the exercise's own rule list is copied, never changed


def test_published_config_has_no_new_rules_or_groups():
    cfg = load_config("configs/per_exercise.yaml")
    ac = assess_config(cfg, "E01")
    assert not any(r.get("channel") in ("angvel", "emb_dist") for r in ac["rules"])
    meas = {"z_rom": 1.0, "z_speed": 0.5, "D_angvel": 3.0, "Dplus_emb_dist": 2.0,
            **{f"D_{c}": 1.0 for c in ac["score"]["angle_channels"]}, f"D_{ac['score']['symmetry_channel']}": 1.0}
    assert set(groups(meas, ac)) == {"angle", "rom", "speed", "symmetry"}


def test_velocity_and_embedding_groups_enter_the_score_with_their_weight():
    ac = assess_config(_cfg(), "E01")
    base = {"z_rom": 0.0, "z_speed": 0.0, f"D_{ac['score']['symmetry_channel']}": 0.0,
            **{f"D_{c}": 0.0 for c in ac["score"]["angle_channels"]}}
    s0, _ = score({**base, "D_angvel": 0.0, "Dplus_emb_dist": 0.0}, ac)
    s1, X = score({**base, "D_angvel": 4.0, "Dplus_emb_dist": 0.0}, ac)
    assert s0 == 100.0 and X["velocity"] == 4.0
    assert np.isclose(s1, 100.0 * (1 - 1 / 6))  # one of six equally weighted groups at d_zero


def test_angular_velocity_channel_is_the_gradient_of_the_main_angle():
    ac = assess_config(_cfg(), "E01")
    n = 60
    world = np.zeros((n, 33, 3))
    world[:, 11], world[:, 12] = [0.2, -0.5, 0], [-0.2, -0.5, 0]  # shoulders
    world[:, 23], world[:, 24] = [0.1, 0.5, 0], [-0.1, 0.5, 0]  # hips
    ang = np.linspace(0, np.pi / 2, n)
    for j, r in ((14, 0.3), (16, 0.6)):  # right elbow / wrist swing out to the side
        world[:, j] = np.stack([-0.2 - r * np.sin(ang), -0.5 + r * np.cos(ang), np.zeros(n)], 1)
    sig = rep_signals(world, 0, n - 1, "right", ac, 30.0)
    main = ac["angle"]
    assert np.allclose(sig["ch"]["angvel"], np.gradient(sig["ch"][main]) * 30.0, equal_nan=True)
    assert np.nanmedian(sig["ch"]["angvel"]) > 0  # the arm rises


def test_aligned_rep_uses_a_given_frame_path():
    ref = {"feat": np.zeros((5, 3)), "n": 5, "channels": ["a"], "main": "a", "duration_s": 4 / 30}
    sig = {"feat": np.zeros((5, 3)), "n": 5, "duration_s": 4 / 30, "ch": {"a": np.arange(5.0)},
           "frame_path": np.array([[0, 0], [1, 0], [2, 1], [3, 2], [4, 3], [4, 4]])}
    al = aligned_rep(sig, ref)
    assert np.allclose(al["curves"]["a"], [0.5, 2, 3, 4, 4])


def test_depth_inputs_are_registered():
    assert LIFTED["xyd_depthanything"] == "depthanything" and INPUT_CHANNELS["xyd_depthanything"] == 3
    assert _cfg()["methods"]["b2_depthanything"]["input"] == "xyd_depthanything"


def test_depth_sampling_reads_the_map_at_the_joint_positions():
    h, w = 28, 42
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    disp = (xx + 100 * yy)[None]  # value encodes the pixel
    uv = np.array([[[(10 + 0.5) / w, (5 + 0.5) / h], [(0.5) / w, (0.5) / h], [np.nan, 0.5]]])
    v = DepthAnything.sample(disp, uv).numpy()[0]
    assert np.isclose(v[0], 10 + 500) and abs(v[1]) < 1e-4 and np.isnan(v[2])
    assert float(DepthAnything.spread(disp)[0]) > 0
