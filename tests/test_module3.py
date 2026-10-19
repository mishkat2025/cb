"""Phase 4, Module III: new joint angles on known geometry, mirroring, the template and its alignment, the library,
the score, rule-based errors, the per-video JSON schema, the library round trip, contained-rep counts and the
view-stability ICC."""
import numpy as np
import pandas as pd
import pytest

from cvdc.assess.angles import (ASSESS_ANGLES, contralateral_elevation, elbow_extension, shoulder_asymmetry,
                                shoulder_elevation, trunk_lateral_lean)
from cvdc.assess.module3 import (assess_config, detect_errors, evaluate, rep_entry, score, set_errors,
                                 validate_report, video_report)
from cvdc.assess.template import (aligned_rep, arm_direction, build_reference, end_range, fit_library, load_library,
                                  on_template, rep_signals, save_library)
from cvdc.data.skeleton import JOINT_SETS, MP33, J, mirror_coco17
from cvdc.eval.metrics import icc_weighted, mean_abs_diff_weighted
from cvdc.eval.recording import counts_per_view

FPS = 30
AC = {"angle": "shoulder_elevation", "channels": ["shoulder_elevation", "elbow_extension", "contralateral_elevation",
                                                  "trunk_lateral_lean", "shoulder_asymmetry"],
      "angle_min_proj": 0.25, "fps": FPS, "band_k": 2.0, "min_defined_frac": 0.3, "min_run_frac": 0.1,
      "min_run_frames": 3, "severity_sd": [1.0, 2.0], "sigma_floor_deg": {"default": 3.0, "rom": 5.0},
      "quality_bins": [50, 80], "quality_names": ["incorrect", "partially_correct", "correct"],
      "score": {"d_zero": 4.0, "weights": {"angle": 1.0, "rom": 1.0, "speed": 1.0, "symmetry": 1.0},
                "angle_channels": ["shoulder_elevation", "elbow_extension", "contralateral_elevation",
                                   "trunk_lateral_lean"], "symmetry_channel": "shoulder_asymmetry"},
      "rules": [{"type": "abduction_range_insufficient", "measure": "rom_shortfall", "joint": "moving_shoulder"},
                {"type": "elbow_bent", "channel": "elbow_extension", "direction": "below", "joint": "moving_elbow"},
                {"type": "too_fast", "measure": "speed", "direction": "fast"}]}


def body(n=1, abd=0.0, elbow_bend=0.0, lean=0.0, hike=0.0, side="right"):
    """(n, 33, 3) upright skeleton (x right, y up, z forward). The ``side`` arm is abducted by ``abd`` degrees in the
    frontal plane with the elbow bent by ``elbow_bend``; the trunk leans by ``lean`` toward +x; the ``side``
    shoulder is raised by ``hike`` degrees about the shoulder midpoint. Other arm hangs."""
    P = np.zeros((n, 33, 3))
    sgn = 1 if side == "right" else -1
    P[:, J["left_hip"]] = [-0.15, 0, 0]
    P[:, J["right_hip"]] = [0.15, 0, 0]
    L = np.radians(lean)
    up = np.array([np.sin(L), np.cos(L), 0.0])
    lat = np.array([np.cos(L), -np.sin(L), 0.0])
    ms = 0.5 * up
    h = np.radians(hike)
    P[:, J[f"{side}_shoulder"]] = ms + 0.2 * (np.cos(h) * sgn * lat + np.sin(h) * up)
    P[:, J[f"{'left' if side == 'right' else 'right'}_shoulder"]] = ms - 0.2 * (np.cos(h) * sgn * lat + np.sin(h) * up)
    a = np.radians(abd)
    arm = np.cos(a) * -up + np.sin(a) * sgn * lat
    sh = P[0, J[f"{side}_shoulder"]]
    el = sh + 0.3 * arm
    P[:, J[f"{side}_elbow"]] = el
    b = np.radians(elbow_bend)
    fwd = np.array([0, 0, 1.0])
    P[:, J[f"{side}_wrist"]] = el + 0.25 * (np.cos(b) * arm + np.sin(b) * fwd)
    o = "left" if side == "right" else "right"
    osh = P[0, J[f"{o}_shoulder"]]
    P[:, J[f"{o}_elbow"]] = osh - 0.3 * up
    P[:, J[f"{o}_wrist"]] = osh - 0.55 * up
    return P


def test_new_angles_known_geometry():
    assert elbow_extension(body(), "right")[0] == pytest.approx(180, abs=1e-6)
    assert elbow_extension(body(elbow_bend=90), "right")[0] == pytest.approx(90, abs=1e-6)
    assert trunk_lateral_lean(body(), "right")[0] == pytest.approx(0, abs=1e-6)
    assert trunk_lateral_lean(body(lean=10), "right")[0] == pytest.approx(10, abs=1e-6)
    assert trunk_lateral_lean(body(lean=10), "left")[0] == pytest.approx(-10, abs=1e-6)
    assert shoulder_asymmetry(body(), "right")[0] == pytest.approx(0, abs=1e-6)
    assert shoulder_asymmetry(body(hike=12), "right")[0] == pytest.approx(12, abs=1e-6)
    assert shoulder_asymmetry(body(hike=12, lean=8), "right")[0] == pytest.approx(12, abs=1e-6)  # lean-independent
    P = body(abd=90)
    assert shoulder_elevation(P, "right")[0] == pytest.approx(90, abs=1e-6)
    assert contralateral_elevation(P, "right")[0] == pytest.approx(0, abs=1e-6)


def test_left_arm_channels_equal_mirrored_right():
    L = body(n=3, abd=70, elbow_bend=20, lean=5, hike=6, side="left")
    M = L[:, [J[n.replace("left", "#").replace("right", "left").replace("#", "right")] for n in MP33]].copy()
    M[..., 0] *= -1  # mirror: x -> -x, left <-> right
    for c, f in ASSESS_ANGLES.items():
        if c == "horizontal_adduction":
            continue
        np.testing.assert_allclose(f(L, "left"), f(M, "right"), atol=1e-9, err_msg=c)
    np.testing.assert_allclose(arm_direction(L, "left"), arm_direction(M, "right"), atol=1e-9)


def test_mirror_coco17_involution():
    X = np.random.default_rng(0).normal(size=(2, 5, 17, 3))
    np.testing.assert_allclose(mirror_coco17(mirror_coco17(X)), X)
    Y = mirror_coco17(X)
    np.testing.assert_allclose(Y[..., 5, 0], -X[..., 6, 0])  # left shoulder <- mirrored right shoulder
    assert len(JOINT_SETS["coco17"]) == 17


def movement(n, peak=160.0, elbow_bend=0.0, pad=0.15):
    """(n, 33, 3) abduction rep: rest -> peak -> rest, with smooth timing."""
    t = np.linspace(0, 1, n)
    prof = np.clip((np.sin(np.pi * np.clip((t - pad) / (1 - 2 * pad), 0, 1))) ** 1.5, 0, 1)
    return np.concatenate([body(1, abd=10 + (peak - 10) * p, elbow_bend=elbow_bend) for p in prof])


def make_ref(n=90):
    W = movement(n)
    return build_reference([rep_signals(W, 0, n - 1, "right", AC, FPS), rep_signals(movement(70), 0, 69, "right", AC, FPS)],
                           AC, FPS)


def test_reference_and_alignment_recover_template():
    ref = make_ref()
    assert ref["n"] == 80 and ref["rom"] == pytest.approx(160, abs=1.5)
    sig = rep_signals(movement(60), 0, 59, "right", AC, FPS)  # same movement, faster
    al = aligned_rep(sig, ref)
    dev = al["curves"]["shoulder_elevation"] - ref["curves"]["shoulder_elevation"]
    assert np.nanmax(np.abs(dev)) < 6.0
    assert al["log2_speed"] == pytest.approx(np.log2(ref["duration_s"] / (59 / FPS)))
    x = np.arange(10.0)
    path = np.array([[0, 0], [1, 0], [2, 1], [3, 2]])
    np.testing.assert_allclose(on_template(x, path, 3), [0.5, 2, 3])


def test_end_range_ignores_single_frame_spike():
    a = np.r_[np.linspace(10, 150, 20), np.full(10, 150.0), np.linspace(150, 10, 20)]  # rise, hold, lower
    a[5] = 179.0
    assert end_range(a) == pytest.approx(150)
    tri = np.r_[np.linspace(10, 150, 20), np.linspace(150, 10, 20)]
    tri[25] = np.nan
    assert 142 < end_range(tri) <= 150


def library_for(ref, reps_by_view):
    curves = {v: [aligned_rep(s, ref)["curves"] for s in ss] for v, ss in reps_by_view.items()}
    scal = {v: {"rom": [s["rom"] for s in ss], "log2_speed": [aligned_rep(s, ref)["log2_speed"] for s in ss]}
            for v, ss in reps_by_view.items()}
    return fit_library(ref, curves, scal, AC["sigma_floor_deg"], min_reps=3)


def expert_reps(rng, k=12, peak_sd=4.0):
    out = []
    for _ in range(k):
        n = int(rng.integers(70, 95))
        out.append(rep_signals(movement(n, peak=160 + rng.normal(0, peak_sd), elbow_bend=abs(rng.normal(0, 4))), 0,
                               n - 1, "right", AC, FPS))
    return out


def test_library_bias_and_floor():
    ref = make_ref()
    rng = np.random.default_rng(1)
    reps_f = expert_reps(rng)
    reps_l = []  # view L reads the elevation 20 deg lower everywhere
    for s in expert_reps(rng):
        s = dict(s, ch=dict(s["ch"], shoulder_elevation=s["ch"]["shoulder_elevation"] - 20), rom=s["rom"] - 20)
        reps_l.append(s)
    lib = library_for(ref, {"F": reps_f, "L": reps_l})
    assert np.allclose(lib["F"]["bias"]["shoulder_elevation"], 0)
    b = lib["L"]["bias"]["shoulder_elevation"]
    assert np.nanmedian(b) == pytest.approx(-20, abs=2)
    assert lib["L"]["rom_bias"] == pytest.approx(-20, abs=3)
    assert np.nanmin(lib["F"]["sigma"]["trunk_lateral_lean"]) >= 3.0  # floor
    assert np.isnan(lib["R"]["bias"]["shoulder_elevation"]).all()  # no expert reps in R


def test_score_and_errors():
    ref = make_ref()
    rng = np.random.default_rng(2)
    lib = library_for(ref, {"F": expert_reps(rng)})
    good = rep_signals(movement(80), 0, 79, "right", AC, FPS)
    s_good, X, meas, errs, _ = evaluate(good, aligned_rep(good, ref), ref, lib["F"], AC)
    assert 0 <= s_good <= 100 and s_good > 75
    assert not [e for e in errs if e["type"] == "elbow_bent"]
    bad = rep_signals(movement(80, peak=100, elbow_bend=60), 5, 79, "right", AC, FPS)
    s_bad, Xb, mb, errs_b, _ = evaluate(bad, aligned_rep(bad, ref), ref, lib["F"], AC)
    assert s_bad < s_good and Xb["angle"] > X["angle"] and Xb["rom"] > X["rom"]
    types = {e["type"]: e for e in errs_b}
    assert {"elbow_bent", "abduction_range_insufficient"} <= set(types)
    e = types["elbow_bent"]
    assert e["joint"] == "right_elbow" and e["deviation_deg"] < 0 and 5 <= e["frames"][0] <= e["frames"][1] <= 79
    assert e["severity"] in ("mild", "moderate", "severe")
    assert types["abduction_range_insufficient"]["measured_deg"] < types["abduction_range_insufficient"]["target_deg"]
    s_all, _ = score({**mb, **{f"D_{c}": 99.0 for c in AC["channels"]}, "z_rom": 99.0, "z_speed": 99.0}, AC)
    assert s_all == pytest.approx(0.0)


def test_json_schema_and_round_trip(tmp_path):
    ref = make_ref()
    lib = library_for(ref, {"F": expert_reps(np.random.default_rng(3))})
    sig = rep_signals(movement(80, peak=110, elbow_bend=50), 0, 79, "right", AC, FPS)
    s, X, meas, errs, _ = evaluate(sig, aligned_rep(sig, ref), ref, lib["F"], AC)
    rep = rep_entry(0, sig, s, meas, X, errs, AC, {"model": "proposed_j1", "fold": 0, "cosine_centred": 0.3, "l2": 2.0})
    meta = {"exercise": "E01_abduction", "video_id": "E01_P99_AF_VFL_GM", "view": "F",
            "angle_source": "3d_mediapipe_world", "template": {"source": "pt_demo"}}
    d = video_report(meta, [rep], set_errors(8, 10), AC)
    assert validate_report(d) and d["set_errors"][0]["type"] == "too_few_reps"
    import json
    validate_report(json.loads(json.dumps(d)))  # survives JSON serialisation
    with pytest.raises(ValueError):
        validate_report({k: v for k, v in d.items() if k != "reps"})
    bad = json.loads(json.dumps(d))
    bad["reps"][0]["score"] = 120
    with pytest.raises(ValueError):
        validate_report(bad)
    bad = json.loads(json.dumps(d))
    bad["reps"][0]["errors"] = [{"type": "elbow_bent", "joint": "right_elbow", "severity": "mild", "frames": [90, 95]}]
    with pytest.raises(ValueError):
        validate_report(bad)
    save_library(tmp_path / "E01", ref, lib, {"exercise": "E01"})
    ref2, lib2, meta2 = load_library(tmp_path / "E01")
    assert meta2["exercise"] == "E01" and ref2["n"] == ref["n"]
    np.testing.assert_allclose(lib2["F"]["sigma"]["elbow_extension"], lib["F"]["sigma"]["elbow_extension"])
    s2, *_ = evaluate(sig, aligned_rep(sig, ref2), ref2, lib2["F"], AC)
    assert s2 == pytest.approx(s)


def test_assess_config_merges_score(monkeypatch):
    cfg = {"assess": {"band_k": 2, "score": {"d_zero": 4}}, "exercises": {"E01": {"assess": {"angle": "x",
                                                                                          "score": {"angle_channels": ["x"]}}}}}
    ac = assess_config(cfg, "E01")
    assert ac["score"] == {"d_zero": 4, "angle_channels": ["x"]} and ac["band_k"] == 2 and ac["angle"] == "x"


def test_counts_per_view():
    own = pd.DataFrame({"start": [30, 120, 210], "end": [90, 180, 270]})  # reps at 1-3, 4-6, 7-9 s (best view F)
    views = {"F": (0.0, 300), "L": (1.5, 300), "R": (-0.5, 220)}
    c = counts_per_view(own, 0.0, views, FPS, 0.1)
    assert c["F"] == (3, 3)
    assert c["L"] == (2, 3)  # L covers aligned -1.5 .. 8.47 s: the last rep (7-9 s) is cut, but less than half
    assert c["R"] == (2, 2)  # R covers aligned 0.5 .. 7.8 s: the last rep is cut by more than half


def test_icc_weighted():
    rng = np.random.default_rng(0)
    perf = rng.normal(0, 1, 50)
    x = np.repeat(perf, 3) + rng.normal(0, 0.5, 150)
    code = np.repeat(np.arange(50), 3)
    icc = icc_weighted(x, code, np.ones((1, 50)))[0]
    # balanced closed form: (MSB - MSW) / (MSB + (k - 1) MSW)
    m = x.reshape(50, 3)
    msb = 3 * ((m.mean(1) - x.mean()) ** 2).sum() / 49
    msw = ((m - m.mean(1, keepdims=True)) ** 2).sum() / 100
    assert icc == pytest.approx((msb - msw) / (msb + 2 * msw))
    assert icc_weighted(np.repeat(perf, 3), code, np.ones((1, 50)))[0] == pytest.approx(1.0)
    w2 = np.ones((1, 50))
    w2[0, :25] = 2  # duplicating sets equals stacking them twice
    xx = np.r_[x, x[:75]]
    cc = np.r_[code, code[:75] + 50]
    assert icc_weighted(x, code, w2)[0] == pytest.approx(icc_weighted(xx, cc, np.ones((1, 75)))[0], rel=1e-9)
    y = rng.normal(size=20)
    brute = np.mean([abs(a - b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j])
    assert mean_abs_diff_weighted(y, np.ones((1, 20)))[0] == pytest.approx(brute)


def test_movement_quality_score():
    from cvdc.data.labels import movement_quality

    df = pd.DataFrame({"PF1": [1, 1, 0], "PF2": [1, 0.5, 0], "PF3": [0, 1, 1], "PF4": [0.3, 1, 1], "CF1": [1, 1, 0],
                       "CF2": [1, 0.5, 0], "CF3": [1, 1, 0]})
    np.testing.assert_allclose(movement_quality(df), [100, 80, 0])  # PF3 / PF4 do not enter
