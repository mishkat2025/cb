"""Full-scope additions (2026-10-20): config inheritance, sync rules S8, adaptive rep segmentation, the new rep signals
and angles, and the MediaPipe Hands crop geometry."""
import numpy as np
import pandas as pd
import pytest

from cvdc.assess.angles import (all_joint_pc1, forearm_elevation, hip_flexion, knee_flexion, trunk_thigh_angle,
                                wrist_extension, wrist_height_2d)
from cvdc.config import deep_merge, load_config, video_root
from cvdc.data.extract import hand_crop, pick_hand
from cvdc.data.reps import adaptive_params, segment_signal, video_info
from cvdc.data.skeleton import J
from cvdc.data.sync import classify

# ------------------------------------------------------------------------------------------------ config


def test_deep_merge_merges_dicts_and_replaces_lists():
    a = {"x": {"p": 1, "q": [1, 2]}, "y": 1}
    b = {"x": {"q": [3], "r": 2}}
    assert deep_merge(a, b) == {"x": {"p": 1, "q": [3], "r": 2}, "y": 1}
    assert a == {"x": {"p": 1, "q": [1, 2]}, "y": 1}  # inputs untouched


def test_full_config_moves_every_output_and_keeps_the_pilot_config():
    pilot, full = load_config("configs/data.yaml"), load_config("configs/data_full.yaml")
    assert pilot["data"]["include_exercises"] == ["E01", "E02"] and pilot["data"]["index_csv"] == "data/index.csv"
    assert not pilot["sync"]["meta"].get("start_time_rule") and "signals" not in pilot["sync"]["motion"]
    # 2026-10-22 file-structure cleanup: the videos moved from D:\cb\mobiphysio to D:\cb\dataset\videos
    assert full["data"]["include_exercises"] == [] and video_root(full).parts[-2:] == ("dataset", "videos")
    assert full["train"]["cvdc_init_from"] is None
    # no full-scope path may point at a pilot location
    paths = [full["data"][k] for k in ("index_csv", "ref_index_csv", "reps_csv", "reps_perview_csv", "reps_ref_csv")]
    paths += [full["pose"]["kp_dir"], full["pose"]["ref_kp_dir"], full["preprocess"]["out_dir"],
              full["preprocess"]["pack_dir"], full["sync"]["cache_dir"], full["sync"]["report_csv"],
              full["sync"]["pairs_csv"], full["lifter"]["out_dir"], full["cvdc"]["pairs"]["pairs_csv"],
              full["eval"]["results_dir"], full["train"]["runs_dir"], *full["outputs"].values()]
    assert all("/full/" in p or p.endswith("/full") for p in paths), paths
    # model keys not overridden are still merged
    assert full["model"] == pilot["model"]

# -------------------------------------------------------------------------------------------- sync S8


def _group(dc, dur, lags, motion=None, gid="E05_P01_VFL"):
    """A F+L+R group: creation-time differences and durations per view, audio lags per pair."""
    idx = pd.DataFrame([dict(video_id=f"{gid}_{v}", exercise=gid[:3], subject="P01", variation="FL", group="expert",
                             view=v, group_id=gid) for v in "FLR"])
    rows = []
    for a, b in (("F", "L"), ("F", "R"), ("L", "R")):
        m = (motion or {}).get((a, b))
        rows.append(dict(group_id=gid, video_a=f"{gid}_{a}", video_b=f"{gid}_{b}", view_a=a, view_b=b,
                         d_creation_s=dc[b] - dc[a], dur_a=dur[a], dur_b=dur[b], audio_lag_s=lags[(a, b)],
                         audio_lag_dev_s=lags[(a, b)], audio_edge=False, motion_confirmed=m is not None,
                         motion_lag_s=m if m is not None else np.nan))
    P = pd.DataFrame(rows)
    P["audio_edge"] = P.audio_edge.astype("boolean")
    P["motion_confirmed"] = P.motion_confirmed.astype("boolean")
    return idx, P


SC = {"audio": {"additivity_ms": 20}, "meta": {"max_dc_s": 3.0}, "audio_lag_weight": 0.04,
      "pair_weights": {"sync": 1.0, "audio": 0.8, "rep": 0.5}}
S8 = {**SC, "meta": {"max_dc_s": 3.0, "windows_override_audio": True, "start_time_rule": True}}
ADDITIVE = {("F", "L"): 0.5, ("F", "R"): 0.8, ("L", "R"): 0.3}


def test_s8a_windows_that_cannot_overlap_beat_additive_audio():
    idx, P = _group({"F": 0, "L": 20, "R": 45}, {"F": 2, "L": 2, "R": 2}, ADDITIVE)
    assert classify(idx, P, SC)[0].iloc[0][["cls", "method"]].tolist() == ["synced", "audio"]   # pilot rule
    assert classify(idx, P, S8)[0].iloc[0][["cls", "method"]].tolist() == ["not-synced", "metadata"]


def test_s8b_start_time_rule_and_its_pair_sources():
    # stop times 8-10 s apart, start times (creation - duration) within 1 s; audio lags not additive
    lags = {("F", "L"): 0.5, ("F", "R"): 0.8, ("L", "R"): 0.9}
    dc, dur = {"F": 0.0, "L": 8.5, "R": 10.0}, {"F": 40.0, "L": 48.0, "R": 50.5}
    idx, P = _group(dc, dur, lags)
    assert classify(idx, P, SC)[0].iloc[0].cls == "unknown"
    G, pairs = classify(idx, P, S8)
    assert G.iloc[0][["cls", "method", "confidence"]].tolist() == ["synced", "metadata_start", "medium"]
    assert set(pairs.pair_source) == {"audio"}          # no motion confirmation: w_audio, never w_sync
    idx, P = _group(dc, dur, lags, motion={("F", "R"): 0.85})
    pairs = classify(idx, P, S8)[1].set_index(["view_a", "view_b"]).pair_source
    assert pairs[("F", "R")] == "sync" and pairs[("F", "L")] == "audio" and pairs[("L", "R")] == "audio"

# --------------------------------------------------------------------------------- adaptive segmentation


def _reps_signal(n_reps=6, valley=False, fps=30, amp=1.0, noise=0.02, seed=0):
    t = np.arange(0, 3.0 * n_reps + 2, 1 / fps)
    s = amp * np.clip(np.sin(2 * np.pi * (t - 1) / 3.0), 0, None) ** 2 * (t > 1) * (t < 1 + 3.0 * n_reps)
    s = s + np.random.default_rng(seed).normal(0, noise, len(t))
    return (-s + 5) if valley else s


@pytest.mark.parametrize("valley", [False, True])
@pytest.mark.parametrize("amp", [0.05, 50.0])
def test_adaptive_segmentation_is_scale_free_and_finds_the_polarity(valley, amp):
    rc = dict(adaptive=True, prominence_frac=0.4, min_amplitude_frac=0.3, polarity="auto", rest_window_s=1.0,
              median_s=0.3, min_distance_s=1.0, rest_frac=0.1, pad_s=0.1, min_rep_s=0.8, max_rep_s=20.0)
    s = _reps_signal(valley=valley, amp=amp, noise=0.02 * amp)
    p = adaptive_params(s, 30, rc)
    assert p["polarity"] == ("valley" if valley else "peak")
    reps, pol = segment_signal(s, 30, rc)
    assert pol == p["polarity"] and len(reps) == 6
    assert adaptive_params(np.zeros(200), 30, rc) is None and segment_signal(np.zeros(200), 30, rc)[0] == []

# --------------------------------------------------------------------------------------- signals, angles


def _pose(n=1):
    P = np.zeros((n, 33, 3))
    P[:, J["left_shoulder"]], P[:, J["right_shoulder"]] = (-1, 0, 0), (1, 0, 0)
    P[:, J["left_hip"]], P[:, J["right_hip"]] = (-1, -3, 0), (1, -3, 0)
    for s, x in (("left", -1), ("right", 1)):
        P[:, J[f"{s}_knee"]] = (x, -6, 0)
        P[:, J[f"{s}_ankle"]] = (x, -9, 0)
        P[:, J[f"{s}_elbow"]] = (x, -2, 0)
        P[:, J[f"{s}_wrist"]] = (x, -4, 0)
    return P


def test_hip_knee_and_trunk_angles_on_known_geometry():
    P = _pose()
    assert np.allclose(hip_flexion(P, "right"), 0) and np.allclose(knee_flexion(P, "left"), 0)
    P[:, J["right_knee"]] = (1, -3, 3)              # thigh forward, horizontal: 90 deg hip flexion
    P[:, J["right_ankle"]] = (1, -6, 3)             # shank down: 90 deg knee flexion
    assert np.allclose(hip_flexion(P, "right"), 90) and np.allclose(knee_flexion(P, "right"), 90)
    Q = _pose()
    Q[:, [J["left_shoulder"], J["right_shoulder"]], 2] = 3     # trunk tilted 45 deg off the thigh line
    assert np.allclose(trunk_thigh_angle(Q), 45, atol=1e-6)


def test_forearm_elevation_and_wrist_height():
    P = _pose()
    assert np.allclose(forearm_elevation(P, "right"), 180)   # forearm hanging down
    P[:, J["right_wrist"]] = (1, 0, 0)                         # forearm up along the trunk axis
    assert np.allclose(forearm_elevation(P, "right"), 0)
    xy = _pose()[..., :2] * (1, -1)                            # image axes: y down
    assert np.allclose(wrist_height_2d(xy, "left"), -4)


def test_trunk_inclination_2d_is_independent_of_facing_direction():
    from cvdc.assess.angles import trunk_inclination_2d
    xy = np.zeros((2, 33, 2))
    for k, sx in enumerate((1, -1)):                  # lying with the head to the right, then to the left
        xy[k, [J["left_hip"], J["right_hip"]]] = (0, 0)
        xy[k, [J["left_shoulder"], J["right_shoulder"]]] = (sx * np.cos(np.radians(30)), -np.sin(np.radians(30)))
    assert np.allclose(trunk_inclination_2d(xy), 30)   # image y points down: shoulders above the hips = positive


def test_per_exercise_dtw_angles_and_their_features():
    from cvdc.data.pairs import ANGLES, angle_features, dtw_angle_names
    full, pilot = load_config("configs/data_full.yaml"), load_config("configs/data.yaml")
    assert dtw_angle_names(pilot, "E01") == pilot["cvdc"]["pairs"]["dtw_angles"]       # the pilot is unchanged
    assert dtw_angle_names(full, "E01") == pilot["cvdc"]["pairs"]["dtw_angles"]
    for ex in ("E07", "E08"):                          # lower-body angles (user, 2026-10-20)
        names = dtw_angle_names(full, ex)
        assert names and all(n.startswith(("hip_", "knee_", "trunk_")) for n in names)
    # E09: the set with the lowest known-warp error (user, eleventh review): the arm angles (A5)
    assert dtw_angle_names(full, "E09") == pilot["cvdc"]["pairs"]["dtw_angles"]
    assert all(set(dtw_angle_names(full, ex)) <= set(ANGLES) for ex in full["exercises"])
    w = np.repeat(_pose(), 5, 0)
    F = angle_features(w, ["hip_flexion_left", "knee_flexion_right", "trunk_thigh"])
    assert F.shape == (5, 3) and np.allclose(F, 0)


def test_all_joint_pc1_follows_the_dominant_motion():
    t = np.linspace(0, 4 * np.pi, 200)
    P = np.repeat(_pose()[..., :2], 200, 0)
    P[:, J["right_wrist"], 1] += 2 * np.sin(t)
    s = all_joint_pc1(P)
    assert abs(np.corrcoef(s, np.sin(t))[0, 1]) > 0.99
    P[5, 0, 0] = np.nan
    assert np.isnan(all_joint_pc1(P)).all()


def _hand(wrist, dorsal_bend, side):
    """Hand landmarks (camera axes: x right, y down, z away) for a hand hanging down (fingers -y), palm facing the
    camera (-z), bent by ``dorsal_bend`` deg toward its back (+z). A right hand hanging with the palm to the camera
    has its thumb (index MCP) on image left; a left hand on image right."""
    H = np.zeros((21, 3))
    b = np.radians(dorsal_bend)
    axis = np.array([0, np.cos(b), np.sin(b)])                   # wrist -> middle MCP: down, tilted to the back
    lat = np.array([1.0, 0, 0]) * (-1 if side == "right" else 1)   # toward the index finger
    H[0] = wrist
    H[9] = wrist + axis
    H[5] = wrist + axis + 0.3 * lat
    H[17] = wrist + axis - 0.3 * lat
    return H


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("bend", [40.0, -30.0, 0.0])
def test_wrist_extension_sign_and_size(side, bend):
    P = np.zeros((33, 3))
    P[J[f"{side}_elbow"]], P[J[f"{side}_wrist"]] = (0, -1, 0), (0, 0, 0)   # forearm pointing down the image (+y)
    P[J[f"{side}_elbow"]] = (0, -1, 0)
    H = _hand(np.zeros(3), bend, side)
    assert np.isclose(wrist_extension(P, H, side), bend, atol=1e-6)
    # only directions count: a hand in its own (translated) frame gives the same angle
    assert np.isclose(wrist_extension(P, H + 5.0, side), bend, atol=1e-6)
    assert np.isclose(wrist_extension(P[:, :2], H[:, :2], side), 0.0, atol=1e-6)   # 2D: unsigned, in-plane only

# --------------------------------------------------------------------------------------------- hands


def test_hand_crop_and_mapping_back_round_trip():
    hc = dict(crop_px=224, crop_scale=2.2, ahead=0.4)
    W, H = 540, 960
    elbow, wrist = (0.40, 0.50), (0.45, 0.60)
    M, c, half, fa = hand_crop(elbow, wrist, W, H, hc)
    assert np.isclose(fa, np.hypot(0.05 * W, 0.1 * H)) and np.isclose(half, 1.1 * fa)
    wpx = np.array(wrist) * (W, H)
    crop_xy = M @ np.r_[wpx, 1]                       # the Pose wrist inside the crop (pixels)
    lm_near = np.zeros((21, 3))
    lm_near[:, :2] = crop_xy / 224                    # a hand whose wrist sits on the Pose wrist
    lm_far = lm_near.copy()
    lm_far[:, 0] += 0.3
    best = pick_hand([(lm_far, np.zeros((21, 3)), "Left", 0.9), (lm_near, np.ones((21, 3)), "Right", 0.8)],
                     wpx, c, half, fa, W, H)
    xyz, world, d, lab, sc = best
    assert np.allclose(xyz[0, :2], wrist, atol=1e-6) and d < 1e-6 and lab == 1 and sc == 0.8 and world[0, 0] == 1
    assert hand_crop((np.nan, 0.5), wrist, W, H, hc) is None


def test_orient_by_flips_an_arbitrary_sign_pc_to_follow_the_angle():
    from cvdc.data.reps import orient, video_angles
    t = np.linspace(0, 4 * np.pi, 200)
    ref = 1 - np.cos(t)
    assert np.allclose(orient(-ref * 0.3, ref), ref * 0.3) and np.allclose(orient(ref, ref, sign=-1), -ref)
    assert np.array_equal(orient(np.zeros(5), np.arange(5.0)), np.zeros(5))   # flat: unchanged
    n = len(t)
    w = np.repeat(_pose(), n, 0)
    w[:, J["left_knee"], 2] = 2 * ref                 # the left hip flexes
    w[:, J["left_ankle"], 2] = w[:, J["left_knee"], 2]
    xy = w[..., :2] * (1, -1)
    xy[:, J["left_knee"], 0] += 0.5 * ref             # some 2D motion for the PC
    pre = dict(world=w, xy=xy, vis=np.ones((n, 33)), t=np.arange(n) / 30)
    for sign in (1, -1):
        pa = dict(name="all_joint_pc1", source="image", orient_by=dict(name="hip_flexion", source="world", sign=sign))
        s = video_angles(pre, pa)["left"]
        assert sign * np.corrcoef(s, ref)[0, 1] > 0.9


def test_video_info_side_from_names_the_moving_limb():
    n = 120
    t = np.linspace(0, 4 * np.pi, n)
    w = np.repeat(_pose(), n, 0)
    w[:, J["left_knee"], 2] = 2 * (1 - np.cos(t))     # the left thigh swings forward
    w[:, J["left_ankle"], 2] = w[:, J["left_knee"], 2]
    xy = w[..., :2] * (1, -1)
    pre = dict(world=w, xy=xy, vis=np.ones((n, 33)), t=np.arange(n) / 30)
    ex = dict(primary_angle=dict(name="all_joint_pc1", source="image",
                                 side_from=dict(name="hip_flexion", source="world")),
              reps=dict(quality_joints=["knee", "ankle"]))
    _, info = video_info(pre, ex)
    assert info["side"] == "left" and info["quality"] == 1.0
    assert info["range_signal"] == "hip_flexion" and info["unit"] == "deg"   # the ranges are hip-flexion ranges
