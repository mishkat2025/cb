"""Phase 6 demo (spec §10): the online session reproduces the batch pipeline (the cached Phase 1 keypoints fed frame
by frame give the batch's own-view reps), and the overlay helpers."""
import types

import numpy as np
import pandas as pd
import pytest

from cvdc.config import data_path, load_config, resolve

CFG = load_config("configs/per_exercise.yaml")
LIB = resolve(CFG["templates"]["library_dir"])
HAVE = LIB.exists() and data_path(CFG, "reps_perview_csv").exists()


def cached_pose(video_id, upto=None):
    idx = pd.read_csv(resolve(CFG["data"]["index_csv"])).set_index("video_id")
    with np.load(resolve(idx.loc[video_id, "kp_path"])) as z:
        t, lm, w, det, hw = z["t"], z["lm2d"], z["world"], z["detected"], tuple(z["frame_hw"])
    n = len(t) if upto is None else int(np.searchsorted(t, upto))
    h2 = np.full((n, 2, 21, 3), np.nan, np.float32)
    return types.SimpleNamespace(frame_hw=hw, arrays=lambda: (t[:n], lm[:n], w[:n], det[:n], h2))


@pytest.mark.skipif(not HAVE, reason="needs the full-scope Phase 1 caches and the Module III library")
@pytest.mark.parametrize("video_id", ["E07_P05_AF_VFL_GM", "E03_P05_AF_VFL_GM", "E09_P02_AF_VFL_GM",
                                      "E05_P01_AR_VFL_GM"])
def test_online_session_final_equals_the_batch_own_view_reps(video_id):
    from demo.pipeline import OnlineSession
    ex = video_id[:3]
    idx = pd.read_csv(resolve(CFG["data"]["index_csv"])).set_index("video_id")
    sess = OnlineSession(CFG, ex, idx.loc[video_id, "view"])
    for upto in (5.0, 10.0):  # partial buffers first, as in the live loop
        sess.update(cached_pose(video_id, upto))
    sess.update(cached_pose(video_id), final=True)
    own = pd.read_csv(data_path(CFG, "reps_perview_csv"))
    own = own[own.video_id == video_id]
    assert [(r["start"], r["end"]) for r in sess.reps] == list(zip(own.start, own.end))
    assert sess.side == own.side.iloc[0]
    rep = sess.set_report(video_id)
    assert rep["n_reps"] == len(own) and all(0 <= r["score"] <= 100 for r in rep["reps"] if r["score"] is not None)


def test_faulty_joints_map_error_joints_to_landmarks():
    from cvdc.data.skeleton import J
    from demo.live import faulty_joints
    e = {"errors": [{"type": "x", "joint": "right_elbow"}, {"type": "y", "joint": "trunk"},
                    {"type": "z", "joint": "knees"}]}
    assert faulty_joints(e) == {J["right_elbow"], J["left_shoulder"], J["right_shoulder"], J["left_hip"],
                                J["right_hip"], J["left_knee"], J["right_knee"]}
    assert faulty_joints(None) == set()


def test_blur_faces_only_changes_the_head_box():
    from demo.live import blur_faces
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
    lm = np.full((33, 5), np.nan, np.float32)
    lm[:11, 0] = 0.5 + rng.normal(0, 0.02, 11)
    lm[:11, 1] = 0.1 + rng.normal(0, 0.01, 11)
    out = blur_faces(img.copy(), lm)
    assert (out[150:] == img[150:]).all() and not (out[:30, 40:60] == img[:30, 40:60]).all()
    state = {}
    blur_faces(img.copy(), lm, state)
    keep = blur_faces(img.copy(), None, state)  # a frame without landmarks reuses the last head box
    assert (keep[150:] == img[150:]).all() and not (keep[:30, 40:60] == img[:30, 40:60]).all()
    assert not (blur_faces(img.copy(), None)[150:] == img[150:]).all()  # no box yet: the whole frame is blurred


def test_mirror33_swaps_sides_and_is_an_involution():
    from cvdc.data.skeleton import J
    from demo.reference import mirror33
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 33, 2))
    Y = mirror33(X)
    assert np.allclose(mirror33(Y), X)
    assert np.allclose(Y[:, J["left_elbow"], 0], -X[:, J["right_elbow"], 0])
    assert np.allclose(Y[:, J["nose"]], X[:, J["nose"]] * [-1, 1])


def test_online_aligner_tracks_the_template_and_waits_at_rest():
    from demo.reference import OnlineAligner
    t = np.linspace(0, np.pi, 60)
    F = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1)  # a rest -> peak -> back movement on the unit circle
    a = OnlineAligner({"feat": F}, {"align": "arm"}, max_step=5)
    assert [a.step(F[0]) for _ in range(20)] == [0] * 20       # at rest: anchored at the template's start
    slow = np.repeat(F, 2, axis=0)                              # the same movement at half speed
    js = [a.step(f) for f in slow]
    assert all(b >= c for b, c in zip(js[1:], js[:-1]))          # monotonic
    err = np.abs(np.array(js) - np.repeat(np.arange(60), 2))
    assert np.median(err) <= 2 and js[-1] >= 55


@pytest.mark.skipif(not HAVE, reason="needs the full-scope caches and the Module III library")
@pytest.mark.parametrize("ex", ["E01", "E02", "E03", "E04"])
def test_reference_player_follows_the_template(ex):
    from cvdc.assess.module3 import assess_config
    from cvdc.assess.template import load_library
    from demo.reference import ReferencePlayer
    ref, lib, _ = load_library(LIB / ex)
    pl = ReferencePlayer(CFG, ex, ref, assess_config(CFG, ex))
    assert pl.xy.shape == (ref["n"], 33, 2) and np.isfinite(pl.xy).all()
    xy, mir = pl.frame(0, "left" if pl.side == "right" else "right")
    assert mir and np.allclose(xy[:, 0].mean(), -pl.xy[0][:, 0].mean())
    assert pl.view in ("front", "oblique")
