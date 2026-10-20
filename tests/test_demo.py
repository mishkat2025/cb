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
    assert (blur_faces(img.copy(), None) == img).all()
