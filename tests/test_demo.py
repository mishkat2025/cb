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


def _arc_tracker(n_lead=8, n_tmpl=60, n_tail=8):
    """A RepTracker on a synthetic live axis: rest lead-in, a rest -> peak -> rest arc on the unit circle, rest tail."""
    from demo.reference import RepTracker
    t = np.r_[np.zeros(n_lead), np.sin(np.linspace(0, np.pi, n_tmpl)) * np.pi / 2, np.zeros(n_tail)]
    F = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1)
    tm = F[n_lead:n_lead + n_tmpl]
    player = types.SimpleNamespace(axis_feat=F, axis_begin=n_lead + 25, template_peak=np.arange(25, 35))
    return RepTracker(player, tm, ac={"align": "arm"}), F, n_lead


def test_rep_tracker_rest_and_rep_phase():
    tr, F, L = _arc_tracker()
    rest = F[0]
    assert [tr.push(rest) for _ in range(30)] == [(False, 0)] * 30   # at rest: rest pose, no rep
    rep = np.repeat(F[L:L + 60], 2, axis=0)                          # one rep at half the template's speed
    out = [tr.push(f) for f in rep]
    ks = [k for inr, k in out if inr]
    assert out[16][0] and all(b >= c for b, c in zip(ks[1:], ks[:-1]))  # in a rep after a few frames; monotonic
    mid = [k for (inr, k), j in zip(out, np.repeat(np.arange(60), 2)) if inr and 15 <= j <= 45]
    assert abs(np.median(mid) - (L + 30)) <= 6                      # the peak matched to the template's peak
    after = [tr.push(rest) for _ in range(12)]
    assert after[-1] == (False, 0)                                  # back at rest


def test_rep_tracker_ignores_moving_away_from_the_peak():
    tr, F, L = _arc_tracker()
    for _ in range(30):
        tr.push(F[0])
    away = np.array([np.cos(-0.5), np.sin(-0.5), 0.0])               # the other way from the rest (arm below rest)
    assert all(not tr.push(away)[0] for _ in range(20))


def test_rep_tracker_open_begin_from_another_rest():
    tr, F, L = _arc_tracker()
    tr.set_rest(F[L + 12])                                          # the patient rests part-way (E03's 90/90 hold)
    for _ in range(20):
        tr.push(F[L + 12])
    out = [tr.push(f) for f in np.repeat(F[L + 12:L + 30], 2, axis=0)]
    first = next(k for inr, k in out if inr)
    assert L + 10 <= first <= L + 25                                # starts near its own pose, not at the PT's rest


@pytest.mark.skipif(not HAVE, reason="needs the full-scope caches and the Module III library")
@pytest.mark.parametrize("ex", ["E01", "E02", "E03", "E04"])
def test_reference_player_shows_the_pt_rep_from_rest_to_rest(ex):
    from cvdc.assess.module3 import assess_config
    from cvdc.assess.template import load_library
    from demo.reference import ReferencePlayer
    ref, lib, _ = load_library(LIB / ex)
    pl = ReferencePlayer(CFG, ex, ref, assess_config(CFG, ex))
    s, p, e = pl.rep
    assert pl.xy.shape == (len(pl.seq), 33, 2) and np.isfinite(pl.xy).all() and pl.seq[pl.peak] == p
    if ex != "E03":  # E01, E02, E04: one contiguous stretch of the demo, rest -> rep -> rest (X7)
        assert (np.diff(pl.seq) == 1).all() and pl.seq[0] <= s and e <= pl.seq[-1] and pl.rest_k == 0
    assert (pl.tmap[:pl.covered[0]] == -1).all() and (pl.tmap[pl.covered[1] + 1:] == -1).all()
    t = pl.tmap[pl.tmap >= 0]
    assert t[0] == 0 and t[-1] == ref["n"] - 1 and (np.diff(t) >= 0).all()
    assert len(pl.axis_feat) == len(pl.axis_display) == len(pl.axis_template) == pl.covered[0] + ref["n"] + (
        pl.n - 1 - pl.covered[1])
    xy, mir = pl.frame(0, "left" if pl.side == "right" else "right")
    assert mir and np.allclose(xy[:, 0].mean(), -pl.xy[0][:, 0].mean())
    assert pl.view in ("front", "oblique")


@pytest.mark.skipif(not HAVE, reason="needs the full-scope caches and the Module III library")
def test_e01_reference_rest_pose_has_the_arm_down():
    """The user's report (2026-10-20): at rest the reference showed the arm ~20-30 deg out (the segmented rep's first
    frame); the PT's arm hangs at ~9 deg (2D) at rest in the video. The display window now starts and ends at rest."""
    from cvdc.assess.module3 import assess_config
    from cvdc.assess.template import load_library
    from cvdc.data.skeleton import J
    from demo.reference import ReferencePlayer
    ref, _, _ = load_library(LIB / "E01")
    pl = ReferencePlayer(CFG, "E01", ref, assess_config(CFG, "E01"))

    def arm(xy):
        down = (xy[J["left_hip"]] + xy[J["right_hip"]]) / 2 - (xy[J["left_shoulder"]] + xy[J["right_shoulder"]]) / 2
        u = xy[J[f"{pl.side}_elbow"]] - xy[J[f"{pl.side}_shoulder"]]
        return np.degrees(np.arccos(u @ down / np.linalg.norm(u) / np.linalg.norm(down)))
    assert arm(pl.xy[0]) < 13 and arm(pl.xy[-1]) < 13 and arm(pl.xy[pl.peak]) > 160


@pytest.mark.skipif(not HAVE, reason="needs the full-scope caches and the Module III library")
def test_e03_reference_rests_at_the_90_90_hold():
    """X8 (user, 2026-10-20): E03's displayed rest pose is the demo's 90/90 hold (the patients hold 90/90 between
    reps; the demo rests with the arm at the side), and the return ends there (the rotation played back)."""
    from cvdc.assess.angles import ASSESS_ANGLES
    from cvdc.assess.module3 import assess_config
    from cvdc.assess.template import load_library
    from demo.reference import ReferencePlayer
    ref, _, _ = load_library(LIB / "E03")
    pl = ReferencePlayer(CFG, "E03", ref, assess_config(CFG, "E03"))
    idx = pd.read_csv(resolve(CFG["data"]["ref_index_csv"]))
    row = idx[(idx.exercise == "E03") & (idx.role == "primary")].iloc[0]
    with np.load(resolve(row.pre_path)) as z:
        w = z["world"].astype(float)
    el, fo = ASSESS_ANGLES["shoulder_elevation"](w, pl.side), ASSESS_ANGLES["forearm_elevation"](w, pl.side)
    rest = pl.seq[pl.rest_k]
    assert el[rest] > 70 and 60 < fo[rest] < 110                    # arm at 90/90, forearm forward
    assert fo[pl.seq[pl.peak]] < 30                                  # forearm up at the peak
    assert pl.seq[-1] == rest and (np.diff(pl.seq[pl.tail[0]:]) == -1).all()  # the return ends at the hold
    assert (pl.tmap[pl.tail[0]:] == -1).all() and pl.rest_label and pl.return_label


# ---------------------------------------------------------------- X11: the assessment is Module III's score
ENC = resolve(CFG["eval"]["results_dir"]) / "completion" / "encode" / "perex_gkf5" / "E03"
LIB_C = resolve(load_config("configs/completion.yaml")["templates"]["library_dir"])


@pytest.mark.skipif(not (ENC.exists() and LIB_C.exists()), reason="needs the completion encodings and fold libraries")
def test_completed_demo_aligns_like_the_batch_run_and_reports_module3_as_the_assessment():
    from cvdc.assess.cvdc_join import align_steps
    from demo.pipeline import ExerciseModel, OnlineSession

    reps = pd.read_csv(data_path(CFG, "reps_csv")).reset_index()
    m = ExerciseModel(CFG, "E03", subject="P07")
    sess = OnlineSession(CFG, "E03", "F", m)
    assert sess.mode == "completed" and "CVDC" in sess.pipeline_label
    idx = pd.read_csv(resolve(CFG["data"]["index_csv"])).set_index("video_id")
    same = 0
    for r in reps[(reps.subject == "P07") & (reps.exercise == "E03") & (reps.view == "F")].head(5).itertuples():
        with np.load(resolve(idx.loc[r.video_id, "pre_path"])) as z:
            pre = {"xy": z["xy"]}
        sp, d = align_steps(m.sequence(pre, r.start, r.end), sess.R[r.side])
        with np.load(ENC / f"fold{m.fold}.npz") as z:
            i = list(z["item"]).index(r.index)
            L = int(z["join_len"][i])
            spb, db = z["join_path"][i, :L].astype(int), z["join_dist"][i, :L]
        same += int(len(sp) == L and (sp == spb).all())
        assert abs(float(d.mean()) - float(np.nanmean(db))) < 0.02  # CPU fp32 vs the batch's GPU bf16
    assert same >= 4
    rep = sess.set_report("test")
    assert rep["set_score_source"] == "module3_with_cvdc" and rep["model"]["role"].startswith("supplementary")


@pytest.mark.skipif(not HAVE, reason="needs the full-scope Phase 1 caches and the Module III library")
def test_without_a_model_the_assessment_is_the_rule_based_module3():
    from demo.pipeline import OnlineSession

    sess = OnlineSession(CFG, "E01", "F")
    rep = sess.set_report("test")
    assert sess.mode == "rules" and rep["set_score_source"] == "module3_rules" and "model" not in rep
