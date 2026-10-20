"""Full-scope Phase 3 (eleventh review, 2026-10-20): GroupKFold-10 balanced for expert / non-expert, the exercise-label
exclusion (D8), the CVDC encoder shared by the Proposed variants (T15), the set-feature main angles of E03-E09 and the
E11 group probe pieces."""
import json
import types

import numpy as np
import pandas as pd
import pytest

from cvdc.config import load_config
from cvdc.data.dataset import exercise_label_ok
from cvdc.data.skeleton import J
from cvdc.eval.baselines import main_angle
from cvdc.eval.splits import group_kfold
from cvdc.train.run import load_cvdc
from cvdc.train.trainer import fold_plan

FULL = load_config("configs/data_full.yaml")
PILOT = load_config("configs/data.yaml")


def toy_meta(n_exp=24, n_non=34):
    rows = [dict(subject=f"P{i:02d}", group="expert", view=v) for i in range(1, n_exp + 1) for v in "FLR"]
    rows += [dict(subject=f"P{i:02d}", group="non-expert", view="F") for i in range(26, 26 + n_non)]
    return pd.DataFrame(rows)


def test_full_scope_uses_groupkfold_10_with_both_groups_in_every_fold():
    meta = toy_meta()
    assert FULL["eval"]["k"] == 10 and PILOT["eval"]["k"] == 5
    plans = fold_plan(FULL, meta)
    grp = meta.drop_duplicates("subject").set_index("subject").group
    assert len(plans) == 10
    tests = [s for p in plans for s in p["test"]]
    assert sorted(tests) == sorted(meta.subject.unique())  # the test folds partition the subjects
    for p in plans:
        n_exp = sum(grp[s] == "expert" for s in p["test"])
        assert n_exp in (2, 3) and len(p["test"]) - n_exp in (3, 4)  # 24 / 34 dealt round-robin over 10 folds
        assert not set(p["test"]) & set(p["train"]) and not set(p["inner_val"]) & set(p["test"])
        assert {grp[s] for s in p["inner_val"]} == {"expert", "non-expert"}


def test_groupkfold_10_is_seeded():
    meta = toy_meta()
    assert group_kfold(meta, 10, 0) == group_kfold(meta, 10, 0)
    assert group_kfold(meta, 10, 0) != group_kfold(meta, 10, 1)


def test_exercise_label_exclusion_only_hits_the_listed_subject():
    m = pd.DataFrame(dict(exercise=["E03", "E03", "E01", "E05"], subject=["P02", "P01", "P02", "P30"]))
    assert exercise_label_ok(FULL, m).tolist() == [False, True, True, True]  # D8: E03 P02 only
    assert exercise_label_ok(PILOT, m).all()                                    # the pilot has no exclusion


def _write_cvdc(d, plan, summary=True):
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"")
    (d / "run_info.json").write_text(json.dumps({k: plan[k] for k in ("inner_train", "inner_val", "test")}),
                                     encoding="utf-8")
    if summary:
        (d / "summary.json").write_text(json.dumps({"best_epoch": 1, "stopped_epoch": 1}), encoding="utf-8")


def test_cvdc_encoder_is_only_shared_after_a_finished_pretraining_on_the_same_subjects(tmp_path):
    plan = fold_plan(FULL, toy_meta())[0]
    data = types.SimpleNamespace(device="cpu")
    quiet = lambda *a: None  # noqa: E731
    _write_cvdc(tmp_path / "a", plan, summary=False)        # interrupted pretraining: best.pt without summary.json
    assert load_cvdc(FULL, data, plan, tmp_path / "a", quiet) is None
    other = fold_plan(FULL, toy_meta())[1]
    _write_cvdc(tmp_path / "b", other)                      # another fold's subjects
    assert load_cvdc(FULL, data, plan, tmp_path / "b", quiet) is None
    assert FULL["methods"]["proposed_lp"]["cvdc_from"] == "proposed_j1"
    assert "cvdc_from" not in PILOT["methods"]["proposed_lp"]  # the pilot is unchanged


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


@pytest.mark.parametrize("ex, expected", [("E03", 180.0), ("E04", 180.0), ("E07", 0.0), ("E08", 0.0), ("E09", 0.0),
                                          ("E05", 0.0)])
def test_set_feature_main_angle_of_every_exercise(ex, expected):
    """Standing pose, arms hanging: forearm pointing down (forearm elevation 180), hips and trunk straight (0),
    upper arm along the trunk (shoulder elevation 0)."""
    a = main_angle(_pose(2), FULL["exercises"][ex], "right")
    assert np.allclose(a, expected, atol=1e-6)


def test_e06_main_angle_comes_from_the_hands_in_the_side_views():
    a = FULL["exercises"]["E06"]["assess"]
    assert a["angle_source"] == "hands" and a["angle_views"] == ["L", "R"]
    assert a["near_hand"] == {"L": "right", "R": "left"}
