"""LOSO per exercise (E1, T17, C7; 2026-10-22): the LOSO plans, the per-fold-encoder shortcut's leakage check and the
encoder map built from the GroupKFold-5 encoders."""
import json

import pandas as pd
import pytest

from cvdc.config import load_config
from cvdc.train.run import check_shortcut_encoder
from cvdc.train.run_loso import exercise_loso, pid_alive, smoke_subjects
from cvdc.train.trainer import fold_plan, loso_plans

CFG = load_config("configs/per_exercise_loso.yaml")


def toy_meta(n_exp=24, n_non=26, ex="E01"):
    rows = [dict(subject=f"P{i:02d}", group="expert", view=v, exercise=ex) for i in range(1, n_exp + 1) for v in "FLR"]
    rows += [dict(subject=f"P{i:02d}", group="non-expert", view="F", exercise=ex) for i in range(26, 26 + n_non)]
    return pd.DataFrame(rows)


def test_loso_config_keeps_the_groupkfold_5_settings():
    base = load_config("configs/per_exercise.yaml")
    assert CFG["eval"]["k"] == base["eval"]["k"] == 5 and CFG["eval"]["split"] == base["eval"]["split"]
    for k in ("train", "heads", "model", "supervised", "cvdc", "methods"):
        assert CFG[k] == base[k], k  # only the split changes
    assert CFG["eval"]["loso"]["exercises"] == ["E01", "E02", "E03", "E04"]


def test_loso_holds_out_every_subject_once_with_a_clean_inner_split():
    meta = toy_meta()
    plans = loso_plans(CFG, meta)
    grp = meta.drop_duplicates("subject").set_index("subject").group
    assert [p["subject"] for p in plans] == sorted(meta.subject.unique())
    for p in plans:
        assert p["test"] == [p["subject"]]
        assert p["subject"] not in p["train"] and p["subject"] not in p["inner_train"] + p["inner_val"]
        assert sorted(p["inner_train"] + p["inner_val"]) == p["train"] and not set(p["inner_train"]) & set(p["inner_val"])
        n_val = {g: sum(grp[s] == g for s in p["inner_val"]) for g in ("expert", "non-expert")}
        n_tr = {g: sum(grp[s] == g for s in p["train"]) for g in ("expert", "non-expert")}
        assert n_val == {g: round(0.2 * n_tr[g]) for g in n_tr}  # the T8 rule per group
    assert loso_plans(CFG, meta) == plans  # seeded


def _info(plan, ex="E01"):
    return {"exercise": ex, "fold": plan["fold"], "train": plan["train"], "inner_train": plan["inner_train"],
            "inner_val": plan["inner_val"], "test": plan["test"]}


def test_shortcut_check_accepts_the_encoder_of_the_fold_that_held_the_subject_out():
    meta = toy_meta()
    gkf = fold_plan(CFG, meta)
    lp = {p["subject"]: {**p, "exercise": "E01"} for p in loso_plans(CFG, meta)}
    for g in gkf:
        for s in g["test"]:
            chk = check_shortcut_encoder(_info(g), lp[s])
            assert chk["held_out"] == [s] and chk["encoder_fold"] == g["fold"]


def test_shortcut_check_refuses_leakage_and_mismatches():
    meta = toy_meta()
    gkf = fold_plan(CFG, meta)
    lp = {p["subject"]: {**p, "exercise": "E01"} for p in loso_plans(CFG, meta)}
    g0, g1 = gkf[0], gkf[1]
    s_tr, s_va = g0["inner_train"][0], g0["inner_val"][0]
    with pytest.raises(AssertionError, match="LEAKAGE"):
        check_shortcut_encoder(_info(g0), lp[s_tr])  # the encoder trained on that subject
    with pytest.raises(AssertionError, match="LEAKAGE"):
        check_shortcut_encoder(_info(g0), lp[s_va])  # ... or early-stopped on it
    with pytest.raises(AssertionError, match="exercise"):
        check_shortcut_encoder(_info(g1, "E02"), lp[g1["test"][0]])
    with pytest.raises(AssertionError, match="views"):
        check_shortcut_encoder({**_info(g1), "views": ["F", "L"]}, lp[g1["test"][0]])


def _write_encoders(root, ex, plans, tamper=None):
    for p in plans:
        d = root / ex / "proposed_j1" / f"fold{p['fold']}" / "cvdc"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"")
        info = _info(p, ex)
        if tamper and p["fold"] == tamper[0]:
            info["inner_train"] = sorted(info["inner_train"] + [tamper[1]])
            info["test"] = [s for s in info["test"] if s != tamper[1]]
        (d / "run_info.json").write_text(json.dumps(info), encoding="utf-8")


def test_encoder_map_uses_the_groupkfold_fold_of_each_subject(tmp_path):
    meta = toy_meta(8, 6, "E03")
    gkf = fold_plan(CFG, meta)
    _write_encoders(tmp_path, "E03", gkf)
    plans, rows = exercise_loso(CFG, meta, "E03", tmp_path, "proposed_j1")
    assert len(plans) == len(rows) == meta.subject.nunique()
    where = {s: g["fold"] for g in gkf for s in g["test"]}
    for p, r in zip(plans, rows):
        assert p["cvdc_encoder_fold"] == r["gkf_fold"] == where[p["subject"]]
        assert r["in_encoder_test"] and not r["in_encoder_inner_train"] and not r["in_encoder_inner_val"]


def test_encoder_map_stops_on_an_encoder_that_differs_from_the_groupkfold_fold(tmp_path):
    meta = toy_meta(8, 6, "E03")
    gkf = fold_plan(CFG, meta)
    _write_encoders(tmp_path, "E03", gkf, tamper=(0, gkf[0]["test"][0]))
    with pytest.raises(SystemExit, match="other"):
        exercise_loso(CFG, meta, "E03", tmp_path, "proposed_j1")


def test_smoke_subjects_take_both_groups_where_both_exist():
    assert smoke_subjects(toy_meta(), 2) == ["P01", "P26"]
    assert smoke_subjects(toy_meta(5, 0), 2) == ["P01", "P02"]


def test_pid_alive_never_kills():
    import os
    assert pid_alive(os.getpid()) and pid_alive(os.getpid())  # still alive after the first check
    assert not pid_alive(None)
