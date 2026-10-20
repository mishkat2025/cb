"""Cross-view protocol (E24): training views restricted without touching the subjects or splits, no reuse of an
encoder trained on other views, CVDC batches of the training views only, and the VCI over restricted pairs."""
import json
import types

import numpy as np
import pandas as pd
import pytest
import torch

from cvdc.config import load_config, resolve
from cvdc.data.dataset import PackData
from cvdc.eval.cross_view import crossview_plans
from cvdc.eval.metrics import vci_from_sums, vci_sums
from cvdc.train.run import reuse_cvdc
from cvdc.train.trainer import check_plan, fold_plan, with_views

CFG = load_config()


def toy_meta():
    rows = []
    for i in range(1, 25):
        rows += [dict(subject=f"P{i:02d}", group="expert", view=v) for v in "FLR" for _ in range(2)]
    for i in range(26, 52):
        rows += [dict(subject=f"P{i:02d}", group="non-expert", view="F")]
    return pd.DataFrame(rows)


def test_subject_mask_restricts_views():
    meta = toy_meta()
    fake = types.SimpleNamespace(meta=meta, device="cpu")
    m_all = PackData.subject_mask(fake, ["P01", "P30"])
    m_lr = PackData.subject_mask(fake, ["P01", "P30"], ["L", "R"])
    assert m_all.sum() == 7 and m_lr.sum() == 4  # P01: 6 reps (2 per view), P30: one front rep
    assert set(meta.view[m_lr.numpy()]) == {"L", "R"} and set(meta.subject[m_lr.numpy()]) == {"P01"}
    assert (PackData.subject_mask(fake, ["P01"], None) == PackData.subject_mask(fake, ["P01"])).all()


def test_crossview_plans_keep_the_folds_and_drop_one_view():
    meta = toy_meta()
    base = fold_plan(CFG, meta)
    for v in "FLR":
        plans = crossview_plans(CFG, meta, v)
        assert len(plans) == len(base)
        for p, b in zip(plans, base):
            assert p["views"] == sorted(set("FLR") - {v})
            for k in ("train", "inner_train", "inner_val", "test"):
                assert p[k] == b[k]  # same subjects as the Phase 3 folds
    with pytest.raises(ValueError):
        crossview_plans(CFG, meta, "B")
    with pytest.raises(AssertionError):
        check_plan(with_views(base[0], []))


def test_reuse_refuses_an_encoder_trained_on_other_views(tmp_path):
    meta = toy_meta()
    p = with_views(fold_plan(CFG, meta)[0], ["L", "R"])
    d = tmp_path / "prev" / "proposed" / "fold0" / "cvdc"
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"")
    (d / "summary.json").write_text(json.dumps({"best_epoch": 1, "stopped_epoch": 1}), encoding="utf-8")
    info = {k: p[k] for k in ("inner_train", "inner_val", "test")}  # same subjects, all views (no "views" key)
    (d / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    cfg = {**CFG, "train": {**CFG["train"], "runs_dir": str(tmp_path), "cvdc_init_from": "prev"}}
    data = types.SimpleNamespace(device="cpu")
    assert reuse_cvdc(cfg, data, p, log=lambda *a: None) is None
    (d / "run_info.json").write_text(json.dumps({**info, "views": ["F", "R"]}), encoding="utf-8")
    assert reuse_cvdc(cfg, data, p, log=lambda *a: None) is None


def test_vci_pair_mask_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 12
    z = rng.normal(size=(n, 5))
    key = np.repeat(np.arange(4), 3)  # 4 rep sets, views F / L / R
    view = np.tile(np.array([0, 1, 2]), 4)
    group = np.repeat([0, 0, 1, 2], 3)
    subj = np.repeat([0, 0, 1, 2], 3)
    held = view == 0  # held-out view F
    ok = held[:, None] != held[None, :]
    s = vci_from_sums(vci_sums(z, key, view, group, subj, 3, pair_ok=ok))
    zn = z / np.linalg.norm(z, axis=1, keepdims=True)
    sim = zn @ zn.T
    same = (key[:, None] == key[None, :]) & (view[:, None] != view[None, :]) & ok
    other = (group[:, None] != group[None, :]) & ok
    assert s["same"] == pytest.approx(sim[same].mean())
    assert s["other"] == pytest.approx(sim[other].mean())
    full = vci_from_sums(vci_sums(z, key, view, group, subj, 3))
    assert full == vci_from_sums(vci_sums(z, key, view, group, subj, 3, pair_ok=np.ones((n, n), bool)))


@pytest.mark.skipif(not (resolve(CFG["preprocess"]["pack_dir"]) / "reps_coco17_T64.npz").exists(),
                    reason="rep pack not built")
def test_cvdc_batches_use_only_the_training_views():
    from cvdc.data.cvdc_batch import CVDCBatcher

    data = PackData(CFG, "rep", "cpu")
    p = with_views(fold_plan(CFG, data.meta)[0], ["L", "R"])
    b = CVDCBatcher(data, CFG, allowed=data.subject_mask(p["inner_train"], p["views"]), seed=0)
    assert b.sets
    seen = set()
    for k, batch in enumerate(b.iterate(augment_views=False)):
        it = batch["item"][~batch["is_syn"]].numpy()
        seen |= set(data.meta.view.iloc[it])
        assert set(data.meta.subject.iloc[batch["item"].numpy()]) <= set(p["inner_train"])
        if k == 3:
            break
    assert seen <= {"L", "R"} and seen
    pairs = pd.read_csv(resolve(CFG["cvdc"]["pairs"]["pairs_csv"]))
    used = {(a, c) for a, c in b.edge_w}
    vv = data.meta.view.to_numpy()
    assert all(vv[a] in "LR" and vv[c] in "LR" for a, c in used)
    assert len(used) // 2 < len(pairs)
    torch.cuda.empty_cache()
