import copy

import pandas as pd
import pytest
import torch

from cvdc.config import load_config
from cvdc.eval.splits import group_kfold
from cvdc.train.trainer import EarlyStopping, check_plan, fold_plan, inner_split, warmup_cosine

CFG = load_config()


def toy_meta():
    rows = []
    for i in range(1, 25):  # experts, 3 views x 2 reps each
        rows += [dict(subject=f"P{i:02d}", group="expert", view=v) for v in "FLR" for _ in range(2)]
    for i in range(26, 52):  # non-experts, front only (26 subjects: 50 in total, as at rep level)
        rows += [dict(subject=f"P{i:02d}", group="non-expert", view="F")]
    return pd.DataFrame(rows)


def test_inner_split_stays_inside_the_training_fold():
    meta = toy_meta()
    grp = meta.drop_duplicates("subject").set_index("subject").group
    plans = fold_plan(CFG, meta)
    assert len(plans) == 5
    for p, (tr, te) in zip(plans, group_kfold(meta, 5, CFG["eval"]["seed"])):
        assert p["train"] == sorted(tr) and p["test"] == sorted(te)
        itr, iva, test = set(p["inner_train"]), set(p["inner_val"]), set(p["test"])
        assert not iva & test and not itr & test  # the inner split never touches the test fold
        assert not iva & itr  # disjoint from the inner training subjects
        assert iva | itr == set(tr)
        assert check_plan(p)
        for g in ("expert", "non-expert"):  # ~20% of each group, stratified
            n_tr = sum(grp[s] == g for s in tr)
            assert sum(grp[s] == g for s in iva) == round(0.2 * n_tr)


def test_inner_split_is_seeded_per_fold():
    meta = toy_meta()
    tr = group_kfold(meta, 5, 0)[0][0]
    assert inner_split(meta, tr, 0.2, 7) == inner_split(meta, tr, 0.2, 7)
    assert inner_split(meta, tr, 0.2, 7)[1] != inner_split(meta, tr, 0.2, 8)[1]
    vals = [p["inner_val"] for p in fold_plan(CFG, meta)]
    assert len({tuple(v) for v in vals}) == 5


def test_check_plan_rejects_leaks():
    p = dict(fold=0, train=["P01", "P02", "P03"], inner_train=["P01", "P02"], inner_val=["P03"], test=["P04"])
    assert check_plan(p)
    with pytest.raises(AssertionError):
        check_plan({**p, "inner_val": ["P04"], "inner_train": ["P01", "P02", "P03"]})  # validation on a test subject
    with pytest.raises(AssertionError):
        check_plan({**p, "inner_val": ["P02", "P03"]})  # overlaps the inner training subjects


def test_smoke_plan_has_three_training_subjects():
    meta = toy_meta()
    (p,) = fold_plan(CFG, meta, smoke=True)
    assert len(p["inner_train"]) == 3 and len(p["inner_val"]) == 2 and len(p["test"]) == 3
    full = fold_plan(CFG, meta)[0]
    assert set(p["inner_train"]) <= set(full["inner_train"]) and set(p["test"]) <= set(full["test"])


def test_early_stopping_keeps_the_best_epoch():
    model = torch.nn.Linear(2, 1)
    es = EarlyStopping(patience=3)
    vals = [5.0, 3.0, 4.0, 2.0, 2.5, 3.0, 2.0, 1.0]  # best at epoch 4; 3 epochs without improvement -> stop at 7
    stopped = None
    for ep, v in enumerate(vals, 1):
        with torch.no_grad():
            model.weight.fill_(float(ep))  # parameters record the epoch
        if es.step(ep, v, model):
            stopped = ep
            break
    assert stopped == 7 and es.best_epoch == 4 and es.best_value == 2.0
    es.restore(model)
    assert torch.all(model.weight == 4.0)  # the restored state is the best epoch's, not the last one's
    s = es.summary(max_epochs=60)
    assert s["best_epoch"] == 4 and s["stopped_epoch"] == 7 and s["early_stopped"]


def test_early_stopping_ignores_non_finite_values_and_min_delta():
    model = torch.nn.Linear(2, 1)
    es = EarlyStopping(patience=2, min_delta=0.1)
    assert not es.step(1, float("nan"), model)
    assert not es.step(2, 1.0, model)
    assert not es.step(3, 0.95, model)  # not better by more than min_delta
    assert es.step(4, 0.92, model)
    assert es.best_epoch == 2


def test_warmup_cosine_schedule():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = warmup_cosine(opt, total_steps=100, warmup_frac=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert max(lrs) == pytest.approx(1.0) and lrs[-1] < 0.01
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))


def test_fold_plan_on_the_rep_level_subjects_is_leak_free():
    meta = toy_meta()
    cfg = copy.deepcopy(CFG)
    cfg["eval"]["k"] = 10
    for p in fold_plan(cfg, meta):
        assert not set(p["inner_val"]) & set(p["test"])
