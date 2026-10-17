import pandas as pd
import pytest

from cvdc.eval.splits import check_no_leakage, describe, fold_of, group_kfold, loso


def toy_meta():
    rows = []
    for i in range(1, 25):  # experts, 3 views x 2 reps each
        rows += [dict(subject=f"P{i:02d}", group="expert", view=v) for v in "FLR" for _ in range(2)]
    for i in range(26, 53):  # non-experts, front only
        rows += [dict(subject=f"P{i:02d}", group="non-expert", view="F")]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("k", [5, 10])
def test_group_kfold_has_no_leakage_and_balances_groups(k):
    meta = toy_meta()
    splits = group_kfold(meta, k, seed=0)
    assert check_no_leakage(splits, meta.subject.unique())
    d = describe(splits, meta)
    assert d.test_experts.max() - d.test_experts.min() <= 1  # experts spread evenly
    non = d.test_subjects - d.test_experts
    assert non.max() - non.min() <= 1
    for tr, te in splits:  # every item of a subject on one side
        assert not meta[meta.subject.isin(te)].subject.isin(tr).any()


def test_loso_holds_out_each_subject_once():
    meta = toy_meta()
    splits = loso(meta)
    assert len(splits) == meta.subject.nunique()
    assert check_no_leakage(splits, meta.subject.unique())
    assert all(len(te) == 1 for _, te in splits)


def test_leakage_is_detected():
    with pytest.raises(AssertionError):
        check_no_leakage([(["P01", "P02"], ["P02"])], ["P01", "P02"])


def test_loso_shortcut_uses_the_fold_that_held_the_subject_out():
    meta = toy_meta()
    gk = group_kfold(meta, 5, seed=0)
    where = fold_of(gk)
    for s, _ in [(te[0], tr) for tr, te in loso(meta)]:
        assert s in gk[where[s]][1] and s not in gk[where[s]][0]  # the encoder never saw the held-out subject
