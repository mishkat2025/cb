import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score

from cvdc.eval.consistency import viewpoint_consistency
from cvdc.eval.metrics import (add_sums, classification, item_weights, regression, subject_draws, vci_from_sums,
                               vci_sums)


def data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.round(rng.uniform(20, 100, n))  # rounded: many ties, like the scores
    p = y + rng.normal(0, 12, n)
    subj = rng.integers(0, 15, n)
    return y, p, np.array([f"P{s:02d}" for s in subj])


def test_point_estimates_match_scipy():
    y, p, _ = data()
    r = regression(y, p, np.ones((1, len(y))))
    assert np.isclose(r["rmse"][0], np.sqrt(np.mean((p - y) ** 2)))
    assert np.isclose(r["mae"][0], np.mean(np.abs(p - y)))
    assert np.isclose(r["pearson"][0], pearsonr(y, p)[0])
    assert np.isclose(r["spearman"][0], spearmanr(y, p)[0])  # ties handled with mid-ranks


def test_bootstrap_weights_equal_explicit_resampling():
    y, p, subj = data()
    subjects = np.unique(subj)
    M = subject_draws(subjects, np.zeros(len(subjects)), 5, seed=3)
    assert (M.sum(1) == len(subjects)).all()
    W = item_weights(subj, subjects, M)
    r = regression(y, p, W)
    for b in range(5):  # replicate b = the data with each subject's items repeated M[b, s] times
        idx = np.concatenate([np.flatnonzero(subj == s).repeat(int(M[b, i])) for i, s in enumerate(subjects)])
        assert np.isclose(r["rmse"][b], np.sqrt(np.mean((p[idx] - y[idx]) ** 2)))
        assert np.isclose(r["pearson"][b], pearsonr(y[idx], p[idx])[0])
        assert np.isclose(r["spearman"][b], spearmanr(y[idx], p[idx])[0])


def test_draws_respect_strata():
    st = np.array([0] * 10 + [1] * 4)
    M = subject_draws(np.arange(14), st, 200, seed=0)
    assert (M[:, :10].sum(1) == 10).all() and (M[:, 10:].sum(1) == 4).all()


def test_constant_prediction_has_no_correlation():
    y, _, _ = data()
    r = regression(y, np.full(len(y), 70.0), np.ones((1, len(y))))
    assert np.isnan(r["pearson"][0]) and np.isnan(r["spearman"][0])


def test_classification_matches_sklearn():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 3, 200)
    p = np.where(rng.random(200) < 0.6, t, rng.integers(0, 3, 200))
    c = classification(t, p, np.ones((1, 200)), 3)
    assert np.isclose(c["macro_f1"][0], f1_score(t, p, average="macro"))
    assert np.isclose(c["kappa"][0], cohen_kappa_score(t, p))
    assert np.isclose(c["accuracy"][0], (t == p).mean())


def test_vci_sums_match_viewpoint_consistency_and_pool_over_folds():
    rng = np.random.default_rng(0)
    folds = []
    for f in range(2):  # two folds, 4 subjects each, rep sets of 3 views
        n_sets = 12
        base = rng.normal(size=(n_sets, 8))
        z = np.repeat(base, 3, 0) + 0.3 * rng.normal(size=(3 * n_sets, 8)) + 2.0
        key = np.repeat(np.arange(n_sets), 3)
        view = np.tile(np.arange(3), n_sets)
        subj = np.repeat(np.arange(4) + 4 * f, 9)  # 3 sets per subject
        group = subj * 10 + (key % 3 > 0)  # two variations per subject
        folds.append((z, key + 100 * f, view, group, subj))
    total = {}
    for z, key, view, group, subj in folds:
        c = z.mean(0)
        s = vci_sums(z, key, view, group, subj, 8, center=c)
        ref = viewpoint_consistency(*(torch.tensor(a) for a in (z, key, view, group)), center=torch.tensor(c))
        v = vci_from_sums(s)
        assert np.isclose(v["vci"], ref["vci"]) and np.isclose(v["same"], ref["same"])
        total = add_sums(total, s)
    pooled = vci_from_sums(total)
    sp = sum(add_sums({}, vci_sums(z, k, v, g, s, 8, z.mean(0)))["same_cnt"].sum() for z, k, v, g, s in folds)
    assert sp == total["same_cnt"].sum()
    assert np.isfinite(pooled["vci"])
    M = np.ones((1, 8))
    assert np.isclose(vci_from_sums(total, M)["vci"][0], pooled["vci"])
