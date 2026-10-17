import numpy as np
import pytest

from cvdc.assess.dtw import dtw, sample_at, warp_positions


def brute_force_dtw(a, b):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def test_identical_sequences_cost_zero_on_the_diagonal():
    a = np.sin(np.linspace(0, 3, 40))[:, None] * [1, 2]
    cost, path = dtw(a, a)
    assert cost == 0
    assert (path[:, 0] == path[:, 1]).all() and len(path) == 40


@pytest.mark.parametrize("seed", range(3))
def test_cost_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=12), rng.normal(size=9)
    assert dtw(a, b)[0] == pytest.approx(brute_force_dtw(a, b))


def test_path_is_monotone_and_complete():
    rng = np.random.default_rng(0)
    _, path = dtw(rng.normal(size=(20, 3)), rng.normal(size=(31, 3)))
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (19, 30)
    steps = np.diff(path, axis=0)
    assert ((steps >= 0) & (steps <= 1)).all() and (steps.sum(1) >= 1).all()


def test_recovers_a_known_time_warp():
    t = np.linspace(0, 1, 80)
    ref = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], 1) * 50  # angle-like curves, degrees
    q = t ** 1.5  # the second sequence runs slow at first, then fast
    warped = np.stack([np.sin(2 * np.pi * q), np.cos(2 * np.pi * q)], 1) * 50
    _, path = dtw(ref, warped)
    pos = warp_positions(path, len(ref))  # where in `warped` each ref frame lands
    true = (t ** (1 / 1.5)) * 79
    assert np.abs(pos - true).mean() < 1.0
    aligned = sample_at(warped, pos)
    assert np.abs(aligned - ref).mean() < 2.0  # degrees, vs ~20 before alignment
    assert np.abs(warped - ref).mean() > 10


def test_band_limits_the_path():
    rng = np.random.default_rng(1)
    _, path = dtw(rng.normal(size=30), rng.normal(size=30), band=2)
    assert np.abs(path[:, 0] - path[:, 1]).max() <= 3
