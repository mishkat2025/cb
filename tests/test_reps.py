import numpy as np
import pytest

from cvdc.data.reps import segment_reps

FPS = 30


def rep_signal(n_reps, rise=2.0, hold=0.3, rest=1.0, low=10.0, high=170.0, lead=2.0, tail=2.0):
    """Rest -> raised-cosine raise -> hold -> lower -> rest, repeated."""
    seg = []
    up = low + (high - low) * (1 - np.cos(np.linspace(0, np.pi, int(rise * FPS)))) / 2
    for _ in range(n_reps):
        seg += [np.full(int(rest * FPS), low), up, np.full(int(hold * FPS), high), up[::-1]]
    return np.concatenate([np.full(int(lead * FPS), low)] + seg + [np.full(int(tail * FPS), low)])


@pytest.mark.parametrize("n", [1, 3, 10, 12])
def test_counts_reps_with_noise(n):
    s = rep_signal(n) + np.random.default_rng(n).normal(0, 3, size=len(rep_signal(n)))
    reps = segment_reps(s, FPS)
    assert len(reps) == n
    for r in reps:
        assert r["start"] < r["peak"] < r["end"]
        assert r["amplitude"] > 140
        assert r["raise_s"] == pytest.approx(2.0 * (0.795 - 0.205) + 0.1, abs=0.15)  # 10%->90% of a 2 s raised cosine, + pad


def test_reps_do_not_overlap_and_cover_peaks():
    s = rep_signal(5, rest=0.0, hold=0.0)  # back-to-back reps
    reps = segment_reps(s, FPS)
    assert len(reps) == 5
    for a, b in zip(reps, reps[1:]):
        assert a["end"] <= b["start"]


def test_incomplete_last_rep_is_dropped():
    # three full reps, then the video ends while the arm is still raised
    s = np.concatenate([rep_signal(3, tail=0.5), np.full(30, 10.0), np.linspace(10, 170, 60), np.full(20, 170.0)])
    assert len(segment_reps(s, FPS)) == 3


def test_small_movements_and_flat_signal_give_no_reps():
    t = np.arange(20 * FPS) / FPS
    assert segment_reps(10 + 15 * np.sin(2 * np.pi * 0.25 * t), FPS) == []
    assert segment_reps(np.full(300, 20.0), FPS) == []


def test_nan_gaps_are_bridged():
    s = rep_signal(3)
    s[100:110] = np.nan
    assert len(segment_reps(s, FPS)) == 3


def test_valley_polarity():
    s = rep_signal(4)
    reps = segment_reps(180 - s, FPS, polarity="valley")
    assert len(reps) == 4 and all(r["peak_val"] < 20 for r in reps)


def test_short_glitch_does_not_split_a_rep():
    s = rep_signal(3, hold=2.0)
    top = np.flatnonzero(s >= 169.9)
    s[top[30]:top[30] + 3] = 40.0                            # 3-frame 3D pose glitch on the plateau
    assert len(segment_reps(s, FPS)) == 3
    assert len(segment_reps(s, FPS, median_s=0)) == 4        # without the median filter the rep splits
