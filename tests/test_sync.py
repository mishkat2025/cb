import numpy as np
import pytest

from cvdc.data.sync import gcc_phat, motion_pair, solve_offsets

MC = dict(hysteresis=0.5, match_tol_s=0.5, min_overlap=0.6, min_matched=4, resid_threshold_ms=42)


def test_gcc_phat_recovers_audio_lag():
    fs = 8000
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 10 * fs)
    for k in rng.integers(0, 9 * fs, 30):                  # clicks
        x[k:k + 40] += rng.normal(0, 1, 40)
    lag = 0.4375
    y = np.concatenate([rng.normal(0, 0.01, int(lag * fs)), x])[: len(x)] + rng.normal(0, 0.01, len(x))
    est, ratio = gcc_phat(x, y, fs, (100, 3800), 3.0)
    assert est == pytest.approx(lag, abs=1 / fs) and ratio > 1.5


def reps_signal(t, offset=0.0):
    """Irregular rep timing so that only the right lag lines the crossings up."""
    period = 3.0 + 0.4 * np.sin(np.arange(20))
    starts = np.cumsum(np.r_[1.0, period[:-1]])
    s = np.zeros_like(t)
    for st in starts:
        u = (t - st - offset) / 2.4
        m = (u >= 0) & (u <= 1)
        s[m] = np.sin(np.pi * u[m]) ** 2
    return s


def test_motion_pair_finds_and_confirms_lag():
    fps = 30
    t = np.arange(int(40 * fps)) / fps
    lag = 0.517
    a = {"wy_r": reps_signal(t), "el_r": 150 * reps_signal(t)}
    b = {"wy_r": 0.7 * reps_signal(t, lag) + 0.2, "el_r": 140 * reps_signal(t, lag)}
    r = motion_pair(a, b, 0.5, 0.3, fps, MC)
    assert r["motion_confirmed"] and r["motion_lag_s"] == pytest.approx(lag, abs=0.012)


def test_motion_pair_rejects_unrelated_recordings():
    fps = 30
    t = np.arange(int(40 * fps)) / fps
    rng = np.random.default_rng(1)
    s2 = np.zeros_like(t)
    for st in np.cumsum(rng.uniform(2.5, 4.5, 12)):
        u = (t - st) / 2.4
        m = (u >= 0) & (u <= 1)
        s2[m] = np.sin(np.pi * u[m]) ** 2
    r = motion_pair({"wy_r": reps_signal(t)}, {"wy_r": s2}, 0.0, 5.0, fps, MC)
    assert not r["motion_confirmed"]


def test_solve_offsets_least_squares_and_connectivity():
    off, rms = solve_offsets(["F", "L", "R"], "F", {("F", "L"): 0.10, ("F", "R"): 0.30, ("L", "R"): 0.20})
    assert off == pytest.approx({"F": 0.0, "L": 0.10, "R": 0.30}) and rms == pytest.approx(0, abs=1e-6)
    off, _ = solve_offsets(["F", "L", "R"], "F", {("L", "R"): 0.2})  # L, R not joined to F
    assert off["F"] == 0 and np.isnan(off["L"]) and np.isnan(off["R"])


def test_weighted_offsets_follow_motion_lags():
    """A wrong-ish audio lag (weight 0.04) barely moves offsets fixed by two motion lags."""
    lags = {("F", "L"): 0.10, ("F", "R"): 0.30, ("L", "R"): 0.25}       # L-R audio lag off by 50 ms
    w = {("F", "L"): 1.0, ("F", "R"): 1.0, ("L", "R"): 0.04}
    off, _ = solve_offsets(["F", "L", "R"], "F", lags, w)
    assert off["L"] == pytest.approx(0.10, abs=0.002) and off["R"] == pytest.approx(0.30, abs=0.002)
