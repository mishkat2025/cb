"""Parse 'Physiotherapist Exercise Marking.csv': 3 judge rows per video x 7 EAAQ items (D4)."""
import numpy as np
import pandas as pd

from .filenames import parse_name

ITEMS = ["PF1", "PF2", "PF3", "PF4", "CF1", "CF2", "CF3"]
COLS = ["video", "judge"] + ITEMS + ["total", "scaled100", "avg", "avg_round"]


def load_scores(csv_path, regex, variations) -> pd.DataFrame:
    """One row per video: avg_round (main label), avg, judge_sd and the judge-mean of each item (0-1)."""
    raw = pd.read_csv(csv_path, header=None, skiprows=2, dtype=str, encoding="utf-8-sig")
    raw = raw.iloc[:, :len(COLS)]
    raw.columns = COLS[: raw.shape[1]]
    raw["video"] = raw["video"].ffill().str.strip()
    for c in COLS[2:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    per = raw.groupby("video", sort=False).agg(
        avg_round=("avg_round", "first"), avg=("avg", "first"), n_judges=("judge", "size"),
        judge_sd=("scaled100", "std"), **{c: (c, "mean") for c in ITEMS})
    parsed = [parse_name(v, regex, variations) for v in per.index]
    per["video_id"] = [p["video_id"] if p else np.nan for p in parsed]
    per["unknown_variation"] = [bool(p) and p["variation"] == "unknown" for p in parsed]
    per["csv_name"] = per.index
    bad = per[per.video_id.isna()]
    return per.dropna(subset=["video_id"]).reset_index(drop=True), list(bad.csv_name)


MQ_ITEMS = ["PF1", "PF2", "CF1", "CF2", "CF3"]


def movement_quality(df, items=MQ_ITEMS):
    """Secondary target (D13, user 2026-10-19): the movement-quality score = the mean of the judge-mean items
    PF1, PF2, CF1, CF2, CF3 (each 0-1), rescaled to 0-100. It leaves out the repetition items PF3 / PF4, which
    a set's views rate differently (E3). The total score stays the official target."""
    return 100.0 * df[list(items)].astype(float).sum(axis=1) / len(items)


def quality_class(score, bins, names):
    """Score -> derived quality class (D6). bins=[50, 80] -> <50, 50-79.99, >=80."""
    score = np.asarray(score, float)
    out = np.asarray(names, object)[np.digitize(score, bins)]
    out[np.isnan(score)] = None
    return out
