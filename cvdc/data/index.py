"""Build the video index (one row per video) and the PT-reference index.

usage: python -m cvdc.data.index [--config configs/data.yaml]
Writes data/index.csv and data/index_ref.csv and prints the inventory report.
"""
import argparse
import re

import numpy as np
import pandas as pd

from ..config import data_root, load_config, resolve
from .filenames import discover_videos
from .labels import ITEMS, load_scores, quality_class


def ref_stem(file_name: str) -> str:
    """'Abduction Adduction.mp4' -> 'Abduction_Adduction' (cache file stem for a PT demo)."""
    return re.sub(r"\s+", "_", re.sub(r"\.mp4$", "", file_name, flags=re.IGNORECASE).strip())


def build_index(cfg) -> tuple[pd.DataFrame, dict]:
    d = cfg["data"]
    root = data_root(cfg)
    rows, failures, skipped = discover_videos(root, d)
    df = pd.DataFrame(rows)
    report = dict(root=str(root), n_files=len(rows) + len(failures), n_parsed=len(rows), failures=failures,
                  skipped_exercises=skipped)
    excl = {e.upper() for e in d.get("exclude_files") or []}
    report["excluded"] = sorted(set(df.video_id) & excl)
    df = df[~df.video_id.isin(excl)].copy()
    report["duplicate_ids"] = sorted(df.video_id[df.video_id.duplicated()].unique())

    scores, bad_csv = load_scores(root / d["scores_csv"], d["filename_regex"], tuple(d["variations"]))
    report["csv_unparsed"] = bad_csv
    report["csv_unknown_variation"] = sorted(scores.csv_name[scores.unknown_variation])
    df = df.merge(scores.drop(columns=["csv_name", "unknown_variation"]), on="video_id", how="left")
    report["missing_score"] = sorted(df.video_id[df.avg_round.isna()])

    lab = cfg["labels"]
    df["score"] = df[lab["score_col"]]
    df["quality"] = quality_class(df.score, lab["quality_bins"], lab["quality_names"])
    df["group"] = np.where(df.subject_num <= d["expert_max_subject"], "expert", "non-expert")
    df["group_id"] = df.exercise + "_" + df.subject + "_V" + df.variation_raw
    df["n_views_group"] = df.groupby("group_id").view.transform("nunique")
    df["rel_path"] = df.folder + "/" + df.file
    kp_dir = cfg["pose"]["kp_dir"]
    df["kp_path"] = kp_dir + "/" + df.video_id + ".npz"
    df["pre_path"] = cfg["preprocess"]["out_dir"] + "/" + df.video_id + ".npz"
    cols = ["video_id", "exercise", "subject", "subject_num", "group", "view", "variation", "gender",
            "group_id", "n_views_group", "score", "quality"] + ITEMS + [
        "avg", "judge_sd", "n_judges", "folder", "file", "rel_path", "kp_path", "pre_path", "size_mb"]
    df = df[cols].sort_values(["exercise", "subject_num", "variation", "view"]).reset_index(drop=True)
    return df, report


def build_ref_index(cfg, exercises) -> pd.DataFrame:
    """PT demo videos named in the per-exercise YAMLs of the given exercises (primary + backups)."""
    root = data_root(cfg) / cfg["data"]["reference_dir"]
    rows = []
    for code in exercises:
        refs = (cfg["exercises"].get(code) or {}).get("references") or {}
        files = ([refs["primary"]] if refs.get("primary") else []) + list(refs.get("backup") or [])
        for i, f in enumerate(files):
            stem = ref_stem(f)
            rows.append(dict(exercise=code, role="primary" if i == 0 else "backup", file=f, ref_id=stem,
                             rel_path=f"{cfg['data']['reference_dir']}/{f}", exists=(root / f).exists(),
                             kp_path=f"{cfg['pose']['ref_kp_dir']}/{stem}.npz",
                             pre_path=f"{cfg['preprocess']['ref_out_dir']}/{stem}.npz"))
    return pd.DataFrame(rows)


def print_report(df, report):
    print(f"dataset root: {report['root']}")
    print(f"files: {report['n_files']}  parsed: {report['n_parsed']} "
          f"({100 * report['n_parsed'] / max(report['n_files'], 1):.1f}%)  failed: {len(report['failures'])}")
    for f in report["failures"]:
        print("  FAIL:", f)
    print("excluded (D7):", report["excluded"], " skipped exercise folders (not included):", report["skipped_exercises"])
    print("duplicate video ids:", report["duplicate_ids"] or "none")
    print("CSV names not parsed:", report["csv_unparsed"] or "none",
          " CSV unknown variations:", report["csv_unknown_variation"] or "none")
    print("local videos without a score:", report["missing_score"] or "none")
    for ex, g in df.groupby("exercise"):
        print(f"\n== {ex}: {len(g)} videos, {g.subject.nunique()} subjects "
              f"({g[g.group == 'expert'].subject.nunique()} expert / {g[g.group == 'non-expert'].subject.nunique()} non-expert)")
        print("  per group x view:", g.groupby(["group", "view"]).size().unstack(fill_value=0).to_dict("index"))
        print("  per variation   :", g.variation.value_counts().sort_index().to_dict())
        print("  quality classes :", g.quality.value_counts().to_dict())
        gs = g.groupby("group_id").agg(views=("view", lambda v: "".join(sorted(v))), grp=("group", "first"))
        print("  multi-view groups:", gs.groupby(["grp", "views"]).size().to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    df, report = build_index(cfg)
    out = resolve(cfg["data"]["index_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    ref = build_ref_index(cfg, sorted(df.exercise.unique()))
    ref.to_csv(resolve(cfg["data"]["ref_index_csv"]), index=False)
    print_report(df, report)
    print(f"\nreference videos ({len(ref)}):")
    print(ref[["exercise", "role", "file", "exists"]].to_string(index=False))
    print(f"\nwrote {out} ({len(df)} rows) and {cfg['data']['ref_index_csv']}")


if __name__ == "__main__":
    main()
