"""Step 0 on the full dataset (all 9 exercises): parse every filename, join the scores CSV, count.

Uses the Phase 1 parser and CSV loader (cvdc.data.filenames / labels), so the rules are the pilot's.
Writes <out>/inventory.csv (one row per video, scores joined) and prints the report. Keypoints and pixels
are not touched. Nothing in the dataset is modified.

usage: python -I scripts/full/step0/inventory.py --videos D:/cb/dataset/videos --meta-root D:/cb/dataset --out outputs/step0_full
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.filenames import discover_exercise_dirs, parse_name  # noqa: E402
from cvdc.data.labels import ITEMS, load_scores, quality_class  # noqa: E402

STRICT = re.compile(r"^E\d{2}_P\d{2}_A[FLR]_V(FL|ML|LL|LJ|HJ|O|LR)_G[MF]\.mp4$")


def readme_table(md_path):
    """Participant x exercise counts from the dataset README (Table 1)."""
    rows = []
    for line in Path(md_path).read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 12 and cells[0].isdigit():
            rows.append(dict(subject=f"P{int(cells[0]):02d}", **{f"E{i:02d}": int(cells[1 + i]) for i in range(1, 10)}))
    return pd.DataFrame(rows).set_index("subject")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder holding the E0x(...) exercise folders")
    ap.add_argument("--meta-root", required=True, help="folder holding the scores CSV and MobiPhysio.md")
    ap.add_argument("--out", default="outputs/step0_full")
    ap.add_argument("--pilot-index", default="data/index.csv")
    a = ap.parse_args()
    cfg = load_config()
    d = cfg["data"]
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)

    # ---- files and parse rate
    dirs = discover_exercise_dirs(a.videos, d["exercise_dir_regex"], d["skip_dir_prefix"])
    rows, fails, nonmp4 = [], [], []
    for code, folder in dirs.items():
        for f in sorted(folder.iterdir()):
            if f.is_dir() or f.suffix.lower() != ".mp4":
                nonmp4.append(str(f))
                continue
            p = parse_name(f.name, d["filename_regex"], tuple(d["variations"]))
            if p is None:
                fails.append(str(f))
                continue
            p.update(folder=folder.name, file=f.name, size_mb=round(f.stat().st_size / 1e6, 2),
                     strict_ok=bool(STRICT.match(f.name)), folder_code=code)
            rows.append(p)
    df = pd.DataFrame(rows)
    n_files = len(rows) + len(fails)
    print(f"exercise folders: {list(dirs)}")
    print(f"files: {n_files}  parsed: {len(rows)} ({100 * len(rows) / max(n_files, 1):.2f}%)  failed: {fails or 'none'}")
    print("non-mp4 entries:", nonmp4 or "none")
    print("names not in the strict E##_P##_A?_V??_G?.mp4 form (parsed by the tolerant regex):",
          df.loc[~df.strict_ok, "file"].tolist() or "none")
    print("exercise code != folder:", df.loc[df.exercise != df.folder_code, "file"].tolist() or "none")
    print("unknown variation codes:", df.loc[df.variation == "unknown", "file"].tolist() or "none")
    dup = df[df.video_id.duplicated(keep=False)]
    print("duplicate video ids:", dup.file.tolist() or "none")
    tiny = df[df.size_mb < 0.5]
    print("files < 0.5 MB:", tiny[["file", "size_mb"]].values.tolist() or "none")
    # same subject + exercise + view, identical byte size (copy candidates; content checked by audio later)
    same = df[df.duplicated(["exercise", "subject", "view", "size_mb"], keep=False)]
    print("same subject/exercise/view with identical size:", same.file.tolist() or "none")

    # ---- subjects
    df["group"] = np.where(df.subject_num <= d["expert_max_subject"], "expert", "non-expert")
    g_per_subj = df.groupby("subject").gender.nunique()
    print("subjects with >1 gender across files:", g_per_subj[g_per_subj > 1].index.tolist() or "none")

    # ---- CSV join
    scores, bad_csv = load_scores(Path(a.meta_root) / d["scores_csv"], d["filename_regex"], tuple(d["variations"]))
    raw = pd.read_csv(Path(a.meta_root) / d["scores_csv"], header=None, skiprows=2, dtype=str, encoding="utf-8-sig")
    print(f"\nCSV: {len(raw)} judge rows, {len(scores)} parsed video names; unparsed names: {bad_csv or 'none'}")
    print("CSV unknown-variation names:", sorted(scores.csv_name[scores.unknown_variation]) or "none")
    print("CSV videos with != 3 judge rows:", scores.loc[scores.n_judges != 3, ["csv_name", "n_judges"]].values.tolist() or "none")
    csv_dups = scores[scores.video_id.duplicated(keep=False)]
    print("CSV duplicate video ids (after normalising case):", csv_dups.csv_name.tolist() or "none")
    noncanon = scores[scores.csv_name.str.strip() != scores.video_id]
    print("CSV names that differ from the canonical id (case/typo):", noncanon[["csv_name", "video_id"]].values.tolist() or "none")
    j = df.merge(scores.drop(columns=["unknown_variation"]).drop_duplicates("video_id"), on="video_id", how="left")
    miss = j[j.avg_round.isna()]
    print(f"local files without a CSV score: {len(miss)}", miss.file.tolist() or "")
    orphan = sorted(set(scores.video_id) - set(df.video_id))
    print(f"CSV rows without a local file: {len(orphan)}", orphan)
    j["score"] = j.avg_round
    j["quality"] = quality_class(j.score, cfg["labels"]["quality_bins"], cfg["labels"]["quality_names"])
    j["group_id"] = j.exercise + "_" + j.subject + "_V" + j.variation_raw
    j["n_views_group"] = j.groupby("group_id").view.transform("nunique")
    # aliases for the Step 0 scripts (sync_av.py)
    j["ex"], j["pid"], j["angle"], j["var"] = j.exercise, j.subject, j.view, j.variation_raw

    # ---- pilot files unchanged?
    pil = pd.read_csv(resolve(a.pilot_index))
    cmp = j[j.exercise.isin(pil.exercise.unique())].merge(pil[["video_id", "size_mb", "score"]], on="video_id",
                                                          how="outer", suffixes=("", "_pilot"), indicator=True)
    print(f"\npilot (E01/E02) check: {len(pil)} pilot rows; only in new folder: "
          f"{cmp.loc[cmp._merge == 'left_only', 'video_id'].tolist() or 'none'}; only in pilot: "
          f"{cmp.loc[cmp._merge == 'right_only', 'video_id'].tolist() or 'none'}; size differs (> 0.01 MB): "
          f"{cmp.loc[(cmp._merge == 'both') & ((cmp.size_mb - cmp.size_mb_pilot).abs() > 0.011), 'video_id'].tolist() or 'none'}; "
          f"score differs: {cmp.loc[(cmp._merge == 'both') & ((cmp.score - cmp.score_pilot).abs() > 1e-6), 'video_id'].tolist() or 'none'}")

    # ---- README Table 1 vs files
    rt = readme_table(Path(a.meta_root) / "MobiPhysio.md")
    cnt = df.groupby(["subject", "exercise"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(index=rt.index.union(cnt.index), columns=rt.columns, fill_value=0)
    diff = (cnt - rt.reindex(cnt.index).fillna(0)).stack()
    diff = diff[diff != 0]
    print("README Table 1 vs files (subject, exercise: files - table):", diff.astype(int).to_dict() or "all equal")

    # ---- per exercise
    summ = []
    for ex, g in j.groupby("exercise"):
        e, n = g[g.group == "expert"], g[g.group == "non-expert"]
        gs = g.groupby("group_id").agg(views=("view", lambda v: "".join(sorted(v))), grp=("group", "first"))
        full = gs[gs.views == "FLR"]
        r = dict(exercise=ex, videos=len(g), subjects=g.subject.nunique(), experts=e.subject.nunique(),
                 non_experts=n.subject.nunique(), videos_expert=len(e), videos_non_expert=len(n),
                 F=int((g.view == "F").sum()), L=int((g.view == "L").sum()), R=int((g.view == "R").sum()),
                 groups=len(gs), groups_multi=int((gs.views.str.len() >= 2).sum()),
                 flr_expert=int((full.grp == "expert").sum()), flr_non_expert=int((full.grp == "non-expert").sum()),
                 mean_expert=round(e.score.mean(), 1), mean_non_expert=round(n.score.mean(), 1) if len(n) else np.nan,
                 ge95=int((g.score >= 95).sum()), ge95_expert_front=int(((e.score >= 95) & (e.view == "F")).sum()),
                 bins_c_p_i="/".join(str(int((g.quality == q).sum())) for q in ["correct", "partially_correct", "incorrect"]))
        summ.append(r)
        print(f"\n######## {ex} ({g.folder.iloc[0]}): {len(g)} videos, {r['subjects']} subjects "
              f"({r['experts']} expert / {r['non_experts']} non-expert)")
        print("  subjects:", " ".join(sorted(g.subject.unique(), key=lambda s: int(s[1:]))))
        print("  view x group:", g.groupby(["group", "view"]).size().unstack(fill_value=0).to_dict("index"))
        print("  variation x group:", g.groupby(["group", "variation"]).size().unstack(fill_value=0).to_dict("index"))
        print("  gender (videos):", g.gender.value_counts().to_dict(), " subjects:",
              g.drop_duplicates("subject").gender.value_counts().to_dict())
        print("  view sets of (subject, variation) groups:", gs.groupby(["grp", "views"]).size().to_dict())
        for grp, gg in g.groupby("group"):
            s = gg.score
            print(f"  {grp:10s} score n={len(s)} mean {s.mean():.1f} median {s.median():.1f} sd {s.std():.1f} "
                  f"min {s.min():.1f} max {s.max():.1f}  ==100 {(s == 100).mean():.1%}  >=95 {(s >= 95).sum()}")
        print("  quality bins (>=80/50-79/<50):", r["bins_c_p_i"])
        print("  10-pt histogram:", np.histogram(g.score, bins=range(0, 110, 10))[0].tolist())
        print("  item means (0-1):", g[ITEMS].mean().round(3).to_dict())
        print("  item share == 1  :", {c: round(float((g[c] >= 0.999).mean()), 3) for c in ITEMS})
        print("  PF3 = 1 exactly when PF4 = 1 (judge means):",
              bool(((g.PF3 >= 0.999) == (g.PF4 >= 0.999)).all()))
        sp = g.groupby("subject").agg(n=("score", "size"), mean=("score", "mean"), pf1=("PF1", "mean"))
        low = sp[(sp.pf1 < 0.5)]
        print("  subjects with mean PF1 < 0.5 (wrong-exercise candidates):",
              low.round(2).reset_index().values.tolist() or "none")
        if len(e):
            st = e.groupby("group_id").score.agg(["size", "var", "min", "max"])
            st = st[st["size"] >= 2]
            floor = np.sqrt((st["var"].fillna(0) * (st["size"] - 1)).sum() / st["size"].sum())
            print(f"  inter-view label noise (expert groups): mean range {(st['max'] - st['min']).mean():.1f}, "
                  f"RMSE floor {floor:.2f}")
            summ[-1]["noise_floor"] = round(float(floor), 2)
    S = pd.DataFrame(summ)
    print("\n=== summary")
    print(S.to_string(index=False))
    S.to_csv(out / "summary_by_exercise.csv", index=False)
    j.to_csv(out / "inventory.csv", index=False)
    print("\nwrote", out / "inventory.csv", len(j), "rows")


if __name__ == "__main__":
    main()
