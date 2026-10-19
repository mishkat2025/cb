"""Copy check for videos without creation_time (the pilot's E01_P60_AF_VLR_GF was a re-encoded copy): GCC-PHAT audio
match against every other video of the same subject and view, in all exercises. A peak ratio far above the usual
1.0-1.3 at a stable lag with similar durations marks a copy. Audio is kept in memory only.

usage: python -I scripts/step0_full/copy_check.py --videos D:/cb/mobiphysio --out outputs/step0_full
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.data.sync import gcc_phat  # noqa: E402
from cvdc.data.video import read_audio_mono  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="outputs/step0_full")
    a = ap.parse_args()
    out = Path(a.out)
    meta = pd.read_csv(out / "video_meta.csv")
    inv = pd.read_csv(out / "inventory.csv").set_index("video_id")
    targets = meta[~meta.has_creation.astype(bool)].video_id.tolist()
    cache = {}

    def audio(v):
        if v not in cache:
            cache[v] = read_audio_mono(Path(a.videos) / inv.folder[v] / inv.file[v], 8000)[0]
        return cache[v]
    rows = []
    for t in targets:
        same = meta[(meta.subject == inv.subject[t]) & (meta.view == inv.view[t]) & (meta.video_id != t)]
        for o in same.video_id:
            xa, xb = audio(t), audio(o)
            if xa is None or xb is None or min(len(xa), len(xb)) < 8000:
                continue
            lag, ratio = gcc_phat(xa, xb, 8000, (100, 3800), 15.0)
            rows.append(dict(video=t, other=o, lag_s=lag, peak_ratio=ratio,
                             dur=meta.set_index("video_id").duration_s[t], dur_other=meta.set_index("video_id").duration_s[o]))
    R = pd.DataFrame(rows).sort_values(["video", "peak_ratio"], ascending=[True, False])
    R.to_csv(out / "copy_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.groupby("video").head(2).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
