"""Container metadata and a decode check for every video (headers + first and last frame; nothing saved).

Per video: fps (average rate), frame count, duration, upright resolution, rotation, audio presence,
creation_time present, and whether the first frame and a frame in the last second decode. Files with the
same subject / exercise / view and the same byte size are compared by MD5 (copy check). GPS tags are never read.

usage: python -I scripts/step0_full/video_meta.py --videos D:/cb/mobiphysio --inv outputs/step0_full/inventory.csv
       --out outputs/step0_full [--exercises E03 ...] [--workers 12]
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd


def meta(path):
    r = dict(file=path.name, opened=False, first_ok=False, last_ok=False, error="")
    try:
        with av.open(str(path)) as c:
            r["opened"] = True
            vs = c.streams.video[0]
            r.update(fps=float(vs.average_rate or 0), frames=int(vs.frames or 0),
                     duration_s=c.duration / 1e6 if c.duration else np.nan,
                     has_audio=bool(c.streams.audio), has_creation=bool(c.metadata.get("creation_time")),
                     codec=vs.codec_context.name, enc_w=vs.codec_context.width, enc_h=vs.codec_context.height)
            for fr in c.decode(vs):
                r["first_ok"] = True
                r["rotation"] = int(fr.rotation or 0)
                break
            if np.isfinite(r["duration_s"]) and r["duration_s"] > 1.5:
                c.seek(int((r["duration_s"] - 1.0) * 1e6), any_frame=False)  # container time base (us)
                for fr in c.decode(vs):
                    r["last_ok"] = True
                    r["last_t"] = float(fr.time) if fr.time is not None else np.nan
                    break
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"[:200]
    rot = r.get("rotation", 0)
    w, h = r.get("enc_w", 0), r.get("enc_h", 0)
    r["width"], r["height"] = (h, w) if rot % 180 else (w, h)
    return r


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--inv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exercises", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    inv = pd.read_csv(a.inv)
    if a.exercises:
        inv = inv[inv.exercise.isin(a.exercises)]
    paths = [Path(a.videos) / f / n for f, n in zip(inv.folder, inv.file)]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(meta, paths))
    M = inv[["video_id", "exercise", "subject", "group", "view", "variation", "size_mb", "file"]].merge(
        pd.DataFrame(rows), on="file")
    same = M[M.duplicated(["exercise", "subject", "view", "size_mb"], keep=False)].copy()
    same["md5"] = [md5(Path(a.videos) / inv.set_index("file").folder[f] / f) for f in same.file]
    M = M.merge(same[["file", "md5"]], on="file", how="left")
    M.to_csv(Path(a.out) / "video_meta.csv", index=False)

    pd.set_option("display.width", 250)
    bad = M[~(M.opened & M.first_ok & M.last_ok) | (M.error != "")]
    print(f"{len(M)} videos; not opened {int((~M.opened).sum())}, first frame fails {int((~M.first_ok).sum())}, "
          f"last second fails {int((~M.last_ok).sum())}")
    if len(bad):
        print(bad[["video_id", "opened", "first_ok", "last_ok", "duration_s", "error"]].to_string(index=False))
    print("no audio:", M.loc[~M.has_audio.astype(bool), "video_id"].tolist() or "none")
    print("no creation_time:", M.loc[~M.has_creation.astype(bool), "video_id"].tolist() or "none")
    print("frame count vs duration x fps off by > 5%:",
          M.loc[(M.frames - M.duration_s * M.fps).abs() > 0.05 * M.frames, "video_id"].tolist() or "none")
    print("identical-size candidates (md5):")
    print(same[["video_id", "size_mb"]].assign(md5=same.md5.str[:12]).to_string(index=False) if len(same) else " none")
    for ex, g in M.groupby("exercise"):
        print(f"\n=== {ex}: {len(g)} videos")
        print("  fps        :", g.fps.round(0).value_counts().sort_index().to_dict())
        print("  resolution :", (g.width.astype(int).astype(str) + "x" + g.height.astype(int).astype(str)).value_counts().to_dict())
        print("  rotation   :", g.rotation.value_counts().to_dict())
        for grp, gg in g.groupby("group"):
            d = gg.duration_s
            print(f"  duration {grp:10s}: min {d.min():.1f}  p25 {d.quantile(.25):.1f}  median {d.median():.1f}  "
                  f"p75 {d.quantile(.75):.1f}  max {d.max():.1f} s  (total {d.sum() / 3600:.2f} h, "
                  f"{int((gg.duration_s * gg.fps).sum())} frames)")
        print(f"  total frames (duration x fps): {int((g.duration_s * g.fps).sum())}")


if __name__ == "__main__":
    main()
