"""Read container metadata (no frames saved) for every video under the dataset root."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def meta(p: Path):
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    rot = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    ok = cap.isOpened()
    cap.release()
    return dict(folder=p.parent.name, file=p.name, opened=ok, fps=round(fps, 3), frames=int(n),
                width=int(w), height=int(h), rotation=rot, duration_s=round(n / fps, 2) if fps else None)


files = sorted(ROOT.glob("*/*.mp4"))
with ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(meta, files))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 200)
print("not opened:", (~df.opened).sum())
for folder, g in df.groupby("folder"):
    print(f"\n=== {folder}: {len(g)} videos")
    print(" fps      :", g.fps.round(1).value_counts().to_dict())
    print(" res(WxH) :", (g.width.astype(str) + "x" + g.height.astype(str)).value_counts().to_dict())
    print(" rotation :", g.rotation.value_counts().to_dict())
    d = g.duration_s
    print(f" duration s: min {d.min():.1f}  p25 {d.quantile(.25):.1f}  median {d.median():.1f}  p75 {d.quantile(.75):.1f}  max {d.max():.1f}  total {d.sum()/3600:.2f} h")
ref = df[df.folder.str.startswith("Reference")]
print("\nReference videos:")
print(ref[["file", "fps", "frames", "width", "height", "rotation", "duration_s"]].to_string(index=False))
