"""Supervisor answer 1: Module III assesses E02 with the 3D horizontal adduction angle (upper arm vs shoulder
line in the transverse plane). How does that angle behave on the data?

Per expert E02 video (moving arm from the rep step): share of frames where the upper arm's projection on the
transverse plane is long enough (>= 0.25 x upper-arm length, i.e. >= ~14.5 deg from vertical), and the angle
at rest (rep starts) vs at the rep peaks. Cross-view: in audio-synced groups, the angle at the same aligned
instants (the reference view's rep peaks) in each view.
usage: python scripts/phase1/e02_angle_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvdc.assess.angles import horizontal_adduction, shoulder_elevation  # noqa: E402
from cvdc.config import load_config, resolve  # noqa: E402

MIN_PROJ = 0.25


def main():
    cfg = load_config()
    fps = cfg["preprocess"]["fps"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    V = pd.read_csv(resolve(cfg["outputs"]["phase1"]) / "reps_per_video.csv")
    R = pd.read_csv(resolve("data/reps.csv"))
    G = pd.read_csv(resolve(cfg["sync"]["report_csv"]))
    e = V[(V.exercise == "E02") & (V.group == "expert")].merge(idx[["video_id", "pre_path", "group_id"]], on="video_id")
    ang, rows = {}, []
    for r in e.itertuples():
        w = np.load(resolve(r.pre_path))["world"].astype(float)
        a = horizontal_adduction(w, r.side, MIN_PROJ)
        ang[r.video_id] = a
        rr = R[R.video_id == r.video_id]
        rows.append(dict(video_id=r.video_id, view=r.view, defined=np.isfinite(a).mean(),
                         elevation_median=np.nanmedian(shoulder_elevation(w, r.side)),
                         rest=np.nanmedian(a[rr.start.values]) if len(rr) else np.nan,
                         peak=np.nanmedian(a[rr.peak.values]) if len(rr) else np.nan,
                         peak_defined=np.isfinite(a[rr.peak.values]).mean() if len(rr) else np.nan))
    D = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("E02 experts, horizontal adduction (deg), per view (medians over videos):")
    print(D.groupby("view")[["defined", "peak_defined", "elevation_median", "rest", "peak"]].median().round(2).to_string())
    # cross-view: angle at the reference view's peaks, read in every view at the same aligned time
    diffs = []
    for g in G[(G.exercise == "E02") & (G.method == "audio")].itertuples():
        vids = idx[idx.group_id == g.group_id].set_index("view")
        if g.ref_view not in vids.index or vids.loc[g.ref_view, "video_id"] not in ang:
            continue
        rv = vids.loc[g.ref_view, "video_id"]
        t_ref = R[R.video_id == rv].peak.values / fps - getattr(g, f"offset_{g.ref_view}_s")
        for v in vids.index:
            vid = vids.loc[v, "video_id"]
            if v == g.ref_view or vid not in ang:
                continue
            k = np.round((t_ref + getattr(g, f"offset_{v}_s")) * fps).astype(int)
            kr = np.round((t_ref + getattr(g, f"offset_{g.ref_view}_s")) * fps).astype(int)
            inside = (k >= 0) & (k < len(ang[vid]))
            k, kr = k[inside], kr[inside]
            d = (ang[vid][k] - ang[rv][kr] + 180) % 360 - 180  # wrapped difference
            diffs += [dict(pair=f"{g.ref_view}-{v}", diff=x) for x in d if np.isfinite(x)]
    X = pd.DataFrame(diffs)
    print("\nsame instant (reference-view rep peaks), view minus reference, deg:")
    print(X.groupby("pair")["diff"].describe(percentiles=[.25, .5, .75]).round(1).to_string())
    X.assign(absdiff=X["diff"].abs()).groupby("pair").absdiff.median().round(1).to_csv(
        resolve(cfg["outputs"]["phase1"]) / "e02_horizontal_adduction_crossview.csv")
    D.to_csv(resolve(cfg["outputs"]["phase1"]) / "e02_horizontal_adduction.csv", index=False)


if __name__ == "__main__":
    main()
