"""P2/P3 check: does downscaling to 960 px or padding to a square change MediaPipe landmarks?

Runs the extractor with 4 settings (960 / full resolution x pad / no pad) on a fixed sample of videos and
compares: detection rate, landmark agreement (torso units), a 2D aspect check (shoulder width / torso
length, which a horizontal squash would change), 3D bone-length stability, temporal jitter and run time.
usage: python scripts/phase1/pose_settings_check.py [--workers 8]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvdc.config import data_root, load_config, resolve  # noqa: E402
from cvdc.data.extract import extract_one  # noqa: E402

SAMPLE = [  # (exercise, group, view, variation): spread over views, light, occlusion, jitter, low resolution
    ("E01", "expert", "F", "FL"), ("E01", "expert", "L", "FL"), ("E01", "expert", "R", "LL"),
    ("E01", "expert", "F", "O"), ("E01", "expert", "L", "HJ"), ("E01", "non-expert", "F", "ML"),
    ("E01", "non-expert", "F", "LR"), ("E02", "expert", "F", "ML"), ("E02", "expert", "L", "LL"),
    ("E02", "expert", "R", "O"), ("E02", "expert", "R", "LJ"), ("E02", "expert", "F", "HJ")]
SETTINGS = {"s960": (960, False), "s960pad": (960, True), "full": (None, False), "fullpad": (None, True)}
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
UP = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP]


def load(f):
    z = np.load(f)
    h, w = z["frame_hw"]
    xy = z["lm2d"][..., :2] * np.array([w, h])
    return dict(xy=xy, world=z["world"], det=z["detected"], t=z["t"])


def torso(xy):
    return np.nanmedian(np.linalg.norm((xy[:, L_SH] + xy[:, R_SH]) / 2 - (xy[:, L_HIP] + xy[:, R_HIP]) / 2, axis=1))


def stats(d):
    xy, w3 = d["xy"], d["world"]
    T = torso(xy)
    sw = np.linalg.norm(xy[:, L_SH] - xy[:, R_SH], axis=1) / T
    ua = np.linalg.norm(w3[:, [L_EL, R_EL]] - w3[:, [L_SH, R_SH]], axis=2)  # (T,2)
    acc = np.linalg.norm(np.diff(xy[:, UP], 2, axis=0), axis=2) / T
    return dict(det=float(d["det"].mean()), shoulder_w_over_torso=float(np.nanmedian(sw)),
                upper_arm_3d_cv=float(np.nanmean(np.nanstd(ua, 0) / np.nanmean(ua, 0))),
                torso_3d_m=float(np.nanmedian(np.linalg.norm((w3[:, L_SH] + w3[:, R_SH]) / 2 - (w3[:, L_HIP] + w3[:, R_HIP]) / 2, axis=1))),
                jitter_torso_per_frame2=float(np.nanmedian(acc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config()
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    root = data_root(cfg)
    rng = np.random.default_rng(0)
    picks = []
    for ex, grp, view, var in SAMPLE:
        c = idx[(idx.exercise == ex) & (idx.group == grp) & (idx.view == view) & (idx.variation == var)]
        picks.append(c.iloc[rng.integers(len(c))])
    picks = pd.DataFrame(picks)
    base = resolve("data/cache/pose_check")
    p = cfg["pose"]
    conf = {k: p[k] for k in ("min_pose_detection_confidence", "min_pose_presence_confidence", "min_tracking_confidence")}
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(name, r.video_id, ms, pad) for name, (ms, pad) in SETTINGS.items() for r in picks.itertuples()]
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [(n, v, pool.submit(extract_one, str(root / picks.set_index("video_id").loc[v, "rel_path"]),
                                   str(base / n / f"{v}.npz"), resolve(p["model"]), ms, pad, conf))
                for n, v, ms, pad in jobs]
        secs = {(n, v): f.result().get("sec", np.nan) for n, v, f in futs}

    rows, agree = [], []
    for r in picks.itertuples():
        D = {n: load(base / n / f"{r.video_id}.npz") for n in SETTINGS}
        for n, d in D.items():
            rows.append(dict(video_id=r.video_id, view=r.view, variation=r.variation, setting=n,
                             sec=secs[(n, r.video_id)], **stats(d)))
        T = torso(D["fullpad"]["xy"])
        for n1, n2 in itertools.combinations(SETTINGS, 2):
            a1, a2 = D[n1]["xy"][:, UP], D[n2]["xy"][:, UP]
            n = min(len(a1), len(a2))
            diff = np.linalg.norm(a1[:n] - a2[:n], axis=2) / T
            agree.append(dict(video_id=r.video_id, pair=f"{n1} vs {n2}", median_diff_torso=float(np.nanmedian(diff)),
                              p90_diff_torso=float(np.nanpercentile(diff, 90))))
    S, A = pd.DataFrame(rows), pd.DataFrame(agree)
    out = resolve(cfg["outputs"]["phase1"])
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "pose_settings_check.csv", index=False)
    A.to_csv(out / "pose_settings_agreement.csv", index=False)
    pd.set_option("display.width", 220)
    print("per setting (mean over the 12 videos):")
    print(S.groupby("setting")[["det", "shoulder_w_over_torso", "upper_arm_3d_cv", "torso_3d_m",
                                "jitter_torso_per_frame2", "sec"]].mean().round(4).to_string())
    print("\nshoulder width / torso length per video (aspect check):")
    print(S.pivot_table(index=["video_id"], columns="setting", values="shoulder_w_over_torso").round(3).to_string())
    print("\ndetection rate per video:")
    print(S.pivot_table(index=["video_id"], columns="setting", values="det").round(3).to_string())
    print("\nlandmark agreement between settings (upper body, torso units):")
    print(A.groupby("pair")[["median_diff_torso", "p90_diff_torso"]].agg(["mean", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
