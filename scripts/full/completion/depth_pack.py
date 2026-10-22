"""Proposal completion (2026-10-22): the B2-DepthAnything input pack (A11).

<pack_dir>/reps_coco17_T64_depthanything.npz, key 'world' (N, T, 17, 3) like the KTPFormer pack (cvdc.data.pack): per
rep, channels 0-1 = the main pack's 2D x, y (the networks' 2D input, unchanged) and channel 2 = DepthAnything's
relative depth at each joint (depth_extract.py: disparity minus the mid-hip's, divided by the video's median map
spread; positive = closer to the camera than the hips), resampled over the rep's frames exactly like the main pack.
Reps of videos without a depth file (the exercises outside E01-E04) get NaN, so any use of them fails loudly.

usage: python scripts/full/completion/depth_pack.py [--config configs/completion.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.preprocess import resample_fixed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/completion.yaml")
    ap.add_argument("--smoke-fill", action="store_true", help="smoke test only: missing depth = 0 instead of NaN "
                                                              "(the real run rewrites the pack without it)")
    a = ap.parse_args()
    cfg = load_config(REPO / a.config)
    pc = cfg["preprocess"]
    pdir, src = resolve(pc["pack_dir"]), resolve(cfg["depth"]["out_dir"])
    base = f"reps_{pc['joint_set']}_T{pc['T_rep']}"
    items = pd.read_csv(pdir / f"{base}.csv")
    with np.load(pdir / f"{base}.npz") as z:
        xy = z["xy"]
    T = pc["T_rep"]
    D = np.full(xy.shape[:3], np.nan, np.float32)
    cache, missing = {}, set()
    for i, r in enumerate(items.itertuples()):
        if r.video_id not in cache:
            cache.clear()
            p = src / f"{r.video_id}.npz"
            if not p.exists():
                missing.add(r.video_id)
                cache[r.video_id] = None
            else:
                with np.load(p) as z:
                    cache[r.video_id] = z["rel"].astype(np.float32) / float(z["scale"])
        d = cache[r.video_id]
        if d is not None:
            D[i] = resample_fixed(d[r.start:r.end + 1], T)
    if a.smoke_fill:
        D = np.nan_to_num(D, nan=0.0)
    W = np.concatenate([xy, D[..., None]], -1).astype(np.float32)
    np.savez_compressed(pdir / f"{base}_depthanything.npz", world=W)
    ok = np.isfinite(W).all(axis=(1, 2, 3))
    by_ex = items.assign(ok=ok, ex=items.video_id.str[:3]).groupby("ex").ok.mean().round(3).to_dict()
    print(f"{base}_depthanything: {W.shape}; share of reps with depth per exercise {by_ex}; videos without a depth "
          f"file: {len(missing)}; depth channel median |d| {np.nanmedian(np.abs(D)):.3f}, p99 "
          f"{np.nanpercentile(np.abs(D), 99):.3f}")


if __name__ == "__main__":
    main()
