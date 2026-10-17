"""Fixed-length network inputs (P5): every rep resampled to T_rep and every video to T_video frames,
for the configured joint set. The full-length uniform-grid sequences stay in data/cache/pre for DTW.

Arrays (float32): xy (N,T,J,2), vis (N,T,J), world (N,T,J,3); plus an index CSV with one row per item
(video_id, rep, start, end on the uniform grid).

usage: python -m cvdc.data.pack [--config C]
Writes data/cache/packs/{reps,videos}_<joint_set>_T<T>.npz and the matching .csv.
"""
import argparse

import numpy as np
import pandas as pd

from ..config import load_config, resolve
from .preprocess import resample_fixed
from .skeleton import JOINT_SETS


def pack(items, T, joints):
    """items: list of (pre_path, start, end) -> dict of stacked fixed-length arrays."""
    xy, vis, w3 = [], [], []
    cache = {}
    for path, s, e in items:
        if path not in cache:
            cache.clear()
            with np.load(path) as z:
                cache[path] = {k: z[k][:, joints] for k in ("xy", "vis", "world")}
        d = cache[path]
        xy.append(resample_fixed(d["xy"][s:e + 1], T))
        vis.append(resample_fixed(d["vis"][s:e + 1], T))
        w3.append(resample_fixed(d["world"][s:e + 1], T))
    return dict(xy=np.asarray(xy, np.float32), vis=np.asarray(vis, np.float32), world=np.asarray(w3, np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pc = cfg["preprocess"]
    joints = JOINT_SETS[pc["joint_set"]]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    reps = pd.read_csv(resolve("data/reps.csv"))
    pre = idx.set_index("video_id").pre_path.map(resolve)
    out = resolve(pc["pack_dir"])
    out.mkdir(parents=True, exist_ok=True)

    reps = reps[reps.video_id.isin(pre.index)].reset_index(drop=True)
    R = pack([(pre[r.video_id], r.start, r.end) for r in reps.itertuples()], pc["T_rep"], joints)
    name = f"reps_{pc['joint_set']}_T{pc['T_rep']}"
    np.savez_compressed(out / f"{name}.npz", **R)
    reps[["video_id", "rep", "start", "peak", "end", "side"]].to_csv(out / f"{name}.csv", index=False)

    vids = idx[idx.pre_path.map(lambda p: resolve(p).exists())].reset_index(drop=True)
    n = {v: int(np.load(pre[v])["t"].shape[0]) for v in vids.video_id}
    V = pack([(pre[v], 0, n[v] - 1) for v in vids.video_id], pc["T_video"], joints)
    vname = f"videos_{pc['joint_set']}_T{pc['T_video']}"
    np.savez_compressed(out / f"{vname}.npz", **V)
    vids.assign(start=0, end=[n[v] - 1 for v in vids.video_id])[["video_id", "start", "end"]].to_csv(
        out / f"{vname}.csv", index=False)
    for nm, d in ((name, R), (vname, V)):
        print(f"{nm}: " + ", ".join(f"{k} {v.shape}" for k, v in d.items()),
              f"| NaN in xy: {int(np.isnan(d['xy']).any(axis=(1, 2, 3)).sum())} items")


if __name__ == "__main__":
    main()
