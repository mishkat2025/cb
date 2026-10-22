"""Proposal completion (2026-10-22): DepthAnything V2 relative depth at the joints of every E01-E04 video (GPU).

For every video with a Phase 1 file (P7), on the Phase 1 30 fps grid (P5): the decoded frame nearest to each grid time
(PyAV, rotation applied; cvdc.data.video), downscaled to ``depth.decode_max_side``, goes through DepthAnything V2 Small
(cvdc.models.depth); the disparity map is sampled at the 17 coco17 joints (their image positions from the Phase 1
keypoints: xy x torso_px + root_px, normalised by the frame size). Stored per grid frame:
  rel    (n, 17) disparity of each joint minus the mid-hip's (positive = closer to the camera than the hips)
  iqr    (n,) the disparity map's interquartile range (the image's own depth scale)
and per video ``scale`` = the median of iqr. The B2-DepthAnything input channel is rel / scale (A11; packed by
depth_pack.py). Frames are decoded and processed in memory only; nothing image-like is saved (D10).

usage: python scripts/full/completion/depth_extract.py [--workers 1] [--limit N] [--exercises E01 ...]
Resumable (videos with an output file are skipped). Writes <depth.out_dir>/<video_id>.npz and a log.
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve, video_root  # noqa: E402
from cvdc.data.skeleton import JOINT_SETS  # noqa: E402

EXS = ["E01", "E02", "E03", "E04"]
_STATE = {}


def _init(dc):
    import torch

    from cvdc.models.depth import DepthAnything
    torch.backends.cudnn.benchmark = True
    _STATE["model"] = DepthAnything(dc, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    _STATE["dc"] = dc


def nearest_frames(path, times, max_side):
    """Yield (k, rgb) for every grid time index k: the decoded frame nearest to times[k] (container clock)."""
    from cvdc.data.video import iter_frames

    k, prev = 0, None
    for t, rgb, _ in iter_frames(path, max_side=max_side):
        while k < len(times) and (prev is None and times[k] <= t or prev is not None and times[k] <= t):
            if prev is None or abs(t - times[k]) <= abs(prev[0] - times[k]):
                yield k, rgb
            else:
                yield k, prev[1]
            k += 1
        prev = (t, rgb)
        if k >= len(times):
            return
    while k < len(times) and prev is not None:  # grid times after the last frame
        yield k, prev[1]
        k += 1


def video_job(args):
    import torch

    from cvdc.models.depth import to_numpy
    vid, src, pre_path, kp_path, dst, joints = args
    m, dc = _STATE["model"], _STATE["dc"]
    t0 = time.time()
    with np.load(pre_path) as z:
        t, xy, root, tp, hw = z["t"].astype(float), z["xy"][:, joints].astype(float), z["root_px"].astype(float), \
            float(z["torso_px"]), z["frame_hw"]
    with np.load(kp_path) as z:
        t0c = float(z["t0"])
    uv = (xy * tp + root[:, None, :]) / np.array([hw[1], hw[0]], float)  # (n, 17, 2) normalised image coordinates
    n = len(t)
    rel = np.full((n, len(joints)), np.nan, np.float32)
    iqr = np.full(n, np.nan, np.float32)
    root_disp = np.full(n, np.nan, np.float32)
    bs = int(dc["batch"])
    buf_k, buf_img = [], []
    lh, rh = list(JOINT_SETS["coco17"]).index(23), list(JOINT_SETS["coco17"]).index(24)

    def flush():
        if not buf_k:
            return
        ks = np.array(buf_k)
        x = np.stack(buf_img)
        if len(x) < bs:  # a fixed batch size keeps cuDNN's tuned kernels (the last batch of a video is padded)
            x = np.concatenate([x, np.repeat(x[-1:], bs - len(x), 0)])
        d = m(x)[:len(ks)]
        s = to_numpy(m.sample(d, uv[ks]))
        rd = 0.5 * (s[:, lh] + s[:, rh])
        rel[ks] = s - rd[:, None]
        root_disp[ks] = rd
        iqr[ks] = to_numpy(m.spread(d))
        buf_k.clear()
        buf_img.clear()

    # decode in a background thread (PyAV decodes and rescales outside the GIL) while the GPU works on the last batch
    import queue
    import threading

    q = queue.Queue(maxsize=4 * bs)

    def producer():
        try:
            for item in nearest_frames(src, t + t0c, int(dc["decode_max_side"])):
                q.put(item)
        except BaseException as e:  # handed to the consumer
            q.put(e)
        finally:
            q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        buf_k.append(item[0])
        buf_img.append(item[1])
        if len(buf_k) == bs:
            flush()
    flush()
    th.join()
    scale = float(np.nanmedian(iqr)) if np.isfinite(iqr).any() else np.nan
    np.savez_compressed(dst, rel=rel.astype(np.float16), iqr=iqr.astype(np.float16), root=root_disp.astype(np.float16),
                        scale=np.float32(scale), t=t.astype(np.float32),
                        settings=np.array(f"DepthAnythingV2-Small input_size={m.size} "
                                          f"decode_max_side={dc['decode_max_side']}"))
    torch.cuda.empty_cache()
    return vid, n, time.time() - t0, int(np.isfinite(rel).all(1).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/completion.yaml")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="first N videos only (speed test)")
    ap.add_argument("--exercises", nargs="+", default=EXS)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    cfg = load_config(REPO / a.config)
    dc = cfg["depth"]
    out = resolve(dc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    idx = idx[idx.exercise.isin(a.exercises) & idx.pre_path.notna()]
    idx = idx[[resolve(p).exists() for p in idx.pre_path]]
    joints = list(JOINT_SETS[cfg["preprocess"]["joint_set"]])
    vr = video_root(cfg)
    jobs = [(r.video_id, str(vr / r.rel_path), str(resolve(r.pre_path)), str(resolve(r.kp_path)),
             str(out / f"{r.video_id}.npz"), joints) for r in idx.itertuples()
            if not (out / f"{r.video_id}.npz").exists()]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} videos to process ({len(idx)} in scope), {a.workers} workers", flush=True)
    frames = 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(dc,)) as pool:
        for i, (vid, n, sec, ok) in enumerate(pool.map(video_job, jobs, chunksize=1)):
            frames += n
            el = time.time() - t0
            print(f"[{i + 1}/{len(jobs)}] {vid}: {n} frames in {sec:.1f} s ({ok} with all joints); total "
                  f"{frames / el:.1f} frames/s, eta {(len(jobs) - i - 1) * el / (i + 1) / 3600:.2f} h", flush=True)
    print(f"done: {frames} frames in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
