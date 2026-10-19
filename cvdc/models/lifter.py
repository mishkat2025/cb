"""2D -> 3D lifter for B2 (A10): KTPFormer (Peng, Zhou & Mok, CVPR 2024, "KTPFormer: Kinematics and Trajectory
Prior Knowledge-Enhanced Transformer for 3D Human Pose Estimation"), the Human3.6M model trained on CPN 2D
detections (243 frames).

The model code is not vendored: it is imported from a local clone of github.com/JihuaPeng/KTPFormer
(``lifter.code_dir``, gitignored). The repository has no licence (GitHub reports none), so neither its code nor
its weights are committed or redistributed; they are used locally for research only. The weights are loaded
with ``torch.load(weights_only=True)`` from a tensors-only copy of the released checkpoint.

Pipeline per video (preprocessed 30 fps grid, coco17 joints, root-centred and torso-scaled 2D):
  1. COCO-17 -> H36M-17 joints (MotionBERT's ``coco2h36m`` convention: pelvis = mid-hip, thorax = mid-shoulder,
     spine = their midpoint, nose -> 'nose', head = mid-eyes);
  2. scale so the median 2D torso length is ``lifter.input_torso`` in H36M screen-normalised units (the person is
     centred at the origin), resample to ``lifter.fps`` (H36M is 50 fps);
  3. overlapping windows of 243 frames (stride ``lifter.stride``, edge-padded), test-time flip averaging as in
     the reference evaluation, overlapping outputs averaged; root (pelvis) = 0;
  4. back to the 30 fps grid and to coco17 order (eyes and ears <- H36M head), root-centred on the mid-hip and
     divided by the median 3D torso length, like the MediaPipe world landmarks (P5).

usage: python -m cvdc.models.lifter [--config C] [--only ID ...] [--skip-existing]
Writes <lifter.out_dir>/<video_id>.npz (pilot: data/cache/lift_ktpformer/) (``world`` (T,17,3) coco17 normalised, ``h36m`` (T,17,3) raw
output) for every preprocessed video and PT demo.
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from ..config import load_config, resolve
from ..data.preprocess import resample_fixed
from ..data.skeleton import JOINT_SETS

H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
H36M_LEFT, H36M_RIGHT = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
# coco17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
DIRECT = {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 9: 0, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}  # h36m <- coco
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def coco17_to_h36m(X):
    """(..., 17, C) coco17 -> (..., 17, C) H36M-17."""
    X = np.asarray(X, float)
    Y = np.zeros_like(X)
    for h, c in DIRECT.items():
        Y[..., h, :] = X[..., c, :]
    Y[..., 0, :] = (X[..., L_HIP, :] + X[..., R_HIP, :]) / 2
    Y[..., 8, :] = (X[..., L_SH, :] + X[..., R_SH, :]) / 2
    Y[..., 7, :] = (Y[..., 0, :] + Y[..., 8, :]) / 2
    Y[..., 10, :] = (X[..., 1, :] + X[..., 2, :]) / 2
    return Y


def h36m_to_coco17(Y):
    """(..., 17, C) H36M-17 -> (..., 17, C) coco17; eyes and ears take the H36M head joint."""
    Y = np.asarray(Y, float)
    X = np.zeros_like(Y)
    for h, c in DIRECT.items():
        X[..., c, :] = Y[..., h, :]
    for c in (1, 2, 3, 4):
        X[..., c, :] = Y[..., 10, :]
    return X


def torso_len(P):
    """Median over frames of |mid-shoulder - mid-hip| for (T, 17, C) coco17 arrays."""
    d = np.linalg.norm((P[:, L_SH] + P[:, R_SH]) / 2 - (P[:, L_HIP] + P[:, R_HIP]) / 2, axis=-1)
    return float(np.median(d))


def windows(n, size, stride):
    """Start indices of windows of ``size`` covering 0..n-1 (the last one ends at n-1; one window if n <= size)."""
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)
    return starts


class KTPFormerLifter:
    def __init__(self, lc, device="cuda"):
        sys.path.insert(0, str(resolve(lc["code_dir"])))
        from common.graph_utils import adj_mx_from_edges  # noqa: E402  (local clone, not vendored)
        from common.model_ktpformer import KTPFormer  # noqa: E402

        F = lc["frames"]
        adj = adj_mx_from_edges(17, [(i, p) for i, p in enumerate(H36M_PARENTS) if p >= 0], sparse=False)
        adj_t = adj_mx_from_edges(F, [(i, i - 1) for i in range(1, F)], sparse=False)
        self.model = KTPFormer(adj, adj_t, num_frame=F, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=7,
                               num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path_rate=0.0)
        sd = torch.load(resolve(lc["weights"]), map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device).eval()
        self.lc, self.device, self.F = lc, device, F

    @torch.no_grad()
    def lift_h36m(self, x2d):
        """x2d: (T, 17, 2) H36M screen-normalised 2D at the lifter's frame rate -> (T, 17, 3) root-relative 3D."""
        T, F = len(x2d), self.F
        pad = max(0, F - T)
        xp = np.concatenate([x2d, np.repeat(x2d[-1:], pad, 0)]) if pad else x2d
        starts = windows(len(xp), F, self.lc["stride"])
        X = torch.tensor(np.stack([xp[s:s + F] for s in starts]), dtype=torch.float32, device=self.device)
        outs = []
        amp = self.lc.get("amp", "off") == "bf16" and self.device == "cuda"
        for i in range(0, len(X), self.lc["batch"]):
            xb = X[i:i + self.lc["batch"]]
            if self.lc.get("flip_test", True):  # flipped copy in the same batch
                xf = xb.clone()
                xf[..., 0] *= -1
                xf[:, :, H36M_LEFT + H36M_RIGHT] = xf[:, :, H36M_RIGHT + H36M_LEFT]
                xb = torch.cat([xb, xf])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                y = self.model(xb).float()
            if self.lc.get("flip_test", True):
                y, yf = y.chunk(2)
                yf[..., 0] *= -1
                yf[:, :, H36M_LEFT + H36M_RIGHT] = yf[:, :, H36M_RIGHT + H36M_LEFT]
                y = (y + yf) / 2
            outs.append(y.float().cpu().numpy())
        Y = np.concatenate(outs)
        acc, cnt = np.zeros((len(xp), 17, 3)), np.zeros(len(xp))
        for s, y in zip(starts, Y):
            acc[s:s + F] += y
            cnt[s:s + F] += 1
        out = (acc / cnt[:, None, None])[:T]
        return out - out[:, :1]  # root (pelvis) = 0

    def lift_coco(self, xy, fps_in):
        """xy: (T, 17, 2) coco17 2D (root-centred, any scale) at ``fps_in`` -> dict(world (T, 17, 3) coco17 root-
        centred / median 3D torso length, h36m (T, 17, 3) raw lifter output at fps_in)."""
        lc = self.lc
        h = coco17_to_h36m(xy)
        h = h - h[:, :1]
        h = h / torso_len(xy) * lc["input_torso"]
        T = len(h)
        n_up = max(2, int(round((T - 1) * lc["fps"] / fps_in)) + 1)
        h_up = resample_fixed(h, n_up)
        y = resample_fixed(self.lift_h36m(h_up), T)
        w = h36m_to_coco17(y)
        w = w - ((w[:, L_HIP] + w[:, R_HIP]) / 2)[:, None]
        return {"world": (w / torso_len(w)).astype(np.float32), "h36m": y.astype(np.float32)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--input-torso", type=float, help="override lifter.input_torso (scale check)")
    ap.add_argument("--out-dir", help="override lifter.out_dir")
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep outputs that already exist with the same input_torso (e.g. copied pilot lifts)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    lc = dict(cfg["lifter"])
    if a.input_torso:
        lc["input_torso"] = a.input_torso
    out = resolve(a.out_dir or lc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    coco = JOINT_SETS["coco17"]
    fps = cfg["preprocess"]["fps"]
    vids = pd.concat([pd.read_csv(resolve(cfg["data"]["index_csv"]))[["video_id", "pre_path"]],
                      pd.read_csv(resolve(cfg["data"]["ref_index_csv"])).rename(columns={"ref_id": "video_id"})[
                          ["video_id", "pre_path"]].drop_duplicates("video_id")])
    vids = vids[vids.pre_path.map(lambda p: isinstance(p, str) and resolve(p).exists())]
    if a.only:
        vids = vids[vids.video_id.isin(a.only)]
    if a.skip_existing:
        def done(v):
            f = out / f"{v}.npz"
            if not f.exists():
                return False
            with np.load(f) as z:
                return float(z["input_torso"]) == float(lc["input_torso"])
        keep = vids.video_id.map(done)
        print(f"skipping {int(keep.sum())} existing outputs")
        vids = vids[~keep]
    lifter = KTPFormerLifter(lc, "cuda" if torch.cuda.is_available() else "cpu")
    t0, n_fr = time.time(), 0
    for i, r in enumerate(vids.itertuples()):
        with np.load(resolve(r.pre_path)) as z:
            xy = z["xy"][:, coco].astype(float)
        res = lifter.lift_coco(xy, fps)
        np.savez_compressed(out / f"{r.video_id}.npz", **res, input_torso=lc["input_torso"])
        n_fr += len(xy)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(vids)} videos, {n_fr / (time.time() - t0):.0f} frames/s", flush=True)
    print(f"lifted {len(vids)} videos ({n_fr} frames at {fps} fps) in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
