"""M19 (proposal completion, 2026-10-22): the CVDC representation inside Module III, as in proposal Fig. 3
("depth-aware fused skeletal embedding -> DTW + joint-angle comparison vs clinical template") and Fig. 2 step 2
("map input to a depth-aware skeleton representation" before the template comparison).

For one fold's Proposed model (the CVDC encoder after fine-tuning, eval.proposed_default):
  1. every rep and the PT template's reps are encoded as sequences of per-time features (STGCN.sequence: the last
     block's features averaged over joints, T' = 16 steps for T_rep = 64), centred with the fold's training mean;
  2. each rep is aligned to the PT reference sequence by DTW in that embedding space (cosine geometry: the centred
     steps are L2-normalised, so the Euclidean DTW cost is a monotone function of 1 - cos);
  3. the step path is turned into a frame path between the rep's frames and the template's frames
     (``frame_path``); Module III reads every joint-angle channel along it (instead of the arm-direction DTW, M8)
     and gets one more channel, ``emb_dist``: per rep frame, the cosine distance of its step to the matched template
     step (``frame_distance``), compared with the >= 95 experts like the posture channels (score group 'embedding',
     rule ``movement_pattern_deviation``).
The PT reference uses exactly Module III's template reps (``ref['spans']``: frames and side of every demo rep used,
E03's working phase and both arms), each packed like a patient rep (coco17, T_rep, 2D xy) and mirrored when its side
differs from the patient's (P8); the reference sequence is their mean. A rep's 16 steps and the template's 16 steps
are both spread uniformly over their frames (step j at pack frame 4j of the T_rep = 64 resampled rep).
"""
import numpy as np
import torch

from .dtw import dtw


def step_frames(n_frames, n_steps, T):
    """Frame position (float) of each of the n_steps encoder steps in a rep of n_frames resampled to T pack frames:
    step j is centred on pack frame j * T / n_steps (two stride-2 convolutions with symmetric padding)."""
    stride = T / n_steps
    return np.arange(n_steps) * stride / (T - 1) * (n_frames - 1)


def align_steps(S_rep, S_ref):
    """DTW of two centred step sequences (n, D), (m, D) on L2-normalised steps. Returns (path (L, 2), the cosine
    distance 1 - cos of every matched pair (L,))."""
    a = S_rep / np.maximum(np.linalg.norm(S_rep, axis=1, keepdims=True), 1e-12)
    b = S_ref / np.maximum(np.linalg.norm(S_ref, axis=1, keepdims=True), 1e-12)
    _, path = dtw(a, b)
    d = 1.0 - np.sum(a[path[:, 0]] * b[path[:, 1]], axis=1)
    return path, d


def frame_path(step_path, n_rep, n_ref, n_steps, T):
    """A monotone frame path (L, 2) from (0, 0) to (n_rep - 1, n_ref - 1) through the matched step centres: the
    piecewise-linear curve through them, rasterised so that both indices advance by at most 1 per point."""
    pr, pt = step_frames(n_rep, n_steps, T), step_frames(n_ref, n_steps, T)
    pts = [(0.0, 0.0)] + [(pr[i], pt[j]) for i, j in step_path] + [(n_rep - 1.0, n_ref - 1.0)]
    out = [(0, 0)]
    for (a0, b0), (a1, b1) in zip(pts[:-1], pts[1:]):
        k = int(max(abs(round(a1) - round(a0)), abs(round(b1) - round(b0)), 1))
        for s in np.linspace(0, 1, k + 1)[1:]:
            p = (int(round(a0 + s * (a1 - a0))), int(round(b0 + s * (b1 - b0))))
            if p != out[-1]:
                out.append(p)
    path = np.asarray(out)
    # monotone and gap-free by construction; guard the corners
    path[:, 0] = np.clip(np.maximum.accumulate(path[:, 0]), 0, n_rep - 1)
    path[:, 1] = np.clip(np.maximum.accumulate(path[:, 1]), 0, n_ref - 1)
    return path


def frame_distance(step_path, d, n_rep, n_steps, T):
    """(n_rep,): the embedding distance at every rep frame: per rep step, the mean distance of its matched pairs,
    linearly interpolated between the step centres."""
    per = np.zeros(n_steps)
    cnt = np.zeros(n_steps)
    np.add.at(per, step_path[:, 0], d)
    np.add.at(cnt, step_path[:, 0], 1)
    per = per / np.maximum(cnt, 1)
    return np.interp(np.arange(n_rep), step_frames(n_rep, n_steps, T), per)


# ---------------------------------------------------------------- encoding (GPU)
@torch.no_grad()
def encode_sequences(model, x, cfg, dev, batch=512):
    """Per-time features (N, T', D) of model inputs x (N, C, T, V) in eval mode (bf16 autocast as in training)."""
    from ..train.trainer import autocast

    model.eval()
    out = []
    for i in range(0, len(x), batch):
        with autocast(cfg, dev):
            out.append(model.encoder.sequence(x[i:i + batch]).float())
    return torch.cat(out).cpu().numpy().astype(np.float64)


def ref_template_packs(cfg, ref_meta, T):
    """The PT template reps of Module III (``ref_meta['spans']``, ``ref_meta['pt_pre_path']``) packed like patient
    reps: {side: (n_reps, T, V, 2) 2D xy}, keyed by the side the rep is used for."""
    from ..config import resolve
    from ..data.preprocess import resample_fixed
    from ..data.skeleton import JOINT_SETS

    joints = JOINT_SETS[cfg["preprocess"]["joint_set"]]
    with np.load(resolve(ref_meta["pt_pre_path"])) as z:
        xy = z["xy"][:, joints].astype(np.float32)
    packs = {}
    for s, e, side in ref_meta["spans"]:
        packs.setdefault(side, []).append(resample_fixed(xy[int(s):int(e) + 1], T))
    return {k: np.asarray(v, np.float32) for k, v in packs.items()}


def ref_sequences(model, packs, cfg, dev):
    """{patient side: (T', D) reference sequence}: the mean over every template rep, mirrored (x -> -x, left / right
    swapped) when the rep is used for the other side than the patient's (P8)."""
    from ..data.skeleton import mirror_coco17

    out = {}
    for patient in ("left", "right"):
        xs = []
        for side, p in packs.items():
            xs.append(p if side == patient else mirror_coco17(p))
        x = torch.from_numpy(np.concatenate(xs)).to(dev).permute(0, 3, 1, 2).contiguous()
        out[patient] = encode_sequences(model, x, cfg, dev).mean(0)
    return out
