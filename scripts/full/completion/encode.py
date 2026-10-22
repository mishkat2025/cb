"""Proposal completion (2026-10-22): encode every E01-E04 rep with each fold's trained models (GPU).

For every exercise and fold of ``--tag`` (GroupKFold-5 ``perex_gkf5`` or LOSO ``perex_loso`` with ``--loso``):
  join   (M19) with eval.proposed_default (Proposed, lambda 1.0): every rep's per-time CVDC features, aligned by DTW
         to the PT template's reference sequence (cvdc.assess.cvdc_join); saves the step path and the cosine
         distance of every matched pair.
  z      (with ``--z``; GroupKFold-5 only) the pooled 256-d embedding of every rep (training and test subjects) for
         B1, B2-KTPFormer, Proposed lambda 1.0, the frozen CVDC encoder and the random-init encoder: the input of the
         depth check (scripts/full/completion/depth_check.py).
Every model is checked against the embeddings it saved at training time (embeddings_final.npz, test items).

usage: python scripts/full/completion/encode.py [--tag perex_gkf5] [--z] [--loso] [--exercises E01 ...]
Writes <results>/completion/encode/<tag>/<EX>/fold<k>.npz.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.assess.cvdc_join import align_steps, encode_sequences, ref_sequences, ref_template_packs  # noqa: E402
from cvdc.assess.module3 import assess_config  # noqa: E402
from cvdc.assess.run import reference_for  # noqa: E402
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402
from cvdc.train.run import method_config  # noqa: E402
from cvdc.train.trainer import fold_plan, forward_all, get_device, loso_plans, seed_everything  # noqa: E402

EXS = ["E01", "E02", "E03", "E04"]
INIT = "init_xy"
Z_METHODS = ["b1", "b2_ktpformer", "proposed_j1", "proposed_lp", INIT]
LMAX = 32  # DTW path length bound for 16 x 16 steps is 31


def load_model(cfg, d, m, fold, dev):
    c = method_config(cfg, "b1" if m == INIT else m)
    if m == INIT:  # as cvdc.train.run builds the init snapshot of the fold's seed
        seed_everything(cfg["train"]["seed"] + fold)
        model = CVDCModel(c).to(dev)
        snap = np.load(d / "embeddings_init.npz")
    else:
        ck = torch.load(d / "finetune" / "best.pt", map_location=dev, weights_only=True)
        model = CVDCModel(c).to(dev)
        model.load_state_dict(ck["model"])
        snap = np.load(d / "embeddings_final.npz")
    return model, c, snap


def check_snapshot(z, items, snap, tag):
    pos = {x: i for i, x in enumerate(items)}
    gap = np.abs(z[[pos[x] for x in snap["item"]]] - snap["z"]).max()
    if gap > 0.05 * max(1.0, np.abs(snap["z"]).max()):
        raise RuntimeError(f"{tag}: embeddings differ from the training-time snapshot by {gap:.3f}")
    return float(gap)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--tag", default="perex_gkf5")
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--z", action="store_true", help="also save pooled embeddings of every method (depth check)")
    ap.add_argument("--exercises", nargs="+", default=EXS)
    ap.add_argument("--device")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    cfg = load_config(REPO / a.config)
    if a.device:
        cfg["train"]["device"] = a.device
    dev = get_device(cfg)
    data = PackData(cfg, "rep", dev, lift="ktpformer" if a.z else None)
    meta = data.meta
    T = cfg["preprocess"]["T_rep"]
    jm = cfg["eval"]["proposed_default"]
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out_root = resolve(cfg["eval"]["results_dir"]) / "completion" / "encode" / a.tag
    for ex in a.exercises:
        ac = assess_config(cfg, ex)
        ref, _ = reference_for(cfg, ex, ac)
        packs = ref_template_packs(cfg, ref, T)
        sel = np.flatnonzero((meta.exercise == ex).to_numpy())
        items = torch.from_numpy(sel).to(dev)
        side = meta.side.to_numpy()[sel]
        mex = meta[meta.exercise == ex]
        plans = loso_plans(cfg, mex) if a.loso else fold_plan(cfg, mex)
        (out_root / ex).mkdir(parents=True, exist_ok=True)
        for p in plans:
            f = p["fold"]
            dst = out_root / ex / f"fold{f}.npz"
            if dst.exists():
                continue
            save = {"item": sel.astype(np.int32), "fold": np.int32(f), "T": np.int32(T)}
            d = root / ex / jm / f"fold{f}"
            if not (d / "done.json").exists():
                print(f"{ex} fold {f}: no finished {jm} run, skipped", flush=True)
                continue
            model, c, snap = load_model(cfg, d, jm, f, dev)
            x = data.inputs(items, c["model"]["input"]).permute(0, 3, 1, 2).contiguous()
            S = encode_sequences(model, x, c, dev)
            gap = check_snapshot(S.mean(1), sel, snap, f"{ex} {jm} fold {f}")
            centre = snap["z_train_mean"].astype(np.float64)
            R = {k: v - centre for k, v in ref_sequences(model, packs, c, dev).items()}
            n_steps = S.shape[1]
            path = np.full((len(sel), LMAX, 2), -1, np.int8)
            dist = np.full((len(sel), LMAX), np.nan, np.float32)
            plen = np.zeros(len(sel), np.int16)
            for i in range(len(sel)):
                pth, dd = align_steps(S[i] - centre, R[side[i]])
                plen[i] = len(pth)
                path[i, :len(pth)] = pth
                dist[i, :len(pth)] = dd
            save.update(join_path=path, join_dist=dist, join_len=plen, n_steps=np.int32(n_steps),
                        join_method=np.array(jm), snapshot_gap=np.float32(gap))
            del model
            if a.z:
                for m in Z_METHODS:
                    dm = root / ex / ("b1" if m == INIT else m) / f"fold{f}"
                    model, c, snap = load_model(cfg, dm, m, f, dev)
                    z = forward_all(model, data, items, c, project=False, heads=False)["z"].cpu().numpy()
                    check_snapshot(z.astype(np.float64), sel, snap, f"{ex} {m} fold {f}")
                    save[f"z_{m}"] = z.astype(np.float32)
                    save[f"centre_{m}"] = snap["z_train_mean"].astype(np.float32)
                    del model
            np.savez_compressed(dst, **save)
            print(f"{ex} fold {f}: {len(sel)} reps, mean path distance {np.nanmean(dist):.3f}, |S - snapshot| "
                  f"{gap:.4f} ({time.time() - t0:.0f} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
