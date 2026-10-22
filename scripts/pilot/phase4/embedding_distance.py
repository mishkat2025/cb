"""Phase 4 item 5: the embedding-to-reference distance (the proposal's "vector difference", E5) for B1, B2 and Proposed.

For every method and GroupKFold-5 fold of Phase 3 step (a2) (runs/phase3a2/<method>/fold<k>/finetune/best.pt):
  - every rep of every subject is encoded (plain packs, eval mode, bf16 as in training): z (256-d);
  - the PT demo's reps of each exercise (the primary demo, M4) are packed exactly like the patients' reps (coco17, T_rep
    frames, 2D xy / MediaPipe world / KTPFormer lifted 3D by the method's input) and encoded; reference embedding =
    the mean of its rep embeddings (E5). For left-arm reps the demo is mirrored (x -> -x, left and right joints
    swapped) before encoding, so a left-arm patient is compared with a left-arm reference (P8);
  - per rep: cosine distance of the centred embeddings, 1 - cos(z - c, z_ref - c) with c = the training fold's mean z
    (E13: the centred 256-d embedding is what the distance uses), and the L2 distance ||z - z_ref||.
Methods: B1, B2-MediaPipe, B2-KTPFormer, Proposed (joint loss, lambda 1.0: ``eval.proposed_default``), the frozen CVDC
encoder (``eval.proposed_alternative``) and, as a floor, the random-init encoder of each fold (xy, the fold's seed).

usage: python scripts/pilot/phase4/embedding_distance.py [--tag phase3a2]
Writes outputs/phase4/embedding_distance_reps.csv (method, fold, video_id, rep, d_cos, d_l2, is_test) and
outputs/phase4/embedding_reference.npz (reference embeddings per method, fold, exercise and side).
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
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.dataset import PackData, to_input  # noqa: E402
from cvdc.data.preprocess import resample_fixed  # noqa: E402
from cvdc.data.skeleton import COCO17_MIRROR, JOINT_SETS, mirror_coco17  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402
from cvdc.train.run import method_config  # noqa: E402
from cvdc.train.trainer import autocast, fold_plan, forward_all, get_device, seed_everything  # noqa: E402

INIT = "init_xy"


def ref_packs(cfg):
    """{exercise: dict(xy, vis, world, lifted) (n_reps, T, 17, C)} of the primary PT demo's reps."""
    pc = cfg["preprocess"]
    T, joints = pc["T_rep"], JOINT_SETS[pc["joint_set"]]
    ref_idx = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    rr = pd.read_csv(resolve("data/reps_ref.csv"))
    out = {}
    for ex in cfg["data"]["include_exercises"]:
        row = ref_idx[(ref_idx.exercise == ex) & (ref_idx.role == "primary")].iloc[0]
        with np.load(resolve(row.pre_path)) as z:
            d = {k: z[k][:, joints] for k in ("xy", "vis", "world")}
        with np.load(resolve(cfg["lifter"]["out_dir"]) / f"{row.ref_id}.npz") as z:
            d["lifted"] = z["world"]
        reps = rr[(rr.ref_id == row.ref_id) & (rr.exercise == ex)]
        if (reps.side != "right").any():
            raise RuntimeError(f"{row.file}: the demo is expected to use the right arm")
        out[ex] = {k: np.asarray([resample_fixed(v[r.start:r.end + 1], T) for r in reps.itertuples()], np.float32)
                   for k, v in d.items()}
    return out


@torch.no_grad()
def encode_ref(model, pack, mode, cfg, dev, mirrored):
    def prep(k, v):
        if not mirrored:
            return v
        return np.ascontiguousarray(v[:, :, COCO17_MIRROR]) if k == "vis" else mirror_coco17(v)

    t = {k: torch.from_numpy(prep(k, v)).to(dev) for k, v in pack.items()}
    x = to_input(t["xy"], t["vis"], t["world"], mode, t["lifted"]).permute(0, 3, 1, 2).contiguous()
    model.eval()
    with autocast(cfg, dev):
        z = model(x, project=False, heads=False)["z"].float()
    return z.mean(0).cpu().numpy().astype(np.float64)


def distances(z, zref, c):
    a, b = z - c, zref - c
    cos = (a @ b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b), 1e-12)
    return 1.0 - cos, np.linalg.norm(z - zref, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="phase3a2")
    ap.add_argument("--methods", nargs="+",
                    default=["b1", "b2_mediapipe", "b2_ktpformer", "proposed_j1", "proposed_lp", INIT])
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    dev = get_device(cfg)
    data = PackData(cfg, "rep", dev, lift="ktpformer")
    meta = data.meta
    plans = fold_plan(cfg, pd.read_csv(resolve("data/reps.csv")))
    refs = ref_packs(cfg)
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out = resolve("outputs/phase4")
    out.mkdir(parents=True, exist_ok=True)
    all_items = torch.arange(len(meta), device=dev)
    left = (meta.side == "left").to_numpy()
    rows, ref_store = [], {}
    for m in a.methods:
        c = method_config(cfg, "b1" if m == INIT else m)
        mode = c["model"]["input"]
        for p in plans:
            f = p["fold"]
            if m == INIT:  # exactly as cvdc.train.run builds the init snapshot of the fold's seed
                seed_everything(cfg["train"]["seed"] + f)
                model = CVDCModel(c).to(dev)
                snap = np.load(root / "b1" / f"fold{f}" / "embeddings_init.npz")
            else:
                d = root / m / f"fold{f}"
                ck = torch.load(d / "finetune" / "best.pt", map_location=dev, weights_only=True)
                if ck["input"] != mode:
                    raise RuntimeError(f"{d}: input {ck['input']} vs {mode}")
                model = CVDCModel(c).to(dev)
                model.load_state_dict(ck["model"])
                snap = np.load(d / "embeddings_final.npz")
            z = forward_all(model, data, all_items, c, project=False, heads=False)["z"].cpu().numpy().astype(np.float64)
            te = snap["item"]
            gap = np.abs(z[te] - snap["z"]).max()  # same model and inputs as the Phase 3 snapshot
            if gap > 0.05 * max(1.0, np.abs(snap["z"]).max()):
                raise RuntimeError(f"{m} fold {f}: embeddings differ from the Phase 3 snapshot by {gap:.3f}")
            centre = snap["z_train_mean"].astype(np.float64)
            dc, dl = np.full(len(z), np.nan), np.full(len(z), np.nan)
            for ex, pack in refs.items():
                sel = (meta.exercise == ex).to_numpy()
                for side, mirrored in (("right", False), ("left", True)):
                    zr = encode_ref(model, pack, mode, c, dev, mirrored)
                    ref_store[f"{m}__fold{f}__{ex}__{side}"] = zr
                    s = sel & (left if side == "left" else ~left)
                    dc[s], dl[s] = distances(z[s], zr, centre)
            test = meta.subject.isin(set(p["test"])).to_numpy()
            rows.append(pd.DataFrame({"method": m, "fold": f, "video_id": meta.video_id, "rep": meta.rep,
                                      "d_cos": dc, "d_l2": dl, "is_test": test}))
            print(f"{m} fold {f}: max |z - snapshot| {gap:.4f}; test d_cos median {np.median(dc[test]):.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            del model
    pd.concat(rows, ignore_index=True).to_csv(out / "embedding_distance_reps.csv", index=False)
    np.savez_compressed(out / "embedding_reference.npz", **ref_store)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
