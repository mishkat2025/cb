"""The embedding-to-reference distance (the proposal's "vector difference", E5) for the per-exercise models (T17).

For every exercise, method and GroupKFold-5 fold of ``runs/full/<tag>/<EX>/<method>/fold<k>/`` (finetune/best.pt):
every rep of that exercise is encoded (plain packs, eval mode, bf16 as in training); the exercise's primary PT demo's
reps are packed like the patients' reps (coco17, T_rep frames, the method's input: 2D xy or KTPFormer lifted 3D) and
encoded, reference embedding = their mean, mirrored (x -> -x, left / right swapped) for reps of the other side than
the demo's (E5, P8); per rep the centred cosine distance 1 - cos(z - c, z_ref - c) (c = the training fold's mean z,
E13) and the L2 distance. Methods: B1, B2-KTPFormer, Proposed (lambda 1.0), the frozen CVDC encoder and, as a floor,
the random-init encoder of each fold (xy, the fold's seed). Embeddings are checked against the training-time snapshot.

usage: python scripts/full/phase3/embedding_distance_perex.py [--config configs/per_exercise.yaml] [--tag perex_gkf5]
       [--device cpu] [--exercises E01 ...] [--loso]
--loso: the LOSO runs of cvdc.train.run_loso (one fold per subject, cvdc.train.trainer.loso_plans), e.g.
--config configs/per_exercise_loso.yaml --tag perex_loso --exercises E01 E02 E03 E04 --loso.
Writes <eval.results_dir>/<tag>/embedding_distance_reps.csv (exercise, method, fold, video_id, rep, side, d_cos,
d_l2, is_test).
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
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase4"))
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.data.preprocess import resample_fixed  # noqa: E402
from cvdc.data.skeleton import JOINT_SETS  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402
from cvdc.train.run import method_config  # noqa: E402
from cvdc.train.trainer import fold_plan, forward_all, get_device, items_of, loso_plans, seed_everything  # noqa: E402
from embedding_distance import distances, encode_ref  # noqa: E402

INIT = "init_xy"
METHODS = ["b1", "b2_ktpformer", "proposed_j1", "proposed_lp", INIT]


def ref_pack(cfg, ex):
    """(dict(xy, vis, world, lifted) of the primary PT demo's reps, the demo's side)."""
    pc = cfg["preprocess"]
    T, joints = pc["T_rep"], JOINT_SETS[pc["joint_set"]]
    ref_idx = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    rr = pd.read_csv(data_path(cfg, "reps_ref_csv"))
    row = ref_idx[(ref_idx.exercise == ex) & (ref_idx.role == "primary")].iloc[0]
    with np.load(resolve(row.pre_path)) as z:
        d = {k: z[k][:, joints] for k in ("xy", "vis", "world")}
    lp = resolve(cfg["lifter"]["out_dir"]) / f"{row.ref_id}.npz"
    with np.load(lp) as z:
        d["lifted"] = z["world"]
    reps = rr[(rr.ref_id == row.ref_id) & (rr.exercise == ex)]
    if reps.side.nunique() != 1:
        raise RuntimeError(f"{row.file}: demo reps on both sides")
    pack = {k: np.asarray([resample_fixed(v[r.start:r.end + 1], T) for r in reps.itertuples()], np.float32)
            for k, v in d.items()}
    return pack, reps.side.iloc[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--tag", default="perex_gkf5")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--exercises", nargs="+")
    ap.add_argument("--device", help="override train.device")
    ap.add_argument("--loso", action="store_true", help="LOSO runs (one fold per subject) instead of GroupKFold-5")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    if a.device:
        cfg["train"]["device"] = a.device
    dev = get_device(cfg)
    data = PackData(cfg, "rep", dev, lift="ktpformer")
    meta = data.meta
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out = resolve(cfg["eval"]["results_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    smoke = a.tag.endswith("_smoke")
    rows = []
    for ex in a.exercises or sorted(meta.exercise.unique()):
        data.set_scope([ex])
        plans = (loso_plans(cfg, meta[meta.exercise == ex]) if a.loso
                 else fold_plan(cfg, meta[meta.exercise == ex], smoke=smoke))
        pack, demo_side = ref_pack(cfg, ex)
        for m in a.methods:
            c = method_config(cfg, "b1" if m == INIT else m)
            mode = c["model"]["input"]
            for p in plans:
                f = p["fold"]
                d = root / ex / ("b1" if m == INIT else m) / f"fold{f}"
                if not (d / "done.json").exists():
                    continue
                if m == INIT:  # as cvdc.train.run builds the init snapshot of the fold's seed
                    seed_everything(cfg["train"]["seed"] + f)
                    model = CVDCModel(c).to(dev)
                    snap = np.load(d / "embeddings_init.npz")
                else:
                    ck = torch.load(d / "finetune" / "best.pt", map_location=dev, weights_only=True)
                    model = CVDCModel(c).to(dev)
                    model.load_state_dict(ck["model"])
                    snap = np.load(d / "embeddings_final.npz")
                items = items_of(data, p["train"] + p["test"])
                z = forward_all(model, data, items, c, project=False, heads=False)["z"].cpu().numpy().astype(np.float64)
                it = items.cpu().numpy()
                pos = {x: i for i, x in enumerate(it)}
                gap = np.abs(z[[pos[x] for x in snap["item"]]] - snap["z"]).max()
                if gap > 0.05 * max(1.0, np.abs(snap["z"]).max()):
                    raise RuntimeError(f"{ex} {m} fold {f}: embeddings differ from the snapshot by {gap:.3f}")
                centre = snap["z_train_mean"].astype(np.float64)
                mm = meta.iloc[it]
                dc, dl = np.full(len(z), np.nan), np.full(len(z), np.nan)
                for side in ("right", "left"):
                    zr = encode_ref(model, pack, mode, c, dev, mirrored=side != demo_side)
                    s = (mm.side == side).to_numpy()
                    dc[s], dl[s] = distances(z[s], zr, centre)
                rows.append(pd.DataFrame({"exercise": ex, "method": m, "fold": f, "video_id": mm.video_id.to_numpy(),
                                          "rep": mm.rep.to_numpy(), "side": mm.side.to_numpy(), "d_cos": dc,
                                          "d_l2": dl, "is_test": mm.subject.isin(set(p["test"])).to_numpy()}))
                print(f"{ex} {m} fold {f}: |z - snapshot| {gap:.4f} ({time.time() - t0:.0f} s)", flush=True)
                del model
    data.set_scope(None)
    pd.concat(rows, ignore_index=True).to_csv(out / "embedding_distance_reps.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
