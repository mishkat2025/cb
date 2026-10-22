"""Cross-view protocol per exercise (T17, D18; twelfth review item 6): the embedding distance to the PT reference (E5)
under the two-view models of runs/full/<tag>/<EX>/heldout_<V>/<method>/fold<k>/ (cvdc.eval.cross_view).

Per exercise, held-out view, method and fold: the exercise's reps of the test subjects (every view) and of the training
subjects' training views are encoded with the run's final checkpoint, the exercise's PT demo reps are encoded the same
way (mirrored for reps of the other side than the demo's), and the centred cosine distance (c = the run's training-view
mean z) and the L2 distance are stored per rep.

usage: python scripts/full/phase3/crossview_distance_perex.py [--config configs/per_exercise.yaml] [--tag perex_crossview]
       [--exercises E01 E02 E03 E04] [--device cpu]
Writes <eval.results_dir>/<tag>/embedding_distance_reps.csv (exercise, heldout, method, fold, video_id, rep, side, d_cos,
d_l2, is_test, train_view).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pilot" / "phase4"))
sys.path.insert(0, str(REPO / "scripts" / "full" / "phase3"))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402
from cvdc.train.run import method_config  # noqa: E402
from cvdc.train.trainer import forward_all, get_device, items_of  # noqa: E402
from embedding_distance import distances, encode_ref  # noqa: E402
from embedding_distance_perex import ref_pack  # noqa: E402

FINAL = ["E01", "E02", "E03", "E04"]  # D18


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise.yaml")
    ap.add_argument("--tag", default="perex_crossview")
    ap.add_argument("--exercises", nargs="+", default=FINAL)
    ap.add_argument("--device", help="override train.device")
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
    rows = []
    for ex in a.exercises:
        data.set_scope([ex])
        pack, demo_side = ref_pack(cfg, ex)
        for vdir in sorted((root / ex).glob("heldout_*")):
            held = vdir.name.split("_", 1)[1]
            for mdir in sorted(p for p in vdir.iterdir() if p.is_dir()):
                m = mdir.name
                c = method_config(cfg, m)
                mode = c["model"]["input"]
                for fd in sorted(mdir.glob("fold*")):
                    if not (fd / "done.json").exists():
                        raise RuntimeError(f"{fd}: run not finished")
                    info = json.loads((fd / "done.json").read_text(encoding="utf-8"))["subjects"]
                    ck = torch.load(fd / "finetune" / "best.pt", map_location=dev, weights_only=True)
                    model = CVDCModel(c).to(dev)
                    model.load_state_dict(ck["model"])
                    snap = np.load(fd / "embeddings_final.npz")
                    items = torch.cat([items_of(data, info["test"]), items_of(data, info["train"], info["views"])])
                    z = forward_all(model, data, items, c, project=False, heads=False)["z"].cpu().numpy().astype(np.float64)
                    it = items.cpu().numpy()
                    pos = {x: i for i, x in enumerate(it)}
                    gap = np.abs(z[[pos[x] for x in snap["item"]]] - snap["z"]).max()
                    if gap > 0.05 * max(1.0, np.abs(snap["z"]).max()):
                        raise RuntimeError(f"{fd}: embeddings differ from the run's snapshot by {gap:.3f}")
                    centre = snap["z_train_mean"].astype(np.float64)
                    mm = meta.iloc[it]
                    dc, dl = np.full(len(z), np.nan), np.full(len(z), np.nan)
                    for side in ("right", "left"):
                        zr = encode_ref(model, pack, mode, c, dev, mirrored=side != demo_side)
                        s = (mm.side == side).to_numpy()
                        dc[s], dl[s] = distances(z[s], zr, centre)
                    test = mm.subject.isin(set(info["test"])).to_numpy()
                    rows.append(pd.DataFrame({"exercise": ex, "heldout": held, "method": m, "fold": int(fd.name[4:]),
                                              "video_id": mm.video_id.to_numpy(), "rep": mm.rep.to_numpy(),
                                              "side": mm.side.to_numpy(), "d_cos": dc, "d_l2": dl, "is_test": test,
                                              "train_view": ~test}))
                    print(f"{ex} held-out {held} {m} {fd.name}: |z - snapshot| {gap:.4f} ({time.time() - t0:.0f} s)",
                          flush=True)
                    del model
    data.set_scope(None)
    pd.concat(rows, ignore_index=True).to_csv(out / "embedding_distance_reps.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
