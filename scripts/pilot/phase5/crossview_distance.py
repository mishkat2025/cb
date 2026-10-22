"""Phase 5 A (E24): the embedding distance to the PT reference (E5) under the cross-view models.

For every held-out view V, method and fold of runs/<tag>/heldout_<V>/<method>/fold<k>/ (cvdc.eval.cross_view):
every rep is encoded with the fold's final checkpoint (finetune/best.pt), the PT demo's reps of each exercise are
encoded the same way (mirrored for left-arm reps; scripts/pilot/phase4/embedding_distance.py), and per rep the centred
cosine distance 1 - cos(z - c, z_ref - c) and the L2 distance ||z - z_ref|| are stored, with c = the training fold's
mean z over its **training views** (the run's embedding snapshot). The PT demo is a front-view clip, so for V = F the
reference itself comes from the view the model never saw.

usage: python scripts/pilot/phase5/crossview_distance.py [--tag phase5_crossview] [--methods ...]
Writes outputs/phase5/crossview/embedding_distance_reps.csv (heldout, method, fold, video_id, rep, d_cos, d_l2,
is_test, train_view): rows of the test subjects (every view) and of the training subjects' training views.
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
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.eval.cross_view import main_methods  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402
from cvdc.train.run import method_config  # noqa: E402
from cvdc.train.trainer import forward_all, get_device  # noqa: E402
from embedding_distance import distances, encode_ref, ref_packs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="phase5_crossview")
    ap.add_argument("--methods", nargs="+")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    dev = get_device(cfg)
    data = PackData(cfg, "rep", dev, lift="ktpformer")
    meta = data.meta
    refs = ref_packs(cfg)
    root = resolve(cfg["train"]["runs_dir"]) / a.tag
    out = resolve("outputs/phase5/crossview")
    out.mkdir(parents=True, exist_ok=True)
    all_items = torch.arange(len(meta), device=dev)
    left = (meta.side == "left").to_numpy()
    rows = []
    for vdir in sorted(root.glob("heldout_*")):
        held = vdir.name.split("_", 1)[1]
        for m in a.methods or main_methods(cfg):
            c = method_config(cfg, m)
            mode = c["model"]["input"]
            for fd in sorted((vdir / m).glob("fold*")):
                if not (fd / "done.json").exists():
                    raise RuntimeError(f"{fd}: run not finished")
                f = int(fd.name[4:])
                ck = torch.load(fd / "finetune" / "best.pt", map_location=dev, weights_only=True)
                if ck["input"] != mode:
                    raise RuntimeError(f"{fd}: input {ck['input']} vs {mode}")
                model = CVDCModel(c).to(dev)
                model.load_state_dict(ck["model"])
                snap = np.load(fd / "embeddings_final.npz")
                z = forward_all(model, data, all_items, c, project=False, heads=False)["z"].cpu().numpy().astype(np.float64)
                gap = np.abs(z[snap["item"]] - snap["z"]).max()
                if gap > 0.05 * max(1.0, np.abs(snap["z"]).max()):
                    raise RuntimeError(f"{fd}: embeddings differ from the run's snapshot by {gap:.3f}")
                centre = snap["z_train_mean"].astype(np.float64)
                dc, dl = np.full(len(z), np.nan), np.full(len(z), np.nan)
                for ex, pack in refs.items():
                    sel = (meta.exercise == ex).to_numpy()
                    for side, mirrored in (("right", False), ("left", True)):
                        zr = encode_ref(model, pack, mode, c, dev, mirrored)
                        s = sel & (left if side == "left" else ~left)
                        dc[s], dl[s] = distances(z[s], zr, centre)
                info = pd.read_json(fd / "done.json", typ="series")["subjects"]
                test = meta.subject.isin(set(info["test"])).to_numpy()
                train_view = meta.subject.isin(set(info["train"])).to_numpy() & meta.view.isin(set(info["views"])).to_numpy()
                keep = test | train_view
                rows.append(pd.DataFrame({"heldout": held, "method": m, "fold": f, "video_id": meta.video_id[keep],
                                          "rep": meta.rep[keep], "d_cos": dc[keep], "d_l2": dl[keep],
                                          "is_test": test[keep], "train_view": train_view[keep]}))
                print(f"held-out {held} {m} fold {f}: max |z - snapshot| {gap:.4f}; test d_cos median "
                      f"{np.median(dc[test]):.3f} ({time.time() - t0:.0f} s)", flush=True)
                del model
    pd.concat(rows, ignore_index=True).to_csv(out / "embedding_distance_reps.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
