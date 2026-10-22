"""Phase 5 B: CVDC pair-source ablation (C5, E25), pretraining only (no fine-tuning), on the GroupKFold-5 folds of
Phase 3 (same subjects, inner validation splits, seeds, epoch budget, schedule and early stopping as the step (a)
CVDC pretraining, T8a).

Per setting of ``ablation.sources`` and fold: CVDC pretraining with ``cvdc.sources`` = the setting's sources, on the
inner training subjects' rep sets, early-stopped on the inner validation subjects' rep sets of the same sources;
then the embedding snapshots of the random-init and the pretrained encoder (test subjects, every view; centre = the
training fold's mean z), as in cvdc.train.run. Settings in ``ablation.reuse`` (the default 'all' sources) are not
retrained: their step (a) runs are used after checking that the configuration is the same.
Runs: runs/<ablation.runs_tag>/<setting>/fold<k>/ (cvdc/, embeddings_init.npz, embeddings_pretrained.npz,
done.json). Report: scripts/pilot/phase5/report_pair_sources.py.

usage: python scripts/pilot/phase5/pair_sources.py [--settings sync sync_audio ...] [--folds 0 1 2 3 4] [--smoke] [--force]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.dataset import PackData  # noqa: E402
from cvdc.models.model import CVDCModel  # noqa: E402
from cvdc.train.pretrain_cvdc import pretrain_cvdc  # noqa: E402
from cvdc.train.trainer import embedding_snapshot, fold_plan, get_device, seed_everything  # noqa: E402


def setting_config(cfg, sources):
    c = copy.deepcopy(cfg)
    c["model"]["input"] = cfg["methods"][cfg["eval"]["proposed_default"]]["input"]
    c["cvdc"]["sources"] = list(sources)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    ap.add_argument("--settings", nargs="+")
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--smoke", action="store_true", help="fold 0 with the smoke subjects, train.smoke.epochs epochs")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.model_config)
    ab = cfg["ablation"]
    settings = a.settings or [s for s in ab["sources"] if s not in ab.get("reuse", {})]
    dev = get_device(cfg)
    data = PackData(cfg, "rep", dev)
    plans = fold_plan(cfg, data.meta, smoke=a.smoke)
    if a.folds is not None and not a.smoke:
        plans = [p for p in plans if p["fold"] in a.folds]
    root = resolve(cfg["train"]["runs_dir"]) / (ab["runs_tag"] + ("_smoke" if a.smoke else ""))
    epochs = cfg["train"]["smoke"]["epochs"] if a.smoke else None
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    for s in settings:
        if s in ab.get("reuse", {}):
            raise SystemExit(f"setting {s!r} reuses runs/{ab['reuse'][s]}; nothing to train")
        c = setting_config(cfg, ab["sources"][s])
        for p in plans:
            out = root / s / f"fold{p['fold']}"
            if (out / "done.json").exists() and not a.force:
                log(f"skip {s} fold {p['fold']} (done)")
                continue
            out.mkdir(parents=True, exist_ok=True)
            log(f"\n=== {s} {c['cvdc']['sources']} fold {p['fold']} ===")
            t0 = time.time()
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            seed = cfg["train"]["seed"] + p["fold"]
            seed_everything(seed)
            np.savez_compressed(out / "embeddings_init.npz", **embedding_snapshot(CVDCModel(c).to(dev), data, p, c))
            model, summ = pretrain_cvdc(c, data, p, out / "cvdc", seed, epochs, log)
            np.savez_compressed(out / "embeddings_pretrained.npz", **embedding_snapshot(model, data, p, c))
            info = json.loads((out / "cvdc" / "run_info.json").read_text(encoding="utf-8"))
            done = {"setting": s, "sources": c["cvdc"]["sources"], "fold": p["fold"], "seed": seed, "subjects": p,
                    "cvdc_stage": summ, "batcher_train": info["batcher_train"], "batcher_val": info["batcher_val"],
                    "wall_s": time.time() - t0,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None,
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            (out / "done.json").write_text(json.dumps(done, indent=1, default=float), encoding="utf-8")
            log(f"=== {s} fold {p['fold']} done in {done['wall_s'] / 60:.1f} min ===")
            del model


if __name__ == "__main__":
    main()
