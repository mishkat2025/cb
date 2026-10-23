"""YAML config loading. Paths in configs are relative to the repo root unless absolute.

A config may start with ``extends: <base.yaml>`` (relative to its own folder): the base is loaded first and the
config's keys are deep-merged over it (dicts merge, everything else replaces). The model config is merged under
the data config chain, so a derived data config (e.g. ``configs/data_full.yaml``, D15) can also override model keys
such as output folders.
"""
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "data.yaml"
DEFAULT_MODEL_CONFIG = REPO / "configs" / "model.yaml"
# data-file locations that used to be hard-coded; a config can move them (D15)
DATA_PATH_DEFAULTS = {"reps_csv": "data/reps.csv", "reps_perview_csv": "data/reps_perview.csv",
                      "reps_ref_csv": "data/reps_ref.csv", "set_features_csv": "data/cache/set_features.csv"}


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_chain(path: Path) -> dict:
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = d.pop("extends", None)
    return deep_merge(_load_chain((path.parent / base).resolve()), d) if base else d


def load_config(path=None, model_path=None) -> dict:
    """Load the main config (following ``extends``), merge the model config (top-level keys model, heads, cvdc,
    supervised, train, ...) under it and attach every per-exercise YAML under cfg['exercises'][code]."""
    path = Path(path or DEFAULT_CONFIG)
    data = _load_chain(path)
    mpath = Path(model_path) if model_path else DEFAULT_MODEL_CONFIG
    cfg = {}
    if mpath.exists():
        cfg = yaml.safe_load(mpath.read_text(encoding="utf-8"))
        cfg["_model_path"] = str(mpath)
    cfg = deep_merge(cfg, data)
    ex_dir = resolve(cfg["data"]["exercise_config_dir"])
    cfg["exercises"] = {p.stem.upper(): yaml.safe_load(p.read_text(encoding="utf-8"))
                        for p in sorted(ex_dir.glob("E*.yaml"))}
    cfg["_path"] = str(path)
    return cfg


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p).resolve()


def _dataset_dir(configured: Path, inside: Path) -> Path:
    """The configured dataset folder, or the same folder inside the repo (``dataset/``) when only that one exists."""
    return inside if not configured.exists() and inside.exists() else configured


def data_root(cfg) -> Path:
    """Folder with the scores CSV, the dataset README and the PT reference folder (D1)."""
    env = os.environ.get("MOBIPHYSIO_ROOT")
    return Path(env).resolve() if env else _dataset_dir(resolve(cfg["data"]["root"]), REPO / "dataset")


def video_root(cfg) -> Path:
    """Folder with the exercise folders E0x(...) (D1): ``data.video_root``, default = the data root."""
    env = os.environ.get("MOBIPHYSIO_VIDEO_ROOT")
    if env:
        return Path(env).resolve()
    vr = cfg["data"].get("video_root")
    return _dataset_dir(resolve(vr), REPO / "dataset" / "videos") if vr else data_root(cfg)


def data_path(cfg, key) -> Path:
    """A data file location from ``data.<key>`` (reps tables etc.), with the pilot's location as default."""
    return resolve(cfg["data"].get(key) or DATA_PATH_DEFAULTS[key])


def exercise_codes(cfg) -> list:
    """All configured exercise codes (E01..E09), the class list of the exercise head (9 classes)."""
    return sorted(cfg["exercises"])


def exercise_cfg(cfg, code) -> dict:
    """Per-exercise config with the global rep defaults filled in."""
    ex = dict(cfg["exercises"].get(code) or {})
    ex["reps"] = {**cfg["reps"], **(ex.get("reps") or {})}
    return ex
