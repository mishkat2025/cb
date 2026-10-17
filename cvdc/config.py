"""YAML config loading. Paths in configs are relative to the repo root unless absolute."""
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "data.yaml"
DEFAULT_MODEL_CONFIG = REPO / "configs" / "model.yaml"


def load_config(path=None, model_path=None) -> dict:
    """Load the main config, merge the model config (top-level keys model, heads, cvdc, supervised, train)
    and attach every per-exercise YAML under cfg['exercises'][code]."""
    path = Path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    mpath = Path(model_path) if model_path else DEFAULT_MODEL_CONFIG
    if mpath.exists():
        cfg.update(yaml.safe_load(mpath.read_text(encoding="utf-8")))
        cfg["_model_path"] = str(mpath)
    ex_dir = resolve(cfg["data"]["exercise_config_dir"])
    cfg["exercises"] = {p.stem.upper(): yaml.safe_load(p.read_text(encoding="utf-8"))
                        for p in sorted(ex_dir.glob("E*.yaml"))}
    cfg["_path"] = str(path)
    return cfg


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p).resolve()


def data_root(cfg) -> Path:
    env = os.environ.get("MOBIPHYSIO_ROOT")
    return Path(env).resolve() if env else resolve(cfg["data"]["root"])


def exercise_codes(cfg) -> list:
    """All configured exercise codes (E01..E09), the class list of the exercise head (9 classes)."""
    return sorted(cfg["exercises"])


def exercise_cfg(cfg, code) -> dict:
    """Per-exercise config with the global rep defaults filled in."""
    ex = dict(cfg["exercises"].get(code) or {})
    ex["reps"] = {**cfg["reps"], **(ex.get("reps") or {})}
    return ex
