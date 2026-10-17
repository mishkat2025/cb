"""MobiPhysio filename parser and runtime discovery of exercise folders (D1, D2).

A name such as ``E01_P01_AF_VFL_GM`` encodes exercise, participant, angle (view), variation and gender.
Exercise folders (``E01(Abduction)``, ...) are discovered at runtime; folders starting with ``__`` are
download-error markers and are skipped. Missing exercises are simply absent, never an error.
"""
import re
from pathlib import Path

DEFAULT_REGEX = (r"^(?P<exercise>E\d{2})[ _-]*(?P<subject>P\d{1,3})[ _-]*A(?P<view>[FLR])[ _-]*"
                 r"V(?P<variation>[A-Z]+)[ _-]*G(?P<gender>[MF])$")
DEFAULT_VARIATIONS = ("FL", "ML", "LL", "LJ", "HJ", "O", "LR")


def parse_name(name: str, regex: str = DEFAULT_REGEX, variations=DEFAULT_VARIATIONS) -> dict | None:
    """Parse a video name (with or without ``.mp4``). Returns None if it does not match.

    Fields are upper-cased, the subject is zero-padded to two digits (P1 -> P01) and an unknown
    variation code is kept in ``variation_raw`` with ``variation = 'unknown'``.
    """
    stem = re.sub(r"\.mp4$", "", name.strip(), flags=re.IGNORECASE).strip()
    m = re.match(regex, stem, re.IGNORECASE)
    if not m:
        return None
    d = {k: v.upper() for k, v in m.groupdict().items()}
    num = int(d["subject"][1:])
    d["subject"] = f"P{num:02d}"
    d["subject_num"] = num
    d["variation_raw"] = d["variation"]
    if d["variation"] not in variations:
        d["variation"] = "unknown"
    d["video_id"] = f"{d['exercise']}_{d['subject']}_A{d['view']}_V{d['variation_raw']}_G{d['gender']}"
    return d


def discover_exercise_dirs(root, dir_regex=r"^E\d{2}\(", skip_prefix="__", include=None) -> dict:
    """Map exercise code -> folder for every exercise folder under ``root`` (filtered by ``include``)."""
    root = Path(root)
    found = {}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith(skip_prefix) or not re.match(dir_regex, p.name, re.IGNORECASE):
            continue
        found.setdefault(p.name[:3].upper(), p)
    if include:
        include = {c.upper() for c in include}
        found = {c: p for c, p in found.items() if c in include}
    return found


def discover_videos(root, cfg_data: dict):
    """Return (rows, failures, skipped_codes) for all .mp4 files in the included exercise folders."""
    all_dirs = discover_exercise_dirs(root, cfg_data["exercise_dir_regex"], cfg_data["skip_dir_prefix"])
    include = cfg_data.get("include_exercises") or []
    dirs = {c: p for c, p in all_dirs.items() if not include or c in set(include)}
    rows, failures = [], []
    for code, folder in dirs.items():
        for f in sorted(folder.glob("*.mp4")):
            d = parse_name(f.name, cfg_data["filename_regex"], tuple(cfg_data["variations"]))
            if d is None:
                failures.append(str(f))
                continue
            d.update(folder=folder.name, file=f.name, size_mb=round(f.stat().st_size / 1e6, 2))
            rows.append(d)
    skipped = sorted(set(all_dirs) - set(dirs))
    return rows, failures, skipped
