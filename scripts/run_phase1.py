"""Phase 1 in one command: index -> keypoints -> preprocessing -> sync -> reps -> fixed-T packs -> checks -> report.

usage: python scripts/run_phase1.py [--config configs/data.yaml] [--workers 12] [--from STEP] [--to STEP]
Each step is resumable (cached keypoints and audio lags are skipped). Steps: index extract preprocess sync reps pack
lift pack_lift checks report. ``lift`` / ``pack_lift`` are the KTPFormer 2D -> 3D lifting for B2 (A10; GPU) and its
fixed-length packs; the pilot ran them in Phase 2. The full-scope run (configs/data_full.yaml) adds the checks in
scripts/phase1_full/ (README there).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STEPS = ["index", "extract", "preprocess", "sync", "reps", "pack", "lift", "pack_lift", "checks", "report"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--workers", default="12")
    ap.add_argument("--from", dest="start", choices=STEPS, default="index")
    ap.add_argument("--to", dest="stop", choices=STEPS, default=STEPS[-1])
    a = ap.parse_args()
    cfg, w = ["--config", a.config], ["--workers", a.workers]
    cmds = {
        "index": ["-m", "cvdc.data.index", *cfg],
        "extract": ["-m", "cvdc.data.extract", *cfg, *w],
        "preprocess": ["-m", "cvdc.data.preprocess", *cfg, *w],
        "reps": ["-m", "cvdc.data.reps", *cfg, *w],
        "sync": ["-m", "cvdc.data.sync", *cfg, *w],
        "pack": ["-m", "cvdc.data.pack", *cfg],
        "lift": ["-m", "cvdc.models.lifter", *cfg, "--skip-existing"],
        "pack_lift": ["-m", "cvdc.data.pack", *cfg, "--lift", "ktpformer"],
        "checks": ["-m", "cvdc.viz.checks", *cfg],
        "report": ["scripts/phase1/report.py", *cfg],
    }
    for step in STEPS[STEPS.index(a.start):STEPS.index(a.stop) + 1]:
        t0 = time.time()
        print(f"\n##### {step}", flush=True)
        subprocess.run([sys.executable, *cmds[step]], cwd=REPO, check=True)
        print(f"##### {step} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
