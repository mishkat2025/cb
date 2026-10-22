"""Proposal completion (2026-10-22): the DepthAnything chain, one GPU job after the other, stopping at the first failure.

  1. depth_extract.py          DepthAnything V2 Small at the joints of every E01-E04 video (resumable)
  2. depth_pack.py             the B2-DepthAnything input pack (A11)
  3. run_per_exercise          B2-DepthAnything, GroupKFold-5 per exercise (runs/full/perex_gkf5/<EX>/b2_depthanything)
  4. run_loso                  B2-DepthAnything, LOSO per exercise (runs/full/perex_loso/<EX>/b2_depthanything)
Every step is resumable, so a relaunch continues where it stopped. Status: <results>/completion/depth_chain_status.json.
Start it detached (it must survive the Claude session; see scripts/full/loso/README.md):
    explorer.exe D:\\cb\\cvdc-physio\\scripts\\full\\completion\\run_depth_chain.cmd
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXS = ["E01", "E02", "E03", "E04"]
PY = str(REPO.parent / ".venv" / "Scripts" / "python.exe")  # the venv (the chain itself may run the base interpreter)
STEPS = [
    ("extract", [PY, "scripts/full/completion/depth_extract.py", "--workers", "1"]),
    ("pack", [PY, "scripts/full/completion/depth_pack.py"]),
    ("gkf5", [PY, "-m", "cvdc.train.run_per_exercise", "--config", "configs/per_exercise.yaml", "--methods",
              "b2_depthanything", "--exercises", *EXS, "--tag", "perex_gkf5"]),
    ("loso", [PY, "-m", "cvdc.train.run_loso", "--methods", "b2_depthanything", "--exercises", *EXS]),
]


def main():
    status_path = REPO / "outputs" / "full" / "results" / "completion" / "depth_chain_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env.pop("__PYVENV_LAUNCHER__", None)  # the steps start the venv's python.exe themselves
    sys.path.insert(0, str(REPO))
    from cvdc.train.run_loso import keep_awake
    keep_awake(True)  # no sleep while the chain runs (as the LOSO runner)
    for name, cmd in STEPS:
        t0 = time.time()
        status["steps"][name] = {"state": "running", "start": time.strftime("%Y-%m-%d %H:%M:%S")}
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(f"\n===== {name}: {' '.join(cmd)} ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
        code = subprocess.call(cmd, cwd=REPO, env=env)
        status["steps"][name].update(state="done" if code == 0 else f"failed (exit {code})",
                                     hours=round((time.time() - t0) / 3600, 2), end=time.strftime("%Y-%m-%d %H:%M:%S"))
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
        if code != 0:
            print(f"===== {name} failed with exit {code}; chain stopped", flush=True)
            sys.exit(code)
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    print("===== chain finished", flush=True)


if __name__ == "__main__":
    main()
