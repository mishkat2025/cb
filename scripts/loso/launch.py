"""Start the LOSO job (python -m cvdc.train.run_loso) as a hidden, detached process and return at once.

Run it through launch.cmd started by Windows Explorer (``explorer.exe <repo>\\scripts\\loso\\launch.cmd``, or a
double-click), so the job is a descendant of Explorer: outside any job object of a terminal or app (the Claude
desktop app and Windows Terminal are packaged apps whose child processes sit in the app's job; WMI's
Win32_Process.Create puts them under the WMI provider's memory quota, which killed a 3 GB test process after 6 s).
The job has no console (DETACHED_PROCESS), so closing a window cannot stop it; stdout and stderr are appended to
runs/full/<tag>/train.log. It runs the base interpreter with the venv's __PYVENV_LAUNCHER__ (what the venv's
python.exe redirector does), so the job is one process whose pid the runner writes to runs/full/<tag>/RUNNING.json.
Resumable: the runner skips finished runs (done.json). Refuses to start while another cvdc GPU job runs.

usage: python scripts/loso/launch.py [--smoke] [-- extra run_loso arguments]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.train.run_loso import pid_alive  # noqa: E402

DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW = 0x00000008, 0x00000200, 0x08000000


def other_gpu_jobs():
    """Command lines of running python processes that run cvdc training / evaluation (one GPU job at a time)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
          "'cvdc\\.(train|eval)\\.|embedding_distance|group_probe|crossview_distance' } | ForEach-Object { "
          "\"$($_.ProcessId) $($_.CommandLine)\" }")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    return [x for x in r.stdout.splitlines() if x.strip() and str(os.getpid()) != x.split()[0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("extra", nargs="*", help="further arguments for cvdc.train.run_loso (after --)")
    a = ap.parse_args()
    tag = "perex_loso" + ("_smoke" if a.smoke else "")
    root = REPO / "runs" / "full" / tag
    root.mkdir(parents=True, exist_ok=True)
    log = root / "train.log"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def note(msg, to_train_log=False):
        """launch.log always; train.log only before the job starts (the job writes there through its own file
        position, so a line appended by another process while it runs would be overwritten)."""
        for path in [root / "launch.log"] + ([log] if to_train_log else []):
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n##### launch {stamp}: {msg}\n")
        print(msg)

    lock = root / "RUNNING.json"
    if lock.exists():
        info = json.loads(lock.read_text(encoding="utf-8"))
        if pid_alive(info.get("pid")):
            note(f"refused: the LOSO job is already running (pid {info['pid']}, started {info.get('started')})")
            return 1
    busy = other_gpu_jobs()
    if busy:
        note("refused: another cvdc GPU job is running: " + " | ".join(busy))
        return 1
    venv_py = sys.executable  # the venv's python.exe (redirector)
    base = getattr(sys, "_base_executable", sys.executable)
    env = os.environ.copy()
    if os.path.normcase(base) != os.path.normcase(venv_py):
        env["__PYVENV_LAUNCHER__"] = venv_py
    cmd = [base, "-u", "-m", "cvdc.train.run_loso"] + (["--smoke"] if a.smoke else []) + a.extra
    note(" ".join(cmd), to_train_log=True)
    with open(log, "ab") as f:
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                             creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
                             close_fds=True)
    note(f"started pid {p.pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
