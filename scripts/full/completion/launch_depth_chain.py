"""Start depth_chain.py as a hidden, detached job and return at once (run it through run_depth_chain.cmd, started by
Windows Explorer, like scripts/full/loso/launch.py: outside the job object of the Claude app or a terminal).

The chain gets its own hidden console (CREATE_NO_WINDOW): unlike DETACHED_PROCESS, its steps and their worker
processes (depth_extract.py uses a process pool) inherit that hidden console instead of opening console windows, and
no window exists that could be closed by accident (the first launch from a visible console stopped after 3 minutes).
Output is appended to <results>/completion/depth_chain.log. Refuses to start while the chain already runs.

usage: python scripts/full/completion/launch_depth_chain.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.train.run_loso import pid_alive  # noqa: E402

CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW = 0x00000200, 0x08000000


def main():
    out = REPO / "outputs" / "full" / "results" / "completion"
    out.mkdir(parents=True, exist_ok=True)
    status = out / "depth_chain_status.json"
    if status.exists():
        info = json.loads(status.read_text(encoding="utf-8"))
        if not info.get("finished") and pid_alive(info.get("pid")):
            print(f"refused: the chain is already running (pid {info['pid']})")
            return 1
    base = getattr(sys, "_base_executable", sys.executable)
    env = os.environ.copy()
    if os.path.normcase(base) != os.path.normcase(sys.executable):
        env["__PYVENV_LAUNCHER__"] = sys.executable  # what the venv's python.exe redirector does
    cmd = [base, "-u", "scripts/full/completion/depth_chain.py"]
    with open(out / "depth_chain.log", "ab") as f:
        f.write(f"\n##### launch {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n".encode())
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                             creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, close_fds=True)
    print(f"started pid {p.pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
