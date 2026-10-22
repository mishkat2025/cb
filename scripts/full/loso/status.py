"""Status of the LOSO job (cvdc.train.run_loso): is it running, how many runs are finished per exercise and method, the
estimated time left, and the end of the log. Reads only files; safe to run at any time.

usage (from cvdc-physio/): python scripts/full/loso/status.py [--smoke] [--lines 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.train.run_loso import pid_alive  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise_loso.yaml")
    ap.add_argument("--tag", default="perex_loso")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lines", type=int, default=8)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(REPO / a.config)
    tag = a.tag + ("_smoke" if a.smoke else "")
    root = resolve(cfg["train"]["runs_dir"]) / tag
    methods = cfg["eval"]["loso"]["methods"]
    log = root / "train.log"
    lock = root / "RUNNING.json"
    print(f"LOSO job {tag} ({root}), checked {time.strftime('%Y-%m-%d %H:%M:%S')}")
    state = "not running"
    if lock.exists():
        info = json.loads(lock.read_text(encoding="utf-8"))
        state = (f"RUNNING (pid {info['pid']}, started {info['started']})" if pid_alive(info["pid"])
                 else f"NOT RUNNING: stale lock of pid {info['pid']} (crashed or killed; relaunch to resume)")
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    exits = [x for x in tail if x.startswith("exit ")]
    if state == "not running" and exits:
        state = f"not running; last launch ended with '{exits[-1]}'"
    print(f"state: {state}")
    folds = root / "folds.csv"
    if not folds.exists():
        print("no folds.csv yet (the job has not built its plans)")
        return
    F = pd.read_csv(folds)
    est_path = resolve(cfg["eval"]["results_dir"]) / "perex_loso_estimate" / "runtime_estimate.csv"
    est = pd.read_csv(est_path).set_index(["exercise", "method"]) if est_path.exists() else None
    rows, left_s, unknown = [], 0.0, False
    for ex, fe in F.groupby("exercise"):
        for m in methods:
            walls = []
            for f in fe.fold:
                d = root / ex / m / f"fold{f}" / "done.json"
                if d.exists():
                    walls.append(json.loads(d.read_text(encoding="utf-8"))["wall_s"])
            n_left = len(fe) - len(walls)
            per = (sum(walls) / len(walls)) if walls else (est.loc[(ex, m), "per_run_s"] if est is not None and
                                                              (ex, m) in est.index else None)
            if n_left:
                if per is None:
                    unknown = True
                else:
                    left_s += n_left * per
            rows.append(dict(exercise=ex, method=m, done=len(walls), of=len(fe),
                             mean_min=round(sum(walls) / len(walls) / 60, 2) if walls else None))
    T = pd.DataFrame(rows)
    piv = T.pivot(index="exercise", columns="method", values="done").reindex(columns=methods)
    piv = piv.astype(str) + "/" + T.pivot(index="exercise", columns="method", values="of").reindex(columns=methods).astype(str)
    print(f"finished runs: {int(T.done.sum())} / {int(T.of.sum())}")
    print(piv.to_string())
    print(f"estimated time left: {left_s / 3600:.1f} h" + (" (+ runs without an estimate)" if unknown else "") +
          ("" if est is not None else " (from finished runs only; no estimate file)"))
    if tail:
        print(f"--- last {a.lines} log lines ({log}) ---")
        print("\n".join(tail[-a.lines:]))


if __name__ == "__main__":
    main()
