"""LOSO per exercise for the final scope E01-E04 (E1, T17, C7; user 2026-10-22).

For every exercise separately (one model per exercise, T17), one fold per subject with >= 1 rep in that exercise: the
subject is the test set, every other subject of the exercise trains, with the T8 inner validation split (20% of each
group's training subjects, seeded per fold) for early stopping. Every other setting is the GroupKFold-5 run's
(``configs/per_exercise.yaml``: video head T13, per-video weights T6, targets standardised on the inner training
subjects T7, T8a / T8b early stopping, no exercise head, E03 P02 kept in score regression D8); only the split changes.

Methods: B1, B2-KTPFormer, Proposed (lambda 1.0, T15) and the frozen-encoder Proposed. **CVDC shortcut (C7):** no
CVDC pretraining here. Each LOSO fold of subject s uses the CVDC encoder that the GroupKFold-5 run pretrained on the
fold where s was held out (runs/full/<eval.loso.encoder_tag>/<EX>/proposed_j1/fold<g>/cvdc/); before training,
every (exercise, subject) pair is checked against that encoder's run_info.json (s in its test fold, not among its
inner training / validation subjects; every subject it saw is a training subject of the LOSO fold) and the check is
written to encoder_map.csv and to each run's done.json (``cvdc_stage.shortcut_check``). A failed check stops the job.

Runs: runs/full/<tag>/<EX>/<method>/fold<k>/ (k = index of the subject in the exercise's sorted subject list;
folds.csv maps folds to subjects). A run is finished when its done.json exists; finished runs are skipped, so the job
resumes where it stopped (a run folder without done.json is a run cut off midway and is deleted and redone).
One GPU job at a time: a lock file (<runs>/<tag>/RUNNING.json) refuses a second instance while the first is alive.
On Windows the process asks the OS not to sleep while it runs (SetThreadExecutionState; released when it exits).

usage: python -m cvdc.train.run_loso [--config configs/per_exercise_loso.yaml] [--exercises E01 ...]
       [--methods b1 b2_ktpformer proposed_j1 proposed_lp] [--tag perex_loso] [--smoke] [--dry-run] [--force]
Detached launch, status and stop: scripts/loso/README.md.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback

import pandas as pd

from ..config import load_config, resolve
from ..data.dataset import PackData
from ..eval.splits import fold_of, subject_table
from .run import check_shortcut_encoder, run_method_fold
from .trainer import fold_plan, get_device, loso_plans


# ---------------------------------------------------------------- process guards (lock, keep awake)
def pid_alive(pid):
    """True if a process with this pid is running (Windows: OpenProcess + GetExitCodeProcess; never os.kill, which
    terminates the process on Windows)."""
    if pid is None:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def acquire_lock(path):
    if path.exists():
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        if pid_alive(info.get("pid")):
            raise SystemExit(f"another LOSO job (pid {info['pid']}, started {info.get('started')}) holds {path}; "
                             "one GPU job at a time")
        print(f"stale lock {path} (pid {info.get('pid')} not running) replaced", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "argv": sys.argv}, indent=1), encoding="utf-8")


def release_lock(path):
    try:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


def keep_awake(on=True):
    """Windows: ask the OS not to sleep while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); the request ends
    with the process. No-op elsewhere."""
    if os.name != "nt":
        return
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


# ---------------------------------------------------------------- plans and the encoder map (C7)
def smoke_subjects(meta, n):
    """Smoke test: per exercise one subject per group (expert, non-expert) where both exist, else the first n."""
    subj = subject_table(meta)
    if subj.group.nunique() > 1:
        pick = [d.subject.iloc[0] for _, d in subj.groupby("group")][:n]
    else:
        pick = list(subj.subject[:n])
    return sorted(pick)


def exercise_loso(cfg, meta_ex, ex, enc_root, enc_method):
    """LOSO plans of exercise ``ex`` with their shortcut encoder (C7) checked. Returns (plans, encoder-map rows)."""
    gkf = fold_plan(cfg, meta_ex)  # the GroupKFold-5 folds of the per-exercise run (same config, seed, inner splits)
    gfold = fold_of([(p["train"], p["test"]) for p in gkf])
    group = subject_table(meta_ex).set_index("subject").group
    plans, rows = [], []
    for p in loso_plans(cfg, meta_ex):
        s = p["subject"]
        g = gfold[s]
        d = enc_root / ex / enc_method / f"fold{g}" / "cvdc"
        info_path = d / "run_info.json"
        if not info_path.exists() or not (d / "best.pt").exists():
            raise SystemExit(f"{ex} {s}: no CVDC encoder in {d} (run the GroupKFold-5 per-exercise job first)")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for k in ("inner_train", "inner_val", "test"):  # the encoder is the recomputed GroupKFold fold
            if sorted(info[k]) != sorted(gkf[g][k]):
                raise SystemExit(f"{ex} {s}: encoder {d} used other {k} subjects than GroupKFold fold {g}")
        p = {**p, "exercise": ex, "cvdc_encoder": str(d), "cvdc_encoder_fold": g}
        chk = check_shortcut_encoder(info, p)  # raises on leakage
        plans.append(p)
        rows.append(dict(exercise=ex, fold=p["fold"], subject=s, group=group[s], gkf_fold=g, encoder_dir=str(d),
                         in_encoder_test=s in info["test"], in_encoder_inner_train=s in info["inner_train"],
                         in_encoder_inner_val=s in info["inner_val"], encoder_subjects_in_loso_train=True,
                         n_train=len(p["train"]), n_inner_train=len(p["inner_train"]),
                         n_inner_val=len(p["inner_val"]), encoder_inner_train=chk["encoder_inner_train"],
                         encoder_inner_val=chk["encoder_inner_val"], check="ok"))
    return plans, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/per_exercise_loso.yaml")
    ap.add_argument("--exercises", nargs="+")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--tag", default="perex_loso")
    ap.add_argument("--smoke", action="store_true", help="2 subjects per exercise, train.smoke.epochs (2) epochs")
    ap.add_argument("--dry-run", action="store_true", help="plans and encoder checks only, no training")
    ap.add_argument("--force", action="store_true", help="rerun runs that have done.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    lc = cfg["eval"]["loso"]
    methods = a.methods or lc["methods"]
    exercises = a.exercises or lc["exercises"]
    unknown = set(methods) - set(cfg["methods"])
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}")
    if "exercise" in cfg["heads"]["targets_rep"]:
        raise SystemExit("per-exercise models have no exercise head (T17)")
    tag = a.tag + ("_smoke" if a.smoke else "")
    root = resolve(cfg["train"]["runs_dir"]) / tag
    lock = root / "RUNNING.json"
    if not a.dry_run:
        acquire_lock(lock)
        keep_awake(True)
    code = 1
    try:
        code = run(cfg, a, methods, exercises, root, tag)
    except SystemExit as e:
        print(f"stopped: {e}", flush=True)
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        if not a.dry_run:
            keep_awake(False)
            release_lock(lock)
        print(f"exit {code} ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    sys.exit(code)


def run(cfg, a, methods, exercises, root, tag):
    lc = cfg["eval"]["loso"]
    lift = sorted({cfg["methods"][m]["input"].split("_", 1)[1] for m in methods if "_" in cfg["methods"][m]["input"]})
    if len(lift) > 1:
        raise SystemExit(f"one lifter per run: {lift}")
    dev = get_device(cfg)
    print(f"=== LOSO per exercise: start {time.strftime('%Y-%m-%d %H:%M:%S')}, pid {os.getpid()}, device {dev}, "
          f"exercises {exercises}, methods {methods}, runs in {root}", flush=True)
    data = PackData(cfg, cfg["train"]["unit"], dev, lift=lift[0] if lift else None)
    enc_root = resolve(cfg["train"]["runs_dir"]) / lc["encoder_tag"]
    root.mkdir(parents=True, exist_ok=True)
    plans_by_ex, rows = {}, []
    for ex in exercises:
        meta_ex = data.meta[data.meta.exercise == ex]
        plans, r = exercise_loso(cfg, meta_ex, ex, enc_root, lc["encoder_method"])
        if a.smoke:
            keep = set(smoke_subjects(meta_ex, lc["smoke_subjects"]))
            plans = [p for p in plans if p["subject"] in keep]
            r = [x for x in r if x["subject"] in keep]
        plans_by_ex[ex] = plans
        rows += r
        n_g = pd.Series([x["group"] for x in r]).value_counts().to_dict()
        print(f"{ex}: {len(plans)} LOSO folds ({n_g}); encoder check passed for every subject (held out in its "
              f"GroupKFold fold, never in the encoder's inner training / validation subjects)", flush=True)
    emap = pd.DataFrame(rows)
    emap.to_csv(root / "encoder_map.csv", index=False)
    emap[["exercise", "fold", "subject", "group", "gkf_fold", "n_train", "n_inner_train", "n_inner_val"]].to_csv(
        root / "folds.csv", index=False)
    todo = [(ex, m, p) for ex in exercises for m in methods for p in plans_by_ex[ex]]
    left = [(ex, m, p) for ex, m, p in todo if a.force or not (root / ex / m / f"fold{p['fold']}" / "done.json").exists()]
    print(f"{len(todo)} runs ({len(todo) - len(left)} finished, {len(left)} to do); encoder map {root / 'encoder_map.csv'}",
          flush=True)
    if a.dry_run:
        return 0
    epochs = cfg["train"]["smoke"]["epochs"] if a.smoke else None
    t0, done_s, n_done = time.time(), {}, 0
    for ex in exercises:
        data.set_scope([ex])
        for method in methods:
            for p in plans_by_ex[ex]:
                out = root / ex / method / f"fold{p['fold']}"
                if (out / "done.json").exists() and not a.force:
                    continue
                if out.exists():  # cut off midway (no done.json) or --force: start the run afresh
                    shutil.rmtree(out)
                print(f"\n=== {ex} {method} fold {p['fold']} ({p['subject']}): {len(p['inner_train'])} inner-train, "
                      f"{len(p['inner_val'])} inner-val subjects; CVDC encoder of GroupKFold fold "
                      f"{p['cvdc_encoder_fold']} ===", flush=True)
                d = run_method_fold(cfg, data, method, p, out, None, epochs, log=lambda *x: print(*x, flush=True))
                n_done += 1
                done_s.setdefault((ex, method), []).append(d["wall_s"])
                el = (time.time() - t0) / 3600
                print(f"=== {ex} {method} fold {p['fold']} ({p['subject']}) done in {d['wall_s'] / 60:.1f} min "
                      f"(peak GPU {d['peak_gpu_gb'] or float('nan'):.2f} GB); {n_done}/{len(left)} runs of this "
                      f"launch, {el:.2f} h since start ===", flush=True)
            print(f"=== {ex} {method} finished ({(time.time() - t0) / 3600:.2f} h since start) ===", flush=True)
        print(f"=== exercise {ex} finished ({(time.time() - t0) / 3600:.2f} h since start) ===", flush=True)
    data.set_scope(None)
    print(f"=== LOSO finished: {n_done} runs in {(time.time() - t0) / 3600:.2f} h ===", flush=True)
    return 0


if __name__ == "__main__":
    main()
