# LOSO per exercise, final scope E01–E04

Leave-one-subject-out within each exercise (one model per exercise, T17), methods B1, B2-KTPFormer, Proposed (λ 1.0)
and the frozen Proposed, every setting as the GroupKFold-5 runs; Proposed reuses the CVDC encoder of the GroupKFold-5
fold that held the subject out (C7, checked per subject). Decisions: `DESIGN_DECISIONS.md` E1, C7, E30.
147 folds (E01 50, E02 24, E03 49, E04 24) × 4 methods = 588 runs into `runs/full/perex_loso/<EX>/<method>/fold<k>/`
(`folds.csv` maps folds to subjects, `encoder_map.csv` holds the encoder check).

All commands from `D:\cb\cvdc-physio`; Python = `D:\cb\.venv\Scripts\python.exe`.

| Step | Command |
|---|---|
| plans + encoder check only | `python -m cvdc.train.run_loso --dry-run` |
| smoke test (2 subjects per exercise, 2 epochs), detached | `explorer.exe D:\cb\cvdc-physio\scripts\full\loso\launch_smoke.cmd` |
| runtime estimate (after the smoke test) | `python scripts\full\loso\runtime_estimate.py` |
| **start / resume the full run, detached** | `explorer.exe D:\cb\cvdc-physio\scripts\full\loso\launch.cmd` (or double-click it) |
| **status** (running?, runs done per exercise × method, time left, log tail) | `D:\cb\.venv\Scripts\python.exe D:\cb\cvdc-physio\scripts\full\loso\status.py` |
| stop | `powershell -NoProfile -ExecutionPolicy Bypass -File D:\cb\cvdc-physio\scripts\full\loso\stop.ps1` |
| distance to the PT reference (after the run; GPU, ~10 min) | `python scripts\full\phase3\embedding_distance_perex.py --config configs\per_exercise_loso.yaml --tag perex_loso --exercises E01 E02 E03 E04 --loso` |
| report | `python scripts\full\loso\report_loso.py` → `outputs\full\results\perex_loso\summary.md` |

**Why Explorer:** the launch goes through `explorer.exe` so the job is a descendant of the Windows shell, outside any
job object. Started from the Claude desktop app or Windows Terminal (packaged apps), a process sits in the app's job
and can be ended with the app (or an app update); `Win32_Process.Create` (WMI) puts it under the WMI provider's memory
quota, which killed a 3 GB test process after 6 s. `launch.py` then starts the job with no console (closing a window
cannot stop it), appends its output to `runs\full\perex_loso\train.log`, and the job keeps the PC from sleeping while
it runs (released when it exits). A reboot or log-off stops it; launch again to resume.

**Resume:** finished runs have `done.json` and are skipped; a run cut off midway (folder without `done.json`) is
deleted and redone. The job refuses to start while it is already running (lock `runs\full\perex_loso\RUNNING.json`)
or while another cvdc GPU job runs.
