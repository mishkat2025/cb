# CVDC physiotherapy assessment on MobiPhysio

Implementation of the proposal "AI-Guided Physiotherapy Assessment for Low-Resource Telehealth through
Cross-View Depth-Consistency Learning" on the MobiPhysio dataset (doi 10.7910/DVN/XSI0QN).

- **Scope now:** E01 (Abduction) and E02 (Adduction) only (`data.include_exercises`). Exercises are
  discovered at runtime, so adding one later is a config change.
- **Status:** Step 0 and Phase 1 (keypoints and preprocessing) are done. See `PROGRESS.md` for the state and
  the phase checklist, and `DESIGN_DECISIONS.md` for every open choice.
- **Privacy:** the videos show faces. Only keypoints are stored; renders are skeleton-only; `data/` and
  `outputs/` are gitignored. Audio is decoded in memory for synchronization and never written; GPS tags in
  the video containers are never read.

## Setup

```bash
# Python 3.11 venv one level above the repo (D:\cb\.venv), created with uv
uv venv ../.venv --python 3.11
uv pip install --python ../.venv/Scripts/python.exe -r requirements.txt
```

MediaPipe model (Tasks API; `mediapipe==1.1.1` has no `solutions` module), placed in `data/models/`:
`pose_landmarker_full.task` from
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`.

Dataset root = the folder that contains this repo (override with the env variable `MOBIPHYSIO_ROOT`). It must
hold the exercise folders (`E01(Abduction)`, ...), `Reference-Model Videos by PTs/` and
`Physiotherapist Exercise Marking.csv`. The raw dataset is never modified.

## Phase 1: keypoints and preprocessing

One command (resumable; cached keypoints are skipped):

```bash
python scripts/run_phase1.py
```

Or step by step (all read `configs/data.yaml`):

| Step | Command | Output |
|---|---|---|
| index | `python -m cvdc.data.index` | `data/index.csv` (one row per video: exercise, subject, view, variation, gender, group, score, EAAQ items, quality class, paths), `data/index_ref.csv` (PT demos) |
| keypoints | `python -m cvdc.data.extract` | `data/cache/kp/<video>.npz`: 2D (x, y, z, visibility, presence) and 3D world landmarks, real frame timestamps; `outputs/phase1/extract_summary.csv`, `detection_by_variation.csv` |
| preprocessing | `python -m cvdc.data.preprocess` | `data/cache/pre/<video>.npz`: uniform 30 fps grid, interpolated, smoothed, root-centred, torso-scaled (2D and 3D) |
| reps | `python -m cvdc.data.reps` | `data/reps.csv` (one row per rep), `outputs/phase1/reps_per_video.csv` |
| sync | `python -m cvdc.data.sync` | `outputs/sync_report.csv` (per group), `outputs/sync_pairs.csv` (per view pair, with the CVDC pair source) |
| packs | `python -m cvdc.data.pack` | `data/cache/packs/reps_coco17_T64.npz`, `videos_coco17_T300.npz` (+ index CSVs) |
| checks | `python -m cvdc.viz.checks` | skeleton-only PNGs in `outputs/checks/` |
| report | `python scripts/phase1/report.py` | summary tables in `outputs/phase1/` |

Measured runtimes (Ryzen 7 7700, 12-14 processes): keypoint extraction 37.6 min for 941 videos + 3 PT demos
(cached afterwards); preprocessing 7 s; reps 4 s; sync 24 s once audio lags are cached (the first run also decodes
the audio of ~830 videos); packs 18 s; checks 8 s. A re-run with all caches takes ~1.5 min.

Phase 1 checks: `scripts/phase1/pose_settings_check.py` (P2/P3: resolution and square padding),
`scripts/phase1/rep_signal_check.py` (P6: rep-count agreement across views per candidate signal) and
`scripts/phase1/rep_alignment_check.py` (P6: do rep peaks coincide across synced views).
Results and open questions: `PROGRESS.md` §5 and §7.

## Tests

```bash
python -m pytest -q tests
```

## Layout

```
configs/            data.yaml (data, labels, pose, preprocessing, reps, sync); exercises/EXX.yaml
cvdc/data/          filenames, labels, index, video (PyAV), extract, skeleton, preprocess, reps, sync, pack
cvdc/assess/        angles (joint-angle math; Module III follows in Phase 4)
cvdc/viz/           skeleton drawing, Phase 1 checks
scripts/            run_phase1.py, phase1/ checks and report, step0/ (Step 0 exploration)
tests/              pytest
```
