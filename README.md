# CVDC physiotherapy assessment on MobiPhysio

Implementation of the proposal "AI-Guided Physiotherapy Assessment for Low-Resource Telehealth through
Cross-View Depth-Consistency Learning" on the MobiPhysio dataset (doi 10.7910/DVN/XSI0QN).

- **Scope now:** E01 (Abduction) and E02 (Adduction) only (`data.include_exercises`). Exercises are
  discovered at runtime, so adding one later is a config change.
- **Status:** Step 0, Phase 1 (keypoints and preprocessing) and Phase 2 (models, CVDC loss, pair builder) are done;
  training at scale (Phase 3) has not started. See `PROGRESS.md` for the state and
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
| sync | `python -m cvdc.data.sync` | `outputs/sync_report.csv` (per group: class, per-view offsets), `outputs/sync_pairs.csv` (per view pair: lag, CVDC pair source `sync`/`audio`/`rep` and weight) |
| reps | `python -m cvdc.data.reps` (after sync) | `data/reps.csv` (one row per rep; synced groups share `rep_uid` and boundaries across views), `data/reps_perview.csv` (each view on its own, for comparison), `outputs/phase1/reps_per_video.csv` |
| packs | `python -m cvdc.data.pack` | `data/cache/packs/reps_coco17_T64.npz`, `videos_coco17_T300.npz` (+ index CSVs) |
| checks | `python -m cvdc.viz.checks` | skeleton-only PNGs in `outputs/checks/` |
| report | `python scripts/phase1/report.py` | summary tables in `outputs/phase1/` |

Measured runtimes (Ryzen 7 7700, 12-14 processes): keypoint extraction 37.6 min for 941 videos + 3 PT demos
(cached afterwards); preprocessing 7 s; reps 4 s; sync 24 s once audio lags are cached (the first run also decodes
the audio of ~830 videos); packs 18 s; checks 8 s. A re-run with all caches takes ~1.5 min.

Phase 1 checks: `scripts/phase1/pose_settings_check.py` (P2/P3: resolution and square padding),
`scripts/phase1/rep_signal_check.py` (P6: rep-count agreement across views per candidate signal),
`scripts/phase1/rep_alignment_check.py` (P6: do rep peaks coincide across synced views),
`scripts/phase1/e02_angle_check.py` (E02 horizontal adduction angle on the data) and
`scripts/phase1/left_arm_scores.py` (expert scores of left- vs right-arm performances).
Results and open questions: `PROGRESS.md` §5 and §8.

## Phase 2: models, CVDC loss and pairs

Config: `configs/model.yaml` (merged with `configs/data.yaml`; pair weights stay in `sync.pair_weights`).

```bash
python -m cvdc.data.pairs                     # rep-level CVDC pairs from the Phase 1 caches (~12 s)
python -m cvdc.models.lifter                  # KTPFormer 3D for B2, all videos (~13 min; see below)
python -m cvdc.data.pack --lift ktpformer     # lifted 3D packs (same items as the main packs)
python scripts/phase2/smoke.py                # smoke run on E01 + E02 (~4 min on an RTX 4060 Ti)
python scripts/phase2/lifter_check.py --all   # view consistency of MediaPipe vs KTPFormer 3D angles
python scripts/phase2/bench.py                # bf16/fp32 step times and the Phase 3 runtime estimate (idle GPU)
```

**KTPFormer (B2's second 3D source; licence: none, research use only, never committed).** Clone the code
and fetch the Human3.6M CPN checkpoint, then convert it to a tensors-only state dict (see A10 in
`DESIGN_DECISIONS.md` for the allow-listed pickle globals):

```bash
git clone https://github.com/JihuaPeng/KTPFormer third_party/KTPFormer   # commit b961951
# model_243_CPN_best_epoch.bin from the Google Drive folder in the KTPFormer README -> data/models/ktpformer/
```

| Piece | Code | Notes |
|---|---|---|
| graph | `cvdc/models/graph.py` | COCO-17 ST-GCN layout, self-loops, uniform / distance / spatial partitions |
| encoder | `cvdc/models/stgcn.py` | ST-GCN (Yan et al. 2018), 10 blocks, 3.07 M parameters, 256-d embedding, shared by all views; input `xy`, `xyv` or `xyz` |
| heads | `cvdc/models/heads.py`, `model.py` | projection 2-layer MLP -> 128-d L2-normalised; linear heads: exercise (9), score, EAAQ items, quality (3), with an exercise embedding |
| CVDC loss | `cvdc/losses/cvdc.py` | weighted multi-positive InfoNCE, tau 0.1, negatives `all_other_groups` / `different_exercise_only`, monitoring stats |
| supervised loss | `cvdc/losses/supervised.py` | MSE on score / 100 and items, cross-entropy on quality and exercise |
| pair builder | `cvdc/data/pairs.py` | `data/cvdc_pairs.csv`, `data/cvdc_members.csv`, DTW-aligned pack, `outputs/phase2/table_pair_counts.csv` |
| synthetic views | `cvdc/data/synthetic.py` | world landmarks -> levelled yaw orbit -> pinhole projection |
| batches | `cvdc/data/dataset.py`, `cvdc_batch.py`, `augment.py` | packs on the GPU; all views of a rep set together; per-pair weight matrix |
| VCI | `cvdc/eval/consistency.py` | viewpoint consistency index: centred 256-d (primary), raw, projection |
| splits | `cvdc/eval/splits.py` | stratified GroupKFold, LOSO, leakage check |
| lifter | `cvdc/models/lifter.py` | KTPFormer 2D→3D wrapper (input `xyz_ktpformer`) |
| targets | `cvdc/losses/supervised.py` | per-fold standardisation of score and items (`TargetScaler`) |

Smoke outputs: `outputs/phase2/smoke_log.txt` (and `smoke_*.csv`, `smoke_summary.json`); skeleton-only figures
`outputs/checks/phase2_synthetic_views.png` and `phase2_cvdc_smoke_curves.png`. Results: `PROGRESS.md` §6.

## Tests

```bash
python -m pytest -q tests
```

## Layout

```
configs/            data.yaml (data, labels, pose, preprocessing, reps, sync); model.yaml (Phase 2); exercises/EXX.yaml
cvdc/data/          filenames, labels, index, video (PyAV), extract, skeleton, preprocess, reps, sync, pack,
                    pairs, synthetic, augment, dataset, cvdc_batch
cvdc/models/        graph, stgcn, heads, model, lifter (KTPFormer wrapper)
cvdc/losses/        cvdc (weighted multi-positive InfoNCE), supervised
cvdc/assess/        angles, dtw (Module III follows in Phase 4)
cvdc/eval/          consistency (VCI), splits (the rest follows in Phase 5)
cvdc/viz/           skeleton drawing, Phase 1 checks
scripts/            run_phase1.py, phase1/ checks and report, phase2/ (smoke, lifter_check, bench), step0/
third_party/        local clones used by wrappers (gitignored)
tests/              pytest
```
