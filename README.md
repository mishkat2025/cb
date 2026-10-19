# CVDC physiotherapy assessment on MobiPhysio

Implementation of the proposal "AI-Guided Physiotherapy Assessment for Low-Resource Telehealth through
Cross-View Depth-Consistency Learning" on the MobiPhysio dataset (doi 10.7910/DVN/XSI0QN).

- **Scope now:** E01 (Abduction) and E02 (Adduction) only (`data.include_exercises`). Exercises are
  discovered at runtime, so adding one later is a config change.
- **Status:** Step 0, Phase 1 (keypoints and preprocessing), Phase 2 (models, CVDC loss, pair builder),
  Phase 3 steps (a) (GroupKFold-5 for the 4 main methods) and (a2) (score-setup diagnosis and fixes, Proposed
  variants) and Phase 4 (Module III: PT-demo reference templates, DTW deviations, rule-based errors, the reference
  score, the embedding-to-reference distance) are done; the ablations (b) and the LOSO table (c) wait for review.
  See `PROGRESS.md` for the state and the phase checklist, and `DESIGN_DECISIONS.md` for every open choice.
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

**KTPFormer (B2's second 3D source; no licence: local non-commercial research use only, never committed or shared).**
Cite: J. Peng, Y. Zhou and P. Y. Mok, "KTPFormer: Kinematics and Trajectory Prior Knowledge-Enhanced Transformer for 3D Human Pose Estimation", Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. Clone the code
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

## Phase 3: training and GroupKFold evaluation (step a)

Config: `configs/model.yaml` (`train`, `methods`, `eval`). Runs (TensorBoard, checkpoints, config snapshots,
predictions, embeddings) go to `runs/` (gitignored); tables to `outputs/results/phase3a/` (gitignored).

```bash
python -m cvdc.train.run --smoke                  # smoke test: fold 0, 3 training subjects, 2 epochs, all 4 methods (~1 min)
python -m cvdc.train.run                          # GroupKFold-5: B1, B2-MediaPipe, B2-KTPFormer, Proposed (background job)
python scripts/phase3/report_phase3a.py           # metrics, baselines, VCI, bootstrap CIs -> outputs/results/phase3a/
tensorboard --logdir runs/phase3a
```

Step (a) ran at commit `05b196d`. Step (a2) changed the config defaults (video head, video batches, T6 weights,
item masks, early-stopping rule, method list, tag `phase3a2`), so reproduce step (a) from that commit.

Single stages: `python -m cvdc.train.pretrain_cvdc --fold K` (CVDC pretraining of one fold) and
`python -m cvdc.train.train_supervised --method b1|b2_mediapipe|b2_ktpformer|proposed --fold K [--linear-probe]`.
`--linear-probe` freezes the encoder (default: full fine-tuning, `train.fine_tune`).

| Piece | Code | Notes |
|---|---|---|
| trainer | `cvdc/train/trainer.py` | folds + inner validation split (T8), early stopping, warm-up + cosine schedule (T9), run directories (TensorBoard, config snapshot, checkpoints) |
| CVDC pretraining | `cvdc/train/pretrain_cvdc.py` | inner training subjects' rep sets; validation = CVDC loss on the inner validation subjects' rep sets |
| supervised | `cvdc/train/train_supervised.py` | rep level (T1), standardised targets (T7), full fine-tuning or linear probe (T10) |
| runner | `cvdc/train/run.py` | all methods x folds, resumable (`done.json`) |
| metrics | `cvdc/eval/metrics.py` | RMSE / MAE / Pearson / Spearman, macro F1 / kappa, VCI sums, subject-level bootstrap (E17) |
| report | `scripts/phase3/report_phase3a.py` | within-group and pooled tables at video (headline, E15) and rep level, baselines (E16), VCI, stopping epochs, runtimes |

### Step (a2): score setup (label diagnostics, video head, T6 weighting, stabilised early stopping)

Tables go to `outputs/results/phase3a2/` (gitignored); runs to `runs/phase3a2/`. The Proposed variants reuse the
step (a) CVDC encoders (`train.cvdc_init_from: phase3a`), so `runs/phase3a/proposed/fold*/cvdc/` must exist (or set
it to `null` to pretrain again).

```bash
python scripts/phase3/label_diagnostics.py        # no training: item variances, score decomposition, set features, set-level baselines (~30 s)
python -m cvdc.train.run --smoke --force          # smoke test of the a2 setup (all 7 methods, ~2 min)
python -m cvdc.train.run                          # GroupKFold-5: B1, Proposed (iii)/(i)/(ii) λ 0.1, 1.0, B2-MediaPipe, B2-KTPFormer (~2 h, background job)
python scripts/phase3/report_phase3a2.py          # within-group tables (video and rep heads), baselines, deltas, VCI, epochs
```

| Piece | Code | Notes |
|---|---|---|
| baselines | `cvdc/eval/baselines.py` | set features (rep count, rep duration, ROM of the main angle), item-activity rule (T14), score decomposition, video-level baselines (E16, E20) |
| video head | `cvdc/models/heads.py` (`VideoHead`) | mean + attention pooling of a video's rep embeddings + set rep count → score and items (T13) |
| supervised | `cvdc/train/train_supervised.py` | video batches, per-video rep weights (T6), item masks (T14), EMA / min-epoch early stopping (T8b), optional + λ·L_cvdc |
| runner | `cvdc/train/run.py` | methods `b1`, `proposed`, `proposed_lp`, `proposed_j01`, `proposed_j1`, `b2_mediapipe`, `b2_ktpformer`; reuses checked CVDC encoders |
| report | `scripts/phase3/report_phase3a2.py`, `scripts/phase3/label_diagnostics.py` | `summary.md`, `label_diagnostics.md` and CSVs |

## Phase 4: Module III (reference-based assessment) and the embedding distance

Module III follows proposal Figure 2: the PT demo of each exercise is the reference template (30 fps, 3D angles,
left arms mirrored); high-scoring expert videos (≥ 95) give per-view bias and tolerance bands; every rep is
DTW-aligned to the template, compared channel by channel, scored 0–100 and checked by YAML rules (joint, frames,
degrees, severity). Posture checks (elbow, other arm, trunk, shoulder symmetry) are centred on the high-scoring experts
(`assess.posture_centre: experts`, a post-hoc change, M14); the main movement curve stays the demo's. Config: `templates` and `assess` in `configs/data.yaml`, `assess.channels / score / rules` in
`configs/exercises/EXX.yaml`. Outputs go to `outputs/phase4/` (gitignored). Needs the Phase 3 step (a2) runs for the
embedding distance.

```bash
python scripts/phase4/embedding_distance.py   # encode all reps + the PT demo per method and fold (~1 min, GPU)
python -m cvdc.assess.run                       # Module III: templates, per-fold libraries, scores, errors, JSON (~20 s)
python -m cvdc.assess.run --posture-centre pt_demo --out outputs/phase4/posture_pt --no-json   # the first run (before M14)
python scripts/phase4/recording_noise.py        # item 6: reps contained in each view vs PF3 / PF4 (~45 s)
python scripts/phase4/report_phase4.py          # agreement, components, errors, distance tracking, view stability,
                                                # posture before / after (M14), exploratory demo-centred speed (M15) (~40 s)
```

| Piece | Code | Notes |
|---|---|---|
| angles | `cvdc/assess/angles.py` | + elbow extension, contralateral elevation, trunk lateral lean, shoulder asymmetry (all for the moving arm) |
| template | `cvdc/assess/template.py` | arm-direction DTW alignment, PT reference (one DBA step), per-view library (bias, bands, ROM / speed tolerances), save / load |
| Module III | `cvdc/assess/module3.py` | deviations, score (M11), rule-based errors (M12), per-video JSON and its validator (M13), `Assessor` for single reps (demo) |
| runner | `cvdc/assess/run.py` | per-fold libraries (training subjects only), `module3_{reps,videos,errors}.csv`, `json/<video>.json`, `library/<EX>.{npz,json}` (all subjects, for the demo) |
| distance | `scripts/phase4/embedding_distance.py` | centred-cosine and L2 distance to the PT reference embedding for B1, B2, Proposed (λ 1.0), frozen CVDC, random init |
| recording | `cvdc/eval/recording.py`, `scripts/phase4/recording_noise.py` | reps fully / half contained in each view of a synced performance |
| report | `scripts/phase4/report_phase4.py` | `summary.md`, CSVs, `figs/template_<EX>.png`, `figs/agreement_scatter.png`; ICC with bootstrap in `cvdc/eval/metrics.py` |

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
cvdc/assess/        angles, dtw, template (reference + library), module3 (score, errors, JSON), run (Phase 4 runner)
cvdc/eval/          consistency (VCI), splits, metrics (bootstrap CIs, ICC), baselines (set features, video-level
                    baselines), recording (reps contained per view)
cvdc/train/         trainer, pretrain_cvdc, train_supervised, run (GroupKFold runner)
cvdc/viz/           skeleton drawing, Phase 1 checks
scripts/            run_phase1.py, phase1/ checks and report, phase2/ (smoke, lifter_check, bench), phase3/ (reports, label
                    diagnostics), phase4/ (embedding distance, recording noise, report), step0/
third_party/        local clones used by wrappers (gitignored)
tests/              pytest
```
