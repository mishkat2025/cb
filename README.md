# CVDC physiotherapy assessment on MobiPhysio

Implementation of the proposal "AI-Guided Physiotherapy Assessment for Low-Resource Telehealth through
Cross-View Depth-Consistency Learning" on the MobiPhysio dataset (doi 10.7910/DVN/XSI0QN).

From one phone video of a patient doing an exercise: 2D pose (MediaPipe) → an ST-GCN encoder trained with Cross-View
Depth-Consistency (CVDC) learning → Module III compares every repetition with a physiotherapist's demo → a 0–100 score
per repetition and the problems found (for example "elbow bent", "trunk leaning").

## Quick start

The trained weights are part of the repository, so there is nothing else to download to run the demo.

**You need:** Python 3.11, git, and about 4 GB of free disk space for PyTorch and the other packages. A GPU is not
needed: the demo runs on the CPU.

```bash
git clone https://github.com/mishkat2025/cb.git
cd cb
python -m venv .venv
.venv\Scripts\activate             # Windows; on macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**macOS:** `requirements.txt` pins the CUDA build of PyTorch (`torch==2.14.1+cu126`), which has no macOS version.
Change that line to `torch==2.14.1` before `pip install`.

### Run the app

```bash
streamlit run demo/app.py
```

It opens in your browser (http://localhost:8501). Choose **Upload my own video**, pick the exercise and where the
camera stood, upload the video and press **Assess this video**. After about a minute you get a score for every
repetition, the problems found, plots, and a four-panel video (faces blurred) saved in `outputs/full/demo/app/`.

| Code | Exercise | | Code | Exercise | | Code | Exercise |
|---|---|---|---|---|---|---|---|
| E01 | Abduction | | E04 | Medial rotation | | E07 | Hip joint flexion |
| E02 | Adduction | | E05 | Circumduction | | E08 | Both leg flexion |
| E03 | Lateral rotation | | E06 | Wrist extension | | E09 | Back extension |

Camera position: `F` front, `L` left side, `R` right side. The final scope of the study is E01–E04; for E05–E09 the
score is the rule-based Module III (joint angles against the physiotherapist's demo), with the network's score shown
as a supplementary output.

### Run from the command line (no browser)

```bash
python -m demo.live --source my_video.mp4 --exercise E01 --view F --headless --no-realtime --save-mp4 out.mp4
```

Prints one line per repetition (score and problems) and writes `out.mp4` with the skeleton drawn and faces blurred.
Leave out `--headless --no-realtime` to watch it in a window at normal speed; `--source 0` uses a webcam.

### The dataset

The MobiPhysio dataset is not in this repository: use your own copy (Harvard Dataverse,
[doi:10.7910/DVN/XSI0QN](https://doi.org/10.7910/DVN/XSI0QN)). The app finds it in either of these places:

```
cb/                     this repository (the dataset may also go inside it, as cb/dataset/)
dataset/                next to it
  Physiotherapist Exercise Marking.csv
  Reference-Model Videos by PTs/
  videos/
    E01(Abduction)/  E02(Aduction)/  ...  E09(Back _Extension)/     the dataset's exercise folders, as downloaded
```

Anywhere else: set `MOBIPHYSIO_ROOT` to the folder with the CSV and `MOBIPHYSIO_VIDEO_ROOT` to the folder with the
exercise folders. With the dataset in place, **From the dataset** in the app lists every video with its participant,
camera and recording condition; the model loaded is the one whose training left that participant out, so the result
is a fair test. On the command line: `python -m demo.live --list E01`, then
`python -m demo.live --select E01 P05 F FL`.

### What is included

Only skeletons (keypoints), tables and model weights: no video frames or faces.

| What | Where | Size |
|---|---|---|
| Trained models: the Proposed ST-GCN + CVDC (λ 1.0), one per exercise and per fold (9 × 5) | `runs/full/perex_gkf5/E0x/proposed_j1/fold<k>/finetune/best.pt`, with `embeddings_final.npz` (the training embeddings: centring and the app's plot) | 610 MB |
| Module III reference libraries | `outputs/full/phase4/library/` (rule-based, E01–E09), `outputs/full/results/completion/module3/library/` (with the CVDC model, E01–E04, per fold) | 2 MB |
| The four-panel video's embedding map (PCA of each fold's training embeddings) | `outputs/full/demo/pca/` | 3 MB |
| Physiotherapist demo skeletons | `data/full/cache/pre_ref/` | 3 MB |
| Video index and repetition tables | `data/full/index.csv`, `index_ref.csv`, `reps.csv`, `reps_ref.csv` | 9 MB |
| MediaPipe Pose, and Hands for E06 (Apache-2.0, Google) | `data/models/pose_landmarker_full.task`, `hand_landmarker.task` | 17 MB |

Retraining or reproducing the results needs the dataset and the steps in the sections below (they start from the
keypoint extraction). Tests: `python -m pytest -q tests` (on a fresh clone 14 are skipped: they need the dataset's
keypoint caches or KTPFormer).

## Project history and full documentation

**Final report (final scope E01–E04, 2026-10-21): [`reports/FINAL_REPORT_E01_E04.md`](reports/FINAL_REPORT_E01_E04.md)**,
full tables in [`reports/final_E01_E04_tables.md`](reports/final_E01_E04_tables.md) (E05–E09 as a supplementary
appendix). **LOSO per exercise (2026-10-22)** is §7 of that report, tables in
[`reports/final_E01_E04_loso_tables.md`](reports/final_E01_E04_loso_tables.md); how to run, check and resume it:
[`scripts/full/loso/README.md`](scripts/full/loso/README.md). The status notes below are the earlier history.

- **Scope:** all nine exercises since 2026-10-20 (`configs/data_full.yaml`, every input and output under
  `data/full/`, `outputs/full/`, `runs/full/`). The E01/E02 **pilot** (`configs/data.yaml`, the default config) and
  all its results stay as they are. Exercises are discovered at runtime, so adding one is a config change.
- **Full scope status:** Phase 1 (keypoints incl. MediaPipe Hands for E06, sync, reps, packs, KTPFormer lifting) is
  done for all nine exercises, with the tenth-review changes and the CVDC pairs (`PROGRESS.md` §5b); nothing is trained on the full scope yet (waits for the final evaluation scheme).
- **Status:** Step 0, Phase 1 (keypoints and preprocessing), Phase 2 (models, CVDC loss, pair builder),
  Phase 3 steps (a) (GroupKFold-5 for the 4 main methods) and (a2) (score-setup diagnosis and fixes, Proposed
  variants) and Phase 4 (Module III: PT-demo reference templates, DTW deviations, rule-based errors, the reference
  score, the embedding-to-reference distance) are done; the ablations (b) and the LOSO table (c) wait for review.
  See `PROGRESS.md` for the state and the phase checklist, and `DESIGN_DECISIONS.md` for every open choice.
- **Privacy:** the videos show faces. Only keypoints are stored; renders are skeleton-only; `data/`, `outputs/` and `runs/` are gitignored except
  the demo's files (Quick start, "What is included": keypoints, tables and weights only). Audio is decoded in memory for synchronization and never written; GPS tags in
  the video containers are never read.

## Setup (the author's machine)

For a fresh clone, the Quick start above is enough. The author's own setup:

```bash
# Python 3.11 venv one level above the repo (D:\cb\.venv), created with uv
uv venv ../.venv --python 3.11
uv pip install --python ../.venv/Scripts/python.exe -r requirements.txt
```

MediaPipe models (Tasks API; `mediapipe==1.1.1` has no `solutions` module), in `data/models/` (both are included in
the repository):
`pose_landmarker_full.task` from
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`,
and for E06 `hand_landmarker.task` (Apache 2.0) from
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`.

**Folder layout (since 2026-10-22).** Everything lives under `D:\cb`:

```
D:\cb\
  cvdc-physio\     this repository (code, configs, docs, reports; data/ outputs/ runs/ are gitignored artefacts)
  dataset\         the MobiPhysio dataset, never modified
    videos\        the nine exercise folders E01(Abduction) ... E09(Back _Extension)   (data.video_root)
    Reference-Model Videos by PTs\, Physiotherapist Exercise Marking.csv, MobiPhysio.md   (data.root)
  docs\            the research proposal, the MobiPhysio Data in Brief paper, the implementation brief
  .venv\           the Python environment
```

`data.root` (env `MOBIPHYSIO_ROOT`) is `../dataset` and `data.video_root` (D1; env `MOBIPHYSIO_VIDEO_ROOT`) is
`../dataset/videos`, in both the pilot and the full-scope configs. The full-dataset Step 0 check:
`scripts/full/step0/README.md`, results in `PROGRESS.md` §3.7.

## Full scope (all nine exercises): Phase 1

Every command takes `--config configs/data_full.yaml`; nothing of the pilot is written (D15). Order, reuse of the
pilot's keypoints and the extra checks: `scripts/full/phase1/README.md`; results in `PROGRESS.md` §5b.

```bash
python scripts/run_phase1.py --config configs/data_full.yaml --to pack      # index ... packs (resumable)
python scripts/run_phase1.py --config configs/data_full.yaml --from lift    # KTPFormer lifting (GPU), lifted packs, checks, report
python scripts/full/phase1/hands_check.py                                    # E06: MediaPipe Hands wrist angle
python scripts/full/phase1/pf1_motion_check.py --exercise E05                # E05 PF1 = 0 non-experts: wrong exercise?
python scripts/full/phase1/compare_pilot.py                                  # E01/E02 vs the pilot, E03-E09 vs Step 0
python scripts/full/phase1/rep_signal_compare.py                             # E03/E06/E09 rep signals by cross-view agreement
python -m cvdc.data.pairs --config configs/data_full.yaml                    # CVDC pairs (per-exercise DTW angles)
python scripts/full/phase1/dtw_angle_check.py                                # known-warp test of the DTW angles
```

## Full scope: Phase 3, one model per exercise (T17)

Supervisor decision (2026-10-20): every exercise E01–E09 gets its own models, trained and tested on its videos only
(GroupKFold-5 per exercise, stratified by expert / non-expert; no exercise head). Config `configs/per_exercise.yaml`;
results in `PROGRESS.md` §6h. A shared all-exercise run was stopped and is archived in `runs/full/aborted_shared_model/`.

```bash
python scripts/full/phase3/runtime_estimate_perex.py                 # runtime estimates (nothing trained)
python -m cvdc.train.run_per_exercise --smoke                         # smoke test: fold 0 of every exercise, 2 epochs (~1 min)
python -m cvdc.train.run_per_exercise                                 # B1, B2-KTPFormer, Proposed (lambda 1.0), frozen Proposed (~8 h)
python -m cvdc.train.run_per_exercise --exercises E07 --methods b1    # a subset
```

## Demo (Phase 6)

No webcam on this machine: video files play at real-time speed (`--source 0` opens a webcam where one exists). One
model per exercise (T17): give the exercise, or pick a dataset video by its fields (the model fold that held its
subject out is loaded; otherwise a leakage warning is shown). **The assessment is always Module III's score**
(X11): with the Proposed model it is the proposal's pipeline (Module III aligns and compares each rep in that fold's
CVDC embedding space, M19 + M20). The app offers no other model (X12): the comparison networks belong to the
evaluation; the rule-based Module III runs only as a fallback when the checkpoint is missing (and as the
developer switches `--method` / `--no-model` of `demo.live`). The network's own learned score is shown
separately, labelled supplementary.

```bash
python -m demo.live --list E03                                   # dataset videos of an exercise
python -m demo.live --select E07 P05 F FL                        # a dataset video (exercise, subject, view, variation)
python -m demo.live --source my_video.mp4 --exercise E01 --view F --save-mp4 out.mp4   # faces blurred in the file
streamlit run demo/app.py                                        # pick a dataset video from a table (or upload one) ->
                                                                 # four-panel video (saved, played in the page) and a
                                                                 # plain-language per-rep report, plots, JSON (X10)
python -m demo.live --select E03 P05 F FL --ghost                # + the PT reference as a ghost (front views)
```

The side panel shows the PT's own skeleton from one real demo rep, rest → peak → rest: at its rest pose while the
patient rests (nothing compared), and following the patient through Module III's template during a rep, with the live
angle differences per joint (X6, X7); the ghost is drawn only during a rep. Hand-test commands (Windows, from `D:\cb`)
and expected results: `demo/TEST_SHEET.md`.

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
| report | `python scripts/pilot/phase1/report.py` | summary tables in `outputs/phase1/` |

Measured runtimes (Ryzen 7 7700, 12-14 processes): keypoint extraction 37.6 min for 941 videos + 3 PT demos
(cached afterwards); preprocessing 7 s; reps 4 s; sync 24 s once audio lags are cached (the first run also decodes
the audio of ~830 videos); packs 18 s; checks 8 s. A re-run with all caches takes ~1.5 min.

Phase 1 checks: `scripts/pilot/phase1/pose_settings_check.py` (P2/P3: resolution and square padding),
`scripts/pilot/phase1/rep_signal_check.py` (P6: rep-count agreement across views per candidate signal),
`scripts/pilot/phase1/rep_alignment_check.py` (P6: do rep peaks coincide across synced views),
`scripts/pilot/phase1/e02_angle_check.py` (E02 horizontal adduction angle on the data) and
`scripts/pilot/phase1/left_arm_scores.py` (expert scores of left- vs right-arm performances).
Results and open questions: `PROGRESS.md` §5 and §8.

## Phase 2: models, CVDC loss and pairs

Config: `configs/model.yaml` (merged with `configs/data.yaml`; pair weights stay in `sync.pair_weights`).

```bash
python -m cvdc.data.pairs                     # rep-level CVDC pairs from the Phase 1 caches (~12 s)
python -m cvdc.models.lifter                  # KTPFormer 3D for B2, all videos (~13 min; see below)
python -m cvdc.data.pack --lift ktpformer     # lifted 3D packs (same items as the main packs)
python scripts/pilot/phase2/smoke.py                # smoke run on E01 + E02 (~4 min on an RTX 4060 Ti)
python scripts/pilot/phase2/lifter_check.py --all   # view consistency of MediaPipe vs KTPFormer 3D angles
python scripts/pilot/phase2/bench.py                # bf16/fp32 step times and the Phase 3 runtime estimate (idle GPU)
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
python scripts/pilot/phase3/report_phase3a.py           # metrics, baselines, VCI, bootstrap CIs -> outputs/results/phase3a/
tensorboard --logdir runs/phase3a
```

Step (a) ran at commit `fdf6244`. Step (a2) changed the config defaults (video head, video batches, T6 weights,
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
| report | `scripts/pilot/phase3/report_phase3a.py` | within-group and pooled tables at video (headline, E15) and rep level, baselines (E16), VCI, stopping epochs, runtimes |

### Step (a2): score setup (label diagnostics, video head, T6 weighting, stabilised early stopping)

Tables go to `outputs/results/phase3a2/` (gitignored); runs to `runs/phase3a2/`. The Proposed variants reuse the
step (a) CVDC encoders (`train.cvdc_init_from: phase3a`), so `runs/phase3a/proposed/fold*/cvdc/` must exist (or set
it to `null` to pretrain again).

```bash
python scripts/pilot/phase3/label_diagnostics.py        # no training: item variances, score decomposition, set features, set-level baselines (~30 s)
python -m cvdc.train.run --smoke --force          # smoke test of the a2 setup (all 7 methods, ~2 min)
python -m cvdc.train.run                          # GroupKFold-5: B1, Proposed (iii)/(i)/(ii) λ 0.1, 1.0, B2-MediaPipe, B2-KTPFormer (~2 h, background job)
python scripts/pilot/phase3/report_phase3a2.py          # within-group tables (video and rep heads), baselines, deltas, VCI, epochs
```

| Piece | Code | Notes |
|---|---|---|
| baselines | `cvdc/eval/baselines.py` | set features (rep count, rep duration, ROM of the main angle), item-activity rule (T14), score decomposition, video-level baselines (E16, E20) |
| video head | `cvdc/models/heads.py` (`VideoHead`) | mean + attention pooling of a video's rep embeddings + set rep count → score and items (T13) |
| supervised | `cvdc/train/train_supervised.py` | video batches, per-video rep weights (T6), item masks (T14), EMA / min-epoch early stopping (T8b), optional + λ·L_cvdc |
| runner | `cvdc/train/run.py` | methods `b1`, `proposed`, `proposed_lp`, `proposed_j01`, `proposed_j1`, `b2_mediapipe`, `b2_ktpformer`; reuses checked CVDC encoders |
| report | `scripts/pilot/phase3/report_phase3a2.py`, `scripts/pilot/phase3/label_diagnostics.py` | `summary.md`, `label_diagnostics.md` and CSVs |

## Phase 4: Module III (reference-based assessment) and the embedding distance

Module III follows proposal Figure 2: the PT demo of each exercise is the reference template (30 fps, 3D angles,
left arms mirrored); high-scoring expert videos (≥ 95) give per-view bias and tolerance bands; every rep is
DTW-aligned to the template, compared channel by channel, scored 0–100 and checked by YAML rules (joint, frames,
degrees, severity). Posture checks (elbow, other arm, trunk, shoulder symmetry) are centred on the high-scoring experts
(`assess.posture_centre: experts`, a post-hoc change, M14); the main movement curve stays the demo's. Config: `templates` and `assess` in `configs/data.yaml`, `assess.channels / score / rules` in
`configs/exercises/EXX.yaml`. Outputs go to `outputs/phase4/` (gitignored). Needs the Phase 3 step (a2) runs for the
embedding distance.

```bash
python scripts/pilot/phase4/embedding_distance.py   # encode all reps + the PT demo per method and fold (~1 min, GPU)
python -m cvdc.assess.run                       # Module III: templates, per-fold libraries, scores, errors, JSON (~20 s)
python -m cvdc.assess.run --posture-centre pt_demo --out outputs/phase4/posture_pt --no-json   # the first run (before M14)
python scripts/pilot/phase4/recording_noise.py        # item 6: reps contained in each view vs PF3 / PF4 (~45 s)
python scripts/pilot/phase4/report_phase4.py          # agreement, components, errors, distance tracking, view stability,
                                                # posture before / after (M14), exploratory demo-centred speed (M15) (~40 s)
```

| Piece | Code | Notes |
|---|---|---|
| angles | `cvdc/assess/angles.py` | + elbow extension, contralateral elevation, trunk lateral lean, shoulder asymmetry (all for the moving arm) |
| template | `cvdc/assess/template.py` | arm-direction DTW alignment, PT reference (one DBA step), per-view library (bias, bands, ROM / speed tolerances), save / load |
| Module III | `cvdc/assess/module3.py` | deviations, score (M11), rule-based errors (M12), per-video JSON and its validator (M13), `Assessor` for single reps (demo) |
| runner | `cvdc/assess/run.py` | per-fold libraries (training subjects only), `module3_{reps,videos,errors}.csv`, `json/<video>.json`, `library/<EX>.{npz,json}` (all subjects, for the demo) |
| distance | `scripts/pilot/phase4/embedding_distance.py` | centred-cosine and L2 distance to the PT reference embedding for B1, B2, Proposed (λ 1.0), frozen CVDC, random init |
| recording | `cvdc/eval/recording.py`, `scripts/pilot/phase4/recording_noise.py` | reps fully / half contained in each view of a synced performance |
| report | `scripts/pilot/phase4/report_phase4.py` | `summary.md`, CSVs, `figs/template_<EX>.png`, `figs/agreement_scatter.png`; ICC with bootstrap in `cvdc/eval/metrics.py` |

## Phase 5: evaluation (cross-view protocol, pair-source ablation, LOSO estimate)

Same GroupKFold-5 folds, inner splits, seeds, budget and schedule as Phase 3 step (a2). Runs go to
`runs/phase5_crossview/` and `runs/phase5_sources/`, tables to `outputs/results/phase5_*` (all gitignored). **Run GPU
jobs one at a time**: two processes sharing the RTX 4060 Ti were slower than running them in sequence (A9).

```bash
# A. cross-view protocol: train on two views, test on the held-out third (F / L / R), 4 main methods x 5 folds
python -m cvdc.eval.cross_view --smoke --force                 # plumbing check (~1 min)
python -m cvdc.eval.cross_view --methods proposed_j1           # CVDC on the two training views + joint fine-tuning (~2.9 h)
python -m cvdc.eval.cross_view --methods b1 b2_mediapipe b2_ktpformer   # (~1.5 h)
python scripts/pilot/phase5/crossview_distance.py                    # embedding distance to the PT reference per model (~6 min)
python scripts/pilot/phase5/report_crossview.py                      # exercise, MQ, VCI, distance view stability -> outputs/results/phase5_crossview/

# B. CVDC pair-source ablation, pretraining only, held-out within-group VCI ((d) = the step (a) encoders, reused)
python scripts/pilot/phase5/pair_sources.py --smoke --force           # plumbing check
python scripts/pilot/phase5/pair_sources.py                           # sync / +audio / +rep / all - rep / synthetic only x 5 folds
python scripts/pilot/phase5/report_pair_sources.py                    # VCI, paired differences, pair counts -> outputs/results/phase5_sources/

# LOSO runtime estimate (nothing is trained)
python scripts/pilot/phase5/loso_estimate.py                          # -> outputs/results/phase5_loso_estimate/
```

| Piece | Code | Notes |
|---|---|---|
| view restriction | `cvdc/train/trainer.py` (`with_views`), `PackData.subject_mask(subjects, views)` | every training input from the training views only (E24) |
| cross-view runner | `cvdc/eval/cross_view.py` | 3 rotations × methods × folds, resumable; CVDC always pretrained on the two training views |
| VCI pooling | `cvdc/eval/consistency.py` (`pooled_vci`) | within-stratum VCI over folds with bootstrap replicates and an optional pair filter |
| reports | `scripts/pilot/phase5/report_crossview.py`, `report_pair_sources.py` | `summary.md` + CSVs; `--smoke` reads the smoke runs |

Measured runtimes (one job at a time): cross-view 3.9 h (Proposed 2.4 h, each baseline ~0.5 h), pair sources 1.8 h
(the 'all' setting reuses the step (a) encoders), distance + reports ~8 min. Results: `PROGRESS.md` §6e–6g.

## Proposal completion (2026-10-22)

The items of the proposal that were missing or partial (`reports/FINAL_REPORT_E01_E04.md` §4–§7, tables
`reports/completion_tables.md`, decisions D19, A11, M19, M20, E31, E32). All commands from the repo root; the full
list is in the final report §9.

```bash
python scripts/full/completion/classification_conditions.py   # precision / recall / F1 / accuracy, recording conditions
python scripts/full/completion/temporal_stability.py          # jitter, jitter variance, smoothness
python scripts/full/completion/encode.py --tag perex_gkf5 --z # GPU: CVDC alignments + embeddings of every fold model
python scripts/full/completion/depth_check.py                 # is the CVDC embedding depth-aware?
python -m cvdc.assess.run --config configs/completion.yaml --exercises E01 E02 E03 E04   # Module III + M19 + M20
python scripts/full/completion/module3_comparison.py          # before / after
python scripts/full/completion/report_completion.py           # assemble reports/completion_tables.md
```

DepthAnything V2 Small (A11; Apache-2.0): `git clone https://github.com/DepthAnything/Depth-Anything-V2
third_party/Depth-Anything-V2` (commit a561b84) and `hf download depth-anything/Depth-Anything-V2-Small
depth_anything_v2_vits.pth --local-dir data/models/depth_anything`. The extraction, packing and the B2-DepthAnything
training (GroupKFold-5 and LOSO) run as one hidden, resumable job: `explorer.exe
D:\cb\cvdc-physio\scripts\full\completion\run_depth_chain.cmd` (log and status in
`outputs/full/results/completion/`), then `python scripts/full/completion/depth_baseline_report.py`.

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
cvdc/eval/          consistency (VCI, pooled within-stratum VCI), splits, metrics (bootstrap CIs, ICC), baselines (set
                    features, video-level baselines), recording (reps contained per view), cross_view (Phase 5 runner)
cvdc/train/         trainer, pretrain_cvdc, train_supervised, run (GroupKFold runner)
cvdc/viz/           skeleton drawing, Phase 1 checks
scripts/            run_phase1.py (Phase 1 for both scopes)
  pilot/            the E01/E02 pilot (frozen, D15): step0/, phase1/ checks and report, phase2/ (smoke, lifter_check,
                    bench), phase3/ (reports, label diagnostics), phase4/ (embedding distance, recording noise,
                    report), phase5/ (pair-source runner, cross-view distance, reports, LOSO estimate)
  full/             the per-exercise full scope: step0/ (full-data check), phase1/ checks, phase3/ (per-exercise
                    reports, embedding distance, group probe, cross-view), phase4/ (Module III report, rule
                    calibration), phase6/ (demo checks), loso/ (LOSO launch, status, report), final/ (final tables),
                    completion/ (the remaining proposal items, 2026-10-22)
demo/               live.py, app.py (Streamlit), export4.py, reference.py, readable.py
reports/            the final report and its generated tables
third_party/        local clones used by wrappers (gitignored)
tests/              pytest
data/  outputs/  runs/   gitignored artefacts except the demo's files (Quick start): the pilot's at the top level (data/index.csv, outputs/phase*, runs/phase*),
                    the full scope's under full/ (D15)
```

**Removed in the 2026-10-22 cleanup** (moved to `D:\cb\_to_delete\`, not deleted; empty that folder when you no longer
need it): the smoke-test runs and their outputs (`runs/full/*_smoke`, `runs/phase*_smoke`,
`outputs/full/results/*_smoke`), the stopped shared-model run (`runs/full/aborted_shared_model`,
`outputs/full/results/aborted_shared_model`), the E09 pair backup (`data/full/backup_pairs_e09_trunk`), the
full-resolution Hands test cache (`data/full/cache/hands_test_fullres`), the download-error markers of the first
partial download and the pytest caches. Only `scripts/full/loso/runtime_estimate.py` and
`scripts/full/phase3/runtime_estimate_perex.py` read any of them (their estimates are already saved); move the
folders back from `_to_delete` to re-run those two.
