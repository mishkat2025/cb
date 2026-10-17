# Design decisions

Every choice the proposal, the dataset paper or the data leave open, with one line of reasoning. Each will be configurable. The config keys are the intended YAML names; the config files are created in Phase 1.

**Status:**
- **decided:** approved by the supervisor, or fixed by the spec
- **proposed:** my recommendation, open to change
- **TBD:** to be set in the phase named

## Data and labels

| # | Decision | Reasoning | Config key | Status |
|---|---|---|---|---|
| D0 | **Scope: E01 and E02 only.** E03 and every other exercise are excluded from extraction, training and evaluation for now. Runtime discovery stays on, with an include list; an empty list means all discovered exercises. | Supervisor, second review (2026-10-17). Adding exercises later must need only a config change. | `data.include_exercises: [E01, E02]` | decided |
| D1 | Dataset root = the folder containing the repo (`D:\cb`). Exercise folders are discovered at runtime by `^E\d{2}\(`; folders starting with `__` (download-error markers) are skipped, and missing exercises are skipped without error. | The data arrives incrementally; adding an exercise must need no code change. | `data.root`, `data.exercise_dir_regex` | decided |
| D2 | Case-insensitive filename regex `E\d{2}_P\d+_A[FLR]_V(FL\|ML\|LL\|LJ\|HJ\|O\|LR)_G[MF]`. Unknown variation codes (e.g. CSV `VP`) are kept as `unknown` and reported. | Parses 1141/1141 files; the CSV has a lowercase typo (`Al`). | `data.filename_regex` | decided |
| D3 | Subject key = P-number. | IDs are unique across the expert (P01–24) and non-expert (P26–62) groups, with the same gender in every exercise, so no composite key is needed. | `data.subject_key` | decided |
| D4 | Main label = CSV "Rounding to 2 Decimal" (mean of 3 judges, 0–100). Item targets = mean of the 3 judges per EAAQ item (0–1). | This is the dataset's own aggregate, and it recomputes exactly. | `labels.score_col`, `labels.items` | decided |
| D5 | PF3 is not used as a separate target. | PF3 = 1 exactly when PF4 = 1 on every judge row. | `labels.items` | proposed |
| D6 | Quality class from score: ≥80 Correct, 50–79 Partially correct, <50 Incorrect. This is our own derivation, not a dataset label. | Spec default. Local balance: E01 170/284/104, E02 149/208/27, E03 84/97/18. | `labels.quality_bins: [50, 80]` | decided |
| D7 | Drop `E01_P60_AF_VLR_GF` (a re-encoded copy of `E01_P60_AF_VML_GF`). | Identical audio at lag 0; keeping it double-counts one performance. | `data.exclude_files` | decided |
| D8 | E03 P02 (wrong exercise in all 18 videos, PF1 = 0) stays in score regression, is excluded from templates, and exercise-classification results are reported with and without it. **Applied only when E03 is in scope (D0).** | Its low scores are real labels, but the movement isn't the intended exercise. | `data.template_exclude_subjects` | decided (for later) |
| D9 | Read frames with real presentation timestamps and honour the rotation flag. Never read container GPS tags. | Frame rates vary (24.9–60, slightly variable); some files carry location tags. | `video.use_pts` | decided |
| D10 | Privacy: keypoints only. No raw frames, crops, faces or audio are saved; renders are skeleton-only; `data/` and `outputs/` are gitignored. | The released videos show faces. | `.gitignore` | decided |

## Pose extraction and preprocessing

| # | Decision | Reasoning | Config key | Status |
|---|---|---|---|---|
| P1 | MediaPipe 1.1.1 Tasks `PoseLandmarker` (VIDEO mode), full model by default (heavy optional). Save 2D (x, y, visibility) and world landmarks. Add `HandLandmarker` for E06 later. | Legacy `solutions` no longer exists in 1.1.1. Full model ≈ 36 fps per CPU process. | `pose.model`, `pose.hands_for` | decided |
| P2 | Downscale frames to 960 px on the long side before pose estimation. Check the effect against full resolution in Phase 1. | 1080p and 4K decode are the bottleneck; the pose model's input is 256 px anyway. | `pose.max_side` | proposed |
| P3 | Check whether padding non-square frames to square changes the landmarks (MediaPipe "NORM_RECT" warning). | Could distort x/y on portrait frames. | `pose.pad_square` | TBD Phase 1 |
| P4 | Joint set: 33 MediaPipe landmarks vs a 17-joint COCO-style subset. | Spec leaves it open. | `model.joint_set` | TBD Phase 1 |
| P5 | Root = mid-hip (mid-shoulder if hips are often out of frame); scale = torso length; interpolate low-visibility joints; light smoothing; fixed T for the network, original length kept for DTW. | Spec; the hip-visibility check comes first. | `preprocess.*` | TBD Phase 1 |
| P6 | Rep segmentation from each exercise's primary angle (peaks/valleys with prominence and minimum distance). Validate against PF4 and the EAAQ "≥10 reps". | Videos hold whole sets; scores are per video. | `reps.*` per exercise | TBD Phase 1 |

## Views and synchronization

| # | Decision | Reasoning | Config key | Status |
|---|---|---|---|---|
| S1 | Classify each group as synced / not-synced / unknown using **audio, then container creation_time, then motion and rep timing**. Store lag, confidence and method in `outputs/sync_report.csv`. | Supervisor change 2; the dataset can't be changed. | `sync.order` | decided |
| S2 | Audio: GCC-PHAT on 8 kHz mono, 100–3800 Hz, ±3 s window. Group-level proof = lag additivity, \|lag_FL + lag_LR − lag_FR\| ≤ 20 ms. | Peak height is weak (quiet room), but peak location matched motion on all 23 motion-confirmed pairs; additivity rejects chance peaks. Passes for 36% of expert groups. | `sync.audio.*` | decided |
| S3 | Creation time: \|ΔC\| ≤ 3 s on all pairs → synced (medium confidence). Recording windows that can't overlap (\|ΔC\| > dur_a + dur_b) → not-synced (high). | No control pair was ever ≤ 3 s (minimum 4 s); ~99% of expert pairs are. E01 non-expert pairs are 11–97 s apart. | `sync.meta.max_dc_s: 3` | decided |
| S4 | Refine lags with motion inside ±0.3 s of the device-corrected audio lag. Pass if the rep-timing residual is under 42 ms (the smallest control value). | Audio and video clocks differ per phone; motion gives the frame-level lag. | `sync.motion.*` | proposed |
| S5 | Per-device audio-to-video offsets (Front→Left +126 ± 28 ms, Front→Right +36 ± 51 ms in Step 0), re-estimated on all motion-confirmed pairs. | A constant per-phone offset was seen. | `sync.device_offsets` | proposed |
| S6 | Number of views is configurable (3 now: F/L/R; the proposal's 4 later). | The dataset has 3 views, not 4. | `views.names` | decided |

## CVDC (contrastive pretraining)

| # | Decision | Reasoning | Config key | Status |
|---|---|---|---|---|
| C1 | Positive pairs at **rep level**: (a) synced groups, reps that coincide after alignment, `w_sync = 1.0`; (b) not-synced or unknown groups, reps paired by index within the same (exercise, subject, variation) and aligned by DTW on 3D joint angles, `w_rep = 0.5`; (c) synthetic rotated views of each rep, `w_syn = 1.0`. | Supervisor change 2. Reps are the unit at which views are comparable. | `cvdc.pairs.*` | decided |
| C2 | Groups synced only by creation time get `w_sync` only if motion refinement (S4) confirms the alignment; otherwise they fall back to rep matching (`w_rep`). | Creation time proves the recordings were simultaneous but not the frame lag. | `cvdc.pairs.meta_requires_motion: true` | decided |
| C3 | Loss = weighted multi-positive InfoNCE (SupCon-style), τ = 0.1. Negatives: `all_other_groups` (default) or `different_exercise_only`. | Spec and supervisor. | `cvdc.loss.*` | decided |
| C4 | Synthetic views: lift to 3D (MediaPipe world landmarks), rotate about the vertical axis (0°, ±45°, ±90° plus jitter), project with a pinhole camera. | Spec fallback, now one of the three pair sources. | `cvdc.synthetic.*` | decided |
| C5 | Ablations: (a) synced only, (b) + rep-matched, (c) + synthetic, (d) synthetic only; plus τ, embedding size, 2 vs 3 views, augmentation on/off. Report pair counts per source and label every result. | Supervisor change 2.5 and the spec. | `ablation.*` | decided |
| C6 | Embedding 256-d, projection 128-d L2-normalised; ST-GCN with 9–10 blocks. | Spec defaults. | `model.*` | decided (defaults) |
| C7 | Pretrain CVDC within each fold on training subjects only. If LOSO is too slow, pretrain once per GroupKFold fold and give each LOSO subject the encoder from the fold where it was held out. | No leakage. Supervisor change 4. | `eval.cvdc_pretrain_scope` | decided |

## Targets and training

| # | Decision | Reasoning | Config key | Status |
|---|---|---|---|---|
| T1 | Training unit: reps (each inherits its video's score as a weak label) by default; whole videos for comparison. | Rep level weakens the duration confound (E01 Spearman score–duration 0.64). | `train.unit: rep\|video` | decided |
| T2 | Rep-level primary targets: total score + **CF1–CF3**. PF items are reported but only learned at set level. | Speed and rep count can't be judged from one rep (supervisor change 3). | `train.targets` | decided |
| T3 | Report total, PF and CF separately. Baselines: predict-the-mean and duration-only. | Supervisor change 3. | `eval.baselines` | decided |
| T4 | Skip a target head in a fold where that target has zero variance in training (e.g. E03 CF3 = 1.0 everywhere). | Nothing to learn; it avoids degenerate loss terms. | `train.skip_constant_targets` | proposed |
| T5 | One shared encoder and heads with an exercise embedding (not per-exercise models). | Small data per exercise; still works as exercises are added. | `model.exercise_embedding` | decided |
| T6 | Weighted score loss against the skew toward high scores. | Spec: scores are heavily skewed. | `train.score_loss_weighting` | TBD Phase 3 |

## Evaluation

| # | Decision | Reasoning | Config key | Status |
|---|---|---|---|---|
| E1 | Subject-independent splits. GroupKFold(k = 5, balanced for expert/non-expert) for development and ablations; **LOSO** for the final B1/B2/Proposed table. Metrics are pooled over folds; paired tests on per-fold values; 95% CIs and effect sizes. | Spec and approved plan. Tiny LOSO folds (single-rep non-experts) make per-fold correlations meaningless, hence pooling. | `eval.split` | decided |
| E2 | Estimate LOSO runtime from the smoke test before running it (fallback C7). | Supervisor change 4. | – | decided |
| E3 | Inter-view label-noise floor (RMSE of each view's score vs its group mean, expert groups): E01 5.28, E02 4.38 (E03 3.14 on the partial data, out of scope for now), reported next to every RMSE. | Supervisor change 4. These groups are the same performance (S3). | – | decided |
| E4 | Stratified reporting by expert/non-expert, exercise, variation and view. | View is confounded with expertise; scores are skewed. | `eval.strata` | decided |
| E5 | **Embedding-to-reference distance:** reference embedding = mean of the PT demo's rep embeddings (mirrored where needed). Per patient rep, report cosine and L2 distance: in the per-rep JSON and live overlay; as Spearman with the expert score and as a single-feature regressor for B1, B2 and Proposed; and with reference points on the UMAP/t-SNE plots. | Supervisor change 9: tests whether CVDC makes the proposal's "vector difference" meaningful. | `assess.embedding_reference.*` | decided |
| E6 | **Headline results:** score prediction (total, PF, CF), the B1/B2/Proposed comparison and the CVDC ablation, the cross-view protocol, the embedding-to-reference distance, and Module III. Exercise classification (2 classes) is reported but is not a headline result. | With only two shoulder exercises, classification is near-trivial (supervisor, second review). | – | decided |
| E7 | Every report states the scope limitations: 2 shoulder exercises only; E02 has no non-experts; the expert/non-expert contrast exists only in E01 and is confounded (single-rep, front-heavy, unsynced non-expert clips). | Keeps the claims honest (`PROGRESS.md` §0). | – | decided |
| E8 | MPJPE is computed only if a 3D reference exists (e.g. triangulated from synced views); otherwise it is stated as not computed. | Spec. | `eval.mpjpe` | TBD Phase 5 |

## Module III (assessment) and templates

| # | Decision | Reasoning | Config key | Status |
|---|---|---|---|---|
| M1 | Primary template = the PT demo video per exercise: resampled to 30 fps (most demos are 20 fps), mirrored for the left arm, using **3D angles** so a front-view template serves side views. | Supervisor change 5. | `templates.source: pt_demo` | decided |
| M2 | Tolerance bands (mean ± k·std) and the **speed reference** come from expert videos scoring ≥ 95 (E01 48, E02 46, E03 18 so far). | One demo gives no spread; the E03 demo is ~2× slower than dataset executions. | `templates.min_score: 95`, `templates.k` | proposed |
| M3 | E03 template = `Later Rotation V3.mp4`: both arms (left mirrored), working phase only (shoulder abduction ≥ 70°). | Closest to dataset posture (mean Chamfer 7.5–10.8° vs 16.7° for v1 and 55–64° for V2, which keeps the arms at the side). The dataset does right-arm 90/90 external rotation. **Applied only when E03 is in scope (D0).** | `templates.files.E03` | decided (for later) |
| M4 | E01 and E02 templates = `Abduction .mp4` and `Aduction.mp4`; the combined `Abduction Adduction.mp4` is a backup. | Separate single-exercise demos exist; all are front view, right arm. | `templates.files` | proposed |
| M5 | E01 rules from its EAAQ: 1.5–2.5 s per raise or lower; ≥10 reps; right arm 0 → 170–180°; other arm at the side; elbows straight; neutral trunk; legs straight. Rules for other exercises wait for the EAAQ appendix; their YAML stubs are marked `TODO: verify when data is available`. | Only E01's EAAQ is in the paper. | `assess.rules.E01` | decided / pending |
| M6 | Angle source: prefer 3D world landmarks, check per-view reliability, record `angle_source` in every JSON. | Step 0 found unstable 3D forearm azimuth in side views. | `assess.angle_source` | proposed |
| M7 | Joint-angle definitions per exercise, score formula and error thresholds. | Spec starting table, refined against data. | `assess.*` | TBD Phase 4 |

## Demo

| # | Decision | Reasoning | Config key | Status |
|---|---|---|---|---|
| X1 | **No webcam.** `demo/live.py --source <video>` plays files at real-time speed, frame by frame, like a camera: skeleton, rep counter, per-rep score, Module III errors and embedding-to-reference distance, all live. | No physical camera on this machine (supervisor change 1). | `demo.*` | decided |
| X2 | Pick a dataset video by parsed fields (exercise, subject, view, variation); warn if the subject was in the loaded checkpoint's training split; optional annotated .mp4 export. | Supervisor change 1. Demo videos should be held-out subjects. | `demo.select`, `demo.save_mp4` | decided |
| X3 | Runs without a checkpoint (rule-based Module III only). | Spec. | – | decided |

## Environment

| # | Decision | Reasoning | Status |
|---|---|---|---|
| V1 | Python 3.11 venv at `D:\cb\.venv` (outside the repo); torch 2.14.1+cu126 (latest stable with a CUDA 12.x build); versions pinned in `requirements.txt`. | MediaPipe and the scientific stack are mature on 3.11; the CUDA 12.x build was approved. | decided |
| V2 | MediaPipe runs on CPU (no GPU delegate in Python on Windows); extraction is parallelised over processes. | ~290 fps total with 8 processes. | decided |
