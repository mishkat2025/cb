# Progress

Last updated 2026-10-17.

**State:** Step 0 is complete for E01, E02 and the partial E03. The plan was approved with the supervisor's changes (§4). **Phase 1 has not started**; it will start in a new session. Nothing in `cvdc/` exists yet.

Read this together with `DESIGN_DECISIONS.md` (every open choice, with its status) and the task specification `../CVDC_implementation_prompt.md`.

---

## 1. Environment

| Item | Value |
|---|---|
| OS / CPU / RAM | Windows 11 Pro, AMD Ryzen 7 7700 (8C/16T), 31 GB |
| GPU | NVIDIA RTX 4060 Ti 16 GB (driver supports CUDA 13.4) |
| Python env | `D:\cb\.venv` (Python 3.11.9, made with `uv`, lives **outside** the repo) |
| Installed | torch 2.14.1+cu126 (GPU matmul verified), mediapipe 1.1.1, opencv-contrib 5.0, PyAV 18.1, numpy, scipy, pandas, matplotlib (`requirements.txt`) |
| Not yet installed | scikit-learn, umap-learn, tensorboard, pyyaml, streamlit, DTW library, pytest (install and pin in Phase 1) |
| MediaPipe | 1.1.1 has only the **Tasks API** (`PoseLandmarker`); there is no `mp.solutions`. Models live in `data/models/` (gitignored): `pose_landmarker_lite.task`, `pose_landmarker_full.task`. The heavy pose model is approved when needed; the hand landmarker comes later for E06. |
| Pose speed | CPU only (no GPU delegate in MediaPipe Python on Windows): ~36 fps per process at 960 px with the full model; 8 processes ≈ 290 fps |
| Camera | **No physical webcam** (only "Camo", a virtual camera in error state), so the demo runs on video files (§4, change 1) |
| Disk | D: ~730 GB free; local videos ~42 GB |
| Git | this repo (`cvdc-physio/`); identity `mishkat <sbmishkat10@gmail.com>` |

## 2. Data on disk

Dataset root = `D:\cb` (the folder that contains this repo). Never modify it.

| Folder / file | Content | Status |
|---|---|---|
| `E01(Abduction)` | 558 videos | complete (matches Dataverse) |
| `E02(Aduction)` | 384 videos | complete |
| `E03(Lateral_Rotation)` | 199 of 550 videos | **partial**: experts P01–P12 (P04 missing `E03_P04_AL_VLL_GM`), P13 has 8 of 18 videos. P14–P24 and all 27 non-experts are missing. |
| `__E03(Lateral_Rotation)` | `E03_P04_AL_VLL_GM.mp4_Error.txt` | download-error marker ("service temporarily unavailable"); not a data folder |
| `Reference-Model Videos by PTs` | 14 physiotherapist demos | see §3.6 |
| `Physiotherapist Exercise Marking.csv` | scores, 3 judges × 7 EAAQ items | all 3,686 videos |
| `MobiPhysio.md` | dataset README | names E08 "Both Leg Flexion" (the paper says Lumbar Flexion) |

More exercises are still downloading. **Don't touch a folder until the user announces it**, then run the Step 0 check on it (§7).

## 3. Step 0 findings

### 3.1 Files and labels
- **Filenames:** 1141/1141 parse (100%), with the case-insensitive regex `E\d{2}_P\d+_A[FLR]_V(FL|ML|LL|LJ|HJ|O|LR)_G[MF]`.
- **Subject IDs are unique and stable across exercises:** P01–P24 are experts (recorded in 2024), P26–P62 are non-experts (recorded in 2022). Each P-number has the same gender everywhere. IDs 25, 44, 51 and 54 were removed by the curators. **Subject key = P-number.**
- **CSV:** 3 rows per video (one per judge) × 7 EAAQ items (PF1–PF4, CF1–CF3), plus total, 0–100 scale and average. Every total and average recomputes exactly.
  - All 1141 local files match a CSV row (case-insensitive).
  - CSV typos: `E01_P26_Al_VFL_GF` (lowercase l; handled) and `E03_P41_AF_VP_GM` (unknown variation "VP"; not local yet).
  - Judge names are written "Dr" in the expert batch and "Dr." in the non-expert batch.
- **Item scales:** PF1 and PF3 are binary; PF2, CF1 and CF2 take 0/0.5/1; PF4 runs 0–1 in 0.1 steps; CF3 takes 0/0.5/0.6/0.66/1.
  - **PF3 = 1 exactly when PF4 = 1** (redundant).
  - PF4 is not a rep count: 117 E02 videos of ~28 s score 0.

| Exercise (local) | Videos | Subjects | Mean score | Bins ≥80 / 50–79 / <50 | Inter-view label noise: mean range / RMSE floor* |
|---|---|---|---|---|---|
| E01 expert | 382 | 24 | 80.1 | 170 / 284 / 104 (both groups) | 8.5 / **5.28** |
| E01 non-expert | 176 | 27 | 44.5 | (included above) | not the same performance |
| E02 expert | 384 | 24 | 74.1 | 149 / 208 / 27 | 5.8 / **4.38** |
| E03 expert (partial) | 199 | 13 | 73.8 | 84 / 97 / 18 | 3.9 / **3.14** |

\*RMSE of each view's score against its own group mean, over expert groups. These groups are simultaneous recordings of one performance (§3.5), so this is irreducible view and rater noise. Report it next to every RMSE.

### 3.2 Videos
- **Frame rate:** 24.9–60 fps, mostly ~30 and slightly variable. Use real frame timestamps, not the frame index divided by fps.
- **Resolution:** mostly 1080×1920 with a 90° rotation flag. Some are 4K, 1440×1440 or 352×640 (the LR variation).
- **Duration:** E01 non-expert videos are **single reps of 1–7 s (median 3 s)**. Expert videos are full sets: E01 11–54 s, E02 19–41 s, E03 16–39 s.
- **Containers are original phone files:** they keep `creation_time` (1 s resolution), the Android version and the capture fps. The expert-session Left camera is tagged as a Pixel 6a. **Some files carry GPS location tags: never read, log or copy them.**
- **Audio** is present in all 1141 videos (48 kHz stereo).
- **`E01_P60_AF_VLR_GF` is a re-encoded copy of `E01_P60_AF_VML_GF`** (identical audio at lag 0, no creation_time). The other 14 LR videos are separate recordings.

### 3.3 Confounds and data issues
1. **Duration / rep-count confound (E01):** Spearman between score and duration is 0.64, because non-experts are single-rep clips and experts are full sets. The CSV shows the same structure for the E03 and E05 non-experts, who aren't downloaded yet.
2. **View is confounded with expertise:** non-experts have all three views only for FL; their other variations are front-only. So E01's front-view mean is 62.7 against ~74–75 for the side views.
3. **Inter-view label noise:** floors in §3.1.
4. **E03 P02 did the wrong exercise in all 18 videos** (PF1 = 0, scores 17–29). P02's mean is 22.6; every other E03 subject is 69–85.
5. **Near-constant items:** E03 CF3 (leg alignment) is 1.0 on all 199 videos, and E01 PF1 averages 0.99.
6. **Some participants use the left arm:** E03 P10 ML, in all three views. Templates need mirroring.
7. **Side-view 3D is shaky:** MediaPipe world landmarks from side views give an unstable forearm azimuth (ROM above 180° in E03). Check per view whether 3D angles are reliable.
8. **Non-square frames:** MediaPipe warns "NORM_RECT without IMAGE_DIMENSIONS". Check whether padding frames to square changes the landmarks.
9. **Low-light Left views** can have poor detection (42–74% of frames in the sample).

### 3.4 E03 check (requested 2026-10-17)
- **Coverage:** 199/199 parse, and all 199 have scores. The 351 E03 CSV rows without a local file are the missing videos listed in §2.
- **Composition:** 13 subjects (P01–P13; P13 incomplete), all experts. Views 70/65/64 (F/L/R). Variations FL 38, ML 37, LL 36, O 37, HJ 26, LJ 25. 63 complete F+L+R groups.
- **Scores:** mean 73.8, median 77.1. The histogram is bimodal only because of P02 (wrong exercise). ≥95: 18 videos. PF3/PF4 = 1 on only 10.6%.
- **Duration:** 16–39 s (median 24). Spearman with score 0.34 (experts only). The non-experts are expected to be short clips, as in E01; verify when they arrive.
- **Views:** balanced for experts. Expect front-heavy non-expert data, as in E01; the CSV list shows the same pattern.
- **Sync:** see §3.5. 18 groups synced by audio, 47 by creation time, 1 unknown.
- **Template:** see §3.6.

### 3.5 Synchronization: all multi-view groups of E01–E03
**Evidence, in the approved order:**
1. **Audio:** GCC-PHAT cross-correlation (8 kHz mono, 100–3800 Hz, ±3 s window). A group counts as proven when its three lags are additive, |lag_FL + lag_LR − lag_FR| ≤ 20 ms. Audio is kept in memory only.
2. **Container `creation_time`.**
3. **Motion:** first principal component of the wrist trajectories, plus rep timing taken at the mid-level crossings, measured as the leftover timing difference after a linear fit. Pass if under 42 ms, the smallest value of any control.

Controls are the same person in a different variation, which are certainly separate recordings.

**Results**
- **creation_time:**
  - Same-group pairs: |ΔC| ≤ 3 s in 936 of 1006. **Every expert pair is under 10 s, and ≥935 of 946 (~99%) are ≤ 3 s.**
  - Controls are never ≤ 3 s (minimum 4 s, median ~15 min).
  - **E01 non-expert views were recorded one after another:** 11–97 s apart, with windows that cannot overlap.
- **Audio:**
  - The peak is weak in height (peak / second peak ≈ 1.1), but its **location is right**. On all 23 pairs the motion test confirmed, audio and motion lags agree within 0.17 s.
  - The lag-additivity check passes for 36% of expert F+L+R groups (median error 32 ms).
  - Each phone has its own audio-to-video offset relative to the Front phone: Front→Left +126 ± 28 ms (Pixel 6a), Front→Right +36 ± 51 ms (n = 8 and 11). Re-estimate on all data in Phase 1.
- **Motion:** E01 Front–Right pairs are at 11–37 ms in 8 of 8. Pairs with a Left view or from E02 are often noisy. In the first Step 0 report, E02 looked "not shown synced"; audio and creation time show this was a weak 2D signal, not a lack of synchronization.

| Groups (≥ 2 views) | synced: audio (high) | synced: creation time (medium) | not synced (creation time) | unknown |
|---|---|---|---|---|
| E01 expert | 49 | 78 | 0 | 2 |
| E01 non-expert | 0 | 0 | 22 | 0 |
| E02 expert | 45 | 82 | 0 | 2 |
| E03 expert (partial) | 18 | 47 | 0 | 1 |

**Conclusion:** the expert sessions were filmed simultaneously by three phones, with lags under 2 s. The E01 non-expert views were not. Files: `outputs/step0/sync_groups_classified.csv` and `av_pairs.csv` (gitignored).

### 3.6 Reference videos (PT demos)
One physiotherapist, one view per video, ~1–4 reps each. Most are 20 fps (resample to 30). There is no score file, and they are not on Dataverse.

| File | Exercise | fps / s | Apparent view | Moving limb |
|---|---|---|---|---|
| `Abduction .mp4` | E01 | 20 / 9.5 | front | right arm |
| `Aduction.mp4` | E02 | 20 / 6.7 | front | right arm |
| `Abduction Adduction.mp4` | E01 + E02 combined | 30 / 16.2 | front | right arm |
| `Lateral Rotation.mp4` | E03 (v1) | 20 / 12.2 | oblique | right arm only |
| `lateral rotation V2.mp4` | E03 (v2) | 20 / 10.2 | front, portrait | both arms, **upper arms at the side** (abduction ~15°) |
| `Later Rotation V3.mp4` | E03 (v3) | 20 / 12.2 | front | both arms, **90/90 position** (abduction ~83°) |
| `Medial Rotation.mp4` | E04 | 20 / 10.6 | oblique | right arm |
| `Circumduction.mp4` | E05 | 20 / 14.9 | side | right arm |
| `Wrist Extension.mp4` | E06 | 30 / 29.0 | front-oblique | both wrists |
| `Hip Joint Flection.mp4` | E07 | 20 / 11.4 | side, lying | right leg |
| `Both Leg Flection.mp4` | E08 | 20 / 14.3 | side, lying | legs |
| `Back Extension.mp4` | E09 | 20 / 16.8 | side, prone | trunk / arms |
| `Straight Leg Rising.mp4`, `Wrist Flexion.mp4` | not among the 9 (phase-I exercises) | 30 | – | – |

**How the E03 versions differ, and which is proposed:**
- The dataset executes E03 as **right-arm external rotation in the 90/90 position**: shoulder abduction ~89° (p5–p95 71–101°), elbow flexed, forearm rising from about −16° to +52°, ~10 reps at ~2.3–2.5 s each.
- Posture-space distance to high-scoring front-view dataset frames (mean Chamfer distance over 3D abduction, elbow and forearm-elevation angles):
  - **V3: 7.5–10.8°**
  - v1: 16.7°
  - V2: 55–64°
- **Proposed template: V3.** Use both arms (left mirrored) and keep only the working phase (abduction ≥ 70°).
- V3 is ~2× slower than the dataset (~5.4 s per rep), so the **speed reference comes from expert videos**, not from the demo.
- The pool behind this comparison is only 4 front-view videos scoring ≥ 80. Re-check when the rest of E03 arrives.

## 4. Approved plan (supervisor review, 2026-10-17)
The base plan is the Step 0 proposal: subject = P-number; GroupKFold (k = 5, balanced for expert/non-expert) for development and ablations; LOSO for the final comparison; rep-level training by default; quality bins 80/50; one shared encoder with an exercise embedding. Changes:

1. **No webcam.** `demo/live.py --source <video.mp4>` plays the file at real-time speed, frame by frame, like a camera stream. It shows the skeleton overlay, rep counter, per-rep score and Module III errors live, and:
   - lets you pick a dataset video by its parsed fields (exercise, subject, view, variation);
   - warns if that subject was in the loaded checkpoint's training split (use held-out subjects or outside videos);
   - can save the annotated output as .mp4.
2. **Sync solved in software:**
   1. Classify every F/L/R group as synced / not-synced / unknown, using audio, then creation_time, then motion and rep timing. Store the lag for synced groups and align frames.
   2. Segment every video into reps.
   3. Build rep-level positive pairs:
      - synced groups: reps that coincide after alignment, weight `w_sync = 1.0`;
      - not-synced and unknown groups: reps paired by index within the same (exercise, subject, variation), aligned with DTW on 3D joint angles, weight `w_rep = 0.5`;
      - synthetic rotated views of each rep, weight `w_syn = 1.0`.
   4. The CVDC loss is a weighted multi-positive InfoNCE using these per-pair weights.
   5. Ablation: (a) synced pairs only, (b) + rep-matched, (c) + synthetic, (d) synthetic only. Report pair counts per source and label every result with its source mix.
   6. Save a sync report (per-group class, lag, confidence, method) in `outputs/`.
3. **Targets:**
   - For rep-level training, **CF1–CF3 are primary targets alongside the total score**, because PF items (speed, rep count) can't be judged from one rep.
   - Report total, PF and CF separately.
   - Keep the duration-only and predict-the-mean baselines.
4. **Evaluation cost:**
   - Estimate LOSO runtime first, from the smoke test.
   - If it's too heavy: run CVDC pretraining once per GroupKFold fold. Each LOSO subject then uses the encoder from the fold where that subject was held out (no leakage). Document this.
   - Report the inter-view label noise as a reference floor next to RMSE.
5. **Templates:** resample PT demos to 30 fps, mirror for the left arm, and use 3D angles so a front-view template works for side views.
6. **E03:** run the Step 0 check (done, §3.4). Repeat for each newly announced exercise.
7. **Approvals:** PyTorch (done), git repo (done), MediaPipe heavy pose model (when needed), hand model (later, for E06). The user is looking for the EAAQ appendix.
8. **Docs:** write this file and `DESIGN_DECISIONS.md`, commit, report the E03 check, then **stop**.
9. **Embedding-to-reference comparison** (the paper's "vector difference"):
   - Encode each PT reference video (per exercise, all its reps, mirrored where needed) with the trained encoder. Reference embedding = mean of its rep embeddings.
   - For every patient rep, compute the cosine and L2 distance to its exercise's reference embedding:
     - include it in the per-rep JSON and the live demo overlay;
     - report how well this distance alone tracks the expert score (Spearman, and as a single-feature regressor) for B1, B2 and Proposed, to test whether CVDC makes the vector difference more meaningful;
     - show reference vs. patient embeddings on the UMAP/t-SNE plots.

## 5. Phase checklist
- [x] **Step 0:** environment, inventory, labels, sync census, references (E01, E02, E03-partial)
- [ ] **Phase 1, keypoints and preprocessing** (new session)
  - [ ] `cvdc/data/`:
    - filename parser (runtime discovery of `E\d{2}(...)` folders; skip `__*` folders and unannounced ones);
    - CSV join;
    - index CSV.
  - [ ] Keypoint extraction for all local videos:
    - Tasks API; 2D (x, y, visibility) and world landmarks;
    - real frame timestamps, rotation handled, GPS never read;
    - square-padding check (§3.3 item 8);
    - per-variation detection rate.
  - [ ] Sync classifier (audio → creation time → motion) for every group:
    - lag refinement with motion within ±0.3 s of the device-corrected audio lag;
    - `outputs/sync_report.csv`.
  - [ ] Preprocessing:
    - root-centring (check hip visibility), torso scaling, interpolation, smoothing;
    - resample to fixed T, keeping the originals.
  - [ ] Rep segmentation for every video:
    - reps per video;
    - plots;
    - compare against PF4 and the EAAQ "≥10 reps".
  - [ ] Skeleton-only check renders in `outputs/checks/`.
- [ ] **Phase 2, models:**
  - ST-GCN with shared weights;
  - projection head;
  - task heads (score, CF1–CF3, PF, exercise, quality);
  - weighted multi-positive InfoNCE;
  - pair builder for the sync / rep / synthetic sources;
  - synthetic views;
  - B2 3D input.
- [ ] **Phase 3, training:**
  - smoke test (3 subjects, 2 epochs);
  - **LOSO runtime estimate**;
  - CVDC pretraining and fine-tuning;
  - B1 and B2.
- [ ] **Phase 4, Module III:**
  - templates (PT demo at 30 fps, mirrored, 3D angles; tolerance bands from expert videos scoring ≥ 95);
  - DTW, angle and angular-velocity deviation;
  - rules;
  - score;
  - JSON with the embedding-to-reference distance.
- [ ] **Phase 5, evaluation:**
  - GroupKFold and LOSO;
  - stats;
  - cross-view protocol;
  - ablations (a)–(d) plus τ, embedding size, augmentation;
  - label-noise floor;
  - baselines;
  - embedding-distance analysis;
  - UMAP/t-SNE with reference embeddings.
- [ ] **Phase 6, demo:**
  - `demo/live.py` on video files (real-time playback, field picker, leakage warning, .mp4 export, embedding-distance overlay);
  - `demo/app.py`.

## 6. Open items
- **EAAQ appendix** for E02–E09: the user is looking for it. Until it arrives, only E01 has questionnaire-derived rules.
- **E03 is incomplete** (199/550, including one failed download). Rerun the check when the user announces the full folder.
- **Other exercises are downloading.** Run the Step 0 check (§7) for each one the user announces.
- **Suggested download order:** E05 (circumduction, 45 subjects, strongly 3D movement), then the rest.

## 7. Re-running the Step 0 check
See `scripts/step0/README.md` for the order and arguments. Minimum for a new exercise: `inventory.py` → `video_meta.py` → `scores.py` → `sync_av.py <EX>` → `sync_classify.py`, plus a pose sample (`select_groups_ex.py` → `extract_pose.py` → `sync_analysis3.py`) and a reference comparison.
