# Progress

Last updated 2026-10-19 (Phase 4).

**State:** Step 0 and **Phase 1 (keypoints and preprocessing) are complete for E01 and E02** (§5), including the supervisor's follow-up changes from the third review (§4, §5.6). The supervisor's warning on expert/non-expert confounding is part of the evaluation plan (§4, fourth review; `DESIGN_DECISIONS.md` E9–E12). **Phase 2 (models, CVDC loss, pair builder) is implemented and smoke-tested** (§6), including the supervisor's fifth-review changes (§4, §6.5: standardised targets, three VCI variants, the KTPFormer lifter for B2, bf16, the runtime estimate). The user's sixth-review decisions (§4) set the Phase 3 run order: **step (a), GroupKFold-5 for the 4 main methods, is done (§6b)**. **Step (a2), the score-setup diagnosis and fixes (label diagnostics, set-level baselines, video head, T6 weighting, stabilised early stopping, Proposed variants), is done (§6c).** The seventh review's decisions are recorded (§4: within-group score prediction is a negative result; Proposed = joint loss λ 1.0; no angle inputs to the heads; the movement-quality score as a secondary target). **Phase 4 (Module III with the PT demo as the reference, the embedding-to-reference distance and its view stability, the recording-noise test) is done for E01 and E02 (§6d)** and stops here for review; the ablations (b) and the LOSO table (c) have not started.

**Scope: E01 and E02 only** (supervisor, 2026-10-17). E03 is excluded from every phase for now; see §0.

Read this together with `DESIGN_DECISIONS.md` (every open choice, with its status) and the task specification `../CVDC_implementation_prompt.md`.

---

## 0. Scope and limitations

**In scope:** E01 (Abduction) and E02 (Adduction). That is 942 videos from 51 subjects: 24 experts, who did both exercises, and 27 non-experts, who did E01 only.

**Excluded for now:** E03 and every other exercise. Nothing is extracted, trained or evaluated on them.
- The E03 Step 0 results (§3.4) and its template decision (V3, §3.6) stay documented for later.
- The pipeline still **discovers exercises at runtime**, with an include list in the config (`data.include_exercises: [E01, E02]`). Adding E03 or others later is a config change, not a code change.

**Main results** (headline):
1. Score prediction: total score, plus the PF and CF items reported separately, **within groups first** (expert-only, non-expert-only), pooled second (E9).
2. The B1 / B2 / Proposed (ST-GCN + CVDC) comparison and the with/without-CVDC ablation.
3. The cross-view protocol: train on two views, test on the third.
4. The embedding-to-reference distance (the "vector difference", §4 item 9).
5. Module III assessment: rule-based errors, rule score vs expert score.

**Secondary:** exercise classification (E01 vs E02). It is reported, but it is not a headline result, because 2 classes are near-trivial.

**Limitations, to state in every report:**
- **Only 2 exercises, both shoulder movements in the same region.** Exercise classification is close to trivial. Findings may not transfer to other body regions, lying exercises (E07–E09) or the wrist (E06).
- **E02 has no non-experts.** All 384 E02 videos are from the 24 experts, scoring 24–100 (mean 74.1). So E02 has no expert vs non-expert contrast, and its low-score examples come only from imperfect expert takes.
- **The expert/non-expert contrast exists only in E01, and it is confounded.** The groups differ in score (~80 vs ~45) and also in clip length (non-experts: single-rep clips of 1–7 s), views (mostly front), recording year and location, phones and recording mode (non-expert views recorded one after another, not synchronized) (§3.3, §3.5). A model can look accurate by detecting the group, so within-group results are primary (E9–E12).
- **3 views, not the proposal's 4.** Synchronization is reconstructed in software (audio, creation time, motion), not given by the dataset.
- **No error-type labels.** Module III errors are rule-based and evaluated qualitatively, plus by how rule deviations correlate with the expert scores.
- **Scores are skewed toward high values.** There is an inter-view label-noise floor of 5.3 points (E01) and 4.4 points (E02) RMSE.
- **Small sample:** 51 subjects (50 at rep level: P39's only clip has no detected rep), and only 24 of them appear in E02.
- **E01 non-experts are almost all single-view for CVDC** (D12): 152 of their 181 rep sets have one view, with no sync/audio pairs and only 71 rep-matched pairs, so CVDC reaches them mostly through synthetic views. Every CVDC ablation reports them as a separate stratum.

---

## 1. Environment

| Item | Value |
|---|---|
| OS / CPU / RAM | Windows 11 Pro, AMD Ryzen 7 7700 (8C/16T), 31 GB |
| GPU | NVIDIA RTX 4060 Ti 16 GB (driver supports CUDA 13.4) |
| Python env | `D:\cb\.venv` (Python 3.11.9, made with `uv`, lives **outside** the repo) |
| Installed | torch 2.14.1+cu126 (GPU matmul verified), mediapipe 1.1.1, opencv-contrib 5.0, PyAV 18.1, numpy, scipy, pandas, matplotlib (`requirements.txt`) |
| Installed in Phase 1 | pyyaml, pytest, scikit-learn, umap-learn, tensorboard, streamlit, dtaidistance (pinned in `requirements.txt`) |
| Installed in Phase 2 | einops 0.8.2, timm 1.0.30 (KTPFormer wrapper); KTPFormer code in `third_party/` and weights in `data/models/ktpformer/` (both gitignored; no licence, A10) |
| MediaPipe | 1.1.1 has only the **Tasks API** (`PoseLandmarker`); there is no `mp.solutions`. Models live in `data/models/` (gitignored): `pose_landmarker_lite.task`, `pose_landmarker_full.task`. The heavy pose model is approved when needed; the hand landmarker comes later for E06. |
| Pose speed | CPU only (no GPU delegate in MediaPipe Python on Windows): ~36 fps per process at 960 px with the full model; 8 processes ≈ 290 fps (Step 0); Phase 1 full run: 12 processes ≈ 315 fps (712,585 frames in 37.6 min) |
| Camera | **No physical webcam** (only "Camo", a virtual camera in error state), so the demo runs on video files (§4, change 1) |
| Disk | D: ~730 GB free; local videos ~42 GB |
| Git | this repo (`cvdc-physio/`); identity `mishkat <sbmishkat10@gmail.com>` |

## 2. Data on disk

Dataset root = `D:\cb` (the folder that contains this repo). Never modify it.

| Folder / file | Content | Status |
|---|---|---|
| `E01(Abduction)` | 558 videos | complete (matches Dataverse) |
| `E02(Aduction)` | 384 videos | complete |
| `E03(Lateral_Rotation)` | 199 of 550 videos | **out of scope for now (§0).** Also partial: experts P01–P12 (P04 missing `E03_P04_AL_VLL_GM`), P13 has 8 of 18 videos. P14–P24 and all 27 non-experts are missing. |
| `__E03(Lateral_Rotation)` | `E03_P04_AL_VLL_GM.mp4_Error.txt` | download-error marker ("service temporarily unavailable"); not a data folder |
| `Reference-Model Videos by PTs` | 14 physiotherapist demos | see §3.6 |
| `Physiotherapist Exercise Marking.csv` | scores, 3 judges × 7 EAAQ items | all 3,686 videos |
| `MobiPhysio.md` | dataset README | names E08 "Both Leg Flexion" (the paper says Lumbar Flexion) |

More exercises are still downloading. **Don't touch a folder until the user announces it**, then run the Step 0 check on it (§9).

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

### 3.4 E03 check (requested 2026-10-17; kept for later, E03 is out of scope for now)
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
- **Template: V3 (confirmed by the supervisor, applied when E03 is brought into scope).** Use both arms (left mirrored) and keep only the working phase (abduction ≥ 70°).
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
10. **Scope: E01 and E02 only** (second review, 2026-10-17). E03 is excluded from all phases for now; runtime discovery stays. The main results are defined in §0, and the limitations are stated in §0.

**Confirmed in the second review** (now "decided" in `DESIGN_DECISIONS.md`):
- the V3 template for E03 (applied later);
- drop the duplicate `E01_P60_AF_VLR_GF`;
- the E03 P02 handling (applied later);
- groups synced only by creation time get `w_sync` only after motion confirms the alignment; otherwise they get `w_rep` (**superseded by the third review, answer 3:** unconfirmed views get audio-lag pairs with `w_audio`).

**Fourth review (2026-10-17): supervisor warning on expert vs non-expert scores.** Experts and non-experts differ in score (~80 vs ~45) and also in clip length, views, location, phones and recording mode, so a model can look accurate by detecting the group instead of judging movement. Added to the evaluation plan (`DESIGN_DECISIONS.md` E9–E12; recorded only, no code yet):
1. Report every score metric within groups (expert-only, non-expert-only) as the primary result; pooled results secondary (E9).
2. Add a **group-only baseline** (predicts each group's mean score) next to the mean and duration-only baselines (E10, T3).
3. Check whether the embeddings separate experts from non-experts more than they separate scores within a group: a linear probe for the group label (balanced accuracy, AUC) next to within-group score probes, for B1, B2 and Proposed, plus a view-matched subset (E11).
4. Keep rep-level training as the default so clip length cannot be a cue (T1, E12); the remaining cues (rep duration, rep count per set, view mix) are listed in E12.

**Seventh review (2026-10-19): decisions on step (a2)** (`DESIGN_DECISIONS.md` E21, T15, T16, D13):
1. **Within-group score prediction is reported as a negative result**, together with the label diagnostics (§6c.1). The step (a) and (a2) results stay in the repo and the reports; nothing is deleted (E21).
2. **Default Proposed = the joint loss with λ = 1.0** (`proposed_j1`; `eval.proposed_default`). The frozen CVDC encoder (`proposed_lp`) is kept as a reported alternative (`eval.proposed_alternative`) (T15).
3. **No angle / ROM inputs to the deep heads.** Angles and ROM belong to Module III (T16).
4. **The total score stays the official target.** A secondary **movement-quality score** = PF1 + PF2 + CF1 + CF2 + CF3, rescaled to 0–100 (100 × their sum / 5), is reported within groups (D13; `cvdc.data.labels.movement_quality`, `labels.movement_quality_items`). It leaves out PF3/PF4, the items that differ between views of one performance (E3).
5. Then Phase 4 (Module III, proposal Figure 2) for E01 and E02, with the PT demo as the reference ("truth"); **no LOSO and no ablations yet** (§6d).

**Sixth review (2026-10-18): decisions before Phase 3** (`DESIGN_DECISIONS.md` A10, E13, T8, E14):
1. **KTPFormer:** keep using it locally for non-commercial research; never commit or share its code or weights; cite the paper. MotionBERT's weights are also Human3.6M-trained, so switching would not remove the non-commercial limit. The user informs the supervisor.
2. **Headline VCI:** within each exercise and expert/non-expert group, mean-centred 256-d. The pooled value is secondary and reported with the random-encoder reference (0.41).
3. **Epochs:** at most 100 CVDC / 60 fine-tuning, with early stopping on an inner validation split taken from the training subjects only (never the test fold); stopping epochs are logged.
4. **Phase 3 run order:** (a) GroupKFold-5 for B1, B2-MediaPipe, B2-KTPFormer and Proposed: within-group results against the mean, group-only and duration-only baselines, plus the within-group VCI; **stop and report**. (b) Only after approval: the ablations on GroupKFold. (c) Then the final LOSO table (per-fold-encoder shortcut) for the 4 main methods.

**Fifth review (2026-10-17): answers to the Phase 2 questions** (applied in §6.5):
1. **Score loss:** standardise the score and the PF/CF targets with each training fold's mean and std only; invert for reporting; check the relative loss weights across heads.
2. **VCI:** report raw 256-d, mean-centred 256-d and the 128-d projection; the **primary VCI is the mean-centred 256-d embedding** (what the heads and the embedding-to-reference distance use); define it once in `DESIGN_DECISIONS.md` (E13).
3. **E01 non-experts mostly single-view:** accepted as a limitation (D12); the synthetic-views ablation must report them separately (C5).
4. **B2 3D input:** download KTPFormer weights and build the wrapper (MotionBERT as fallback, with reasons); report B2 with MediaPipe-3D and KTPFormer-3D. DepthAnything optional, later, on a small subset only. Check and record the licence of any downloaded weights (A10).
5. **bf16** mixed precision for training; re-measure step time (A9).
6. **Runtime estimate** for CVDC pretraining + fine-tuning, GroupKFold and LOSO with the per-fold-encoder shortcut (E2).
7. **Rep-matched pairs** stay at w_rep = 0.5, but the ablation reports with and without them (C5).

**Third review (2026-10-17): answers to the Phase 1 questions** (applied in Phase 1, §5.6):
1. **E02 rep signal** (wrist-path PC1) is approved for **rep segmentation only**. Module III must use joint angles: for E02 the **3D horizontal adduction angle** (upper arm vs shoulder line in the transverse plane, world landmarks).
2. **Left-arm performances:** mirror and assess normally. First check whether expert scores (total, PF, CF) are lower for them; flag `arm_side` in the Module III output only if they are. Record the result.
3. **Creation-time groups without motion confirmation:** use the audio-based lag for rep-level pairs, with its own weight **`w_audio = 0.8`** (configurable), and add it to the ablation as its own source.
4. **Speed:** tolerances from high-scoring expert data; keep the EAAQ 1.5–2.5 s as a documented alternative config.
5. **Rep consistency (before Phase 2):** in synced groups (audio / creation time / motion), rep boundaries must be identical across views: segment on the best view of the group (highest signal quality, usually Front), transfer the boundaries with the group's lag; per-view segmentation only for not-synced groups. Report the new cross-view agreement and how many reps per group change.

## 5. Phase 1 results (2026-10-17)
How to run: `python scripts/run_phase1.py` (README). Tables are in `outputs/phase1/`; skeleton-only figures are in `outputs/checks/`; the sync report is `outputs/sync_report.csv` (per group) and `outputs/sync_pairs.csv` (per view pair). Decisions P1–P8, S4, S5, C2, D11 and M5a in `DESIGN_DECISIONS.md` carry the numbers.

### 5.1 Keypoints
- **941/941 videos and 3 PT demos**, 712,585 frames, in 37.6 min with 12 CPU processes (MediaPipe full model; frames decoded with PyAV, which gives real timestamps and honours the rotation flag). Code: `cvdc/data/{filenames,labels,index,video,extract}.py`; index `data/index.csv`.
- **Settings check (P2, P3):** on 12 videos, 960 px vs full resolution and padded vs unpadded frames differ by ~1% of torso length, and padding does not change the shoulder-width/torso ratio, so: 960 px, no padding.
- **Detection:** 98.5% of frames on average; 28 videos below 90%.

  | Variation | FL | ML | LJ | O | LL | HJ | LR |
  |---|---|---|---|---|---|---|---|
  | E01 mean frame detection | 99.6% | 99.8% | 100% | 99.0% | 96.0% | 95.6% | 100% |
  | E02 mean frame detection | 99.5% | 99.8% | 100% | 99.1% | 96.9% | 95.1% | – |

- **8 videos have a pose in < 50% of frames** and are excluded downstream (P7): `E01_P10/P11/P12_AL_VLL`, `E01_P16/P20_AL_VHJ`, `E02_P10/P11_AL_VLL`, `E02_P16_AL_VHJ`. All are left-view low-light or high-jitter takes; the other views of those groups are kept.
- **Bug found and fixed:** upright (rotation 0) portrait videos produced padded, non-contiguous RGB rows, which MediaPipe read as garbage. This hit 4 files: both single-exercise PT demos (`Abduction .mp4`, `Aduction.mp4`) and `E01/E02_P13_AL_VLJ`, all at 0% detection. Frames are now always made contiguous, the 4 were re-extracted (now 99.6–100%), and a regression test covers it.
- **Side views self-occlude one arm:** in AL videos the subject's left elbow/wrist reach visibility ≥ 0.5 in only 4–20% of frames; in AR videos the right elbow/wrist in 40–71%. So the AL camera faces the subject's right side. Hips are visible in ≥ 94% of frames in every view, so the root is mid-hip.

### 5.2 Preprocessing
- 30 fps grid from the real timestamps (`cvdc/data/preprocess.py`). Frames without a pose are interpolated. Low-visibility joints are interpolated only for runs ≤ 0.5 s, because longer runs are self-occlusion and interpolating them froze the moving arm (P5). Light Savitzky-Golay smoothing; mid-hip root; torso scaling in 2D and 3D.
- Upper-body joint-frames interpolated: mean 4.2%, median 0.65%. Low-visibility joint-frames kept with MediaPipe's estimate: mean 9.8% (L views ~23%, R views ~11%).
- Fixed-length packs (`cvdc/data/pack.py`, coco17 joints, P4): `reps_coco17_T64` (7,095 reps) and `videos_coco17_T300` (933 videos), with no NaN. T_rep = 64 matches the median rep length of 64 frames (p5 37, p95 89).

### 5.3 Repetitions (`cvdc/data/reps.py`, P6, P6b)
- **Signals:** E01 uses 3D shoulder elevation (prominence 30°). E02 uses the wrist path's first principal component in the trunk frame, oriented toward the midline: E02 is a **cross-body adduction with the arm low** (elevation only ~10–45°; the PT demo shows the same), so no single angle segments it consistently across views. This non-angle signal is approved **for segmentation only** (answer 1).
- **Synced groups share one rep set** (answer 5, P6b): reps are segmented on the group's best view (F in 116/127 E01 and 115/127 E02 groups) and transferred to the other views; reps share a `rep_uid`. Not-synced, unknown and single-view videos are segmented on their own.
- **Counts:** set rep count (best view) median 10 for E01 and E02 experts; E01 non-experts 1 (clips of 1–2 reps). Training reps per view median 9, because an edge rep that some view did not record is dropped from all views. 7,095 training reps (7,827 with per-view segmentation). 18 videos have 0 reps, all E01 non-expert clips.
- **Cross-view agreement in synced groups: 100%** (training reps and set counts) by construction; before: exactly equal 32% / 15%, within ±1 69% / 42% (E01 / E02). Transferred boundaries coincide on the aligned clock within 0.96 frames. Not-synced and unknown groups are unchanged. Details and per-group changes in §5.6.
- **PF4:** for E01 experts the set rep count behaves like 10 × PF4: when PF4 = 1, 98% of videos have ≥ 10 reps; otherwise the median |reps − 10·PF4| is 1 (71% within ±1); Spearman 0.36. For E02 there is no relation (PF4 = 0 on many 10-rep videos; D11).
- **Timing (all expert reps, median):** E01 rep 2.4 s, 10–90% raise 0.77 s, hold 0.87 s, lower 0.77 s; E02 1.8 / 0.60 / 0.60 / 0.57 s. Faster than the EAAQ's 1.5–2.5 s per raise, so speed tolerances come from expert data (answer 4, M5a). E01 3D peak elevation median 154° (p95 170°).
- **Moving arm:** ~14% of expert performances use the **left** arm (all three views agree, including the front, so it is not a label swap). They are mirrored and assessed normally; no `arm_side` flag (answer 2, §5.6, P8).
- **Spot checks:** `outputs/checks/reps_<EX>_{expert,non-expert,extremes}.png` and `skeleton_*.png` (8-frame skeleton montages with the rep signal; PT demos as `skeleton_ref_*`); `sync_*.png` shows the shared rep spans on the aligned clock. The remaining failures are high-jitter and some left-view takes where MediaPipe loses the arm, plus low-amplitude E02 takes.

### 5.4 Synchronization (`cvdc/data/sync.py`)
- **Group classes reproduce Step 0 exactly:** E01 expert 49 audio / 78 creation time / 2 unknown; E01 non-expert 22 not synced; E02 45 / 82 / 2.
- **Motion refinement:** 217 of 818 view pairs confirmed (synced groups: F–L 40%, F–R 31%, L–R 14%). On separate recordings (2,006 control pairs) the test falsely confirms 0.7% even with a ±5 s search. Confirmed three-view lags agree to a median of 3.2 ms (least-squares residual).
- **Device offsets** (video lag − audio lag) re-estimated: F→L +93 ms, F→R +57 ms, L→R −44 ms, with an SD of ~50 ms.
- **Aligned views (C2, answer 3):** all 94 audio groups get three aligned views. In creation-time groups, views joined by motion-confirmed pairs give `sync` pairs (E01: 51 of 78 groups have ≥ 2 such views, E02: 24 of 82); every other view is aligned with its device-corrected audio lag and gives `audio` pairs. Offsets come from a weighted least-squares fit (motion lags weight 1, audio lags 0.04; S7).
- **CVDC view pairs** (`outputs/sync_pairs.csv`, `pair_source`, `pair_weight`): E01 239 `sync` (1.0) / 132 `audio` (0.8) / 66 `rep` (0.5); E02 171 / 204 / 6. `rep` pairs are now only the not-synced E01 non-expert groups and the 2 + 2 unknown groups.
- Figures: `outputs/checks/sync_*.png` (the rep signal of each view on the aligned clock).

### 5.5 Tests
`python -m pytest -q tests`: 58 tests, all passing. They cover the filename parser, quality bins, angle geometry (elevation, signed frontal angle, horizontal adduction, wrist-path PC1), rep segmentation on synthetic signals (counts with noise, back-to-back reps, incomplete last rep, glitch robustness, NaN gaps) and boundary transfer between views, preprocessing (irregular timestamps, gaps, short vs long low-visibility runs, root/scale), sync (GCC-PHAT lag, motion lag and rejection, offset least squares, weighted offsets) and frame contiguity.

### 5.6 Third-review changes (answers 1–5)
- **Answer 1, E02 assessment angle:** `horizontal_adduction` in `cvdc/assess/angles.py` (0° out to the side, 90° forward, > 90° across the body; NaN when the upper arm is within ~14.5° of vertical). On the data (`scripts/phase1/e02_angle_check.py`): front views go from ~18° at rest to ~135° at the rep peak. **The 3D angle is view-dependent:** at the same instants side views read 48° (L) and 41° (R) lower than the front (median), and in R views it is undefined at 80% of rep peaks (upper arm near vertical, median elevation 11°). Phase 4 has to handle this (M7a).
- **Answer 2, left-arm scores** (`scripts/phase1/left_arm_scores.py`; unit = performance, views averaged; one-sided Mann-Whitney, Holm over 2 exercises × total/PF/CF, rule fixed before running):

  | Exercise | Left / right performances | Total score L vs R | PF L vs R | CF L vs R (p uncorrected → Holm) |
  |---|---|---|---|---|
  | E01 | 15 (10 subjects) / 114 | 75.3 vs 80.9 (p 0.28) | 0.76 vs 0.78 | 0.74 vs 0.85 (0.028 → 0.14) |
  | E02 | 22 (10 subjects) / 107 | 71.4 vs 74.9 (p 0.10) | 0.64 vs 0.66 | 0.81 vs 0.86 (0.014 → 0.08) |

  No score is significantly lower after correction, so **no `arm_side` flag** (`assess.flag_arm_side: false`). Borderline: CF is lower in both exercises before correction; within-subject comparisons (10 / 9 subjects) are not significant.
- **Answer 3, audio pairs:** see §5.4; `sync.pair_weights` = sync 1.0, audio 0.8, rep 0.5, synthetic 1.0; ablation sources (C5): sync, + audio, + rep, + synthetic, synthetic only.
- **Answer 4, speed:** `assess.speed.source: expert_data` (reps of videos scoring ≥ 95), `eaaq_phase_s: [1.5, 2.5]` kept as the alternative for E01 (E02: none until the EAAQ appendix). Expert reference (93 videos ≥ 95): 10–90% raise median 0.80 s (p5–p95 0.57–1.13) for E01 and 0.63 s (0.40–0.97) for E02; lowering 0.77 / 0.60 s.
- **Answer 5, rep consistency:** per synced group (views summed), comparing the final rep sets with the earlier per-view segmentation (`data/reps_perview.csv`, `outputs/phase1/table_reps_changes.csv`):

  | | E01 synced (127 groups) | E02 synced (127 groups) |
  |---|---|---|
  | Groups whose per-view counts changed | 86 | 108 |
  | Count change per group (median / p90 / max) | 1 / 5.4 / 15 | 2 / 9.4 / 13 |
  | Groups that lost an edge rep not recorded by every view | 96 | 96 |
  | Training reps before → after | 3,558 → 3,291 | 3,926 → 3,461 |
  | Exactly equal counts across views, before → after | 32% → 100% | 15% → 100% |

  Mean |count change| per transferred view: E01 L 0.8, R 1.3; E02 L 1.3, R 2.8 (the R view over-counted most). `reps.transfer_keep: covered` keeps edge reps in the views that recorded them, if identical rep sets are not required.



## 6. Phase 2 results (2026-10-17)
How to run: `python -m cvdc.data.pairs`, then `python scripts/phase2/smoke.py` (README). Config: `configs/model.yaml`. Decisions A1–A9 and the updated C1, C3, C4, C6 in `DESIGN_DECISIONS.md`. Outputs: `outputs/phase2/` (pair counts, smoke log and tables), `outputs/checks/phase2_*.png` (skeleton only).

**The smoke run is a plumbing check, not a result:** one subject-held-out split (40 training subjects, 10 held out: P07, P08, P12, P14, P18 experts; P40, P42, P48, P53, P58 non-experts), 10 CVDC epochs, 4 supervised epochs, no tuning.

### 6.1 What was built
- **ST-GCN encoder** (`cvdc/models/stgcn.py`, `graph.py`): Yan et al. 2018, 10 blocks (64×4, 128×3, 256×3), temporal kernel 9, residuals, learnable edge importance, global pooling → 256-d. COCO-17 graph with self-loops and the spatial partition (self / centripetal / centrifugal; A9). 3.07 M parameters. All views go through the same weights. Input `xy` (default), `xyv` or `xyz` (B2, world landmarks).
- **Projection head**: Linear-ReLU-Linear → 128-d, L2-normalised. **Task heads** (linear, A3): exercise (9 classes), score, EAAQ items (CF1–CF3 at rep level; PF1, PF2, PF4 added at set level), quality (3), with an exercise embedding for all but the exercise head.
- **CVDC loss** (`cvdc/losses/cvdc.py`): weighted multi-positive InfoNCE, τ = 0.1, `negatives: all_other_groups | different_exercise_only`, same-group other reps neutral (A4), normalisation A6. Monitoring: positive/negative similarity, projection and embedding std (collapse), alignment, uniformity.
- **Pair builder** (`cvdc/data/pairs.py`, A5): rep sets from shared `rep_uid` (sync / audio sources) and index pairing with DTW alignment (rep source). **Synthetic views** (`cvdc/data/synthetic.py`, C4, A8). **Batches** (`cvdc/data/cvdc_batch.py`): all loaded views of a rep set together, 2 synthetic views per set, per-pair weights from `sync.pair_weights` (A7), augmentations with a shared temporal crop (A2). Sources are switchable for the ablations (C5).

### 6.2 Pair counts (rep level; `outputs/phase2/table_pair_counts.csv`)

| Exercise / group | Reps | Rep sets (3 / 2 / 1 views) | sync pairs (1.0) | audio pairs (0.8) | rep pairs (0.5) |
|---|---|---|---|---|---|
| E01 expert | 3,347 | 1,148 (1,052 / 95 / 1) | 2,114 | 1,084 | 53 |
| E01 non-expert | 231 | 181 (21 / 8 / 152) | 0 | 0 | 71 |
| E02 expert | 3,517 | 1,194 (1,130 / 63 / 1) | 1,563 | 1,837 | 53 |
| **All** | **7,095** | **2,523 (2,203 / 166 / 154)** | **3,677** | **2,921** | **177** |

- Synthetic pairs are made on the fly: 2 views per rep set per batch, so every rep set has positives, including the 154 single-view ones (152 are non-expert front-only clips).
- **The E01 non-experts contribute almost no real cross-view pairs** (71 rep-matched pairs from 22 not-synced groups). For them, CVDC relies on synthetic views.
- `rep` pairs: 71 from the 22 not-synced non-expert groups, 106 from the 4 unknown expert groups (E01 P09 VFL, P13 VLJ; E02 P13, P18 VLJ). The unknown groups have unequal rep counts per view (e.g. 9 / 10 / 8), so index pairing may pair neighbouring reps; at weight 0.5 and 2.6% of the edges, this is left as approved.

### 6.3 Checks
- **DTW alignment of rep pairs** (A5): on synced pairs given a known time warp, DTW on raw 3D angles recovers it to a median 2.1 frames for E01 (4.9 without DTW) but does not help E02 (3.2 vs 3.2), whose angles change little. 3D angles are view-biased: the DTW cost of the same rep seen from two views (median ~32°) is no lower than that of the next rep (~31–34°), so the cost is not used to filter pairs.
- **Synthetic views** (`outputs/phase2/smoke_synthetic_check.csv`): the yaw-0 projection of the world landmarks lies a median 0.05 (F, R) / 0.06 (L) torso lengths per joint from the real 2D keypoints of the same rep (mean 0.06 / 0.08 / 0.07 for F / L / R), so the pinhole model matches the real geometry.
  - **Found and fixed:** the world landmarks lean in depth by the camera's pitch (median 14.8° F, 7.9° L, 12.8° R), so rotating about the camera's y axis tipped the person over in side views. Views now orbit the person's vertical axis and keep the camera pitch (A8; test added).
  - `outputs/checks/phase2_synthetic_views.png` shows one rep's real F/L/R views and its synthetic ±45° / ±90° views; the synthetic side views look like the real L / R views.
- **CVDC pretraining** (all sources, training subjects: 2,021 rep sets, ~303 items per batch, 32 batches per epoch): loss 4.90 → 3.19 over 10 epochs; positive cosine 0.94 → 0.86, negative 0.81 → 0.20; projection std 0.033 → 0.076 (1/√128 = 0.088 when spread evenly), so no collapse. Curves: `outputs/checks/phase2_cvdc_smoke_curves.png`.
- **Viewpoint consistency, held-out subjects** (`smoke_viewpoint_consistency.csv`; VCI = same rep, other view − other groups, cosine):

  | | random init | after 10 CVDC epochs (held out) | (training subjects) |
  |---|---|---|---|
  | VCI, projection (128-d) | 0.00 | **0.53** (0.81 vs 0.29) | 0.65 |
  | VCI, embedding z (256-d) | 0.00 | 0.15 (0.97 vs 0.82) | 0.15 |

  At random init every embedding is nearly identical (cosine ≈ 1), so 0.00 is only a floor. The pretraining separates other groups from same-rep views on subjects it never saw. The 256-d embedding is post-ReLU (non-negative), so all its cosines are high and its VCI is compressed (open item for Phase 5, §8). Held-out same-rep cosine on z by pair source: sync 0.968, audio 0.964, rep 0.973.
- **Supervised smoke** (rep level, 4 epochs, lr 1e-3, from the CVDC encoder and from scratch for xy / xyz (B2) / xyv): all run, losses fall, held-out exercise accuracy 0.88–0.94 at the last epoch. **No variant beats predict-the-mean on held-out score RMSE yet** (mean baseline 15.8; scratch xy 15.8–18.1, xyz 15.4–16.7, xyv 15.5–19.8; CVDC-initialised 17.4–23.1, the least stable). The training score loss stays near the score variance.
  - **Plumbing check:** labels match `index.csv`. On 512 training reps the model fits the quality class to 100% and the score to 7–9 RMSE (mean baseline 15.0), so the heads can learn. The score term is MSE on score / 100, whose gradients are small next to the cross-entropy terms. Phase 3 should standardise the score target and tune lr / epochs, linear probe vs fine-tuning (§8).
- **Ablation switches run:** `negatives: different_exercise_only`, `sources: [synthetic]` (2,021 sets, one real view each) and `sources: [sync]` (1,254 sets, no synthetic) each train for an epoch without errors.
- **Speed** (RTX 4060 Ti, bf16): CVDC step 289 ms for ~303 items (9.6 s per epoch on 40 subjects), supervised step 237 ms for 256 reps (~5 s per epoch); peak GPU memory 4.0 GB. The step is limited by memory bandwidth (BatchNorm, elementwise and layout kernels; fp32 is 2× slower), not by batch building (20 ms). These give the per-epoch numbers for the Phase 3 LOSO runtime estimate.

### 6.4 Tests
`python -m pytest -q tests`: **107 tests, all passing** (58 from Phase 1, 49 new). The new tests cover:
- graph adjacency and partitions;
- encoder shapes for every input mode and T, shared weights across views, gradient flow, and that the exercise head is blind to the exercise embedding;
- the CVDC loss: a hand-computed value; invariance to batch order; positives pulled together on a toy batch; weights scaling the terms; zero-weight pairs being neutral; negative modes; anchors without positives; collapse detection;
- the pair builder (edges, sets, reference view) and the positive-weight rules;
- DTW (brute-force cost, path validity, recovery of a known warp);
- synthetic views (rotation, quarter turn, yaw-0 shape, levelling);
- augmentations (shared vs independent crops, per-item spatial changes, joint dropout).

### 6.5 Fifth-review changes (2026-10-17)
1. **Standardised targets** (T7, `TargetScaler` in `cvdc/losses/supervised.py`): score and items are z-scored with the training subjects' mean and std, and predictions are inverted for reporting; constant items are skipped (T4). In the smoke, the training score loss now drops below the mean predictor (z-units 0.87 → 0.66 in 4 epochs from the CVDC encoder; 1.85 → 0.91 from scratch). On the 10 held-out subjects the score RMSE stays around predict-the-mean and moves by 2–5 points between epochs (last epoch: B2-MediaPipe 15.4, B2-KTPFormer 17.0, Proposed 18.0, B1 18.2, xyv 18.2, vs 15.8 for the mean; experts only: B2-MediaPipe 15.1 vs 15.2 for the expert mean). Nothing can be read from 4 untuned epochs. **Head weights:** per-head gradient norms on the encoder (512 random training reps): at initialisation items 13.8 > score 8.1 > exercise 1.7 > quality 0.7; after 4 epochs score 1.55 > items 0.50 > exercise 0.31 > quality 0.18. Score drives the encoder most once training is under way, so the weights stay 1 / 1 / 0.5 / 0.5.
2. **VCI** (E13, `cvdc/eval/consistency.py`): three variants, primary = mean-centred 256-d. Held-out subjects after 10 CVDC epochs (`outputs/phase2/smoke_viewpoint_consistency.csv`):

   | Stratum (held out) | centred z (primary): random init → CVDC | raw z | projection | same-rep view pairs |
   |---|---|---|---|---|
   | E01 expert | −0.01 → **0.16** | 0.00 → 0.03 | 0.00 → 0.25 | 672 |
   | E02 expert | −0.04 → **0.06** | 0.00 → 0.03 | 0.00 → 0.22 | 740 |
   | E01 non-expert | 0.33 → 0.48 | 0.00 → 0.07 | 0.00 → 0.49 | 13 (rep-matched only) |
   | pooled | 0.41 → 0.80 | 0.00 → 0.15 | 0.00 → 0.53 | 1,425 |

   **The pooled VCI is inflated by exercise separation:** "other groups" then include the other exercise, and a random encoder already scores 0.41 pooled but about 0 within each exercise. I propose the **within-stratum** values as the headline VCI (E13). The E01 non-expert stratum rests on 13 pairs. Training subjects after CVDC: E01 expert 0.22, E02 expert 0.18 (centred).
3. **E01 non-experts:** recorded as limitation D12 (§0) and as a required separate stratum in every CVDC ablation (C5).
4. **KTPFormer for B2** (A10, `cvdc/models/lifter.py`):
   - **Licence:** the GitHub repository has **no licence** (API `license: null`), so by default all rights are reserved; the Google Drive weights state none; the model is trained on Human3.6M (non-commercial research licence). Used locally for research only; code (`third_party/KTPFormer`, commit b961951) and weights (`data/models/ktpformer/`) are gitignored and never redistributed. MotionBERT (Apache-2.0) was not needed.
   - **Weights:** `model_243_CPN_best_epoch.bin` (Human3.6M, CPN 2D input, 243 frames; 423 MB; sha256 8687583f34f60abeb6b9feececdd08ca3d88669ee92e9c4d916d3cf27bfdd9de). It pickles a NumPy random state, so its pickle globals were listed and allow-listed for `weights_only=True` loading (no arbitrary code), then re-saved as a tensors-only state dict. 35.2 M parameters, no missing or unexpected keys.
   - **Wrapper:** coco17 → H36M-17 (MotionBERT convention), 30 → 50 fps, 243-frame windows with stride 81, flip averaging, bf16, back to 30 fps and coco17, normalised like the world landmarks. All 936 videos (941 minus the 8 low-detection ones, plus the 3 PT demos) lifted in 13 min (860 frames/s); packs `reps_coco17_T64_ktpformer.npz` and `videos_coco17_T300_ktpformer.npz`, no NaN. Input mode `model.input: xyz_ktpformer`.
   - **Check** (`scripts/phase2/lifter_check.py`, no labels): input scale 0.30 chosen on 40 groups. On all 246 synced 3-view groups (2,148 rep sets), the cross-view difference of the same 3D angle at the same instants, KTPFormer vs MediaPipe world: E01 shoulder elevation **12.9° vs 15.1°**; E01 elbow 24.6° vs 14.7°; E02 horizontal adduction 70.4° vs 50.5°; E02 elbow 16.4° vs 12.0°. KTPFormer has steadier bones (upper-arm length CV 0.03 vs 0.07) and half the wrist jitter. **So the H36M-trained lifter is not more view-consistent than MediaPipe on this data except for E01 shoulder elevation**; B2 is reported with both inputs.
   - DepthAnything: not built (later, small subset).
5. **bf16** (A9): enabled for training and evaluation. Idle-GPU step times (`outputs/phase2/bench_steps.csv`), bf16 vs fp32: CVDC step 234 vs 439 ms (281 items); supervised rep step 211 vs 404 ms (256 reps); video step 118 vs 228 ms (32 videos). `torch.compile` is unavailable (no Triton on Windows).
6. **Runtime estimate** (E2, `scripts/phase2/bench.py`, `outputs/phase2/runtime_estimate.csv`). Assumptions: bf16, CVDC 100 epochs (64 rep sets per batch), fine-tuning 60 epochs (256 reps or 32 videos per batch), one held-out evaluation per epoch, no early stopping; actual fold sizes (GroupKFold-5 stratified: 40 training subjects per fold; rep-level LOSO: 50 folds of 49 training subjects):

   | Protocol (rep level) | B1 or B2 (each) | Proposed (CVDC + fine-tune) |
   |---|---|---|
   | GroupKFold-5 | 0.43 h | 1.49 h (pretraining 0.21 h per fold) |
   | LOSO-50, per-fold encoder (5 pretrains + 50 fine-tunes, C7) | 4.86 h | **5.92 h** |
   | LOSO-50, full (a pretraining per LOSO fold; reference) | 4.86 h | 17.8 h |

   - **Main LOSO table** (B1, B2-MediaPipe, B2-KTPFormer, Proposed; rep level, per-fold encoder): **≈ 20.5 h**.
   - Each GroupKFold ablation setting (C5 source mixes, τ, embedding size, 2 vs 3 views, augmentation): ≈ 1.5 h. The 6 source mixes alone ≈ 9 h.
   - Video-level runs (T1 comparison): B1/B2 2.85 h and Proposed 3.91 h each on LOSO; 0.25 / 1.31 h on GroupKFold.
   - Early stopping or fewer epochs scale these linearly; linear-probe fine-tuning is cheaper (forward-only encoder).
7. **Rep-matched pairs:** w_rep stays 0.5; the ablation adds (d′) all sources without rep-matched pairs next to (d), and (b) vs (c), so every source comparison is reported with and without them (C5).

Also added: subject-independent splits with a leakage check (`cvdc/eval/splits.py`: stratified GroupKFold, LOSO, the LOSO-shortcut fold map), the input mode `xyz_ktpformer`, `python -m cvdc.data.pack --lift ktpformer`, and 15 tests (splits and leakage, target scaler, centred VCI, lifter mapping, windowing and a run on the downloaded weights): **122 tests, all passing**.

## 6b. Phase 3 step (a) results: GroupKFold-5, main methods (2026-10-18)
How to run: `python -m cvdc.train.run` then `python scripts/phase3/report_phase3a.py` (README). Full tables: `outputs/results/phase3a/summary.md` and CSVs (gitignored). Decisions T6, T8a, T9–T12, E3 (update), E15–E19 in `DESIGN_DECISIONS.md`. Same folds, inner splits (32 inner-train / 8 inner-val subjects per fold), budget, schedule and encoder for every method; test predictions pooled over the 5 folds; 95% CIs from a subject-level bootstrap (2,000 draws). All §0 limitations apply (2 shoulder exercises, confounded expert/non-expert contrast, E02 experts only, GroupKFold not LOSO).

**Total score, video level (headline, E15), within groups: RMSE [95% CI] / Spearman ρ** (915 test videos with ≥ 1 rep)

| Method | E01 expert (377 videos) | E02 expert (381) | E01 non-expert (157) |
|---|---|---|---|
| B1 (xy, scratch) | 13.29 [10.31, 17.30] / 0.06 | 17.70 [13.47, 21.86] / 0.04 | 23.12 [18.51, 27.66] / −0.18 |
| B2-MediaPipe (xyz) | 13.84 [11.30, 17.24] / −0.05 | 16.59 [13.01, 20.25] / 0.13 | 21.88 [16.57, 27.06] / −0.07 |
| B2-KTPFormer (lifted xyz) | 14.22 [11.42, 17.48] / 0.01 | 16.17 [12.94, 19.64] / 0.18 | 20.77 [16.19, 25.57] / −0.25 |
| Proposed (xy, CVDC + fine-tune) | 13.33 [10.39, 17.28] / 0.06 | 16.06 [13.06, 19.24] / 0.20 | 17.66 [12.74, 22.13] / −0.10 |
| predict-the-mean | 13.07 [10.32, 16.78] | 16.02 [13.19, 18.97] | 32.20 [29.16, 35.16] |
| group-only (within-group mean) | 12.45 [9.37, 16.59] | 16.02 [13.53, 18.74] | 7.77 [5.74, 9.64] |
| duration-only | 13.24 [10.32, 16.99] / −0.06 | 16.26 [13.54, 19.10] / −0.12 | 27.89 [24.91, 30.68] / 0.22 |
| label-noise floor (E3) | 5.28 | 4.38 | – (expert-based) |

- **No method beats the within-group mean (group-only baseline) in any group.** Within-group Spearman ρ is ≤ 0.20 everywhere and every CI includes 0 or nearly so. Proposed is the best model in E02 experts (16.06 vs 16.02 for the mean) and E01 non-experts, but still at or above the baselines. Rep-level, CF1–CF3, pooled, paired-difference and classification tables are in `summary.md`.
- **Within-group VCI (centred 256-d, held-out subjects, final encoders; random-init reference in brackets):** B1 0.020 / 0.024 / 0.110 (−0.019 / 0.024 / 0.072); B2-MediaPipe 0.001 / 0.025 / −0.003; B2-KTPFormer 0.014 / 0.048 / −0.033; **Proposed 0.087 [0.027, 0.123] / 0.119 [0.054, 0.146] / 0.146 [−0.011, 0.261]**, and 0.298 / 0.257 / 0.296 after CVDC pretraining alone (E01 expert / E02 expert / E01 non-expert). CVDC makes the embedding view-consistent; fine-tuning removes about two thirds of it. Pooled VCI (secondary) 0.69–0.74 for every final model vs 0.401 for the random encoder (exercise separation).
- **Stopping epochs:** every run stopped early. Fine-tuning kept epochs 2–22 (B1 2, 9, 12, 3, 13; B2-MP 2, 7, 15, 5, 7; B2-KTP 15, 13, 15, 7, 17; Proposed 2, 8, 22, 17, 14) of 60; CVDC kept 53, 62, 75, 80, 77 of 100. The inner-validation loss jumps between epochs (score RMSE e.g. 15 → 24 → 16), so early stopping often keeps a barely trained model.
- **Wall time** (RTX 4060 Ti, bf16, sequential): B1 6.4 min, B2-MediaPipe 6.1 min, B2-KTPFormer 8.2 min, Proposed 54.1 min (CVDC 46.0) for all 5 folds; peak GPU memory 4.24 GB.
- **Surprise (labels):** across the views of every expert performance, PF1, PF2 and CF1–CF3 are identical; only PF3/PF4 (repetitions) differ, so the 5.28 / 4.38 floor comes entirely from the rep-count items and the CF floor is 0 (E3).
- **Paired differences** (video level, same bootstrap draws; `paired_deltas.csv`): against the group-only baseline every method has a **higher** RMSE in E01 experts (+0.84 to +1.77, all CIs above 0) and E01 non-experts (+9.9 to +15.4); in E02 experts the CIs include 0 (Proposed +0.04 [−1.82, 1.87]). Against B1, Proposed is lower in E01 non-experts (−5.47 [−7.80, −3.20]) and E01 pooled (−2.06 [−3.04, −1.10]), about the same in E02 (−1.63 [−3.38, 0.52]) and E01 experts (+0.04 [−0.44, 0.58]). B2-MediaPipe and B2-KTPFormer are not clearly different from B1 in any group (all CIs include 0).
- **CF1–CF3** (video level, 0–1 scale): no method is below the group-only baseline on any CF item in any group, except B1 on E01-expert CF3 (0.171 vs 0.177), B1 on E02 CF3 (0.0969 vs 0.0969) and Proposed on E02 CF3 (0.083 vs 0.097, ρ 0.33). E02 CF3 varies in only 2 of the 24 subjects (step (a2), §6c), so this is not evidence of a learned item.
- **Pooled results look better only because of the group:** E01 pooled Pearson r 0.56–0.68 for the models against 0.82 for the group-only baseline, and RMSE 14.7–16.8 against 11.3. Pooled correlations measure the expert/non-expert separation (E9, E10), not movement quality.
- **Classification (secondary):** derived quality class from the quality head, within groups: macro F1 0.25–0.35, Cohen's κ −0.13 to 0.12. Exercise accuracy (video level, pooled) 0.93–0.98.
- **Inner validation diagnostic** (`stopping_runtime.csv`): the kept epoch's rep-level validation RMSE is close to the RMSE of the inner-training mean on the same validation subjects in most folds (e.g. B1 15.1 vs 14.6, 18.1 vs 17.9) and lower in a few (B1 fold 2: 14.3 vs 17.9; Proposed fold 4: 13.3 vs 15.8). The validation subjects mix both groups, so a lower value there can come from the group difference alone. The training score loss falls to ~0.16 (z-units) while the validation score loss stays at 0.7–0.9, so the networks fit the training subjects without generalising within groups.

**Conclusion of step (a) (see §6c for the follow-up):** CVDC does what it is meant to do for the embedding (within-group VCI 0.26–0.30 after pretraining vs ~0 for every encoder trained without it), and Proposed has the lowest RMSE of the four models in E02 experts and E01 non-experts. But **no model predicts the score within a group better than the within-group mean**, so step (a) shows no evidence yet that any network judges movement quality. Likely causes, to test before the ablations: (1) the score setup: every rep inherits its video's score, while the repetition items (PF3/PF4), which one rep cannot show, carry much of the within-group variance (measured in §6c); (2) unstable early stopping that keeps epoch 2–3 models; (3) fine-tuning that undoes most of CVDC's view consistency. Step (a2) (§6c) addresses these; the ablations (b) and LOSO (c) wait.

## 6c. Phase 3 step (a2): score setup diagnosed and changed (2026-10-18)
User request after step (a): diagnose the score setup and fix it (items 1–6 below); re-run GroupKFold-5 for B1 and the Proposed variants, then B2-MediaPipe and B2-KTPFormer; **no ablations, no LOSO**. Decisions T6 (now decided), T8b, T13–T15, E16 (update), E20 in `DESIGN_DECISIONS.md`. How to run: README "Step (a2)". Tables: `outputs/results/phase3a2/label_diagnostics.md` and `summary.md` (+ CSVs; gitignored). Same folds, inner splits, encoder, optimiser and schedule as step (a); all §0 limitations apply.

### 6c.1 Label diagnostics (no training; `scripts/phase3/label_diagnostics.py`)
Units: the 915 evaluated videos. The total score is exactly (100/7) × the sum of the 7 items, so each item's covariance with the score splits the within-group score variance exactly (shares add up to 100%).

| Stratum (videos / subjects) | score SD | PF1 | PF2 | PF3 | PF4 | CF1 | CF2 | CF3 | PF3 + PF4 |
|---|---|---|---|---|---|---|---|---|---|
| E01 expert (377 / 24) | 12.34 | 6.5% | 10.2% | 35.8% | 14.1% | 13.0% | 12.1% | 8.3% | **49.9%** |
| E02 expert (381 / 24) | 15.78 | 18.1% | 5.3% | 27.9% | 33.5% | 2.0% | 12.1% | 1.2% | **61.4%** |
| E01 non-expert (157 / 26) | 7.65 | 0% | 12.2% | 0% | 0.6% | **55.6%** | **26.7%** | 4.9% | 0.6% |

- **Experts:** half (E01) to 61% (E02) of the within-group score variance is the repetition items PF3/PF4, which a single rep cannot show. The CF items carry 33% (E01) and 15% (E02); in E02, CF1 and CF3 carry 2% and 1%. **Non-experts:** the score varies through CF1 and CF2 (82%); their PF items are nearly constant (single-rep clips: PF1 = 1 and PF3 = 0 everywhere, PF4 = 0 in all but 3 videos).
- **Items with ~0 within-group variance** (T14: off the modal value in videos of fewer than 5 subjects): E01 experts PF1 (6 videos of 2 subjects); E02 PF2 (15 videos / 4 subjects), CF3 (32 / 2); E01 non-experts PF1 (constant), PF2 (4 / 1), PF4 (3 / 2), CF3 (13 / 3). They are no longer learned or reported in that group. E02 CF3, where Proposed beat the group mean in step (a) (§6b), is among them.
- **Set-level features do not predict the score within groups** (within-group least squares, same folds, video level; RMSE vs the group-only baseline; out-of-fold R²):

  | Stratum | group-only | rep count only | set features (count + mean rep duration + ROM) |
  |---|---|---|---|
  | E01 expert | 12.45 [9.38, 16.57] | 12.25 [9.06, 16.49], R² 0.02 | 12.23 [9.05, 16.53], R² 0.02, ρ 0.17 |
  | E02 expert | 16.01 [13.53, 18.73] | 16.85 [14.11, 19.75], R² −0.14 | 16.93 [13.88, 20.08], R² −0.15 |
  | E01 non-expert | 7.78 [5.86, 9.59] | 7.76 [5.97, 9.47], R² −0.03 | 7.48 [5.92, 9.04], R² 0.04, ρ 0.26 |

  The Phase 1 rep count hardly varies within the expert groups (E01 SD 0.55 reps, 8–12; E02 SD 1.37) and does not predict PF4 out of fold (E01 RMSE 0.168 vs 0.170 for the group mean; E02 worse than the mean, as D11 found). So the repetition items that dominate the expert score variance are not explained by the detected rep count either; together with the inter-view floor (E3: the views of one performance get different PF3/PF4) this points to rater judgement of the repetitions that the keypoints do not capture. **The one set-level signal:** ROM of the shoulder elevation predicts non-expert CF1 (ρ 0.46, R² 0.15; with all three features R² 0.24, RMSE 0.308 vs 0.362).
- **E02 ROM** uses the 3D horizontal adduction (M7a) as a circular range; it is view-dependent (median 116° F, 53° L, 36° R) and undefined in 35% of rep frames, so it is a weak feature for E02.

### 6c.2 What changed in training (items 2–6)
- **Video head (T13):** mean + attention pooling of all rep embeddings of a video, plus the standardised set rep count and an exercise embedding → video score and PF1, PF2, PF4, CF1–CF3. The rep head stays; L_sup = L_rep + L_video; batches hold whole videos.
- **T6 decided:** rep weight = 1 / n_reps of its video, so every video counts once.
- **T8b:** EMA-smoothed (β 0.7) inner-validation objective, no epoch before 10 kept, patience 15, max 60. **Kept epochs now 12–59 (median 27)** instead of 2–22; 28 of the 35 runs stopped early.
- **T14:** items that do not vary within a group (6c.1) are masked out of the loss for that group and not reported there.
- **Proposed variants (T15)**, all from the same step (a) CVDC encoder per fold (reused after a subject check): (i) frozen encoder + heads, (ii) full fine-tuning with + λ·L_cvdc (λ 0.1, 1.0), (iii) full fine-tuning.
- **Baselines** are fitted on videos now (E16 update), and rep-count-only and set-features were added (E20).
- Runtime: 133 min for the 7 methods × 5 folds (B1 17, Proposed (iii) 14, (i) 9, (ii) 31 and 33, B2-MediaPipe 15, B2-KTPFormer 15 min); peak GPU memory 6.9 GB (joint variants).

### 6c.3 Results (GroupKFold-5, video level, within groups; `outputs/results/phase3a2/summary.md`)
**Total score: RMSE [95% CI] / Spearman ρ.** Video head = the set-level head; rep head = the mean of the video's rep predictions.

| Method | head | E01 expert (377) | E02 expert (381) | E01 non-expert (157) |
|---|---|---|---|---|
| B1 | video | 13.58 [10.98, 17.07] / 0.10 | 17.29 [13.76, 20.70] / 0.13 | 12.08 [9.12, 15.00] / −0.03 |
| Proposed (iii) full fine-tune | video | 14.38 [11.85, 17.61] / 0.06 | 17.97 [13.93, 22.12] / 0.15 | 12.55 [10.58, 14.40] / −0.11 |
| Proposed (i) frozen encoder | video | 14.02 [10.95, 18.12] / 0.01 | 16.04 [12.85, 19.61] / 0.12 | 11.99 [9.41, 14.36] / −0.14 |
| Proposed (ii) λ = 0.1 | video | 14.44 [11.80, 18.00] / 0.07 | 17.10 [13.56, 21.04] / 0.20 | 13.99 [10.65, 17.27] / −0.13 |
| Proposed (ii) λ = 1.0 | video | 13.64 [10.71, 17.21] / 0.16 | 15.86 [12.59, 19.57] / 0.28 | 13.68 [10.56, 16.72] / −0.21 |
| Proposed (ii) λ = 1.0 | rep | 13.71 [10.82, 17.21] / 0.12 | 15.04 [11.91, 18.51] / 0.32 | 14.71 [10.90, 18.61] / −0.26 |
| B2-MediaPipe | video | 13.30 [10.12, 17.50] / 0.18 | 17.51 [14.65, 20.44] / 0.04 | 13.61 [10.42, 16.78] / −0.00 |
| B2-KTPFormer | video | 14.84 [11.67, 18.15] / 0.08 | 17.33 [14.12, 20.45] / 0.06 | 14.27 [10.43, 17.74] / −0.18 |
| group-only (within-group mean) | – | **12.45** [9.38, 16.57] | **16.01** [13.53, 18.73] | **7.78** [5.86, 9.59] |
| rep-count-only | – | 12.25 [9.06, 16.49] / 0.07 | 16.85 [14.11, 19.75] / −0.22 | 7.76 [5.97, 9.47] / −0.01 |
| set-features | – | 12.23 [9.05, 16.53] / 0.17 | 16.93 [13.88, 20.08] / −0.02 | 7.48 [5.92, 9.04] / 0.26 |
| predict-the-mean / duration-only | – | 15.03 / 16.24 | 16.06 / 16.29 | 27.94 / 13.28 |

All rep-head rows and the remaining combinations are in `summary.md`.

- **Still no model beats the within-group mean in any group, with either head.** Paired RMSE differences to the group-only baseline:
  - E01 experts +0.53 to +2.39 for every model and head; the CIs are above 0 except for Proposed λ = 1.0 (video head +1.20 [−0.20, 2.86], rep head +1.26 [−0.04, 2.83]) and B2-MediaPipe's rep head (+0.53 [−0.22, 1.51]);
  - E02 experts −0.97 [−3.46, 1.36] at best (Proposed λ = 1.0, rep head), and every CI includes 0;
  - E01 non-experts +4.2 to +9.0, all CIs above 0.

  In the E01 groups no model beats the set-features or rep-count-only baselines either. In E02 those two baselines are themselves worse than the group mean (16.93 / 16.85), so some models have a lower RMSE than them (e.g. λ = 1.0 rep head vs set-features −1.89 [−4.13, 0.05]), but still not lower than the group mean.
- **Within-group Spearman ρ** ranges from −0.29 to 0.32. Three of the 42 intervals (7 methods × 2 heads × 3 groups) exclude 0: B2-MediaPipe video head in E01 experts (0.18 [0.02, 0.31]), Proposed λ = 1.0 rep head in E02 (0.32 [0.02, 0.59]), and B2-KTPFormer rep head in E01 non-experts (**−0.29** [−0.50, −0.03]). About 2 of 42 are expected by chance at 95%, so none is evidence on its own.
- **Against step (a):** E01 non-expert RMSE dropped a lot (B1 23.1 → 12.1, Proposed 17.7 → 12.6, B2-MediaPipe 21.9 → 13.6), because every video now counts once and non-expert predictions are no longer pulled toward the expert level. This is a level (calibration) gain, not ranking (ρ ≤ 0), and still far above the group mean (7.8). Expert RMSEs moved by less than 2 points, mostly upward (video head; E01: B1 13.29 → 13.58, Proposed 13.33 → 14.38, B2-MediaPipe 13.84 → 13.30; E02: B1 17.70 → 17.29, Proposed 16.06 → 17.97).
- **Video head vs rep head:** the video head is lower in E01 non-experts for every method (e.g. B1 12.08 vs 14.51) and about the same or slightly higher in the expert groups. **The set rep count did not help**, consistent with 6c.1.
- **Proposed vs B1 (same head):** the only CI below 0 is Proposed λ = 1.0's rep head in E02 (−1.74 [−3.61, −0.07]); the frozen-encoder video head in E02 is −1.25 [−2.70, 0.08]. Elsewhere the differences include 0 or favour B1 (E01 non-experts, λ = 0.1 video head +1.91 [0.17, 3.49]). B2-MediaPipe and B2-KTPFormer are not better than B1 in any group.
- **Items (video head, varying items only):** no model is clearly below the group-only baseline on any item. The few lower values are within 0.01: E01-expert CF3 0.174 (frozen, B2-MediaPipe) vs 0.177; non-expert CF1 0.354 (frozen, λ = 0.1) vs 0.362. On non-expert CF1, the set-features baseline (0.308, ρ 0.53) beats every network. On E01-expert PF4, the networks score 0.195–0.272 against the group mean's 0.170.
- **VCI after fine-tuning (centred 256-d, E01 expert / E02 expert / E01 non-expert):**
  - frozen encoder 0.298 / 0.257 / 0.296 (= after CVDC, by construction);
  - **λ = 1.0: 0.244 / 0.256 / 0.166**;
  - λ = 0.1: 0.162 / 0.183 / 0.165;
  - full fine-tuning 0.142 / 0.126 / 0.086 (step (a): 0.087 / 0.119 / 0.146);
  - B1 0.073 / 0.027 / 0.102; B2-MediaPipe 0.049 / 0.004 / 0.106; B2-KTPFormer 0.099 / 0.001 / 0.073.

  The joint loss with λ = 1.0 keeps most of CVDC's view consistency, and its score is not worse than full fine-tuning (best or tied-best among the Proposed variants in both expert groups).
- **Inner-validation diagnostic** (`stopping_runtime.csv`): at the kept epoch, the video-head RMSE on the inner validation subjects (11.7–19.1) is below the inner-training pooled mean (17.1–20.4) in most runs, but below the inner-training **group** mean (12.2–16.3) in only 10 of 35 runs. The validation signal itself mostly rewards between-group calibration.

**Conclusion of step (a2):**
- The score setup is now sound: video-level targets, every video weighted once, only items that vary, stable stopping with real training lengths.
- **The fixes did not produce any within-group score prediction.** The label diagnostics point to the labels, not to the training:
  - the expert score varies mostly through the repetition items, which neither the detected rep count nor any set-level feature predicts out of fold, and which differ between views of the same performance (E3);
  - the CF items that do vary are predicted no better than their group mean, by the networks or by the simple features. The one exception is non-expert CF1, where the shoulder-elevation ROM carries signal (set-features R² 0.24) that the networks do not pick up.
- CVDC remains the only reliable difference between methods, in view consistency; **λ = 1.0 joint training keeps it at no cost in score.**

Proposals for the review (not applied):
1. Make λ = 1.0 joint training the Proposed default for the ablations (b) and LOSO (c).
2. Report within-group score prediction as a negative result, and lead with the measurable outcomes: VCI, the cross-view protocol, and Module III rule deviations against CF items.
3. Test whether explicit angle / ROM inputs (Module III features) let a head reach the set-features level on non-expert CF1.
4. Check with the supervisor whether PF3/PF4 can be re-derived from the detected rep count, or should be dropped from within-group targets.

## 6d. Phase 4: Module III and the embedding-to-reference distance (2026-10-19)
User request (seventh review): Module III as in proposal Figure 2 for E01 and E02, with the **PT demo as the reference ("truth")**; score every video by its distance from the reference and check agreement with the physiotherapists; the embedding distance for B1, B2 and Proposed (λ 1.0), including its view stability; a recording-noise test for PF3/PF4. **No LOSO, no ablations.** Decisions M2, M6, M7, M7a, M8–M13, E5, E22, E23 in `DESIGN_DECISIONS.md`. How to run: README "Phase 4". Tables: `outputs/phase4/summary.md` + CSVs, figures `outputs/phase4/figs/` (curves only, no frames), one JSON per video in `outputs/phase4/json/` (all gitignored). Same GroupKFold-5 folds as Phase 3; video level; 95% CIs from the subject bootstrap (2,000 draws). All §0 limitations apply.

### 6d.1 What was built
- **Reference Template Library** (`cvdc/assess/template.py`, M8): per exercise, the primary PT demo's reps on the 30 fps grid, 3D angles for the moving arm (left arms mirrored), averaged after DTW alignment: E01 `Abduction .mp4` (2 reps, 4.2 and 3.2 s → 112 frames, end range 161.0°), E02 `Aduction.mp4` (2 reps, 2.3 and 2.2 s → 68 frames, end-range horizontal adduction 150.9°). **Per view**, the experts' videos scoring ≥ 95 give the view bias (how a side camera reads the movement differently from the front camera that filmed the demo) and the tolerance band (±2 SD of the experts per template frame) (M2, M9). In the front view the experts' mean curves lie on the demo for shoulder elevation, trunk lean and the other arm; their elbows are 5–10° more bent than the demo's. Side views read the same performances differently (e.g. E01 R-view peak elevation ~25° lower; E02 R-view horizontal adduction ~60° lower) and have much wider bands (`figs/template_E01.png`, `template_E02.png`).
- **Comparison** (`cvdc/assess/module3.py`): every rep DTW-aligned to the template on the upper-arm direction; per channel (E01: shoulder elevation, elbow extension, other-arm elevation, trunk lateral lean, shoulder asymmetry = the left/right symmetry channel; E02: + horizontal adduction as the main angle, M7) the angle deviation in degrees and in expert-SD units; the ROM as the end-range angle vs the view-corrected target (M10); speed (demo / rep duration) and the angular-velocity deviation from the demo along the DTW path.
- **Errors** (M12): YAML rules → `abduction_range_insufficient` / `adduction_range_insufficient`, `elbow_bent`, `other_arm_raised`, `trunk_lean`, `shoulder_asymmetry`, `arm_elevation_deviation` (E02), `too_fast`, `too_slow`, and the set rule `too_few_reps`, each with joint, video frame range, measured / target / deviation (deg or s) and severity.
- **Score** (M11): 100 × (1 − mean of four capped distances: angle deviation, ROM shortfall, speed, symmetry); set score = mean rep score. Defaults chosen before looking at agreement; nothing is fitted to the physiotherapists' scores except the ≥ 95 selection of the band experts.
- **Out of fold:** the libraries are refitted per GroupKFold fold on the training subjects (E01 F/L/R: 137–185 / 85–113 / 93–103 expert reps per fold; E02: 127–216 / 79–116 / 43–70, from as few as 4–9 subjects per view). An all-subject library is saved for the demo (`outputs/phase4/library/`).
- **Output:** 915 per-video JSON reports (spec §8 shape, validated), including the Proposed (λ 1.0) embedding distance from the fold where the subject is held out. Runtime 16 s for 7,095 reps.
- **Tests:** 12 new (angles on known geometry, mirrored left arms, template recovery under time warps, end range, library bias and floors, score and errors on synthetic reps, JSON schema, library round trip, contained-rep counts, weighted ICC vs the closed form, MQ): **159 tests, all passing.**

### 6d.2 Reference score vs the physiotherapists (item 4; `summary.md`, `agreement*.csv`)
**Within groups, Spearman ρ [95% CI] of the set score with each label** ('·' = the item does not vary in the group, T14):

| Label | E01 expert (377 videos) | E02 expert (381) | E01 non-expert (157) | E01 pooled (secondary) |
|---|---|---|---|---|
| total score | 0.03 [−0.10, 0.17] | 0.13 [−0.06, 0.31] | −0.02 [−0.26, 0.22] | 0.59 [0.52, 0.66] |
| **movement-quality score (MQ)** | **0.18 [0.01, 0.34]** | **0.30 [0.09, 0.49]** | −0.02 [−0.26, 0.22] | 0.62 [0.53, 0.71] |
| PF1 | · | 0.16 [0.05, 0.28] | · | −0.03 |
| PF2 (speed) | **0.24 [0.07, 0.38]** | · | · | 0.72 |
| PF3 / PF4 | −0.10 / −0.09 | −0.03 / −0.02 | · | 0.20 / 0.57 |
| CF1 / CF2 / CF3 | 0.02 / 0.06 / 0.11 | 0.09 / 0.21 [−0.01, 0.40] / · | −0.07 / 0.03 / · | −0.07 / 0.55 / −0.07 |

- **The reference score separates the groups, not the performances within them.** Set score 82.5 ± 9.1 (E01 experts), 73.3 ± 9.8 (E02 experts), 50.3 ± 12.0 (E01 non-experts); expert vs non-expert AUC 0.98 (1.00 in the view-matched front-view FL subset, 24 / 19 videos). Within groups it agrees weakly with MQ (E01 0.18, E02 0.30) and E01's speed item PF2 (0.24), and not with the total score (CIs include 0 everywhere). For non-experts total and MQ rank identically (PF3 = PF4 = 0), and neither agrees.
- **By view** (experts, ρ with MQ): E01 F 0.29 [0.09, 0.47], L 0.10, R 0.19; E02 F 0.25, L 0.34 [0.10, 0.56], R 0.33 [0.11, 0.52].
- **Components** (`agreement_components.csv`; 9 components × 3 groups × 2 labels = 54 intervals, so ~3 exclusions of 0 are expected by chance): the **angular-velocity deviation from the demo** agrees in E02 experts (0.24 [0.06, 0.41]) and E01 non-experts (0.40 [0.15, 0.60]); the end-range angle in E01 experts (0.10 [0.00, 0.21]). In E01 non-experts a **higher** end range goes with **lower** scores (ρ −0.31 [−0.57, 0.01]), against the ROM-shortfall rule; the plane of elevation, checked as a possible reason, carries no signal (M7).
- **As a predictor** (out-of-fold within-group line on the set score): RMSE total 12.52 / 16.20 / 7.92 vs the group mean 12.45 / 16.01 / 7.78 (Δ +0.07 [−0.00, 0.19], +0.19 [−0.14, 0.62], +0.14 [−0.04, 0.37]); MQ Δ −0.06 [−0.45, 0.38], −0.08 [−0.52, 0.44], +0.20. **No gain over the within-group mean**, consistent with E21.
- **Errors** (`errors_summary.csv`, rep rates): the rules describe the expert/non-expert difference (E01 non-experts vs experts: range insufficient 71% vs 3%, too fast 81% vs 7%, elbow bent 57% vs 29%, trunk lean 29% vs 10%). **Within experts they fire as often in the videos scoring ≥ 95 as overall** (E01 elbow bent 26% vs 29%; E02 elbow bent 40% vs 48%, shoulder asymmetry 36% vs 33%, arm elevation 27% vs 40%), so they are not specific to worse performances; their video rates correlate with total / MQ at |ρ| ≤ 0.25 (largest: E02 arm elevation 0.20, non-expert trunk lean −0.25, i.e. the wrong sign).

### 6d.3 Embedding distance to the PT reference (item 5; `embedding_tracking.csv`, `view_stability.csv`)
The demo's reps are encoded with each fold's final checkpoint (mirrored for left-arm reps); centred cosine distance; standardised per fold and group with training subjects before pooling (E5). The embeddings reproduce the Phase 3 snapshots to bf16 precision.

**(a) Tracking the physiotherapists** (closeness = −distance; ρ with total / MQ; CIs in `summary.md`):

| Method | E01 expert | E02 expert | E01 non-expert |
|---|---|---|---|
| B1 | 0.04 / **0.19 [0.04, 0.33]** | 0.10 / −0.01 | 0.10 / 0.10 |
| B2-MediaPipe | −0.08 / −0.04 | −0.01 / −0.06 | 0.02 / 0.02 |
| B2-KTPFormer | 0.10 / 0.15 [0.01, 0.28] | 0.09 / 0.04 | 0.14 / 0.14 |
| **Proposed (λ 1.0)** | 0.02 / 0.01 | 0.17 [−0.08, 0.40] / 0.13 | −0.10 / −0.10 |
| Proposed alt. (frozen CVDC) | −0.02 / 0.02 | 0.16 / 0.21 [−0.01, 0.41] | −0.03 / −0.03 |
| random-init encoder | 0.04 / −0.01 | 0.02 / −0.06 | −0.24 [−0.45, 0.02] / −0.25 |
| Module III reference score | 0.03 / 0.18 | 0.13 / **0.30** | −0.02 / −0.02 |

No distance tracks the total score within any group (all CIs include 0), and none beats the within-group mean as an out-of-fold predictor (Proposed Δ RMSE +0.33 [0.03, 0.76] / +0.41 [−1.11, 1.84] / +0.17 [−0.38, 0.92]). The L2 distance gives the same picture.

**(b) View stability: the same performance filmed from F / L / R** (synced expert groups; ICC(1) of the distance across views, higher = more stable; ΔICC vs B1 with paired CIs; E22):

| Method | E01 rep-level ICC | E02 rep-level ICC | ΔICC vs B1 (E01 / E02) | E01 / E02 video-level ICC | R_view (E01 / E02, rep) |
|---|---|---|---|---|---|
| B1 | 0.02 [−0.09, 0.13] | 0.11 [−0.08, 0.30] | – | −0.01 / 0.10 | 0.92 / 0.92 |
| B2-MediaPipe | 0.07 | 0.03 | +0.06 [−0.09, 0.18] / −0.08 [−0.34, 0.17] | 0.08 / 0.03 | 0.92 / 0.98 |
| B2-KTPFormer | 0.12 | 0.01 | +0.10 [−0.07, 0.25] / −0.10 [−0.31, 0.11] | 0.13 / −0.04 | 0.90 / 0.98 |
| **Proposed (λ 1.0)** | **0.32 [0.21, 0.43]** | **0.62 [0.51, 0.70]** | **+0.31 [0.17, 0.43] / +0.51 [0.34, 0.67]** | 0.27 / 0.68 | 0.75 / 0.58 |
| Proposed alt. (frozen CVDC) | 0.30 | 0.68 | +0.29 [0.16, 0.45] / +0.57 [0.38, 0.73] | 0.25 / 0.73 | 0.70 / 0.53 |
| random-init encoder | −0.14 | 0.10 | −0.15 / −0.01 | −0.18 / 0.16 | 1.07 / 0.93 |
| Module III score | 0.54 | 0.50 | – | 0.52 / 0.44 | 0.67 / 0.70 |
| Module III score without speed | 0.24 | 0.31 | – | 0.27 / 0.34 | 0.88 / 0.84 |
| Module III end-range angle | −0.02 | −0.11 | – | −0.05 / −0.21 | 0.99 / 1.12 |

- **CVDC makes the vector difference view-stable**, as hypothesised: Proposed's ICC is 0.31 (E01) and 0.51 (E02) above B1's, with CIs well above 0, at rep and video level; B2 (3D input without CVDC) is no more stable than B1. The views also rank performances alike under CVDC (between-view Spearman, rep level: E02 0.62–0.66 for Proposed vs 0.38 / 0.05 / −0.06 for B1 on F–L / F–R / L–R; E01 0.47 / 0.46 / 0.21 vs 0.30 / 0.16 / −0.09).
- Module III's full score looks as stable (0.54 / 0.50), but synced views share the rep boundaries (P6b), so its speed part is identical across views by construction; **without speed it drops to 0.24 / 0.31**, below Proposed's distance. Its 3D end-range angle is not view-stable at all (ICC ≤ 0).
- **Stable but not clinically informative:** within groups the CVDC distance does not track the physiotherapists' scores (6d.3a). CVDC fixes the view dependence of the vector difference; it does not make the difference measure movement quality as rated.

### 6d.4 Recording-noise test (item 6; `recording_noise.json`, E23)
Per synced expert performance, the reps each view's video fully contains (the best view's reps on the aligned clock):

| | E01 (127 performances, 371 views) | E02 (127, 375) |
|---|---|---|
| performances with different PF4 across views / PF3 / contained count | 100 / 56 / 97 | 71 / 36 / 96 |
| view pairs with different PF4: higher-PF4 view holds **more / the same / fewer** complete reps | **47% / 35% / 18%** (209 pairs) | **39% / 45% / 16%** (163 pairs) |
| 'more' under the within-performance permutation null | 28% (p < 0.001) | 28% (p = 0.001) |
| within-performance Spearman PF4 vs contained count | 0.26 | 0.21 |
| share of the between-view variance of the total score explained by the count | 18% | 3% |
| PF3 = 1 when ≥ 10 / < 10 reps are contained | 57% / 9% | 32% / 18% |

- **Partly.** Between-view PF3/PF4 differences follow the reps each video actually contains more often than chance, clearly so in E01 (the higher-PF4 view holds more complete reps 2.6× as often as fewer). But a third (E01) to almost half (E02) of the PF4 differences occur between views that contain the same number of complete reps, so they are rater variation, not recording. Counting at least half-recorded reps instead makes the views almost equal (88–93% of pairs), so the view differences are partially cut edge reps. In E01 the contained count explains 18% of the between-view variance of the total score; in E02 3%, where PF4 does not behave like a rep count anyway (D11).

### 6d.5 Conclusion of Phase 4
- Module III runs end to end as in proposal Figure 2 (template library → DTW alignment → angle, ROM, speed and symmetry deviations → localised errors with joint, frames, degrees and severity → score → JSON).
- Against the PT reference, the score and the rules capture the **expert/non-expert difference** (AUC 0.98), but within groups they agree only weakly with the movement-quality score (E01 0.18, E02 0.30) and not with the total score, and the rules fire as often in the best expert videos as elsewhere.
- **CVDC's clear, measurable effect is view stability**: the embedding distance to the reference is far more stable across F / L / R with CVDC (ICC +0.31 / +0.51 over B1), and more stable than the 3D-angle Module III score without timing. It does not make the distance track the physiotherapists' scores within groups.
- The between-view PF3/PF4 label differences are partly real recording truncation (E01 especially) and partly rater variation.

Questions for the review (not applied):
1. **Rule specificity:** within experts, the rules fire as often in ≥ 95 videos as overall (e.g. E02 elbow bent 40%). The centre is the demo (whose elbows are straighter than the experts'), and the band is ±2 expert SD per frame. Keep "PT = truth" and report the rates as they are, widen k, or centre the posture channels on the experts?
2. **Speed reference:** the score's speed part is centred on the experts (M5a), but the angular-velocity deviation from the **demo** is the component that agrees best (E02 experts 0.24, non-experts 0.40). Switch the speed centre to the demo? This would be a post-hoc change, so it would need a fresh check (e.g. on LOSO).
3. **Non-expert end range:** a higher end range goes with lower non-expert scores (ρ −0.31). Can the physiotherapists say what CF1–CF3 mean for E01 (the EAAQ figure lists the criteria but not their item numbers), so the rules can follow the items?
4. Single-view use (live demo): the library has per-view bands, so the demo needs the camera view (default front); OK?

## 7. Phase checklist
All phases run on **E01 + E02 only** (§0). Code stays exercise-agnostic, with exercises discovered at runtime and filtered by `data.include_exercises`.

- [x] **Step 0:** environment, inventory, labels, sync census, references (E01, E02, E03-partial)
- [x] **Phase 1, keypoints and preprocessing** (2026-10-17; results in §5)
  - [x] `cvdc/data/`:
    - filename parser (runtime discovery of `E\d{2}(...)` folders, filtered by `data.include_exercises` = [E01, E02]; skip `__*` folders);
    - CSV join;
    - index CSV.
  - [x] Keypoint extraction for all local videos:
    - Tasks API; 2D (x, y, visibility) and world landmarks;
    - real frame timestamps, rotation handled, GPS never read;
    - square-padding check (§3.3 item 8);
    - per-variation detection rate.
  - [x] Sync classifier (audio → creation time → motion) for every group:
    - lag refinement with motion within ±0.3 s of the device-corrected audio lag;
    - `outputs/sync_report.csv`.
  - [x] Preprocessing:
    - root-centring (check hip visibility), torso scaling, interpolation, smoothing;
    - resample to fixed T, keeping the originals.
  - [x] Rep segmentation for every video:
    - reps per video;
    - plots;
    - compare against PF4 and the EAAQ "≥10 reps".
  - [x] Skeleton-only check renders in `outputs/checks/`.
  - [x] Third-review changes (§5.6): E02 horizontal adduction angle, left-arm score check, `audio` pair source (`w_audio` 0.8), best-view rep segmentation with boundary transfer.
- [x] **Phase 2, models** (2026-10-17; results in §6):
  - [x] ST-GCN with shared weights (`cvdc/models/`);
  - [x] projection head;
  - [x] task heads (score, CF1–CF3, PF, exercise, quality);
  - [x] weighted multi-positive InfoNCE (`cvdc/losses/cvdc.py`);
  - [x] pair builder for the sync / audio / rep / synthetic sources (`cvdc/data/pairs.py`);
  - [x] synthetic views (`cvdc/data/synthetic.py`);
  - [x] B2 3D input (`model.input: xyz`, MediaPipe world landmarks);
  - [x] KTPFormer 2D→3D lifter for B2 (`cvdc/models/lifter.py`, input `xyz_ktpformer`; fifth review, §6.5);
  - [ ] DepthAnything channel: later, optional, on a small subset only (fifth review);
  - [x] fifth-review changes: standardised targets, three VCI variants, bf16, runtime estimate (§6.5).
- [ ] **Phase 3, training** (run order E14):
  - [x] trainer (`cvdc/train/`): rep-level training as the default (T1, E12); targets standardised per training fold (T7); bf16; early stopping on an inner validation split of the training subjects, stopping epochs logged (T8, T8a); TensorBoard, checkpoints, config snapshot per run; full fine-tuning or linear probe (T10);
  - [x] smoke test (3 subjects, 2 epochs; `python -m cvdc.train.run --smoke`);
  - [x] **LOSO runtime estimate** (§6.5; upper bound without early stopping);
  - [x] **(a) GroupKFold-5: B1, B2-MediaPipe, B2-KTPFormer, Proposed** (CVDC pretraining + fine-tuning): within-group results vs mean / group-only / duration-only baselines, within-group VCI; **stop and report**; done 2026-10-18, results in §6b;
  - [x] **(a2) score setup** (user request after step (a)): label diagnostics and set-level baselines (rep-count-only, set-features); video head (mean + attention pooling + set rep count, T13); T6 per-video weighting; early stopping min 10 / patience 15 / EMA (T8b); items only where they vary (T14); Proposed variants (i) frozen, (ii) λ 0.1 / 1.0, (iii) full fine-tune (T15); re-run of B1, Proposed variants, B2-MediaPipe, B2-KTPFormer on GroupKFold-5; **stop and report**; done 2026-10-18, results in §6c;
  - [ ] (b) after approval: CVDC ablations on GroupKFold (C5);
  - [ ] (c) then the final LOSO table (per-fold-encoder shortcut, C7) for the 4 main methods.
- [x] **Phase 4, Module III** (2026-10-19, E01 + E02; results in §6d):
  - [x] templates (PT demo at 30 fps, mirrored, 3D angles; per-view bias and tolerance bands from expert videos scoring ≥ 95, refitted per fold);
  - [x] DTW alignment, angle deviation, ROM shortfall, speed and angular-velocity deviation, left/right symmetry;
  - [x] rules (joint, frames, degrees, severity);
  - [x] reference score per rep and per video, clinical agreement within groups (total, movement-quality score, items);
  - [x] JSON with the embedding-to-reference distance;
  - [x] embedding distance for B1, B2, Proposed (λ 1.0): score tracking and view stability;
  - [x] recording-noise test for PF3/PF4.
- [ ] **Phase 5, evaluation:**
  - **within-group metrics first** (expert-only, non-expert-only), pooled second (E9);
  - group-only baseline next to mean and duration-only (E10);
  - group-separability probe on the embeddings vs within-group score probes, incl. a view-matched subset (E11);
  - GroupKFold and LOSO;
  - stats;
  - cross-view protocol;
  - ablations (a)–(e) and (d′) (with and without rep-matched pairs), plus τ, embedding size, augmentation; E01 non-experts as a separate stratum (C5, D12);
  - VCI: centred 256-d (primary), raw, projection; within strata (E13);
  - B2 with MediaPipe-3D and KTPFormer-3D inputs (A10);
  - label-noise floor;
  - baselines (mean, duration-only, group-only);
  - embedding-distance analysis (within-group tracking and view stability done in Phase 4, §6d; repeat on LOSO);
  - UMAP/t-SNE with reference embeddings.
- [ ] **Phase 6, demo:**
  - `demo/live.py` on video files (real-time playback, field picker, leakage warning, .mp4 export, embedding-distance overlay);
  - `demo/app.py`.

## 8. Open items and questions
**Answered in the third review (2026-10-17):** the four Phase 1 questions (§4, applied in §5.6).

**Answered in the fifth review (2026-10-17):** the four Phase 2 questions (§4, applied in §6.5).

**Answered in the sixth review (2026-10-18):** KTPFormer licence (local non-commercial research use, cite the paper; the user informs the supervisor), headline VCI (within-group, centred), epoch budget with early stopping, Phase 3 run order (§4).

**Recorded, not open:** KTPFormer is less view-consistent than MediaPipe on 3 of 4 angles (only E01 shoulder elevation improves; §6.5); B2 is reported with both inputs.

**Addressed in Phase 4:** the E02 horizontal adduction angle's view dependence (per-view bias and bands, M7a, M8); the left-arm CF difference stays borderline (no flag under the agreed rule).

**Addressed in step (a2) (§6c):** the step (a) open items (noisy early stopping → T8b; linear probe and the joint objective → Proposed variants; T6 → per-video weighting, decided).

**Answered in the seventh review (2026-10-19)** (the step (a2) questions; §4): within-group score prediction is reported as a negative result with the label diagnostics (E21); Proposed = joint loss λ 1.0, frozen encoder as the alternative (T15); no angle / ROM inputs to the heads (T16); total score official, movement-quality score secondary (D13). Still my choices, not reviewed: the T14 thresholds (≥ 5 subjects off the mode, SD ≥ 0.05), the EMA factor 0.7 (T8b), and the reuse of the step (a) CVDC encoders (T15).

**Open after Phase 4 (for the review; §6d.5):**
- Rule specificity: within experts the rules fire as often in ≥ 95 videos as overall; keep the demo as the centre, widen k, or centre the posture channels on the experts?
- Speed reference: switch the score's speed centre from the experts (M5a) to the demo, whose angular-velocity deviation agrees best? (Post-hoc; would need a fresh check.)
- Non-expert end range goes against the ROM rule (ρ −0.31): the item numbering of the E01 EAAQ (which criterion is CF1, CF2, CF3) would let the rules follow the items.
- The demo needs the camera view for the per-view library (default front).
- Score and rule defaults (equal weights, d_zero 4, k 2, σ floors, run length) are my choices (M9–M12), not tuned.

**Other open items:**
- **EAAQ appendix** for E02–E09: the user is looking for it. Until it arrives, only E01 has questionnaire-derived rules.
- **E03 is out of scope for now**, and it is incomplete (199/550, including one failed download). If it comes back into scope, rerun the Step 0 check on the full folder and add it to `data.include_exercises`.
- **Other exercises** stay out of scope until the supervisor says otherwise. When one is announced and brought into scope, run the Step 0 check (§9).

## 9. Re-running the Step 0 check
See `scripts/step0/README.md` for the order and arguments. Minimum for a new exercise: `inventory.py` → `video_meta.py` → `scores.py` → `sync_av.py <EX>` → `sync_classify.py`, plus a pose sample (`select_groups_ex.py` → `extract_pose.py` → `sync_analysis3.py`) and a reference comparison.
