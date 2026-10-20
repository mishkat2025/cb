# Progress

Last updated 2026-10-20 (full scope, one model per exercise (T17): step 1 done and reported (§6h.9); Module III E03–E09 with the M18 calibration; demo; step 6, the per-exercise cross-view protocol, running).

**State (2026-10-20):** the scope is now **all nine exercises** (user, ninth review, §4). The full-data Step 0 check (§3.7) found E03–E09 usable with caveats; the user brought all seven in. **Phase 1 is done for the full scope (§5b):** keypoints for all 3,685 videos (MediaPipe Hands for E06), sync, reps, packs and the KTPFormer lift, verified with the pilot's checks plus a comparison with the pilot and with Step 0. The E05 PF1 check (user answer 5) found no wrong exercise (§5b.7). **Tenth review (§4) applied:** E06's wrist angle is assessed in the side views only (§5b.3); the rep-phase fix is accepted (P6c); the rep signals of E03 and E09 were re-chosen by cross-view agreement (now forearm elevation and the 2D trunk inclination, §5b.10); the CVDC pairs are built with per-exercise DTW angles (§5b.11). **Work stops here, as asked: nothing is trained on the full scope until the user confirms the final evaluation scheme**, and no LOSO is started (the final table is planned as GroupKFold-10 until the supervisor decides, E1). **All E01/E02 results stay as the pilot**, unchanged; the full scope runs from `configs/data_full.yaml` into new folders (`data/full/`, `outputs/full/`, `runs/full/`; D15).

**Update (2026-10-20, eleventh and twelfth reviews, §4, §6h):** E09's rep-matched pairs now use the arm angles, the set with the lowest known-warp error (§6h.1). A GroupKFold-10 run of one model shared by all nine exercises was started and then **stopped by the user** (supervisor decision: **one model per exercise**, T17); its partial outputs are kept, labelled as aborted (§6h.2). The per-exercise runner (`python -m cvdc.train.run_per_exercise`) is smoke-tested on all nine exercises, the runtimes are estimated (§6h.4: main runs ~7.7 h, cross-view ~14–17 h, LOSO per exercise ~38 h with the shortcut), **step 1, GroupKFold-5 per exercise for B1, B2-KTPFormer, Proposed (λ 1.0) and the frozen Proposed, is done** (6.7 h; results §6h.9), Module III runs for E03–E09 (§6h.6, calibration M18 §6h.8), the demo is built (§6h.7), and **step 6, the per-exercise cross-view protocol, is running**.

**Pilot state (E01/E02, unchanged since 2026-10-19):** Step 0 and **Phase 1 (keypoints and preprocessing) are complete for E01 and E02** (§5), including the supervisor's follow-up changes from the third review (§4, §5.6). The supervisor's warning on expert/non-expert confounding is part of the evaluation plan (§4, fourth review; `DESIGN_DECISIONS.md` E9–E12). **Phase 2 (models, CVDC loss, pair builder) is implemented and smoke-tested** (§6), including the fifth-review changes (§6.5). Phase 3 **step (a)** (§6b) and **step (a2)** (§6c) are done; the seventh review's decisions are recorded (§4). **Phase 4 (Module III, embedding-to-reference distance, recording-noise test) is done** (§6d), with the eighth review's changes (§6d.6–6d.7). **Phase 5 has started:** **A, the cross-view protocol** (train on two views, test on the held-out third, 4 main methods, GroupKFold-5) is done (§6e); **B, the CVDC pair-source ablation on the VCI** is done (§6f); the **LOSO runtime is estimated** (§6g: ~16 h for the 4 methods). **Work stops here, as asked: LOSO has not started**, and the other ablations (τ, embedding size, 2 vs 3 views, augmentation) have not run.

**Scope:** the pilot is **E01 and E02** (supervisor, 2026-10-17; `configs/data.yaml`). **Since 2026-10-20 the full scope is all nine exercises** (`configs/data_full.yaml`, `data.include_exercises: []`; D0).

Read this together with `DESIGN_DECISIONS.md` (every open choice, with its status) and the task specification `../CVDC_implementation_prompt.md`.

---

## 0. Scope and limitations

**Scope change (user, 2026-10-20):** expand from E01/E02 to **all usable exercises** now that the full dataset is on disk. First step, done: the Step 0 check of E03–E09 (§3.7), with a verdict per exercise. LOSO is not started. The E01/E02 work below is kept as the **pilot**: its results, tables and caches are not deleted or overwritten, and full-scope work writes to new output folders (D15). **Approved and started (ninth review, §4): all nine exercises; Phase 1 of the full scope is done (§5b).** The limitations below describe the pilot; the full scope adds those of §3.7.9 (E04, E06, E07 experts only; E07–E09 all male; score skew; PF3/PF4 view noise) and the E06 wrist-angle problem (§5b.3).

**Pilot scope:** E01 (Abduction) and E02 (Adduction). That is 942 videos from 51 subjects: 24 experts, who did both exercises, and 27 non-experts, who did E01 only.

**Not in the pilot:** E03–E09. They have full-scope Phase 1 data (§5b); nothing is trained or evaluated on them yet.
- The E03 Step 0 results (§3.4, partial data; superseded by §3.7) and its template decision (V3, §3.6, re-confirmed in §3.7.7) stay documented.
- The pipeline **discovers exercises at runtime**, with an include list in the config (`[E01, E02]` in the pilot config, `[]` = all in `configs/data_full.yaml`); bringing in E03–E09 was a config change plus their rep signals (M16).

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
| Disk | D: is a **7200 rpm HDD** (WD10EZEX); full dataset 154 GB (E03–E09 117 GB). Jobs that stream whole videos (audio census) are disk-bound: 21 min for E03–E09; running two such jobs at once stalls both (seen on 2026-10-20). Pose extraction is CPU-bound (~16 MB/s), so the disk is not its limit. |
| Git | this repo (`cvdc-physio/`); identity `mishkat <sbmishkat10@gmail.com>` |

## 2. Data on disk

**Layout since 2026-10-20 (full download):** the nine exercise folders are in **`D:\cb\mobiphysio\`**; the scores CSV, `MobiPhysio.md` and the PT reference folder stay in `D:\cb`. The old partial folders `D:\cb\E01(…)`, `E02(…)`, `E03(…)` are gone. Never modify any of it.

| Folder / file | Content | Status |
|---|---|---|
| `mobiphysio\E01(Abduction)` … `E09(Back _Extension)` | 558 / 384 / 550 / 383 / 511 / 354 / 251 / 356 / 339 videos = **3,686** | **complete**: every count equals the Dataverse listing and the README's Table 1 per participant and exercise. E01/E02 files are byte-size and score identical to the pilot's (§3.7). |
| `__E03(Lateral_Rotation)`, `__E03(Lateral_Rotation)_Error.txt`, `___All_Errors.txt` | download-error markers of the first, partial download | stale: `E03_P04_AL_VLL_GM.mp4` is now present in `mobiphysio\E03(…)` |
| `Reference-Model Videos by PTs` | 14 physiotherapist demos | see §3.6 and §3.7.7 |
| `Physiotherapist Exercise Marking.csv` | scores, 3 judges × 7 EAAQ items | all 3,686 videos; every video has exactly 3 judge rows |
| `MobiPhysio.md` | dataset README | names E08 "Both Leg Flexion" (the paper says Lumbar Flexion; §3.7.7) |

**Video root:** `data.video_root` (D1) points the full-scope config at `D:\cb\mobiphysio\`; the pilot config keeps the old layout (its `data/index.csv` and caches are untouched and are not re-indexed).

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

### 3.4 E03 check (requested 2026-10-17, on the partial download; superseded by §3.7)
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
- The pool behind this comparison is only 4 front-view videos scoring ≥ 80. Re-check when the rest of E03 arrives. **Re-checked on the full data in §3.7.7: V3 is still the closest.**

### 3.7 Step 0 on the full dataset: E03–E09 (2026-10-20)
User request (2026-10-20): run the Step 0 check on every new exercise with the same method, report usable / usable with caveats / not usable per exercise, estimate the full-scope runtimes, and stop; do not start LOSO; keep the E01/E02 pilot untouched. How to run: `scripts/step0_full/README.md`. Outputs: `outputs/step0_full/` (gitignored; nothing in `outputs/step0`, `outputs/phase*`, `data/index.csv` or the pilot caches was rewritten). The scripts reuse the Phase 1 code (parser, CSV loader, extractor, preprocessing, rep segmenter, sync tests).

#### 3.7.1 Files and labels
- **3,686 / 3,686 filenames parse (100%)** with the D2 regex. One name is not in the strict form: **`E03_P41_AF_VP_GM`** (variation code `VP`, not one of the seven; the CSV has the same name). It is a single front-view non-expert clip, kept with variation `unknown` (D16). Every exercise code matches its folder; no duplicate ids, no file under 0.5 MB, no non-video files.
- **Scores CSV:** all 3,686 files have a score and every CSV row has a file; 3 judge rows per video. The only name mismatch is the known `E01_P26_Al_VFL_GF` typo.
- **Subjects:** 58 participants (P01–P24 experts, P26–P62 non-experts; 25, 44, 51, 54 absent), each with one gender in every exercise. Per-participant counts equal the README's Table 1 for every exercise.
- **The E01/E02 files in `mobiphysio\` are the pilot's files:** the same 942 names (941 in the pilot index + the dropped LR copy, D7), identical sizes and scores.
- **Bad files: none.** All 3,686 open and decode their first frame and a frame in their last second (8 clips under 1.5 s were only checked at the start). All have audio. 8 have no `creation_time`: the 4 pilot ones plus `E03_P34_AF_VLL`, `E03_P60_AF_VLL`, `E05_P34_AF_VLL`, `E05_P60_AF_VML` (non-experts).
- **Duplicates: no new ones.** MD5 of every same-subject, same-view, same-size pair: only the known `E01_P60_AF_VLR` = `VML` copy; the 3 new same-size pairs (`E03_P53_AF` LJ/O, `E04_P19_AL` LL/ML, `E05_P20_AR` LL/ML) differ in content. None of the 36 new LR videos is an audio copy of another variation. The 4 new files without `creation_time` match no other video of the same subject and view (best audio peak ratio 1.2–2.7; the known copy has 284).
- **Partial or separate takes:** `E05_P19_AF_VLJ_GM` (expert) lasts 4.2 s against 26–27 s for its L/R partners (it ends with them: the front phone caught only the last seconds); the R views of `E05_P03` LL and ML were recorded 24 s and 61 s away from F/L (separate takes).

| Exercise | Videos | Subjects exp / non-exp | Views F / L / R | Complete F+L+R groups exp / non-exp | Mean score exp / non-exp | ≥ 95 (front, exp) | Bins ≥ 80 / 50–79 / < 50 | Label-noise floor* | Median duration exp / non-exp (s) |
|---|---|---|---|---|---|---|---|---|---|
| E03 Lateral rotation | 550 | 24 / 27 | 255 / 148 / 147 | 125 / 17 | 75.4 / 44.4 | 32 (18) | 135 / 293 / 122 | 3.38 | 26.3 / 3.1 |
| E04 Medial rotation | 383 | 24 / 0 | 129 / 126 / 128 | 125 / 0 | 75.8 / – | 24 (10) | 112 / 271 / 0 | 2.45 | 25.0 / – |
| E05 Circumduction | 511 | 23 / 22 | 233 / 141 / 137 | 119 / 16 | 87.0 / 40.6 | 133 (44) | 254 / 153 / 104 | 4.76 | 26.8 / 2.9 |
| E06 Wrist extension | 354 | 22 / 0 | 119 / 117 / 118 | 114 / 0 | 94.4 / – | 253 (92) | 322 / 32 / 0 | 3.16 | 28.8 / – |
| E07 Hip joint flexion | 251 | 14 / 0 | 83 / 84 / 84 | 83 / 0 | 82.5 / – | 45 (15) | 144 / 98 / 9 | 2.76 | 40.3 / – |
| E08 Both leg flexion | 356 | 14 / 16 | 160 / 98 / 98 | 84 / 14 | 93.4 / 32.9 | 168 (61) | 234 / 47 / 75 | 4.04 | 45.0 / 4.5 |
| E09 Back extension | 339 | 14 / 13 | 150 / 94 / 95 | 83 / 11 | 83.6 / 9.5 | 103 (35) | 179 / 60 / 100 | 2.68 | 47.2 / 3.8 |

\*As in §3.1: RMSE of each view's score against its expert group mean.

- **Variations:** experts have FL/ML/LL/LJ/HJ/O (no LR) in every exercise; non-experts have all seven but most variations front-only, as in E01. **E07–E09 are all male** (and E08/E09 non-experts too).
- **Label structure** (`labels_check.csv`): PF3 = 1 exactly when PF4 = 1 in every exercise (D5 holds). Share of the within-expert score variance carried by the repetition items PF3 + PF4: E03 0.41, **E04 0.84**, **E05 0.80**, E06 0.59, E07 0.56, **E08 0.81**, E09 0.54 (pilot: E01 0.50, E02 0.61). PF1, PF2 and CF1–CF3 are identical across the views of every expert performance in E04 and E06–E09 (as in the pilot); in E03 CF2 and in E05 CF1/CF2 differ across views in some groups. Items active under the T14 rule (expert strata): E03 PF4 CF1 CF2; E04 PF4 CF1 CF2; E05 PF4 CF1 CF2 CF3; E06 CF1 CF2 only; E07 PF2 PF4 CF2 CF3; E08 PF4 CF3; E09 PF1 PF2 PF4 CF1 CF2.
- **Score skew:** E06 experts 49% at exactly 100 (SD 8.0, minimum 67.6; no video below 50); E08 experts 56% at 100 (SD 9.3); E04 SD 8.7, minimum 57.1 (no video below 50); E09 experts 31% at 100. **E09 non-experts score a median of 0** (mean 9.5).
- **Subjects whose mean PF1 is below 0.5 in an exercise** (the E03 P02 signature; D14): E03 P02 (PF1 = 0 on all 18, mean 22.6; known); E05 non-experts P30, P34, P57, P58, P59 (PF1 0) and P32 (0.38); E08 non-experts P29, P38, P41; **E09 expert P21** (PF1 = 0 on all 18 videos, mean 52.0) and E09 non-experts P29, P32, P35, P37, P38, P39, P41, P42. Side note on the pilot: in E02, experts P07 and P10 also have PF1 = 0 on all their videos (mean 48.0 / 52.4); nothing in the pilot results changes. PF1's meaning is not known (EAAQ pending, §3.7.8), and the motion shows it is not always "wrong exercise" (E09 P21, §3.7.5).

#### 3.7.2 Videos
- 25–60 fps (E07: all 30 fps), mostly 1080×1920 with a rotation flag; the non-expert batch also has landscape, 1440×1440, 720p and the 352×640 LR files, as in E01.
- E03–E09 hold **2.35 M frames** (all nine exercises 3.07 M).

#### 3.7.3 Synchronization: all multi-view groups of E03–E09
Same evidence order as §3.5 (audio → creation time → motion), with two changes (S8):

| Exercise, group | synced: audio | synced: creation time | synced: creation time, start times | not synced | unknown |
|---|---|---|---|---|---|
| E03 expert | 38 | 90 | 1 | 0 | 0 |
| E03 non-expert | 0 | 0 | 0 | 22 | 0 |
| E04 expert | 42 | 86 | 0 | 0 | 1 |
| E05 expert | 49 | 72 | 3 | 1 | 1 |
| E05 non-expert | 0 | 0 | 0 | 17 | 0 |
| E06 expert | 38 | 80 | 2 | 0 | 0 |
| E07 expert | 35 | 46 | 2 | 0 | 1 |
| E08 expert | 34 | 48 | 2 | 0 | 0 |
| E08 non-expert | 0 | 0 | 0 | 14 | 0 |
| E09 expert | 42 | 40 | 1 | 0 | 1 |
| E09 non-expert | 0 | 0 | 0 | 11 | 0 |

- **Experts: simultaneous, as in the pilot.** 2,179 of 2,222 expert view pairs have |ΔC| ≤ 3 s; audio additivity passes for 30–51% of expert F+L+R groups (median error 20–36 ms); audio lags vs F median 0.46 s, max 2.56 s.
- **creation_time stamps the end of the recording more often than the start.** On audio-synced pairs, ΔC fits "end time" (ΔC = Δduration − lag) to a median 0.7 s (p90 1.3 s) against 0.9 s (p90 2.3 s) for "start time", and fits it better in 68% of pairs. In the long lying exercises the phones stopped up to ~15 s apart, so the plain rule (|ΔC| ≤ 3 s, in effect comparing stop times) missed groups whose start times agree. New rule (S8): synced (medium) when every pair's start times (creation − duration) agree within 3 s: 11 groups. It passes 2 of 5,902 control pairs (0.03%; the plain rule 0), so it stays medium confidence until motion confirms.
- **Non-experts: never simultaneous.** Their views were recorded 8–473 s apart, in windows that cannot overlap: 64 non-expert groups are not synced. One E03 group (P59 FL) passed the audio test (peak ratio 2.6 on all three pairs, additivity error 0.1 ms) although its three ~2 s clips were recorded 14–39 s apart: the same sound cue in separate takes. **Non-overlapping recording windows now override audio** (S8). The offsets also change between exercises for the same subject (P59 F→R: 75 s in E01, 39 s in E03, 30 s in E05), so these are not simultaneous recordings with offset clocks.
- **Unknown (4 expert groups):** E04 P12 LL, E05 P03 LL (R view 24 s away), E07 P19 LJ (L+R only), E09 P05 ML (R view 16–17 s away). **Not synced (1 expert group):** E05 P03 ML (R view 61 s later).
- **Motion test on the pose sample** (§3.7.4; pairs searched within ±0.3 s of the device-corrected audio lag; Phase 1 rep-timing test, < 42 ms): confirmed E03 36%, E04 26%, E05 5%, E06 3%, E07 24%, E08 21%, E09 48% of same-group pairs; **controls 2 of 258 testable pairs (0.8%; pilot 0.7%)**. Low for E05 (circular arm path) and E06 (small wrist movement), where audio and creation time carry the classification. In the two sampled unknown groups motion confirmed 0 of 3 and 1 of 3 pairs, so they stay unknown. Device offsets from the motion-confirmed pairs: F→L +100 ms, F→R +36 ms, L→R −61 ms (n = 20 / 23 / 19), consistent with the pilot's +93 / +57 / −44 ms (S5).

#### 3.7.4 Pose sample: detection and posture
Sample (`pose_sample.csv`, seeded): per exercise 6 expert subjects × 2 variations × 3 views (36 videos, variations spread over all six), 8 more front-view expert videos scoring ≥ 80 (reference pool), and for E03/E05/E08/E09 one non-expert F+L+R group plus 8 non-expert single clips; E09 P21 FL added; E03 P02 was drawn anyway. **355 videos (337k frames) and the 9 PT demos of E03–E09**, Phase 1 settings (full model, 960 px, PyAV timestamps), 22 min with 12 processes (256 frames/s). Keypoints in `data/cache/step0_full/` (gitignored).
- **Detection:** mean share of frames with a pose: F ≥ 99.5% and R ≥ 98.3% in every exercise; L 88.4% (E04), 88.5% (E05), 90.6% (E07), 92.0% (E09), 92.7% (E06), 93.5% (E08), 98.9% (E03). HJ is the weakest variation in most exercises (84–99%; E04 LL 87.5%). 19 of 355 videos fall below 90% and 3 below 50% (`E04_P11_AL_VLL` 0%, `E07_P08_AL_VHJ` 29%, `E05_P13_AL_VHJ` 47%): all left-view LL/HJ takes, the pilot's pattern (P7).
- **E07–E09 are lying exercises, and MediaPipe detects them well.** The 2D trunk axis lies 79–106° from image vertical in E07/E08 (supine, body across the frame in all three views) and 52–61° in E09 (prone, trunk raised); detection 98.3–99.9% in F/R and 90.6–93.5% in L. The far-side limbs are often low-visibility: E08 knees visible (≥ 0.5) in 58/68% of F frames, E09 knees 60/52% of F frames; E07 arms 39–60% in F (not needed). 3D angles of lying people cannot be checked against a 3D reference (none exists; E8), but the 3D hip angle gives consistent rep counts across views (E08 92%, §3.7.5).
- **Upper-limb side views self-occlude the far arm**, as in the pilot (AL: left elbow/wrist visible in 9–21% of frames in E03–E05).
- **Root:** both hips visible in ≥ 87% of frames in every exercise and view, so the mid-hip root (P5) works for all nine.

#### 3.7.5 Rep segmentation feasibility
Phase 1 segmenter (P6) with per-video adaptive thresholds (prominence 0.4×, minimum amplitude 0.3× the signal's p5–p95 range; oriented so the first second is the rest level). Feasibility = the share of synced expert F+L+R groups of the sample (9–12 per exercise) whose three per-view counts lie within ±1, the P6 criterion; subjects with mean PF1 < 0.5 are left out. Pilot reference (per-view segmentation, before boundary transfer): E01 69%, E02 42%. Synced groups will again share one rep set segmented on their best view (P6b). `rep_signal_agreement.csv`:

| Exercise | Best signal (3D unless stated) | Within ±1 | Exact | Median reps per view | Others |
|---|---|---|---|---|---|
| E03 | wrist-path PC1 of the moving arm | **90%** | 60% | 9 | all-joint 2D PC1, 2D wrist height 80%; forearm elevation 70% |
| E04 | all-joint 2D PC1 / wrist-path PC1 / forearm elevation | **64%** | 55% | 9 | shoulder elevation 36% |
| E05 | 2D wrist height / wrist-path PC1 | **82%** | 27% | 10 | shoulder elevation, forearm elevation 73% |
| E06 | 2D wrist height | **75%** | 58% | 9 | wrist angle from the Pose hand points: 25% (3D), 33% (2D) |
| E07 | all-joint 2D PC1 | **70%** | 30% | 9 | summed hip flexion 60% |
| E08 | hip flexion | **92%** | 67% | 10 | summed knee flexion 92% |
| E09 | hip angle / 2D trunk inclination | **78%** | 22% | 10 | trunk–thigh angle 56% |

- Expert sets hold a median of 9–10 reps per view, matching the EAAQ's "≥ 10" (PF4). Non-expert clips hold 1–3 reps (E08/E09: 0–1).
- As for E02, the best signals of E03–E05 and E07 are not joint angles: they are for **segmentation only**; Module III keeps joint angles (§3.7.7).
- **E03 P02** gives 2 reps per view against 9 for everyone else; **E09 P21** gives 10 per view and the back-extension posture (§3.7.6).

#### 3.7.6 Known issues
- **E03 P02 (wrong exercise): confirmed on the full data.** PF1 = 0 on all 18 videos, mean 22.6 (every other E03 expert: subject means 68.6–85.1, no video below 64.3), 2 reps per view. Excluded from the templates and from the exercise labels, kept in score regression (D8).
- **E06 needs MediaPipe Hands.** The Pose hand points (index, pinky) give no usable wrist angle: cross-view rep agreement 25–33%. Module III's wrist-extension angle needs the Hands model (`hand_landmarker.task`, ~7.8 MB, Google), which is **not downloaded and needs approval** (P1). Rep segmentation and the networks can use Pose alone (2D wrist height 75%).
- **E07–E09 lie down** (E07/E08 supine, E09 prone); detection is fine (§3.7.4).
- **E09 P21 is not a wrong-exercise case:** PF1 = 0 on all 18 videos, but the posture matches the back-extension demo and the high scorers (Chamfer 2–6° against 4.1° for a high scorer against the rest of the pool) with 10 reps per view (one high-jitter left-view video looks like the E08 demo instead, probably mis-tracked). What PF1 rates is unknown until the EAAQs arrive (§3.7.8).
- **Other issues found:** the `VP` variation (`E03_P41_AF_VP_GM`); `E05_P19_AF_VLJ_GM` is a 4.2 s partial take; the E05 P03 R views are separate takes; the E09 non-experts score a median of 0 and 8 of 13 have PF1 ≈ 0; E06 and E08 experts are about half at 100 and E04 has no low scores (little score variance for regression); E07 has only 14 subjects, all male, and no non-experts; the pipeline config needs the new video root (§2).

#### 3.7.7 PT references and the joint angles for Module III
Chamfer distance in a per-exercise 3D angle space (moving side first) between the demo's frames and the frames of the sample's expert videos scoring ≥ 80 (front view; all views in brackets). Spaces: upper limb = shoulder elevation, elbow flexion, forearm elevation and azimuth in the trunk frame; E06 = Pose-point wrist deviation, elbow flexion, shoulder elevation; E07/E08 = hip and knee flexion of both legs; E09 = trunk–thigh angle, hip and knee flexion (`ref_compare.csv`).

| Demo | Exercise | Apparent view, posture | Reps found, s per rep | Expert s per rep (sample) | Chamfer front (all) | Pool front / all |
|---|---|---|---|---|---|---|
| `Later Rotation V3.mp4` | E03 | front, upright, 90/90 | 2, 4.8 s | 1.8 | **13.5** (17.2) | 13 / 19 |
| `Lateral Rotation.mp4` (v1) | E03 | oblique | 2, 4.9 s | 1.8 | 19.1 (21.0) | 13 / 19 |
| `lateral rotation V2.mp4` | E03 | front, arms at the side | 2, 2.5 s | 1.8 | 55.3 (52.9) | 13 / 19 |
| `Medial Rotation.mp4` | E04 | oblique, right arm | 1, 4.1 s | 2.0 | 16.5 (20.9) | 9 / 10 |
| `Circumduction.mp4` | E05 | side, right arm | 4, 2.7 s | 2.1 | 23.1 (19.2) | 19 / 37 |
| `Wrist Extension.mp4` | E06 | front-oblique, both wrists | 2, 12.0 s | 1.7 | 6.0 (7.0) | 19 / 40 |
| `Hip Joint Flection.mp4` | E07 | side, supine, right leg | 1, 3.1 s | 2.9 | 7.8 (13.9) | 14 / 26 |
| `Both Leg Flection.mp4` | E08 | side, supine, both legs | 2, 5.9 s | 3.5 | 9.5 (10.5) | 19 / 40 |
| `Back Extension.mp4` | E09 | side, prone | 2, 4.9 s | 3.4 | 2.9 (5.6) | 15 / 29 |

- **E03: V3 is still the closest** (13.5° vs 19.1° for v1 and 55.3° for V2), now on 13 high-scoring front videos instead of 4 (the numbers are not comparable with §3.6's 7.5–10.8°: this space adds the forearm azimuth). The dataset is right-arm external rotation in 90/90: shoulder elevation median 89°, forearm elevation from ~130° (forward-down) to ~27° (up). V3 is 2.6× slower than the experts (4.8 vs 1.8 s per rep), so speed comes from expert data (M5a).
- **E04:** the demo is oblique and holds the upper arm lower (elevation median 40° vs 69° in the dataset): use the per-view bias from the experts, as in M8. The dataset is internal rotation with the arm abducted ~70°: forearm elevation from ~70° (about horizontal) to ~145° (down).
- **E05:** elevation 19–152° in the demo vs 7–163° in the dataset; the circle's wide posture space explains the larger distance.
- **E06:** posture matches (6.0°), but the demo holds each extension (12 s per rep vs 1.7 s): speed from expert data.
- **E07:** the moving hip flexes to ~125° with the knee bent to ~110–125° while the other leg stays straight (hip 13°, knee 9° in the demo; median 13° / 14° in the dataset).
- **E08 = "Both Leg Flexion"** (README, demo name): supine, both hips and knees flex together (hips to ~125–145°, knees to ~130–145°), i.e. knees to chest, which flexes the lumbar spine. The paper's "Lumbar Flexion" names the same videos.
- **E09:** prone trunk extension, trunk–thigh angle 4–49° in the demo vs 2–45° in the dataset; knees stay straight on the floor.
- **Demos are slower than the experts in every exercise** (1.1–7×), so the speed reference comes from expert data everywhere (M5a).

**Joint angles proposed for Module III** (3D world landmarks, moving side, left side mirrored; M16, to be refined against the data when each exercise is brought in):

| Exercise | Main angle (ROM, rep score) | Posture checks |
|---|---|---|
| E03 Lateral rotation | forearm rotation about the upper arm, read as forearm elevation in the trunk frame (~130° → ~27°) | shoulder abduction ~90° (90/90), elbow flexion ~90°, shoulder asymmetry (shrug), trunk lateral lean |
| E04 Medial rotation | forearm rotation, as forearm elevation in the trunk frame (~70° → ~145°) | shoulder elevation (~70° in the dataset), elbow flexion, trunk lean |
| E05 Circumduction | shoulder elevation over the circle (ROM) + wrist-path circularity and amplitude (two principal components of the wrist path) | elbow extension, trunk lean, shoulder hitch |
| E06 Wrist extension | wrist extension: forearm vs hand, **from MediaPipe Hands**, both wrists | elbow flexion, forearm position (shoulder elevation) |
| E07 Hip joint flexion | hip flexion of the moving leg (trunk vs thigh, 0 → ~125°) | knee flexion of the moving leg; the other leg's hip and knee staying extended (pelvis stability) |
| E08 Both leg flexion | hip flexion of both legs | knee flexion of both legs, left/right symmetry |
| E09 Back extension | trunk extension (trunk vs thigh line, 0 → ~45°) | knee flexion (legs straight on the floor), left/right symmetry |

#### 3.7.8 EAAQ questionnaires for the other exercises
- **They are not in the dataset.** The Dataverse dataset holds exactly 3,688 files: 3,686 videos, 1 markdown file (the README) and 1 CSV (the scores) (Dataverse API, file counts by content type, read 2026-10-20; nothing downloaded). The local CSV has item numbers only ("Primary Factors 1–4", "Control Factors 1–3"), no question text.
- The Data in Brief paper says one sample questionnaire is in Fig. 4 (E01) and "the others are provided in the Appendix", but the local PDF (16 pages) has no appendix; it points to supplementary material online (doi:10.1016/j.dib.2026.112635). The publisher's page refused automated access (HTTP 403), so I could not check it. **The E02–E09 EAAQs are most likely in the article's online supplementary material**; someone with access needs to download it.

#### 3.7.9 Verdict per exercise
**None is unusable. All seven are usable with caveats**:

| Exercise | Verdict | Why usable | Caveats |
|---|---|---|---|
| **E03** Lateral rotation | usable with caveats | 24 experts + 27 non-experts, 125 synced expert F+L+R groups, detection ≥ 98.9% in every view, best rep signal 90%, V3 template confirmed | P02 wrong exercise (excluded from templates and exercise labels); `VP` clip; non-experts are single-rep, mostly front-only, not synced (the E01 confound); V3 2.6× slower than the experts; CF2 differs across views in some groups |
| **E04** Medial rotation | usable with caveats | 24 experts, 128 synced groups, F/R detection ≥ 99.6% | experts only (no group contrast); narrow scores (SD 8.7, none below 57) with 84% of the variance in PF3/PF4; rep agreement 64% (between E01 and E02); L view 88% detection; demo posture differs (arm lower) |
| **E05** Circumduction | usable with caveats | 23 experts + 22 non-experts, 124 synced groups, rep signal 82% | 6 non-experts with PF1 ≈ 0; 80% of the expert variance in PF3/PF4; E05_P19 partial front take; E05 P03 separate R takes; low motion-sync confirmation (5%); CF1/CF2 differ across views in some groups |
| **E06** Wrist extension | usable with caveats; Module III blocked until MediaPipe Hands is approved | 22 experts, 120 synced groups, rep counting from wrist height 75% | Pose hand points give no usable wrist angle; experts only; 49% of videos at 100 and none below 67.6, so little score variance; only CF1/CF2 active; demo 7× slower |
| **E07** Hip joint flexion | usable with caveats | 14 experts, 83 synced groups, supine poses detected 91–100%, demo matches (7.8°) | smallest set: 14 subjects, all male, experts only; rep agreement 70% (moderate) |
| **E08** Both leg flexion | usable with caveats | 14 experts + 16 non-experts, 84 synced groups, rep signal 92%, demo matches (9.5°) | experts 56% at 100 with 81% of the variance in PF3/PF4; only PF4/CF3 active for experts; all male; 3 non-experts with PF1 ≈ 0; knees often low-visibility in F |
| **E09** Back extension | usable with caveats | 14 experts + 13 non-experts, 83 synced groups, rep signal 78%, demo matches (2.9°) | non-experts score a median of 0 (8 of 13 with PF1 ≈ 0), so the non-expert stratum has almost no score range; expert P21 PF1 = 0 everywhere (movement correct); all male |

**Across the full scope:** 58 subjects; 756 expert multi-view groups in E03–E09, 751 of them synced; 64 non-expert groups, none synced. The pilot's limitations carry over (no error labels, 3 views, scores skewed, PF3/PF4 view noise), and the repetition items dominate the expert score in E04, E05 and E08 even more than in E01/E02, so within-group score prediction is likely to stay hard (E21).

#### 3.7.10 Runtime estimates for the full scope (all 9 exercises)
From the pilot's measured timings (`scripts/step0_full/runtime_estimate.py`, `runtime_estimate.csv`): one GPU job at a time; training time proportional to the number of training reps, with the pilot's epochs to early stopping. **Estimated training reps: 26,564** (E01/E02 measured 7,095; E03–E09 = expert videos × the sample's median 9–10 reps per video × 0.9 for dropped edge reps + 1 rep per non-expert clip): **3.74× the pilot**.

| Item | Estimate | Basis |
|---|---|---|
| Phase 1 pose extraction of E03–E09 (2.35 M frames) | **2.1–2.6 h** | pilot 315 frames/s; this session's sample 256 frames/s (12 CPU processes) |
| + MediaPipe Hands pass for E06 | ~0.3 h | assumed to cost as much as the pose pass (not measured) |
| + audio sync stage of Phase 1 | ~0.4 h | the E03–E09 audio census took 21 min (disk-bound) |
| + KTPFormer lifting (B2) | 0.8 h (GPU) | 850 frames/s (A10) |
| **Main GroupKFold-5 run** (B1, B2-MediaPipe, B2-KTPFormer, Proposed λ 1.0, with the CVDC pretraining) | **7.9 h** (B1 1.0, B2-MP 0.9, B2-KTP 0.9, Proposed 2.1, CVDC pretraining 2.9) | pilot step (a2) + step (a) pretraining 2.1 h × 3.74 |
| **Cross-view protocol** (3 rotations × 5 folds × 4 methods) | **14.6 h** | pilot 3.9 h × 3.74 |
| **LOSO** (58 folds, per-fold-encoder shortcut C7, 4 methods) | **70.5 h central** (49–97 h; B1 14.7, B2-MP 13.4, B2-KTP 13.5, Proposed 28.9) | pilot E26 cost per inner-training rep and epoch × 58 folds × ~20,900 inner-training reps (4.35× the pilot's LOSO) |

- Without E06: 7.0 h, 13.0 h and 62.9 h.
- Uncertain by at least ±30%: more data per epoch may change when early stopping fires, and the rep counts come from a sample. **Full-scope LOSO is about 3 days of GPU** (the pilot's 16 h × 4.35); options in §8.

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

**Scope change (user, 2026-10-20): full dataset.** The full MobiPhysio dataset (all 9 exercises) is downloaded. Expand from E01/E02 to all usable exercises; do not start LOSO. Step 1: the Step 0 check of every new exercise (E03–E09) with the same method (parse rate, counts, CSV join, duplicates and bad files, F/L/R groups and sync classes, rep-segmentation feasibility, match to the PT demos and the Module III angles, the known issues E03 P02 / E06 Hands / E07–E09 lying, and whether the other EAAQs exist in the dataset), a verdict per exercise, and runtime estimates for Phase 1 extraction, the main GroupKFold runs, the cross-view protocol and LOSO; then stop. Keep every E01/E02 result as the pilot and write new results to new folders. Done: §3.7 (D0, D14–D17, S8, M16, E27).

**Twelfth review (user, 2026-10-20): one model per exercise (supervisor decision; final plan)** (`DESIGN_DECISIONS.md` T17, E1, E14, E28; applied in §6h):
0. **Stop the shared-model run**; keep its partial outputs in their own folder, labelled as aborted. Keep the code changes (E09 arm angles, set-feature angles, group probe, D8).
1. For **each exercise E01–E09 separately**, using only that exercise's videos: B1, B2-KTPFormer, Proposed (CVDC pretraining on that exercise's multi-view pairs + λ = 1.0 joint fine-tuning) and the frozen-encoder Proposed as the reported alternative. **No exercise-classification head.** Same training settings as the pilot otherwise.
2. **GroupKFold-5 per exercise**, grouped by subject, balanced expert / non-expert where both exist; CVDC pretraining inside each fold on training subjects only.
3. Report per exercise: within-group score results (expert / non-expert) vs the mean, group-only, duration-only, rep-count-only and set-feature baselines; the movement-quality score; the within-group VCI; the view stability of the distance to the PT reference; the group probe. Flag exercises with few subjects (e.g. E07: 14) as low-confidence.
4. The pilot (shared model, E01/E02) results stay as a reference only.
5. One GPU job at a time; CPU work in parallel: Module III for E03–E09 (incl. the E06 side-view wrist angle) and the Phase 6 demo code (video-file input; the demo loads the selected exercise's model).
6. After the main runs: the per-exercise cross-view protocol (rotate F / L / R) for B1, B2-KTPFormer and Proposed, on exercises with synced multi-view data.
7. LOSO per exercise later (supervisor / cloud decision); do not start it.
Before launching: runtime estimates for 1–2, 6 and per-exercise LOSO, a smoke test of the per-exercise runner; record the decision, commit and report after each step.

**Eleventh review (user, 2026-10-20):** (1) E09 alignment angles: let the warp-recovery test decide, also testing trunk–thigh + knees + arm angles combined; use the lowest median warp error and rebuild the E09 pairs (done, §6h.1). (2) Final scheme GroupKFold-10 (subject-grouped, balanced for expert / non-expert), full scope; LOSO not now. (3) Start full-scope training for B1, B2-MediaPipe, B2-KTPFormer, Proposed (λ 1.0) and the frozen Proposed (started, then **superseded by the twelfth review**, §6h.2).

**Tenth review (user, 2026-10-20): answers to the full-scope Phase 1 questions** (`DESIGN_DECISIONS.md` P1c, P6c, P6d, A5, M16; applied in §5b.3, §5b.10, §5b.11):
1. **E06:** the wrist angle is assessed in the **side views only**; front-view E06 is assessed without the wrist angle, stated as a limitation.
2. **The rep-phase fix (P6c) is accepted.**
3. **Compare Step 0's alternative rep signals for E03 / E06 / E09 on all synced groups**, choosing by cross-view rep agreement (±1) on synced groups only (criterion fixed before the run, no score results used); report the comparison and apply the winner.
4. **Per-exercise angles for rep-matched pairing** (lower-body / trunk angles for E07–E09).
Then re-run the affected Phase 1 steps, update the docs, commit, **stop: no training until the user confirms the final evaluation scheme**.

**Ninth review (user, 2026-10-20): answers to the full-data Step 0 questions (§8)** (`DESIGN_DECISIONS.md` D0, D1, D8, D14, D15, P1, P1c, P6c, S8, S9, M16, E1; applied in §5b):
1. **Bring in all seven exercises (E03–E09).**
2. **Download the MediaPipe Hands model for E06** (done: `hand_landmarker.task`, Apache 2.0; P1).
3. **A separate video-folder setting** (videos in `D:\cb\mobiphysio\`; `data.video_root`) and **all full-scope outputs in new folders**; the pilot's outputs are not touched (`configs/data_full.yaml`, D15).
4. **Both sync-rule changes accepted** (S8): non-overlapping recording windows override audio, and the start-time rule. Groups added only by the start-time rule get `w_sync` only after motion confirms the alignment; otherwise `w_audio` / `w_rep` as before.
5. **Do not exclude subjects only because PF1 = 0.** Exclude from templates and exercise labels only where the movement itself shows a wrong exercise (as E03 P02). Run that check on the 6 E05 non-experts and report (§5b.7).
6. **LOSO: the supervisor decides.** Plan **GroupKFold-10 as the final table** for now; do not start LOSO.
7. **Accept the proposed joint angles and rep signals as starting definitions; document them** (M16, `configs/exercises/E03…E09.yaml`).
8. The user tries to get the EAAQ appendix.

Then: run Phase 1 for E03–E09 (keypoints incl. Hands for E06, sync, reps, KTPFormer lift, packs), verify with the pilot's checks, report, commit, and **stop before training**.

**Eighth review (2026-10-19): answers to the Phase 4 questions** (`DESIGN_DECISIONS.md` M14, M15, M7, X4; applied in §6d.6–6d.7):
1. **Keep the PT demo as the reference curve, but centre the posture checks (e.g. elbow) on the high-scoring experts (≥ 95)**, fitted per fold on training subjects; marked as a post-hoc change, reported before / after (M14).
2. **Do not switch the main speed reference:** expert-centred speed stays primary; the demo-centred speed is reported only as a clearly labelled post-hoc exploratory analysis (M15).
3. **CF item meanings:** the user asks the supervisor (dataset author); left as is for now (M7).
4. **Live demo:** default view = front, with a `--view` option (X4, `demo.default_view`).

**Phase 5 request (2026-10-19):** start Phase 5 for E01 and E02, in order, and stop and report after B: (A) the cross-view protocol on GroupKFold-5 (train on two views, test on the held-out third, rotating F / L / R) for B1, B2-MediaPipe, B2-KTPFormer and Proposed (λ 1.0), reporting on the held-out view the exercise classification, the movement-quality score within groups, the VCI and the view stability of the embedding distance to the PT reference (§6e, E24); (B) the pair-source ablation on the VCI only (CVDC pretraining + held-out within-group VCI, no fine-tuning; sync only / + audio / + rep / + synthetic / synthetic only; pair counts per source) (§6f, E25); then estimate the LOSO runtime of the 4 main methods with early stopping and the per-fold-encoder shortcut, **without starting LOSO** (§6g, E26).

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



## 5b. Phase 1 on the full scope: E01–E09 (2026-10-20)
User request (ninth review, §4): run Phase 1 for E03–E09 (keypoints incl. MediaPipe Hands for E06, sync, reps, KTPFormer lift, packs), verify with the pilot's checks, report, commit, **stop before training**. How to run: `scripts/phase1_full/README.md` (config `configs/data_full.yaml`, D15). **Nothing of the pilot was written**: the full scope lives in `data/full/`, `outputs/full/` (gitignored). E01/E02 are part of the full scope (all nine exercises in one index), reusing the pilot's keypoints. Decisions: D0, D1, D8, D14, D15, D17, P1, P1c, P6c, S8, S9, M16, E1 in `DESIGN_DECISIONS.md`.

### 5b.1 What ran
| Step | Result | Time |
|---|---|---|
| Index | 3,686 files parse (100%), 3,685 after the known duplicate (D7); 12 PT-demo rows (11 files) | seconds |
| Keypoints | 3,685 / 3,685 videos and 11 demos; **3.06 M frames** (E03–E09 2.35 M). Reused by copy: the pilot's 941 E01/E02 files and the Step 0 sample's 355 E03–E09 files (same extractor settings, checked per file). 2,432 new extractions incl. MediaPipe Hands for all 354 E06 videos and the E06 demo | 2.1 h (12 processes, ~265 frames/s) |
| Preprocessing | 3,655 files; **41 videos excluded** for pose in < 50% of frames (P7): the pilot's 8 + 33 new, all left-view LL/HJ takes except one non-expert front clip (`E08_P27_AF_VLL`). The 936 E01/E02 files (933 videos + 3 demos) are **byte-identical** to the pilot's | minutes |
| Sync | audio for 2,431 E03–E09 videos (E01/E02 lags reused from the pilot's cache), creation time, start time (S8), motion with the extended candidates (S9), controls | 23 min (disk-bound) |
| Reps | 26,560 training reps in 3,644 videos (after the tenth-review signal change, §5b.10) | 9 s |
| Packs | `reps_coco17_T64` (26,560) and `videos_coco17_T300` (3,644), no NaN; CVDC pairs (§5b.11) | 75 s + 2 min |
| KTPFormer lift | 2,719 new videos (2.27 M frames, 854 frames/s); the pilot's 936 E01/E02 lifts reused (inputs identical); lifted packs, no NaN | 44 min (GPU) |
| Checks | 149 skeleton-only PNGs (`outputs/full/checks/`), report tables (`outputs/full/phase1/`) | 1 min |

The Step 0 estimates (§3.7.10: extraction 2.1–2.6 h + 0.3 h Hands, audio 0.4 h, lifting 0.8 h) held.

### 5b.2 Keypoints
- **Detection** (mean share of frames with a pose): F ≥ 98.3% and R ≥ 98.8% in every exercise; L 90.8% (E09), 91.5% (E07), 91.8% (E08), 94.6–97.4% (E01–E06). 145 videos below 90%. By variation: LJ, LR, ML, FL ≥ 96% everywhere; **HJ is the weakest** (85.6% E07, 86.1% E08, 87.5% E09, 94–96% E01–E06), then LL (95–99%). Same pattern as the pilot and the Step 0 sample (`table_detection_by_variation.csv`).
- **Root:** both hips visible in ≥ 88.8% of frames in every exercise × view (lowest: E07–E09 left views), so the mid-hip root holds for all nine (P5). Upper-body joints low-visibility: ~25% of joint-frames in the lying exercises (E07–E09; the far arm, not needed there), ≤ 23% elsewhere (side views).
- **Excluded (< 50% detection, P7): 41 videos** (`preprocess_summary.csv`), 40 of them left-view LL or HJ takes; the other views of those groups are kept.

### 5b.3 E06: MediaPipe Hands (P1c; user answer 2)
- **Detection** (354 videos): a hand on the body's left / right side in 81 / 78% of frames in F, 57 / 70% in L, 76 / 67% in R; both hands 69 / 52 / 59%. By variation: FL 89 / 87%, ML 86 / 87%, O 78 / 79%, LJ 67 / 69%, LL 63 / 65%, **HJ 37 / 36%**. The hand's wrist landmark lies a median 0.19–0.38 forearm lengths from the Pose wrist. Renders: `hands_E06_*.png` (skeleton + hand landmarks; the hands sit on the forearm ends).
- **The wrist-extension angle from the hand landmarks is not view-consistent on this data** (`scripts/phase1_full/hands_check.py`, `hands_check.log`). In the 112 synced expert F+L+R groups (one performance seen by three phones):

  | Signal (0.3 s median, frames with the hand found in both views) | Spearman across views | Mean \|difference\| |
  |---|---|---|
  | Hands angle, 3D signed (+ = extension) | 0.01 | 39° |
  | Hands angle, 3D unsigned | 0.12 | 20° |
  | Hands angle, 2D (image) | 0.20 | 18° |
  | Pose-point angle (index / pinky), 2D | 0.08 | 14° |
  | **Pose wrist height (E06's rep signal)** | **0.88** | – |

  Comparing the front view with the side view that sees the same hand (the L camera sees the right hand, the R camera the left): 3D unsigned 0.23–0.29, signed 0.17–0.18, 2D 0.10–0.20 against 0.73–0.82 for the wrist height. Rep counts from the Hands angle agree within ±1 across views in 13.5% of groups (Pose-point angle 36%, the pipeline's wrist height 62%). **Why:** in E06 the arms point forward, toward the front camera, so the wrist bends mostly in depth for that view (`hands_E06_median_E06_P17_VO.png`); the signed angle needs the palm normal, which flips with MediaPipe's per-hand depth (a group's front view reads −50 to −100°, its side views +30 to +50°, during the same reps); and the far hand is occluded in each side view. Within a single view the angle does show the reps (`E06_P05_AF_VFL`: clean peaks of +50–75°). **Crops from full-resolution frames did not help** (test on 25 groups / 75 videos, `--hands-full-res`: detection unchanged, 2D Spearman 0.24 vs 0.22 on the same groups).
- Rep segmentation of E06 does not use the hands (Pose wrist height, M16), so Phase 1 is complete. **Decision (tenth review): the E06 wrist angle is assessed in the side views only**, on the hand that camera sees (L → the body's right hand, R → the left), as the unsigned angle in the image plane; **front-view E06 is assessed without the wrist angle**, stated as a limitation (`configs/exercises/E06.yaml`, implemented in Phase 4).

### 5b.4 Preprocessing
Upper-body joint-frames interpolated: mean 4.2% (median 0.6%); low-visibility joint-frames kept with MediaPipe's estimate: mean 13% (lying exercises ~25%: the far arm). Rep length at 30 fps: p5 / p50 / p95 = 38 / 68 / 123 frames, so T_rep = 64 stays (pilot median 64).

### 5b.5 Synchronization (S1–S9)
| Exercise, group | synced: audio | synced: creation time | synced: start time (S8b) | not synced | unknown |
|---|---|---|---|---|---|
| E01 expert / non-expert | 49 / 0 | 78 / 0 | 1 / 0 | 0 / 22 | 1 / 0 |
| E02 expert | 45 | 82 | 1 | 0 | 1 |
| E03 expert / non-expert | 38 / 0 | 90 / 0 | 1 / 0 | 0 / 22 | 0 |
| E04 expert | 42 | 86 | 0 | 0 | 1 |
| E05 expert / non-expert | 48 / 0 | 73 / 0 | 3 / 0 | 1 / 17 | 1 / 0 |
| E06 expert | 38 | 80 | 2 | 0 | 0 |
| E07 expert | 35 | 46 | 2 | 0 | 1 |
| E08 expert / non-expert | 34 / 0 | 48 / 0 | 2 / 0 | 0 / 14 | 0 |
| E09 expert / non-expert | 42 / 0 | 40 / 0 | 1 / 0 | 0 / 11 | 1 / 0 |

- **E03–E09 reproduce the Step 0 classes in all 820 groups**; one E05 group moved from audio to creation time (same class). In the pilot exercises the start-time rule moved the two groups S8 predicted (`E01_P09_VFL`, `E02_P18_VLJ`) from unknown to synced; nothing else changed class.
- **Start-time groups (13) get `w_sync` only where motion confirms** (user answer 4): of their 37 view pairs, 9 are motion-confirmed (`sync`) and 28 aligned by device-corrected audio lags (`audio`).
- **Motion test** (S4, S9): 875 of 3,220 view pairs confirmed; share of synced-group pairs confirmed: E09 52%, E04 47%, E01 43%, E07 36%, E08 34%, E03 29%, E02 15%, **E05 10%, E06 7%** (circle and wrist: low, as Step 0 found). **Controls (separate recordings): 40 of 5,237 testable pairs falsely confirmed (0.8%; pilot 0.7%)** with the extended candidates; control residual p1 45 ms, median 192 ms.
- **Device offsets** (video lag − audio lag, motion-confirmed simultaneous pairs): F→L +97 ms (n 341, SD 62), F→R +50 ms (n 324, SD 40), L→R −54 ms (n 194); pilot +93 / +57 / −44.
- **CVDC view pairs** (`outputs/full/sync_pairs.csv`): E03–E09 `sync` 1,247 (649 with motion-confirmed lags), `audio` 960, `rep` 195. Not-synced or unknown expert groups (7): `E01_P13_VLJ`, `E02_P13_VLJ`, `E04_P12_VLL`, `E05_P03_VLL`, `E05_P03_VML` (separate R takes), `E07_P20_VFL`, `E09_P05_VML`.

### 5b.6 Repetitions (P6, P6b, P6c)
- Signals as approved (answer 7, M16): wrist-path PC1 (E03), all-joint 2D PC1 (E04, E07), 2D wrist height (E05, E06), 3D hip flexion (E08, E09); E01/E02 unchanged. Thresholds adaptive per video (the Step 0 method). **Port check:** on the Step 0 sample the pipeline reproduces Step 0's per-video counts (all but 3 sub-2 s clips Step 0 skipped) and its cross-view agreement exactly (90 / 64 / 82 / 75 / 70 / 92 / 78% on Step 0's own groups).
- **Fix found in the spot checks: the rep phase.** Step 0's rule took the level of the first second as rest. Many videos start mid-movement (E08 often with the knees already flexed, E06 with the arms already raised), so 33–72% of videos were segmented half a cycle off (a "rep" from one flexed position to the next; counts unaffected, so the agreement check did not show it; `reps_E08_expert.png` before the fix). **Change (P6c, after seeing the renders):** a fixed polarity (rep = the movement away from rest) for the signals with a physical direction (E05 / E06 wrist height up, E08 / E09 hip flexion up), and the PC signals oriented by the exercise's main angle (E03: rises as the forearm goes up; E04: as it goes down; E07: with the moving leg's hip flexion). Every synced group's best view now uses the same phase.

  | Cross-view agreement, all synced expert F+L+R groups, per-view counts within ±1 (PF1 < 0.5 subjects left out, as Step 0) | E03 | E04 | E05 | E06 | E07 | E08 | E09 |
  |---|---|---|---|---|---|---|---|
  | Step 0 sample (9–12 groups) | 90% | 64% | 82% | 75% | 70% | 92% | 78% |
  | All groups, Step 0 rule (first-second rest) | 61% | 79% | 82% | 65% | 71% | 74% | 56% |
  | All groups, P6c (groups) | 55% (114) | 77% (119) | 78% (112) | 62% (112) | 72% (79) | 71% (77) | 63% (72) |
  | **All groups, final: P6c + the re-chosen E03 / E09 signals (§5b.10)** | **57%** | **77%** | **78%** | **62%** | **72%** | **71%** | **78%** |

  **The Step 0 samples were optimistic for E03, E08 and E09 and pessimistic for E04**: on Step 0's own groups the pipeline gives Step 0's numbers, so the drop is sampling, not the port. Synced groups share the best view's reps (P6b), so their training reps are identical across views by construction (100%); the per-view agreement still matters for the non-expert clips (segmented on their own) and as a signal-quality measure. Remaining failures in the renders: double counts in noisy left views (e.g. `E03_P20_AL_VML`, 18 reps), E06 videos where the arms stay up (no wrist-height reps: 11 E06 videos with 0 reps), high-jitter takes.
- **Counts** (final, after §5b.10): median set rep count 9–10 per expert video in every exercise (EAAQ "≥ 10"); non-expert clips 1 (E08 0–1). Training reps: E03 3,284 + 178 non-expert, E04 3,042, E05 3,318 + 205, E06 2,733, E07 2,016, E08 2,270 + 122, E09 2,230 + 68; E01 3,578 and E02 3,516 as in the pilot (26,560 in all). Videos with 0 reps: 92 (non-expert clips 81, of them 34 E09 clips, most barely moving: hip range < 15°, the E09 non-experts score a median of 0; E06 experts 11).
- **Set rep count vs PF4** (experts, Spearman, final signals): E08 0.41, E07 0.25, E09 0.20, E05 0.17, E06 0.16, E03 0.01, E04 0.01 (pilot E01 0.37, E02 −0.02). When PF4 = 1, the set has ≥ 10 reps in 92% (E08), 84% (E05), 81% (E03), 80% (E07), 77% (E09) of videos, but only 38% (E06) and 31% (E04): PF4 is not a rep count there either (D11).
- **Timing** (median rep, experts): E03 2.1 s, E04 1.9, E05 1.8, E06 2.5, E07 3.1, E08 3.5, E09 3.4 (`table_rep_timing*.csv`; speed reference from the videos scoring ≥ 95).
- Moving side: left in 12–29% of upper-limb expert videos (E04 29%); E06 left 64% (both wrists move; the side with the larger wrist-height range). PT demos: E03 V3 2 reps, E04 1, E05 4, E06 2, E07 2 (Step 0: 1; the demo starts mid-flexion, which the first-second rule mistook for rest), E08 2, E09 2.

### 5b.7 E05 PF1 check (user answer 5)
The six E05 non-experts with mean PF1 < 0.5 (P30, P34, P57, P58, P59: PF1 = 0; P32: 0.38). Script and rule: `scripts/phase1_full/pf1_motion_check.py` (rule in its docstring, fixed before running on these subjects). **Revision before the run:** the first design counted circles of the 3D wrist path in every view, but a high-scoring expert's front views give ~0 turns (the sagittal circle is flattened by MediaPipe's front-view depth) against ~9 in the side view, and the non-experts are mostly front-view; so the check uses (A) a posture match of each front-view clip to the five upper-limb exercises (leave-subject-out nearest pool, calibrated on the non-experts with PF1 ≥ 0.5: E01 / E03 / E05 clips assigned to their own exercise in 87 / 90 / 99% of cases) and (B) circles in the side views (all 261 reference E05 side clips ≥ 0.5 turns).

| Subject | Mean PF1 | Mean score | Front clips assigned to E05 | Side-view turns | Verdict |
|---|---|---|---|---|---|
| P30 | 0.00 | 42.5 | 6 / 6 | 2.70, 0.78 | circumduction |
| P32 | 0.38 | 33.0 | 6 / 6 | 2.00, 2.14 | circumduction |
| P34 | 0.00 | 30.2 | 7 / 7 | 1.61, 1.92 | circumduction |
| P57 | 0.00 | 30.3 | 5 / 5 | 1.73, 1.65 | circumduction |
| P58 | 0.00 | 31.0 | 5 / 5 | 0.44, 0.61 | circumduction (weakest) |
| P59 | 0.00 | 27.6 | 5 / 5 | 1.64, 1.58 | circumduction |

**None of the six shows a wrong exercise, so none is excluded** (D14); they stay in the templates' candidate pools (they never reach ≥ 95) and in the exercise labels. P58's circles are the smallest (two compact or erratic paths, `pf1_E05_wrist_paths.png`). PF1 = 0 for these subjects therefore rates something other than the exercise type (the EAAQ text is still missing, §3.7.8).

### 5b.8 E01/E02: consistency with the pilot (`scripts/phase1_full/compare_pilot.py`)
Keypoints and preprocessed files identical (936 / 936). Sync: only the two start-time groups changed. In every other group the per-video counts are identical (set count 100%, per-view count 100%, training count 99.4% of 927 videos); 716 of 7,088 rep boundaries moved (707 in transferred views, 9 in best views where an edge rep dropped and the reps were re-indexed), a median 1 frame: the device offsets were re-estimated on all nine exercises (F→L +97 vs +93 ms, F→R +50 vs +57 ms). The pilot's results stay as they are; the full-scope E01/E02 data differ from them only by these shifts and the two groups.

### 5b.10 Rep signals of E03 / E06 / E09 re-chosen by cross-view agreement (tenth review, item 3; P6d)
`scripts/phase1_full/rep_signal_compare.py` (`rep_signal_compare.csv`). **Criterion, fixed before the run (no score or label used):** all synced expert F+L+R groups with keypoints in all three views, leaving out only the subject excluded from the exercise labels for a wrong exercise (E03 P02, D8); per view, the candidate's rep count segmented on its own with the pipeline's settings (adaptive thresholds, rep phase P6c: each candidate oriented per side to rise with the movement: E03 the forearm going up, E06 the 2D wrist height, E09 the trunk–thigh angle); the share of groups whose three counts lie within ±1; ties by exact agreement, then the current signal. Candidates: Step 0's per region.

| Exercise (groups) | Candidates: within ±1 / exact (current in bold) | Winner |
|---|---|---|
| E03 (114) | forearm elevation 57.0 / 21.9; **wrist-path PC1 55.3 / 18.4**; all-joint PC1 52.6; elbow flexion 48.2; 2D wrist height 47.4; forearm azimuth 36.8; shoulder elevation 31.6 | **forearm elevation (3D)** |
| E06 (112) | **2D wrist height 61.6 / 24.1**; all-joint PC1 58.0; forearm elevation 55.4; elbow flexion 41.1; Pose-point wrist deviation 2D 28.6, 3D 23.2 | **2D wrist height (unchanged)** |
| E09 (78) | 2D trunk inclination 76.9 / 35.9; 2D shoulder height 67.9; all-joint PC1 67.9; **hip flexion 61.5 / 24.4**; trunk–thigh angle 55.1; summed hip flexion 52.6; knee flexion 11.5 | **2D trunk inclination** |

- The comparison reproduces the pipeline's numbers for the current signals (E03 55.3%, E06 61.6%), so it measures the same thing. **Applied:** E03 now segments on the forearm elevation, which is also its Module III main angle (no longer a segmentation-only signal); E09 on the 2D trunk inclination (`trunk_inclination_2d`, segmentation only; the moving side named by the hip-flexion range). The E03 margin is small (2 of 114 groups).
- **After the re-run** the pipeline's own per-view agreement equals the comparison (E03 57.0%, E09 76.9%; with E09 P21 left out as in §5b.6: 77.8%). Renders (`reps_E03_expert.png`, `reps_E09_expert.png`): E03 reps are the forearm-up phases, E09 reps the trunk-raised plateaus; two noisy left views of E03 still double-count. Effects: E03 training reps 3,184 → 3,284 (experts), 183 → 178 (non-experts); E09 2,376 → 2,230 and 86 → 68 (more E09 non-expert clips without a rep: most of them barely move).

### 5b.11 CVDC pairs with per-exercise DTW angles (tenth review, item 4; A5)
`python -m cvdc.data.pairs --config configs/data_full.yaml` (Phase 2's pair builder, data only; nothing trained). Rep-matched pairs (not-synced and unknown groups) are paired by index and time-aligned by DTW on 3D angles; each exercise now has its own angles (`pairs.dtw_angles` in its YAML): E03 / E04 forearm elevation + shoulder elevation, E05 / E06 shoulder elevation + elbow (as the pilot), **E07 / E08 hip + knee flexion, E09 trunk–thigh angle + knee flexion**, both sides; E01 / E02 keep the pilot's default.

| | Reps | Rep sets (3 / 2 / 1 views) | sync pairs | audio pairs | rep pairs |
|---|---|---|---|---|---|
| All nine exercises | 26,560 | 9,435 (8,225 / 675 / 535) | 14,503 | 10,445 | 402 (281 DTW-aligned members) |

Per exercise: `outputs/full/phase2/table_pair_counts.csv`. Rep pairs: E01 99, E02 27, E03 48, E04 2, E05 117, E06 0, E07 30, E08 45, E09 34 (mostly non-expert clips; 535 single-view sets get positives only from synthetic views, as the pilot's E01 non-experts, D12).

**Check of the DTW angles** (`scripts/phase1_full/dtw_angle_check.py`; 40 synced groups per exercise, where the matching rep is known): (1) the pilot's known-warp test (A5): the other view's rep replayed slow-then-fast (t^1.5) and re-aligned by DTW; median error in frames:

| Exercise | Per-exercise angles | Pilot's upper-limb angles | Without DTW |
|---|---|---|---|
| E03 | **3.1** | 6.0 | 6.2 |
| E04 | **2.6** | 5.1 | 5.3 |
| E07 | **4.1** | 5.8 | 9.1 |
| E08 | **4.4** | 5.5 | 10.4 |
| E09 | 7.6 | **4.9** | 10.5 |

(2) Whether the same rep in the other view is cheaper than the neighbouring rep: 44–64% with the per-exercise angles vs 38–53% with the upper-limb ones, near chance as in the pilot (3D angles are view-biased, A5), which is why pairs are made by index and the DTW cost is not used to pick them. **E09 is the exception:** every trunk / lower-body set tested recovers the warp worse than the arm angles (a separate run on 327 rep pairs, `dtw_angle_check_E09.csv`: trunk–thigh + knees 7.4 frames, hips + knees 7.5, trunk + hips 7.8, hips 8.1, trunk alone 10.8, i.e. worse than no DTW at 9.8, vs 4.8 for the arm angles): during back extension the trunk holds still on a plateau while the arms keep moving. As instructed, E09 uses the best trunk / lower-body set (trunk–thigh + knees); it affects 34 rep pairs (§8).

### 5b.9 Tests
`python -m pytest -q tests`: **187 tests, all passing** (22 new in `tests/test_full_scope.py`; + the 2D trunk inclination and the per-exercise DTW angles in the tenth-review update): config inheritance and that no full-scope path points into a pilot folder; S8a and S8b with their pair sources; adaptive segmentation (scale-free, polarity); `orient_by`; hip / knee / trunk-thigh / forearm-elevation angles, 2D wrist height and all-joint PC1 on known geometry; the signed wrist-extension angle (sign and size for both hands, frame independence); the hand-crop geometry and mapping back; `side_from`.

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
**Note:** §6d.1–6d.5 describe the first Phase 4 run (commit `bbb0f6a`), with every channel centred on the PT demo. After the eighth review the posture checks are centred on the high-scoring experts (M14, post-hoc); §6d.6 gives before / after, and `outputs/phase4/summary.md` now shows the after state (the before run is kept in `outputs/phase4/posture_pt/`). The embedding-distance (§6d.3) and recording-noise (§6d.4) results do not depend on it.

User request (seventh review): Module III as in proposal Figure 2 for E01 and E02, with the **PT demo as the reference ("truth")**; score every video by its distance from the reference and check agreement with the physiotherapists; the embedding distance for B1, B2 and Proposed (λ 1.0), including its view stability; a recording-noise test for PF3/PF4. **No LOSO, no ablations.** Decisions M2, M6, M7, M7a, M8–M13, E5, E22, E23 in `DESIGN_DECISIONS.md`. How to run: README "Phase 4". Tables: `outputs/phase4/summary.md` + CSVs, figures `outputs/phase4/figs/` (curves only, no frames), one JSON per video in `outputs/phase4/json/` (all gitignored). Same GroupKFold-5 folds as Phase 3; video level; 95% CIs from the subject bootstrap (2,000 draws). All §0 limitations apply.

### 6d.1 What was built
- **Reference Template Library** (`cvdc/assess/template.py`, M8): per exercise, the primary PT demo's reps on the 30 fps grid, 3D angles for the moving arm (left arms mirrored), averaged after DTW alignment: E01 `Abduction .mp4` (2 reps, 4.2 and 3.2 s → 112 frames, end range 161.3°), E02 `Aduction.mp4` (2 reps, 2.3 and 2.2 s → 68 frames, end-range horizontal adduction 150.9°). **Per view**, the experts' videos scoring ≥ 95 give the view bias (how a side camera reads the movement differently from the front camera that filmed the demo) and the tolerance band (±2 SD of the experts per template frame) (M2, M9). In the front view the experts' mean curves lie on the demo for shoulder elevation, trunk lean and the other arm; their elbows are 5–10° more bent than the demo's. Side views read the same performances differently (e.g. E01 R-view peak elevation ~25° lower; E02 R-view horizontal adduction ~60° lower) and have much wider bands (`figs/template_E01.png`, `template_E02.png`).
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

### 6d.6 Eighth review: posture checks centred on the experts (M14, post-hoc), before / after
Before = the first run (every channel centred on the demo + view bias); after = the posture channels (E01: elbow, other arm, trunk lean, shoulder asymmetry; E02: + arm elevation) centred on the ≥ 95 experts' mean curve in each view, fitted per fold on the training subjects. The main movement curve (E01 shoulder elevation, E02 horizontal adduction), the ROM target, speed, alignment and bands are unchanged. Same folds and bootstrap draws (`posture_before_after_*.csv`).

| | E01 expert | E02 expert | E01 non-expert |
|---|---|---|---|
| ρ set score vs total: before → after (Δ [CI]) | 0.03 → 0.04 (0.01 [−0.02, 0.05]) | 0.13 → 0.16 (0.03 [−0.02, 0.08]) | −0.02 → −0.03 (−0.01 [−0.04, 0.03]) |
| ρ vs MQ: before → after (Δ [CI]) | 0.18 → 0.17 (−0.01 [−0.04, 0.01]) | 0.30 → 0.34 [0.12, 0.52] (0.03 [−0.00, 0.07]) | −0.02 → −0.02 |
| set score mean: before → after | 82.5 → 84.1 | 73.3 → 76.2 | 50.3 → 51.0 |
| score view stability (ICC, rep level) | 0.54 → 0.51 | 0.50 → 0.55 | – |
| as a predictor (Δ RMSE vs group mean, after) | +0.07 [−0.01, 0.17] | +0.16 [−0.19, 0.59] | +0.12 [−0.04, 0.32] |

**Posture-rule rates** (share of reps; in videos scoring ≥ 95 / in videos < 80), before → after:

| Rule | E01 expert | E02 expert | E01 non-expert (all) |
|---|---|---|---|
| elbow bent | 26% / 29% → 12% / 11% | 40% / 48% → **12% / 30%** | 57% → 21% |
| shoulder asymmetry | 5% / 6% → 2% / 4% | 36% / 36% → **2% / 14%** | 6% → 3% |
| other arm raised | 11% / 10% → 11% / 8% | 18% / 20% → 17% / 15% | 9% → 7% |
| trunk lean | 8% / 10% → 7% / 9% | 8% / 18% → 6% / 19% | 29% → 26% |
| arm elevation (E02) | – | 27% / 45% → 23% / 38% | – |

- **Agreement with the physiotherapists does not change** (all paired Δρ within ±0.03, CIs include 0).
- **The posture rules become specific in E02:** they now fire mostly in the lower-scoring videos (shoulder asymmetry 2% of reps in ≥ 95 videos vs 14% in videos < 80; elbow bent 12% vs 30%). In E01 they fire less often, but no more often in the lower-scoring videos (elbow bent 12% vs 11%). The range, speed and set rules are unchanged by construction.
- Most of the drop is the demo-vs-expert offset that the first run counted as error (the demo's elbows are 5–10° straighter than the experts'). Since the centre was chosen after seeing the rates, this is reported as a post-hoc change, and the before run stays reproducible (`--posture-centre pt_demo`).

### 6d.7 Eighth review: speed centred on the demo (M15, post-hoc exploratory only)
The primary speed component stays expert-centred (M5a). Exploratory only (`speed_exploratory.csv`): the speed distance centred on the demo instead does **not** agree better (E01 experts PF2 −0.14 [−0.39, 0.14] vs 0.28 [0.07, 0.45] expert-centred; total / MQ within ±0.22 with CIs including 0 in every group). A score variant with the demo-centred speed is worse on PF2 (Δρ −0.31 [−0.57, −0.03]) and not better elsewhere (Δ −0.14 to −0.06). So the Phase 4 signal comes from the **angular-velocity deviation from the demo along the DTW path** (E02 0.24 [0.06, 0.41], non-experts 0.40 [0.15, 0.60]), which also carries amplitude differences, not from the demo's speed itself. Being post-hoc, none of these intervals is confirmatory.

## 6e. Phase 5 A: cross-view protocol, GroupKFold-5 (2026-10-19)
User request (Phase 5, in order A then B, stop after B): train on two views, test on the held-out third (rotate F / L / R) for B1, B2-MediaPipe, B2-KTPFormer and Proposed (λ 1.0); report on the held-out view the exercise classification, the movement-quality score within groups, the VCI and the view stability of the embedding distance to the PT reference. Decision E24 in `DESIGN_DECISIONS.md`. How to run: README "Phase 5". Tables: `outputs/results/phase5_crossview/summary.md` + CSVs (gitignored). All §0 limitations apply.

### 6e.1 Setup
- Same GroupKFold-5 folds, inner splits, seeds, epoch budget, early stopping (T8a, T8b) and schedule as step (a2). **Each run sees only its two training views**: supervised items, inner validation, target scalers, the video head's count statistics, CVDC pretraining and its validation, the joint loss's CVDC batches and the embedding centre (`trainer.with_views`; tests in `tests/test_crossview.py`, including CVDC batches drawn only from the training views).
- **Proposed pretrains CVDC afresh on the two training views** in every rotation and fold (15 pretrainings): the step (a) encoders saw all three views, and `reuse_cvdc` now refuses an encoder trained on other views. Synthetic views stay on (rotations of the training views' world landmarks; they can reach the held-out camera's angle).
- **What still crosses from the held-out camera is timing, not poses:** synced groups share rep boundaries segmented on their best view (P6b), and rep-matched pairs are DTW-aligned onto the set's reference view (A5).
- Every run predicts all views of its test subjects, so each of the 915 evaluated videos (F 373 / L 270 / R 272) is scored **once as a held-out view** (the rotation that held its view out; each by a model that saw neither the subject nor the camera) and twice as a seen view. The step (a2) models (trained on all three views) are the 'all-view' reference.
- **MQ** (D13) comes from the video head's items: 100 × mean(PF1, PF2, CF1, CF2, CF3), with items that do not vary in the video's group (T14, not learned there) set to their training mean in that group. The group-only baseline is fitted on the training subjects' training-view videos.
- **VCI pairings:** held-out↔seen = same rep seen by the held-out camera and a seen camera, minus other groups with one member in each; seen↔seen = two different seen cameras, with the other-group pairs restricted the same way. Pooled over rotations both pairings hold the same camera pairs, and for an all-view model they agree exactly (checked).
- Runtime (one GPU job at a time): B1 0.53 h, B2-MediaPipe 0.48 h, B2-KTPFormer 0.48 h, Proposed 2.40 h (CVDC 1.53 h) for 15 runs each; 3.9 h in total. Kept fine-tuning epochs median 32 / 26 / 23 / 18; CVDC kept epochs median 54 (30–70).
- **Finding on compute:** two GPU processes at once were slower than running them in sequence (CVDC epoch 14.4 s shared vs 5.3 s alone; A9), so every Phase 5 job was chained.
- Tests: `tests/test_crossview.py` (the view restriction, the plans keeping the Phase 3 folds, no reuse of an encoder trained on other views, CVDC batches of the training views only, the VCI pair filter against brute force); `python -m pytest -q tests`: **165 passed**.

### 6e.2 Exercise classification on the held-out view (secondary, E6)
Video level, pooled over both exercises and groups, accuracy [95% CI] / macro F1:

| Method | held-out view | seen views (same models) | all-view model | held-out − seen | vs B1 (held-out) |
|---|---|---|---|---|---|
| B1 | 0.903 [0.867, 0.937] / 0.900 | 0.971 / 0.970 | 0.979 / 0.979 | −0.068 [−0.099, −0.041] | – |
| B2-MediaPipe | 0.793 [0.751, 0.838] / 0.788 | 0.972 / 0.971 | 0.984 / 0.983 | −0.179 [−0.222, −0.135] | −0.109 [−0.150, −0.067] |
| B2-KTPFormer | 0.827 [0.788, 0.866] / 0.821 | 0.956 / 0.955 | 0.973 / 0.972 | −0.128 [−0.167, −0.091] | −0.075 [−0.126, −0.025] |
| **Proposed** | **0.968 [0.952, 0.983] / 0.968** | 0.983 / 0.982 | 0.991 / 0.991 | **−0.014 [−0.026, −0.001]** | **+0.066 [0.035, 0.098]** |

- Per held-out view (F / L / R): B1 0.93 / 0.83 / 0.93; B2-MediaPipe 0.78 / 0.78 / 0.82; B2-KTPFormer 0.84 / 0.89 / 0.76; Proposed 0.98 / 0.96 / 0.97. Per group, Proposed 0.963 / 0.987 / 0.936 (E01 expert / E02 expert / E01 non-expert).
- **The 3D-input baselines generalise worst to a new camera.** A likely reason: MediaPipe world landmarks (and the lifted 3D) are expressed in the camera's frame (A8 found the camera pitch in them), so a new camera shows the model a rotated skeleton it never saw.

### 6e.3 Movement-quality score within groups (held-out view)
Video level, RMSE [95% CI] / Spearman ρ:

| Method | E01 expert (377) | E02 expert (381) | E01 non-expert (157) |
|---|---|---|---|
| B1 | 14.68 [11.22, 18.70] / 0.00 | 14.88 [11.64, 18.37] / −0.02 | 12.53 [9.74, 15.44] / 0.04 |
| B2-MediaPipe | 15.98 [13.38, 19.51] / 0.06 | 16.58 [14.31, 18.91] / 0.05 | 12.37 [9.67, 15.09] / 0.12 |
| B2-KTPFormer | 15.76 [12.68, 19.73] / 0.08 | 16.65 [14.09, 19.01] / −0.00 | 11.91 [9.74, 14.27] / 0.07 |
| **Proposed** | 12.86 [9.17, 17.40] / 0.15 | 12.16 [9.08, 15.29] / 0.07 | 12.91 [9.57, 16.51] / −0.01 |
| group-only (within-group mean) | **11.67** [7.31, 16.66] | **11.32** [7.97, 14.69] | **10.81** [8.10, 13.41] |
| predict-the-mean | 12.49 | 12.10 | 25.11 |
| *seen views, same models: B1 / B2-MP / B2-KTP / Proposed* | 12.53 / 12.04 / 12.45 / 12.31 | 12.29 / 11.45 / 12.22 / 11.55 | 11.55 / 11.60 / 11.70 / 11.42 |

Paired MQ RMSE differences:

| | E01 expert | E02 expert | E01 non-expert |
|---|---|---|---|
| held-out − seen: B1 | +2.15 [0.80, 4.13] | +2.59 [0.40, 4.70] | +0.99 [−0.64, 2.45] |
| held-out − seen: B2-MediaPipe / B2-KTPFormer | +3.94 / +3.31 | +5.12 / +4.43 | +0.77 / +0.21 |
| held-out − seen: **Proposed** | **+0.55 [0.22, 0.96]** | **+0.61 [−0.05, 1.22]** | +1.50 [0.22, 2.60] |
| Proposed − B1 (held-out) | **−1.81 [−3.63, −0.51]** | −2.71 [−6.00, 0.02] | +0.38 [−1.30, 1.72] |
| Proposed's gap − B1's gap | **−1.60 [−3.31, −0.43]** | −1.97 [−4.25, 0.37] | +0.51 [−0.68, 1.58] |
| Proposed − group-only (held-out) | +1.20 [0.34, 2.49] | +0.84 [−0.26, 2.11] | +2.10 [0.27, 3.81] |

- **CVDC shrinks the cross-view degradation of the MQ prediction** (E01 experts clearly, E02 experts by a similar amount but with a CI touching 0). B1 loses 2.2–2.6 points on an unseen camera and B2 loses 3.3–5.1 (worst on held-out L: B2 E02 RMSE 22–23 vs ~12 on seen views); Proposed loses about 0.6.
- **The gain is in calibration, not ranking.** No method beats the within-group mean on the held-out view: every Δ vs group-only is positive (+0.84 to +5.33), and only two CIs include 0 (Proposed in E02, B2-KTPFormer in E01 non-experts). Held-out ρ ≤ 0.15 with every CI including 0 (Proposed E01 0.15 [−0.05, 0.33]; vs B1 +0.15 [−0.02, 0.28]). This is the same negative result as E21.
- **E01 non-experts show no CVDC benefit.** Proposed's gap +1.50 [0.22, 2.60] is not smaller than B1's. 116 of their 157 videos are front views, so "held-out F" means training on just 21 + 21 non-expert side videos.
- Total score (secondary, `summary.md`): the same pattern. No model beats the group mean; Proposed has the lowest expert RMSE (15.11 / 17.74 vs B1 16.00 / 20.64).

### 6e.4 VCI on the held-out view (centred 256-d, held-out subjects)

| Model | E01 expert | E02 expert | E01 non-expert |
|---|---|---|---|
| random init, held-out↔seen | 0.060 | 0.093 | 0.257 |
| B1, held-out↔seen | 0.066 [−0.006, 0.122] | 0.028 [−0.010, 0.063] | 0.073 [−0.114, 0.201] |
| B2-MediaPipe / B2-KTPFormer, held-out↔seen | 0.011 / 0.059 | 0.023 / 0.022 | 0.066 / 0.066 |
| Proposed after CVDC (2 views), held-out↔seen | 0.268 [0.157, 0.349] | 0.203 [0.131, 0.240] | 0.310 [0.119, 0.407] |
| **Proposed final, held-out↔seen** | **0.213 [0.133, 0.266]** | **0.209 [0.141, 0.241]** | 0.209 [0.064, 0.302] |
| Proposed final, seen↔seen | 0.259 [0.181, 0.310] | 0.248 [0.173, 0.275] | 0.260 [0.117, 0.334] |
| Proposed all-view model, same pairs | 0.258 | 0.271 | 0.187 |
| same-rep pairs (held-out↔seen) | 6,502 | 6,906 | 142 |

- **Proposed − B1 (held-out↔seen): +0.15 [0.08, 0.19], +0.18 [0.10, 0.22], +0.14 [−0.01, 0.26].** The CVDC embedding stays view-consistent for a camera it never saw. B1 and B2 are at the random-encoder level.
- The unseen camera costs Proposed a little: held-out↔seen is 0.04–0.05 below seen↔seen (all three CIs below 0), and 0.05–0.06 below the all-view model on the same pairs (expert CIs below 0).
- Per held-out view, Proposed's fold-mean held-out↔seen VCI is F 0.28 / L 0.17 / R 0.20 (E01 experts) and 0.23 / 0.21 / 0.20 (E02).
- **E01 non-experts: no evidence for CVDC.** Under the cross-camera pairing the random encoder already scores 0.257 [0.020, 0.398] on their 142 rep-matched pairs, so Proposed's 0.209 is not above it.

### 6e.5 View stability of the embedding distance to the PT reference
Synced expert performances, ICC(1) [95% CI] of the standardised distance, rep level (video level in `summary.md`):

| Model | E01 held-out↔seen | E01 seen↔seen | E01 all three | E02 held-out↔seen | E02 seen↔seen | E02 all three |
|---|---|---|---|---|---|---|
| B1 | 0.04 [−0.06, 0.14] | 0.04 | 0.03 | 0.01 [−0.11, 0.11] | 0.16 | 0.05 |
| B2-MediaPipe | −0.06 | −0.05 | −0.06 | 0.01 | 0.17 | 0.05 |
| B2-KTPFormer | 0.04 | 0.11 | 0.07 | −0.07 | 0.13 | −0.02 |
| **Proposed** | **0.19 [0.09, 0.30]** | 0.37 [0.26, 0.50] | 0.23 | **0.40 [0.29, 0.51]** | 0.48 [0.39, 0.55] | 0.43 |
| Proposed all-view model (Phase 4) | 0.32 | 0.32 | 0.32 | 0.62 | 0.62 | 0.62 |

- **Proposed − B1, held-out↔seen: ΔICC +0.15 [0.04, 0.26] (E01), +0.40 [0.24, 0.55] (E02)**; video level 0.14 / 0.44 vs B1 0.03 / 0.01. B2 is not more stable than B1.
- With an unseen camera Proposed's distance is less stable than with all views seen (ΔICC −0.14 [−0.21, −0.05] / −0.22 [−0.32, −0.09]), and less stable than between its own training views.
- The PT demo is a front-view clip, so with F held out the reference itself comes from the unseen camera.

### 6e.6 Conclusion of A
- **On an unseen camera, CVDC is the only method that holds up.** It keeps exercise accuracy (0.968 vs B1 0.903, B2 0.79–0.83), keeps the embedding view-consistent (held-out VCI 0.21 vs ≤ 0.07) and keeps the reference distance view-stable (ICC 0.19 / 0.40 vs ≤ 0.04). Its MQ prediction degrades about a quarter as much as B1's (gap +0.55 / +0.61 vs +2.15 / +2.59 points).
- **3D input (B2) generalises worst across cameras**, worse than plain 2D.
- **Not solved:** on the held-out view no method predicts MQ (or the total score) within a group better than the group mean, and CVDC's advantage there is calibration, not ranking. Seeing the camera still helps Proposed a little (VCI −0.05, ICC −0.14 / −0.22 vs the all-view model).
- **E01 non-experts:** no CVDC benefit; they are mostly front-only and enter CVDC almost only through synthetic views (D12).

## 6f. Phase 5 B: CVDC pair-source ablation on the VCI (2026-10-19)
User request: CVDC pretraining plus the held-out within-group VCI, no fine-tuning, for sync only / + audio / + rep / + synthetic / synthetic only, with pair counts per source. Decision E25. Tables: `outputs/results/phase5_sources/summary.md` + CSVs (gitignored).

**Setup**
- Same folds, inner splits, seeds, budget, schedule, augmentations, synthetic-view settings and early stopping (T8a) as the step (a) pretraining; only `cvdc.sources` changes.
- **(d) all sources is the step (a) pretraining itself**, reused after checking that its configuration equals today's.
- **(d′) all without rep-matched pairs** was added because C5 requires every source comparison with and without them.
- The VCI is scored on identical same-rep pairs for every setting (all sources in `cvdc_members.csv`), so the evaluation does not depend on the training sources.
- New pretraining: 108 min for 25 runs.

**Pair counts per source:** positive pairs in one training epoch, all subjects (per-fold inner-training counts in `pair_counts.csv`):

| Setting | E01 expert: sets / sync / audio / rep / synthetic | E02 expert | E01 non-expert |
|---|---|---|---|
| (a) sync only | 898 / 2,114 / 0 / 0 / 0 | 637 / 1,563 / 0 / 0 / 0 | 0 sets |
| (b) + audio | 1,128 / 2,114 / 1,084 / 0 / 0 | 1,174 / 1,563 / 1,837 / 0 / 0 | 0 sets |
| (c) + rep-matched | 1,147 / 2,114 / 1,084 / 53 / 0 | 1,193 / 1,563 / 1,837 / 53 / 0 | 29 / 0 / 0 / 71 / 0 |
| (d) + synthetic (all) | 1,148 / 2,114 / 1,084 / 53 / 7,842 | 1,194 / 1,563 / 1,837 / 53 / 8,228 | 181 / 0 / 0 / 71 / 643 |
| (d′) all − rep-matched | 1,148 / 2,114 / 1,084 / 0 / 7,770 | 1,194 / 1,563 / 1,837 / 0 / 8,156 | 181 / 0 / 0 / 0 / 543 |
| (e) synthetic only | 1,148 / 0 / 0 / 0 / 3,444 | 1,194 / 0 / 0 / 0 / 3,582 | 181 / 0 / 0 / 0 / 543 |

Real pair counts match Phase 2 (§6.2). Synthetic pairs are one epoch's random draw (2 views per set).

**VCI after pretraining** (centred 256-d, held-out subjects, VCI [95% CI]; same-rep pairs 3,251 / 3,453 / 71):

| Setting | E01 expert | E02 expert | E01 non-expert | kept CVDC epochs (folds) |
|---|---|---|---|---|
| random init | −0.019 [−0.116, 0.059] | 0.024 [−0.051, 0.065] | 0.072 [−0.188, 0.257] | – |
| (a) sync only | 0.244 [0.095, 0.358] | 0.305 [0.151, 0.368] | 0.146 [−0.085, 0.334] | 30, 29, **3**, 35, 25 |
| (b) + audio | 0.280 [0.132, 0.377] | 0.284 [0.151, 0.351] | 0.304 [0.032, 0.456] | 44, 24, 14, 19, 17 |
| (c) + rep-matched | 0.273 [0.117, 0.378] | 0.296 [0.152, 0.382] | 0.301 [0.014, 0.440] | 28, 41, 27, 20, 30 |
| (d) + synthetic = all (step a) | 0.298 [0.192, 0.379] | 0.257 [0.156, 0.326] | 0.296 [0.095, 0.383] | 53, 62, 75, 80, 77 |
| (d′) all − rep-matched | 0.278 [0.168, 0.353] | 0.241 [0.141, 0.315] | 0.333 [0.110, 0.450] | 40, 48, 58, 44, 94 |
| (e) synthetic only | 0.116 [0.016, 0.175] | 0.131 [0.054, 0.180] | 0.302 [0.072, 0.415] | 95, 93, 78, 95, 90 |

Paired differences (centred VCI; E01 expert / E02 expert / E01 non-expert):
- + audio: +0.037 [−0.016, 0.086] / −0.021 [−0.070, 0.051] / +0.158 [0.023, 0.241]
- + rep-matched: −0.007 / +0.012 / −0.003 (all CIs include 0)
- + synthetic on top of real pairs, (d) − (c): +0.025 [−0.049, 0.095] / −0.039 [−0.152, 0.066] / −0.005
- rep-matched given synthetic, (d) − (d′): +0.020 [−0.001, 0.043] / +0.016 [−0.010, 0.039] / −0.037 [−0.111, 0.057]
- synthetic only − all: **−0.182 [−0.253, −0.110] / −0.126 [−0.202, −0.044]** / +0.005 [−0.115, 0.115]

**Findings**
- **The real synchronized pairs alone give the experts' full view consistency.** Sync only reaches 0.24 / 0.31, and every real-pair mix reaches 0.24–0.30. No added source changes the expert VCI by an amount whose CI excludes 0.
- **Synthetic views alone give about half** (0.12 / 0.13; −0.18 and −0.13 vs all sources), and on top of the real pairs they add nothing measurable. Rep-matched pairs (53 + 53 expert pairs, under 2% of the experts' real pairs) have no measurable effect.
- **Caveat: sync-only fold 2 kept epoch 3.** On its small, noisy validation set the agreed rule (T8a: patience 10, no smoothing) stopped at epoch 13. That fold's E01-expert VCI is 0.065, against 0.22–0.40 in the other folds, which pulls the sync-only mean down. The rule was not changed after seeing this (see §8).
- **E01 non-experts (71 rep-matched pairs from ~22 subjects): the stratum cannot separate the sources.**
  - Every setting except sync only gives ~0.30, including (b), which has **no** non-expert positives at all; so their VCI is transfer from the experts' pairs.
  - The one CI that excludes 0 (+ audio vs sync only, +0.158) involves the epoch-3 fold, and the fold values swing from −0.20 to 0.67 (the random encoder's alone from −0.20 to 0.43).
  - Synthetic views, the only positives most non-expert sets get (D12), do not raise their VCI: synthetic only 0.302 vs 0.304 for (b).

## 6g. LOSO runtime estimate (no LOSO run; 2026-10-19)
`scripts/phase5/loso_estimate.py` → `outputs/results/phase5_loso_estimate/` (E26). 50 rep-level folds, inner split by the T8 rule (median 39 inner-training / 10 inner-validation subjects, 5,568 inner-training reps per fold vs 4,185–4,835 in GroupKFold). Epoch time per inner-training rep and epochs to early stopping come from the step (a2) runs. One GPU job at a time. Proposed = joint loss λ 1.0, fine-tuned from the step (a) CVDC encoder of the GroupKFold fold that held the subject out (C7), so no CVDC pretraining:

| Method | s / epoch (median fold) | epochs run in step (a2) | central (mean epochs) | low (fewest) | cap (60 epochs) |
|---|---|---|---|---|---|
| B1 | 5.1 | 35, 39, 60, 42, 60 | 3.4 h | 2.5 h | 4.3 h |
| B2-MediaPipe | 5.2 | 43, 33, 49, 37, 48 | 3.1 h | 2.4 h | 4.4 h |
| B2-KTPFormer | 5.6 | 29, 38, 49, 42, 37 | 3.1 h | 2.3 h | 4.7 h |
| Proposed (λ 1.0) | 10.6 | 27, 42, 60, 34, 60 | 6.6 h | 4.1 h | 8.9 h |
| **4 main methods** | | | **16.2 h** | 11.4 h | 22.3 h |

- Without the shortcut, Proposed would add 9.4 h of CVDC pretraining.
- The Phase 2 estimate (20.5 h) assumed full budgets and no joint loss. The joint loss doubles Proposed's epoch cost, so early stopping saves less than expected.
- The LOSO runner is not written yet: `fold_plan` supports GroupKFold only, and the shortcut needs the subject → encoder map (`splits.fold_of`) in place of the current subject check in `reuse_cvdc`.

## 6h. Full scope, Phase 3: E09 DTW angles, the aborted shared-model run, one model per exercise (2026-10-20)
Decisions T17, E1, E28 (new / updated), A5, D8, E11, E20 in `DESIGN_DECISIONS.md`. Code: `cvdc/train/run_per_exercise.py`, `configs/per_exercise.yaml`, `scripts/phase3_full/` (report, group probe, runtime estimate).

### 6h.1 E09 alignment angles by the warp-recovery test (eleventh review, item 1)
`scripts/phase1_full/dtw_angle_check.py --only E09` with every earlier set plus trunk–thigh + knees + arm angles (same 40 synced groups and 327 rep pairs as §5b.11; `outputs/full/phase1/dtw_angle_check_E09_v2.log`). Median error of the recovered known warp (t^1.5), frames:

| Angle set | Median warp error | Same rep cheaper than the next |
|---|---|---|
| **arm angles (shoulder elevation + elbow, both sides)** | **4.78** | 46.5% |
| trunk–thigh + knees + arm angles | 6.01 | 44.0% |
| trunk–thigh + knees (the previous choice) | 7.39 | 50.8% |
| hips + knees | 7.48 | 47.1% |
| trunk–thigh + hips | 7.84 | 54.7% |
| hips | 8.14 | 53.8% |
| trunk–thigh | 10.81 | 55.0% |
| no DTW | 9.79 | – |

**E09 now uses the arm angles** (`configs/exercises/E09.yaml`). Pairs rebuilt (`python -m cvdc.data.pairs --config configs/data_full.yaml`): edges, rep sets and pair counts unchanged; only the 25 DTW-aligned E09 members (from the 34 E09 rep pairs) and their DTW costs changed (checked against a backup, `data/full/backup_pairs_e09_trunk/`).

### 6h.2 The shared all-exercise run (GroupKFold-10), stopped
Built and started per the eleventh review (commit `fb7da25`): GroupKFold-10 over the 58 subjects, stratified by expert / non-expert (2–3 experts and 3–4 non-experts per test fold), one model for all nine exercises with a 9-class exercise head; E03 P02 masked out of the exercise labels (D8); the frozen Proposed sharing the λ = 1.0 run's CVDC encoder per fold; set features with the E03–E09 main angles (E20). **Stopped by the user after B1 fold 0** (60 of 60 epochs, 16.9 min; fold 1 at epoch 30). Partial outputs: `runs/full/aborted_shared_model/` and `outputs/full/results/aborted_shared_model/` (with `ABORTED.md`); nothing from it is reported. The code it needed stays (report and group-probe scripts, D8 mask, set-feature angles).

### 6h.3 One model per exercise: runner and smoke test (twelfth review)
- `python -m cvdc.train.run_per_exercise` (config `configs/per_exercise.yaml`): per exercise, GroupKFold-5 over that exercise's subjects (stratified by group where both exist), inner validation 20% per group, then for each method and fold the same `run_method_fold` as the pilot, with the data scoped to the exercise (`PackData.set_scope`), so the supervised items, the target scalers, the video head's count statistics, the CVDC rep sets (pretraining and joint loss), the embedding centre and the test predictions all come from that exercise. No exercise head and no exercise embedding; everything else as the pilot's step (a2). Methods: B1, B2-KTPFormer, Proposed (λ 1.0; CVDC pretrained per fold), frozen Proposed (reuses that encoder).
- Subjects per exercise (with ≥ 1 rep): E01 50 (24 / 26), E02 24, E03 49 (24 / 25), E04 24, E05 44 (23 / 21), E06 22, **E07 14**, E08 30 (14 / 16), E09 25 (14 / 11). Low-confidence flags (< 15 subjects): exercise E07; strata E07, E08, E09 experts (14) and E09 non-experts (11).
- **Smoke test** (`--smoke`: fold 0, 3 training subjects, 2 epochs; all 9 exercises × 4 methods, 1.2 min; `outputs/full/results/perex_smoke_train.log`): every run finished; every run's test predictions, embeddings and CVDC rep sets hold only its exercise; no run has exercise-head outputs; the frozen variant reused the λ = 1.0 run's CVDC encoder in all nine exercises. Peak GPU memory 8.1 GB (λ 1.0).
- Tests: `tests/test_phase3_full.py` (per-exercise config, a model without exercise head, the exercise scope on every item selection, per-exercise plans balanced by group, GroupKFold-10, D8, the encoder sharing, the set-feature angles); `python -m pytest -q tests`: **202 passed**.

### 6h.4 Runtime estimates (before launching; E28)
`scripts/phase3_full/runtime_estimate_perex.py`: measured cost per epoch and inner-training rep (pilot step (a2): B1 0.92 ms, B2-KTPFormer 0.99, Proposed λ 1.0 1.93, frozen 0.40; the aborted full-scope B1 fold 0.93 ms) and per CVDC rep set (4.34 ms), epochs to early stopping from the same runs; exact counts from each exercise's plans. One GPU job at a time.

| Step | Central | Range (fewest epochs – cap) |
|---|---|---|
| **1–2: GroupKFold-5 per exercise** (B1 1.1, B2-KTPFormer 1.0, CVDC 2.9, Proposed λ 1.0 2.1, frozen 0.6 h) | **7.7 h** | 5.7–10.0 h |
| **6: cross-view per exercise** (3 rotations × 5 folds; B1, B2-KTPFormer, Proposed with a fresh two-view CVDC) | **~14 h** (17.2 h by set counts) | 12.9–22.4 h |
| **7: LOSO per exercise** (288 folds; B1, B2-KTPFormer, Proposed, frozen), per-fold-encoder shortcut (C7) | **37.5 h** | 26–50 h |
| 7 without the shortcut (a CVDC pretraining per LOSO fold) | 60.8 h | 45–79 h |

Per exercise (1–2): 0.6 h (E07) to 1.0 h (E01). The cross-view figure by set counts assumes CVDC costs scale with the rep sets of the two training views (2.9× the all-view pretraining); in the pilot it cost 2.0×, so ~14 h is likelier. Uncertainty ≥ ±30%.

### 6h.5 Step 1 launched
`python -m cvdc.train.run_per_exercise --tag perex_gkf5` (exercises E01 → E09, per exercise B1 → B2-KTPFormer → Proposed λ 1.0 → frozen; log `runs/full/perex_gkf5/train.log`). Results will be reported per exercise as each finishes (§3 of the twelfth review). Meanwhile on the CPU: Module III for E03–E09 and the Phase 6 demo.

### 6h.6 Module III for E03–E09 (CPU, during the training; twelfth review item 5)
`python -m cvdc.assess.run --config configs/per_exercise.yaml` (all nine exercises, each with its own GroupKFold-5 folds, 75 s; M17) and `python scripts/phase4_full/report_module3.py` (`outputs/full/phase4/summary.md`). The generalised code reproduces the pilot's E01/E02 Module III output exactly. Templates from the PT demos: E03 V3 both arms over the 90/90 phase (4 reps, end range of the forearm elevation 19.0°), E04 1 rep (147.2°), E05 4 reps (shoulder elevation 154.0°), E06 2 reps (wrist angle 74.3° in the front-oblique demo; the target is the side-view experts'), E07 2 reps (hip flexion 124.7°), E08 2 reps (127.6°), E09 2 reps (trunk–thigh 48.9°); these match the Step 0 descriptions (§3.7.7).

**Agreement of the set score with the physiotherapists, within groups** (Spearman [95% CI]; ⚠ = < 15 subjects):

| Stratum | total score | MQ |
|---|---|---|
| E01 expert / non-expert | 0.04 [−0.09, 0.19] / −0.03 | 0.17 [−0.00, 0.33] / −0.03 |
| E02 expert | 0.16 [−0.04, 0.36] | **0.34 [0.12, 0.52]** |
| E03 expert / non-expert (after M18) | **0.23 [0.05, 0.38]** / **0.43 [0.20, 0.60]** | **0.22 [0.03, 0.39]** / **0.42 [0.19, 0.59]** |
| E04 expert | 0.17 [−0.05, 0.37] | 0.07 |
| E05 expert / non-expert | 0.10 / **0.38 [0.05, 0.60]** | 0.05 / **0.36 [0.04, 0.59]** |
| E06 expert | **0.22 [0.04, 0.40]** | **0.23 [0.04, 0.42]** |
| E07 expert ⚠ | **0.19 [0.04, 0.35]** | 0.13 [−0.03, 0.30] |
| E08 expert ⚠ / non-expert | −0.04 / **−0.33 [−0.51, −0.06]** | −0.02 / **−0.33** |
| E09 expert ⚠ / non-expert ⚠ | 0.13 / **0.71 [0.24, 0.83]** | 0.18 / **0.71** |

- **Weak positive agreement in most strata, none strong; one negative** (E08 non-experts: a higher Module III score goes with lower physiotherapist scores). The E09 non-expert value rests on 11 subjects whose clips mostly barely move (median score 0), so it mainly separates moving from not moving.
- **As a predictor** (out-of-fold within-group line vs the within-group mean): no stratum gains with a CI below 0; the largest gains are the non-experts of E03 (−0.82 [−1.54, 0.11]) and E05 (−0.83 [−2.01, 0.45]); E04 MQ and E08 expert lines are slightly worse than the mean (CIs above 0). As in the pilot (E21), the reference score does not beat the within-group mean.
- **Expert vs non-expert separation** (context): AUC E01 0.98, E09 0.94, E08 0.73, E05 0.71, E03 0.61. Outside E01/E09 the groups score much closer under Module III than under the physiotherapists (e.g. E03 74.7 vs 70.7; physiotherapists 75.4 vs 44.4).
- **Rule specificity** (share of reps, experts' videos ≥ 95 vs < 80): specific in E02 (posture rules, as after M14), E07 (`other_hip_flexed` 13% vs 26%, `other_knee_flexed` 13% vs 25%, `too_fast` 1% vs 30%), E06 (`too_fast` 2% vs 22%), E04 (`shoulder_elevation_deviation` 16% vs 26%); **not specific** in E03 (`lateral_rotation_range_insufficient` fired in 38% of the best videos' reps: the V3 demo reaches 19° of forearm elevation while the experts stop at ~27°, so the demo-centred ROM target, M10, was beyond most experts; **after the M18 calibration 7% vs 6%**, §6h.8) and E08 (`knee_flexion_deviation` 11% vs 0%, the latter from only 17 videos). `too_fast` fires in 28–81% of non-expert reps everywhere (short clips, quick reps).
- **View stability** over synced expert performances (rep level, ICC(1)): the full score 0.51 (E03 after M18; 0.33 before) – 0.68 (E04), but without its speed group (shared rep timing, P6b) 0.03 (E06) – 0.45 (E04; E03 0.26 after M18, 0.07 before); the end-range angle −0.11 – 0.30. Module III's 3D measures stay view-dependent, as in the pilot.

### 6h.8 Module III rule calibration (post-hoc, M18) and the E08 non-expert sign (user, 2026-10-20; CPU only)
`scripts/phase4_full/rule_calibration.py` → `outputs/full/phase4/calibration/summary.md`; before = targets on the PT demo (`outputs/full/phase4/calibration_pt_demo/`, `--set E03:rom_centre=pt_demo --set 'E08:channel_centre={knee_flexion: pt_demo}'`), after = on the ≥ 95 experts per view (fitted per fold on the training subjects; now the default output). Checked: the before run reproduces the earlier E03 output exactly, and the after run changes only E03.

| Rule | version | experts ≥ 95 | experts < 80 | non-experts | agreement with total (experts / non-experts) |
|---|---|---|---|---|---|
| E03 `lateral_rotation_range_insufficient` | PT demo (before) | 38% | 35% | 48% | 0.19 / 0.40 |
| | **experts (after)** | **7%** | 6% | 7% | 0.23 / 0.43 (Δ +0.04 [−0.04, 0.11] / +0.02 [−0.05, 0.09]) |
| E08 `knee_flexion_deviation` | PT demo (before) | 16% | 0% (17 videos) | 6% | −0.01 / −0.35 |
| | experts (after = the M14 default already) | 11% | 0% (17 videos) | 4% | −0.04 / −0.33 |

- **E03:** the range rule no longer flags the best videos (38% → 7%) but is still not more frequent in the lower-scoring ones (6%); the mean expert set score rises 74.7 → 81.0, the view stability of the rep score improves (ICC 0.33 → 0.51; without speed 0.07 → 0.26), agreement changes within noise.
- **E08 knee rule:** its target was already the experts' curve (E08's knee is a posture channel under M14), so nothing new was applied; the demo-centred version fires more (16% vs 11%). Its 0% in the experts' videos < 80 comes from 17 videos, so the rule's specificity cannot be judged in E08 (56% of its expert videos score 100).
- **E08 non-experts (ρ −0.33): the speed group carries the sign.** Without it the set score agrees positively (+0.22 [−0.05, 0.41]); without any other group it stays negative. The better non-experts (P27, P45–P50; physiotherapists 45–60) do 1.2–2 reps per clip at ~2 s per rep and are flagged `too_fast` in 42–100% of their reps; the lowest (P29, P38, P41: PF1 ≈ 0, scores 0–8) do one slow ~3–4 s rep, close to the experts' tempo, with normal end ranges, so Module III rates them higher. The rep count per clip (ρ +0.60) and the speed ratio (+0.54) are what tracks the physiotherapists here; the knee deviation goes the right way (ρ −0.46). **Decision (user, 2026-10-20): the speed reference stays expert-centred (M5a, M15; no post-hoc change), and this explanation is reported as a finding**: for the E08 non-experts the expert tempo reference rates the slow single reps of the lowest-scoring clips as close and the quicker, more numerous reps of the better clips as too fast.

### 6h.9 Step 1 results: one model per exercise, GroupKFold-5 (2026-10-20)
180 runs (9 exercises × 4 methods × 5 folds) in **6.7 h** (estimate 7.7 h, 5.7–10.0 h). Then `scripts/phase3_full/embedding_distance_perex.py` (225 encoder-folds, 4 min, GPU), `group_probe.py` (4.5 min) and `report_perex.py` → **`outputs/full/results/perex_gkf5/summary.md`** (+ CSVs). Video level, within groups, 95% CIs from the subject bootstrap (2,000 draws, paired). ⚠ = < 15 subjects: E07, E08, E09 experts (14), E09 non-experts (11). Every exercise's models saw only its own videos (T17).

**Total score, video head: RMSE / ρ** (rep head and CIs in `score_metrics.csv`):

| Stratum (subjects) | B1 | B2-KTP | Prop λ1 | Prop frozen | group-only | set-feat. | floor |
|---|---|---|---|---|---|---|---|
| E01 expert (24) | 13.29 / 0.08 | 14.67 / −0.02 | 13.46 / 0.12 | 14.72 / −0.03 | **12.45** | 12.22 / 0.18 | 5.28 |
| E01 non-expert (26) | 11.44 / 0.11 | 15.68 / −0.18 | 15.48 / −0.18 | 14.15 / −0.01 | **7.78** | 7.48 / 0.26 | – |
| E02 expert (24) | 18.89 / 0.13 | 17.52 / 0.16 | 16.78 / 0.27 | 15.77 / 0.21 | **16.01** | 16.96 | 4.38 |
| E03 expert (24) | 20.08 / 0.03 | 20.88 / 0.03 | 16.52 / 0.27 | 15.76 / 0.24 | **14.71** | 15.26 | 3.38 |
| E03 non-expert (25) | 16.53 / −0.12 | 14.77 / −0.09 | 17.92 / −0.07 | 16.22 / −0.10 | **9.32** | 9.23 | – |
| E04 expert (24) | 9.08 / 0.37 | 8.83 / 0.33 | **8.06 / 0.52** | 8.76 / 0.29 | 8.94 | 8.90 | 2.45 |
| E05 expert (23) | 16.33 / 0.29 | 17.60 / 0.02 | 13.37 / 0.26 | 14.16 / 0.02 | **11.75** | 11.96 | 4.76 |
| E05 non-expert (21) | 20.41 / 0.31 | 23.13 / 0.07 | 27.86 / 0.25 | 25.63 / 0.38 | **12.45** | 12.06 | – |
| E06 expert (22) | 9.25 / −0.14 | 8.85 / 0.03 | 9.58 / 0.03 | 8.73 / 0.08 | **8.35** | 9.56 | 3.16 |
| E07 expert ⚠ (14) | 16.94 / −0.11 | 14.81 / 0.28 | 17.03 / −0.03 | 15.42 / −0.02 | **14.76** | 15.29 | 2.76 |
| E08 expert ⚠ (14) | 12.38 / 0.25 | 10.94 / 0.15 | 13.64 / 0.30 | 17.23 / 0.05 | 9.57 | **8.75** | 4.04 |
| E08 non-expert (16) | 25.17 / 0.50 | 33.11 / 0.26 | 29.31 / 0.67 | 32.46 / 0.53 | 20.09 | **16.80** | – |
| E09 expert ⚠ (14) | 24.97 / 0.06 | 24.28 / 0.17 | 21.59 / 0.02 | 23.80 / 0.07 | 17.38 | **17.16** | 2.68 |
| E09 non-expert ⚠ (11) | 33.41 / 0.09 | 36.09 / 0.01 | 23.34 / 0.69 | 26.74 / 0.70 | 13.28 | **10.55** | – |

- **No model beats the within-group mean in any of the 14 strata** (paired RMSE CI below 0 in 0 strata for every method; above 0 in 9–10). Closest: E04 Proposed −0.88 [−2.08, 0.39] (ρ 0.52), E02 frozen −0.24 [−2.28, 1.75]. Pooled over the exercises within each group, every model is worse than the group mean (experts +1.7 [1.0, 2.5] for Proposed to +3.3 for B1; non-experts +7.6 to +10.8). **The pilot's negative result (E21) holds for every exercise.**
- **Proposed vs B1 (experts):** lower RMSE pooled −1.57 [−2.27, −0.86], and in E02 (−2.11), E03 (−3.56), E04 (−1.02), E05 (−2.96) (CIs below 0); better ranking in E02 (ρ +0.14 [0.01, 0.25]) and E03 (+0.25 [0.08, 0.42]); within-exercise ranking pooled over experts (centred ρ) 0.22 [0.14, 0.29] for Proposed vs 0.11 for B1, 0.14 for B2-KTPFormer, 0.14 for the set features. **Non-experts: Proposed is worse than B1** (E01 +4.0, E05 +7.5 RMSE; calibration on few short clips).
- **MQ** (`summary.md` §2): within experts no model beats the group mean (CIs include 0 or lie above it); in **E08 and E09 non-experts** (⚠ E09) B1 and Proposed do (E08 MQ −6.7 / −7.2, E09 −6.2 Proposed, −8.0 frozen; ρ 0.65–0.78). The E09 non-experts score a median of 0 and most clips barely move, so this mainly separates moving from not moving (the random-init score probe reaches R² 0.56 there too).

**Within-group VCI (centred 256-d, held-out subjects)**:

| Stratum | random init | after CVDC | B1 | B2-KTP | Prop λ1 |
|---|---|---|---|---|---|
| E01 / E02 / E03 / E04 / E05 / E06 expert | −0.17 – 0.08 | **0.51–0.62** | 0.06–0.30 | −0.06–0.22 | **0.41–0.57** |
| E07 / E08 / E09 expert ⚠ | **0.66 / 0.72 / 0.42** | 0.73 / 0.68 / 0.70 | 0.23 / 0.20 / 0.32 | 0.32 / 0.17 / 0.23 | 0.57 / 0.59 / 0.64 |
| non-expert strata (19–71 rep-matched pairs) | 0.02–0.56 | 0.18–0.43 | 0.09–0.31 | −0.19–0.22 | 0.17–0.37 |

- **Upper-limb exercises: CVDC makes the embedding view-consistent** (after CVDC 0.51–0.62 vs a random-init floor ≤ 0.08), and the λ 1.0 joint loss keeps most of it (0.41–0.57) against 0.06–0.30 for B1 and less for B2-KTPFormer, as in the pilot.
- **Lying exercises (E07–E09): the random encoder is already view-consistent** (0.42–0.72), and trained B1 / B2 lose it; CVDC stays at about the random level (0.68–0.73). So in E07–E09 the VCI cannot show a CVDC gain over an untrained network; a likely reason is that synced views share rep boundaries (P6b), so the temporal profile of a rep is identical across views, and in the lying exercises the movement's timing dominates an untrained encoder's embedding (not tested).

**View stability of the distance to the PT reference** (synced expert performances, rep level, ICC(1); ΔICC vs B1 [95% CI]):

| Exercise | random init | B1 | B2-KTP | Prop λ1 (Δ vs B1) | Prop frozen (Δ vs B1) |
|---|---|---|---|---|---|
| E01 | 0.02 | 0.13 | 0.03 | 0.69 (**+0.56 [0.43, 0.68]**) | 0.74 (**+0.61**) |
| E02 | −0.09 | 0.45 | 0.19 (−0.26) | 0.73 (**+0.28 [0.15, 0.46]**) | 0.72 (**+0.27**) |
| E03 | −0.02 | 0.14 | −0.08 (−0.22) | 0.55 (**+0.41 [0.20, 0.63]**) | 0.65 (**+0.51**) |
| E04 | 0.19 | 0.35 | 0.15 | 0.78 (**+0.42 [0.27, 0.60]**) | 0.84 (**+0.49**) |
| E05 | 0.14 | 0.30 | 0.15 | 0.34 (+0.04 [−0.18, 0.24]) | 0.47 (+0.17 [−0.02, 0.35]) |
| E06 | −0.42 | 0.58 | 0.50 | 0.75 (+0.17 [−0.08, 0.58]) | 0.77 (+0.19) |
| E07 ⚠ | **0.95** | 0.54 | 0.63 | 0.82 (**+0.28 [0.08, 0.52]**) | 0.93 (**+0.39**) |
| E08 ⚠ | **0.96** | 0.47 | 0.41 | 0.88 (**+0.42 [0.26, 0.59]**) | 0.92 (**+0.46**) |
| E09 ⚠ | **0.89** | 0.79 | 0.55 (−0.24) | 0.88 (**+0.09 [0.02, 0.16]**) | 0.87 (**+0.07**) |

- **CVDC makes the vector difference view-stable** in 7 of 9 exercises (Proposed vs B1, CIs above 0; not E05 and E06); B2-KTPFormer is less stable than B1 in E02, E03, E09. In E07–E09 the random-init encoder is as stable as Proposed (see the VCI caveat).

**Group probe (E11)** (`summary.md` §5): expert vs non-expert is separable from every encoder's embeddings, **the random-init encoder included** (pooled AUC 0.87 for all five; front-view FL view-matched 0.77 random init to 0.86 frozen CVDC; E01 0.95–1.00, E09 0.90–0.99, E03 0.72–0.81, E05 0.76–0.83, E08 0.76–0.92). The group signal is in the keypoint sequences themselves (clip length, recording set-up), not produced by the training or by CVDC. Within groups the score probes find little: R² ≤ 0 in most expert strata (exceptions E04: B1 0.09, Proposed 0.21, frozen 0.13; E05 B1 0.15), positive for the E08 / E09 non-experts (moving vs not moving).

**Training:** CVDC kept epochs 8–99 (median ~55), fine-tuning 10–60; Proposed λ 1.0 takes 20–39 min per exercise (5 folds incl. CVDC), B1 4–10 min. Peak GPU memory 8.2 GB.

**Conclusions of step 1:**
1. One model per exercise does not change the main negative result: **no network predicts the score (or MQ, for experts) within a group better than the within-group mean**, in any exercise.
2. Among the networks, Proposed (λ 1.0) is the most accurate for the experts (pooled RMSE −1.6 vs B1, better in E02–E05) and ranks best within exercises, but still not better than the mean; it is worse than B1 for the non-experts.
3. **CVDC's measurable effect is view consistency**: within-group VCI and the view stability of the distance to the PT reference are clearly higher than B1 / B2 in the upper-limb exercises and E07–E09 (the distance), but the lying exercises have a high random-encoder floor.
4. The expert / non-expert groups are separable even from untrained embeddings, so pooled results would mostly measure the group (E9–E11).

### 6h.7 Phase 6 demo code (CPU, during the training; twelfth review item 5)
`demo/pipeline.py` (engine), `demo/live.py` (video file at real-time speed, or `--source 0`), `demo/app.py` (Streamlit); decision X5.
- `python -m demo.live --select E07 P05 F FL` (or `--source file.mp4 --exercise E07 [--view L]`): skeleton overlay (hand landmarks for E06; the joints of the last rep's errors in red), exercise / view / rep counter, each rep's Module III score, quality and raw errors (`type · joint · deviation · frames (severity)`), the model's rep score and distance to the PT reference, a live plot of the main angle and the last rep against its template band, FPS and the checkpoint; the set score at the end (model video head if loaded, else Module III); keys q / r / s; `--save-mp4` with blurred faces; `--no-model` runs Module III only (X3); `--list E03` lists dataset videos.
- **Check:** on `E07_P05_AF_VFL_GM` (headless, Module III only, 34 s for a 38 s video including MediaPipe) the final reps equal the batch own-view segmentation exactly (11 reps); set score 78.7. A first version locked the moving side at the first rep and picked the wrong leg (the subject moves the left leg in the first 6–8 s), which gave scores of 11–24 with `other_hip_flexed` errors; fixed by re-deriving side and reps from the whole buffer every 0.5 s (X5).
- Tests (`tests/test_demo.py`): the online session fed the cached Phase 1 keypoints of E03 / E05 / E07 / E09 videos (in partial buffers, then final) reproduces the batch own-view reps and side; the error-joint mapping; the face blur touches only the head box.
- **With a trained model** (E01 Proposed λ 1.0, fold 0): `--select E01 P11 F FL --fold 0` loads the checkpoint, prints rep scores (Module III and model), the distance to the PT reference and the model's set score (89.8, MQ 94.5; Module III 72.5), and shows the **leakage warning** (P11 was in fold 0's training split; without `--fold` the fold that held P11 out is chosen). Reps whose boundaries move as more video arrives are re-reported as "updated".
- **Streamlit app** checked in the browser: dataset video E01 P15 F FL (held out by fold 0; physiotherapists 76.67): 10 reps, Module III set score 88.0, model set score 82.8 (MQ 85.9), per-rep table, template plots, the embedding plot; results survive widget reruns; no server errors.

## 7. Phase checklist
The pilot phases ran on **E01 + E02** (§0); the full scope (all nine) has Phase 1 (§5b). Code stays exercise-agnostic, with exercises discovered at runtime and filtered by `data.include_exercises`.

- [x] **Step 0:** environment, inventory, labels, sync census, references (E01, E02, E03-partial)
- [x] **Step 0 on the full dataset (E03–E09)** (2026-10-20; §3.7): inventory and CSV join, decode and copy checks, sync classes for all groups, 355-video pose sample (detection, posture, rep signals, motion sync with controls), PT-demo comparison, label structure, EAAQ search, verdicts, full-scope runtime estimates. Answers: ninth review (§4).
- [x] **Phase 1 on the full scope (E01–E09)** (2026-10-20; §5b): video root and new folders (`configs/data_full.yaml`); Pose for 3,685 videos + Hands for E06 (on wrist crops); sync with S8 and the extended motion candidates; reps with the approved signals, adaptive thresholds and the rep-phase fix (P6c); packs; KTPFormer lift; renders and report; E05 PF1 motion check; E06 Hands check; E01/E02 vs the pilot and E03–E09 vs Step 0; 20 new tests.
- [x] **E09 DTW angles by the warp test** (eleventh review; §6h.1): arm angles, pairs rebuilt.
- [x] ~~Shared all-exercise GroupKFold-10 run~~ (eleventh review): started, **stopped** (twelfth review: one model per exercise); partial outputs archived (§6h.2).
- [ ] **One model per exercise (T17; §6h):** runner + smoke test [x]; runtime estimates [x]; (1–2) GroupKFold-5 main runs [x] (6.7 h); per-exercise report [x] (§6h.9); Module III E03–E09 [x] (§6h.6; rule calibration M18, §6h.8); Phase 6 demo [x] (§6h.7); (6) cross-view per exercise [running; smoke test passed]; (7) LOSO per exercise: not started (supervisor / cloud).
- [x] **Tenth-review changes** (2026-10-20; §5b.3, §5b.10, §5b.11): E06 wrist angle in side views only (Phase 4); rep-phase fix accepted; E03 / E09 rep signals re-chosen by cross-view agreement; per-exercise DTW angles; CVDC pairs built for the full scope; 2 more tests (187). **Not started:** any training on the full scope (waits for the user's confirmation of the final evaluation scheme).
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
  - [x] **LOSO runtime re-estimate with early stopping and the shortcut** (Phase 5, §6g, E26);
  - [x] **(a) GroupKFold-5: B1, B2-MediaPipe, B2-KTPFormer, Proposed** (CVDC pretraining + fine-tuning): within-group results vs mean / group-only / duration-only baselines, within-group VCI; **stop and report**; done 2026-10-18, results in §6b;
  - [x] **(a2) score setup** (user request after step (a)): label diagnostics and set-level baselines (rep-count-only, set-features); video head (mean + attention pooling + set rep count, T13); T6 per-video weighting; early stopping min 10 / patience 15 / EMA (T8b); items only where they vary (T14); Proposed variants (i) frozen, (ii) λ 0.1 / 1.0, (iii) full fine-tune (T15); re-run of B1, Proposed variants, B2-MediaPipe, B2-KTPFormer on GroupKFold-5; **stop and report**; done 2026-10-18, results in §6c;
  - [ ] (b) CVDC ablations on GroupKFold (C5): **pair sources done (Phase 5 B, §6f)**; τ, embedding size, 2 vs 3 views, augmentation not run;
  - [ ] (c) then the final LOSO table (per-fold-encoder shortcut, C7) for the 4 main methods: **runtime estimated (§6g, ~16 h), not started; the LOSO runner is not written yet**.
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
  - GroupKFold (done for the main table, §6b–6c) and LOSO (estimated, §6g; not started);
  - stats;
  - [x] cross-view protocol (GroupKFold-5, 4 main methods; §6e, E24);
  - [x] pair-source ablations (a)–(e) and (d′) on the VCI, E01 non-experts as a separate stratum (§6f, E25); [ ] τ, embedding size, 2 vs 3 views, augmentation (C5);
  - VCI: centred 256-d (primary), raw, projection; within strata (E13);
  - B2 with MediaPipe-3D and KTPFormer-3D inputs (A10);
  - label-noise floor;
  - baselines (mean, duration-only, group-only);
  - embedding-distance analysis (within-group tracking and view stability done in Phase 4, §6d; repeat on LOSO);
  - UMAP/t-SNE with reference embeddings.
- [ ] **Phase 6, demo** (default view front, `--view F|L|R`, X4):
  - `demo/live.py` on video files (real-time playback, field picker, leakage warning, .mp4 export, embedding-distance overlay);
  - `demo/app.py`.

## 8. Open items and questions
**Answered (user, 2026-10-20, after M18):** the E08 speed reference stays expert-centred; the speed-group explanation of the E08 non-experts' negative agreement is a reported finding (§6h.8, M18).

**Open after the twelfth review (2026-10-20):**
1. **LOSO per exercise:** the supervisor's (or cloud) decision; estimated 37.5 h with the shortcut (§6h.4); not started.
2. **EAAQs for E02–E09** (§3.7.8): still pending.

**Answered in the eleventh and twelfth reviews (2026-10-20):** the evaluation scheme (one model per exercise, GroupKFold-5 per exercise; T17), the E09 DTW angles (by the warp test: arm angles), the GroupKFold folds, the noise floors for E03–E09 (E3, `eval.noise_floor`), the exercise-label exclusion (E03 P02; no exercise head in the per-exercise models).

**Answered in the tenth review (2026-10-20):** the four questions after Phase 1 on the full scope (§4; applied in §5b.3, §5b.10, §5b.11): E06 wrist angle in the side views only; the rep-phase fix accepted; E03 / E06 / E09 rep signals compared and re-chosen by cross-view agreement; per-exercise DTW angles.

**Answered in the ninth review (2026-10-20):** the eight questions after the full-data Step 0 (§4; applied in §5b): all seven exercises in; Hands for E06; video root and new folders; both sync rules (start-time groups get `w_sync` only with motion confirmation); PF1 = 0 is no exclusion reason (E05 check: none excluded); GroupKFold-10 planned, LOSO pending; the angles and rep signals accepted as starting definitions; the EAAQ appendix pending.

**Answered in the third review (2026-10-17):** the four Phase 1 questions (§4, applied in §5.6).

**Answered in the fifth review (2026-10-17):** the four Phase 2 questions (§4, applied in §6.5).

**Answered in the sixth review (2026-10-18):** KTPFormer licence (local non-commercial research use, cite the paper; the user informs the supervisor), headline VCI (within-group, centred), epoch budget with early stopping, Phase 3 run order (§4).

**Recorded, not open:** KTPFormer is less view-consistent than MediaPipe on 3 of 4 angles (only E01 shoulder elevation improves; §6.5); B2 is reported with both inputs.

**Addressed in Phase 4:** the E02 horizontal adduction angle's view dependence (per-view bias and bands, M7a, M8); the left-arm CF difference stays borderline (no flag under the agreed rule).

**Addressed in step (a2) (§6c):** the step (a) open items (noisy early stopping → T8b; linear probe and the joint objective → Proposed variants; T6 → per-video weighting, decided).

**Answered in the seventh review (2026-10-19)** (the step (a2) questions; §4): within-group score prediction is reported as a negative result with the label diagnostics (E21); Proposed = joint loss λ 1.0, frozen encoder as the alternative (T15); no angle / ROM inputs to the heads (T16); total score official, movement-quality score secondary (D13). Still my choices, not reviewed: the T14 thresholds (≥ 5 subjects off the mode, SD ≥ 0.05), the EMA factor 0.7 (T8b), and the reuse of the step (a) CVDC encoders (T15).

**Answered in the eighth review (2026-10-19)** (the Phase 4 questions; §4, §6d.6–6d.7): posture checks centred on the experts (M14, post-hoc, before / after reported); speed stays expert-centred, demo-centred speed exploratory only (M15); the demo defaults to the front view with `--view` (X4).

**Questions after Phase 5 A and B (not applied):**
1. **LOSO:** start the final LOSO table for the 4 main methods? Estimated ~16 h of GPU (11–22 h), run as one job at a time; the LOSO runner (shortcut encoder map, inner splits per LOSO fold) still has to be written and smoke-tested first.
2. **CVDC early stopping in the ablations:** the agreed rule (T8a: patience 10, no smoothing) kept epoch 3 in sync-only fold 2 (§6f), which pulls that setting's VCI down. Keep T8a as decided (as now), or give CVDC pretraining the T8b smoothing / minimum epochs for the remaining ablations? Changing it would also change (d) and need re-runs.
3. **Synthetic views and the unseen camera:** in A the held-out camera was reachable during CVDC only through synthetic views, while B shows synthetic views add nothing on top of real pairs between seen cameras. A cross-view run of Proposed without synthetic views (~2.4 h) would show whether they are what carries CVDC to the unseen camera. Run it?
4. **Remaining C5 ablations** (τ, embedding size, 2 vs 3 views, augmentation): run them on GroupKFold before LOSO, or after?
5. **E01 non-experts:** no CVDC benefit in A or B (mostly front-only, 71 rep-matched pairs). Report as a limitation (D12) and keep them as a separate stratum, as now?

**Still open after Phase 4:**
- **E01 CF item meanings** (which EAAQ criterion is CF1, CF2, CF3): the user asks the supervisor (dataset author). Until then the rules do not follow the items, and the non-expert end-range finding (a higher end range goes with lower scores, ρ −0.31) stays unexplained.
- E01 posture rules are still not more frequent in lower-scoring expert videos (§6d.6).
- Score and rule defaults (equal weights, d_zero 4, k 2, σ floors, run length) are my choices (M9–M12), not tuned.

**Other open items:**
- **EAAQ appendix** for E02–E09: not in the dataset (§3.7.8); most likely in the paper's online supplementary material. Until it arrives, only E01 has questionnaire-derived rules.
- **E03–E09:** Phase 1 done (§5b); training waits for the answers above.

## 9. Re-running the Step 0 check
**Full dataset (2026-10-20):** `scripts/step0_full/README.md` (order, arguments, run times; outputs in `outputs/step0_full/`).

**Pilot:** see `scripts/step0/README.md` for the order and arguments. Minimum for a new exercise: `inventory.py` → `video_meta.py` → `scores.py` → `sync_av.py <EX>` → `sync_classify.py`, plus a pose sample (`select_groups_ex.py` → `extract_pose.py` → `sync_analysis3.py`) and a reference comparison.
