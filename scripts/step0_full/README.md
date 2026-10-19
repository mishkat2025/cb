# Step 0 on the full dataset (all 9 exercises)

Step 0 check of the full MobiPhysio download (2026-10-20, `PROGRESS.md` §3.7). Same method as the pilot Step 0
(`scripts/step0/`), reusing the Phase 1 parser, CSV loader, extractor, preprocessing, rep segmenter and sync tests from
`cvdc/`. Keypoints only: no frames or audio are written; GPS tags are never read. **Nothing here touches the pilot's
artefacts** (`data/index.csv`, `data/cache/kp`, `outputs/phase*`, `outputs/step0`): results go to `outputs/step0_full/`
and keypoints of the pose sample to `data/cache/step0_full/` (both gitignored).

Videos now live in `D:\cb\mobiphysio\E0x(...)`; the scores CSV, `MobiPhysio.md` and the PT reference folder stay in
`D:\cb`. Run from the repo root with the venv's Python (`D:\cb\.venv\Scripts\python.exe -I ...`):

| Order | Command | What it does | Time (this machine) |
|---|---|---|---|
| 1 | `scripts/step0_full/inventory.py --videos D:/cb/mobiphysio --meta-root D:/cb --out outputs/step0_full` | parse every filename, join the CSV, counts, README Table 1 check, pilot-file check | seconds |
| 2 | `scripts/step0_full/video_meta.py --videos D:/cb/mobiphysio --inv outputs/step0_full/inventory.csv --out outputs/step0_full` | container metadata + first/last-second decode check, MD5 of same-size candidates | ~10 min |
| 3 | `MOBIPHYSIO_ROOT=D:/cb/mobiphysio scripts/step0/sync_av.py outputs/step0_full/inventory.csv outputs/step0_full 12 E03 ... E09` | audio GCC-PHAT lags, creation-time differences, LR copy check (pilot script) | ~21 min (HDD-bound) |
| 4 | `scripts/step0_full/pose_sample.py --videos D:/cb/mobiphysio --meta-root D:/cb --inv outputs/step0_full/inventory.csv --out outputs/step0_full` | select the pose sample, MediaPipe Pose (Phase 1 settings), preprocessing, detection / visibility / posture tables | ~22 min (355 videos + 9 PT demos) |
| 5 | `scripts/step0_full/pose_analysis.py --out outputs/step0_full` | rep signals and cross-view rep agreement, motion sync test with controls, PT-demo comparison (needs 6 first for the sync classes) | ~2 min |
| 6 | `scripts/step0_full/sync_classify.py --out outputs/step0_full` | audio -> creation time -> motion classes for every multi-view group; run before 5 and again after it | seconds |
| 7 | `scripts/step0_full/copy_check.py --videos D:/cb/mobiphysio --out outputs/step0_full` | audio copy check of the videos without creation_time | ~1 min |
| 8 | `scripts/step0_full/labels_check.py --out outputs/step0_full` | item activity (T14 rule), repetition-item variance share, items identical across views | seconds |
| 9 | `scripts/step0_full/runtime_estimate.py --out outputs/step0_full [--exercises ...]` | runtime estimates from the pilot's measured timings | seconds |

Main outputs: `inventory.csv`, `summary_by_exercise.csv`, `video_meta.csv`, `av_pairs.csv`, `av_videos.csv`,
`sync_groups_classified.csv`, `pose_sample.csv`, `pose_summary.csv`, `rep_counts_sample.csv`,
`rep_signal_agreement.csv`, `motion_pairs_sample.csv`, `ref_compare.csv`, `labels_check.csv`, `copy_check.csv`,
`runtime_inputs.csv`, `runtime_estimate.csv`, plus a `.log` per step.
