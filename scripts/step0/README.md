# Step 0 inspection scripts

Exploratory scripts used for the Step 0 data check (see `PROGRESS.md` §3). They will be superseded by
`cvdc/data/` in Phase 1 (filename parser, sync classifier, rep segmentation). Keypoints only: no frames
or audio are ever written; GPS tags in the video containers are never read.

Dataset root = env `MOBIPHYSIO_ROOT`, default: the folder containing this repo. Run from the repo root
with the venv's Python (`D:\cb\.venv\Scripts\python.exe -I ...`). `$O` = an output folder, e.g. `outputs/step0`.

| Order | Script | What it does |
|---|---|---|
| 1 | `inventory.py <root> $O/inventory.csv` | parse every filename, counts per exercise/view/variation/participant, F+L+R groups |
| 2 | `video_meta.py <root> $O/video_meta.csv` | fps, frame count, resolution, rotation, duration (headers only) |
| 3 | `scores.py "<root>/Physiotherapist Exercise Marking.csv" $O/inventory.csv $O/video_meta.csv` | parse the 3-judge x 7-item CSV, join to local files, per-exercise score stats, quality bins |
| 4 | `select_groups_ex.py $O/inventory.csv $O/list.txt <EX> <n_pid> <max_pnum> [seed]` | sample complete groups (2 variations per participant -> controls) |
| 5 | `extract_pose.py data/models/pose_landmarker_full.task $O/kp 960 8 $O/list.txt` | MediaPipe PoseLandmarker (Tasks API) -> .npz keypoints (2D + world) |
| 6 | `sync_av.py $O/inventory.csv $O 8 [EX ...]` | audio GCC-PHAT lags, creation_time differences, LR-copy check (all groups) |
| 7 | `sync_analysis3.py $O/kp $O` | motion + rep-timing sync test on the sampled groups |
| 8 | `sync_classify.py $O` | combine audio -> metadata -> motion into synced / not-synced / unknown per group |
| 9 | `sync_eval.py $O` | audio/metadata/motion agreement diagnostics |
| 10 | `ref_analysis.py <kp_ref>` | posture, apparent view, moving limb, reps of the PT reference videos |
| 11 | `e03_ref_compare.py`, `e03_ref_chamfer.py` | compare the three E03 reference versions with dataset executions (3D angles) |
| 12 | `noise_floor.py $O/inventory_scored.csv` | inter-view label noise (RMSE floor) per exercise |
