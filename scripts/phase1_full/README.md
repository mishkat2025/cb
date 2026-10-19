# Phase 1 on the full scope (all nine exercises)

Phase 1 (keypoints, preprocessing, sync, reps, packs, KTPFormer lifting) for the full MobiPhysio dataset, run on
2026-10-20 after the user's answers to the full-data Step 0 questions (`PROGRESS.md` §5b). Everything reads
`configs/data_full.yaml`, which `extends` the pilot config and moves every input and output to `data/full/`,
`outputs/full/` and `runs/full/` (D15). **Nothing in the pilot's `data/index.csv`, `data/cache/*`, `data/reps*.csv`,
`outputs/phase*`, `outputs/sync_*.csv` or `runs/` is written.** Keypoints only: no frames, crops or audio are saved;
GPS tags are never read.

Run from the repo root with the venv's Python (`D:\cb\.venv\Scripts\python.exe`):

| Order | Command | What it does | Time (this machine) |
|---|---|---|---|
| 0 | copy `data/cache/kp/*.npz`, `data/cache/step0_full/kp/*.npz` → `data/full/cache/kp/`; the two `kp_ref` folders → `data/full/cache/kp_ref/`; `data/cache/sync/audio_pairs.csv` → `data/full/cache/sync/` | optional: reuse the pilot's E01/E02 keypoints and audio lags and the Step 0 pose sample (same extractor settings; the extractor checks the settings stamp and skips them) | seconds |
| 1 | `scripts/run_phase1.py --config configs/data_full.yaml --to extract` | index (3,685 videos, 12 PT demos), MediaPipe Pose for every video not cached, and MediaPipe Hands on wrist crops for E06 (P1c) | see `PROGRESS.md` §5b |
| 2 | `scripts/run_phase1.py --config configs/data_full.yaml --from preprocess --to pack` | preprocessing, sync (audio → creation time → start time → motion; S8, S9), reps (adaptive per-video thresholds for E03–E09, P6c), fixed-length packs | |
| 3 | optional: copy the pilot's `data/cache/lift_ktpformer/E0[12]_*.npz` and the 3 E01/E02 demo lifts → `data/full/cache/lift_ktpformer/` **only if** `scripts/phase1_full/compare_pilot.py` reports the E01/E02 preprocessed files identical | reuse the pilot's KTPFormer lifts | seconds |
| 4 | `scripts/run_phase1.py --config configs/data_full.yaml --from lift` | KTPFormer lifting (GPU; `--skip-existing`), lifted packs, skeleton-only checks, report tables | |
| 5 | `scripts/phase1_full/hands_check.py` | E06: hand detection, the Hands wrist-extension angle vs the Pose-point angle (cross-view rep agreement and same-instant agreement in synced groups), skeleton + hand renders | minutes |
| 6 | `scripts/phase1_full/pf1_motion_check.py --exercise E05` | user answer 5: do the E05 non-experts with mean PF1 < 0.5 perform a circumduction? (rule fixed in the script's docstring) | seconds |
| 7 | `scripts/phase1_full/compare_pilot.py` | E01/E02 in the full scope vs the pilot (identical keypoints / preprocessing / reps except where the S8 sync rules changed a group); Phase 1 vs Step 0 per exercise | seconds |

Outputs: `outputs/full/phase1/` (tables and logs), `outputs/full/checks/` (skeleton-only PNGs),
`outputs/full/sync_report.csv`, `outputs/full/sync_pairs.csv`, `data/full/` (index, reps tables, caches, packs).
