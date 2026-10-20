# Demo test sheet (final scope E01–E04)

Commands to check the demo by hand (`demo/live.py`, `demo/app.py`). Run from the repo root with the venv
(`D:\cb\.venv\Scripts\python.exe`). One model per exercise (T17): `--select` loads the GroupKFold-5 fold that held the
subject out, so these subjects are **held out** of the model they are scored with (no leakage warning). Videos play at
real-time speed unless `--no-realtime`; `--headless` runs without a window. Saved `.mp4` files blur faces (D10);
`--save-mode skeleton` draws on a black frame instead. Outputs (reports, videos) go to `outputs/full/demo/`
(gitignored). Decisions: X1–X6 in `DESIGN_DECISIONS.md`.

## 1. Live demo, one held-out video per exercise

| Exercise | Command | Expected (checked 2026-10-20) |
|---|---|---|
| E01 abduction | `python -m demo.live --select E01 P03 F FL --ghost` | model fold 0; 9 reps; Module III set 84.7, model 80.2 (physiotherapists 60.5) |
| E02 adduction | `python -m demo.live --select E02 P07 F ML --ghost` | fold 2; 14 reps; 73.3 / 95.5 (47.6) |
| E03 lateral rotation | `python -m demo.live --select E03 P05 F FL --ghost` | fold 1; 10 reps; 85.5 / 61.4 (69.0) |
| E04 medial rotation | `python -m demo.live --select E04 P10 F FL --ghost` | fold 1; 10 reps; 81.1 / 87.3 (81.9) |
| E04, side view | `python -m demo.live --select E04 P10 L FL --ghost` | ghost stays off ("the L view is not the reference's oblique view"); the panel says to compare the 3D-angle differences |
| E03 non-expert clip | `python -m demo.live --select E03 P30 F FL` | fold 4; 1 rep; `too_few_reps` severe |
| Module III only | `python -m demo.live --select E01 P03 F FL --no-model` | no checkpoint needed (X3) |
| Leakage warning | `python -m demo.live --select E01 P11 F FL --fold 0` | prints `WARNING: LEAKAGE ...` (P11 trained fold 0) |

What to look at in the window:
- **Left (the video):** the skeleton (joints of the last rep's errors in red); the live angle difference at each
  channel's joint (`elev -7`, `elbow +12`, ...: patient − Module III target at the aligned template frame, **green
  inside the band, red outside**); with `--ghost` the PT reference as a faint cyan skeleton, scaled to the patient's
  torso and centred on the mid-hip (front-view videos only unless `--ghost-any-view`); top: rep counter, rep score,
  errors (`type . joint . deviation . frames (severity)`), the model's rep score and distance to the PT reference,
  the live main angle and the last rep against its band.
- **Right (side panel):** the PT reference skeleton **time-aligned to the current rep** (open-end DTW on Module III's
  alignment feature), its file and camera view, `mirrored` when the patient uses the other side, the template frame,
  and per channel `patient / target (diff)`. For a side-view video (or a non-front demo, E04 oblique) it says that
  the picture comes from another view and only the angle differences (view-corrected, M8) compare.
- Keys: `q` quit, `r` reset, `s` save the report (JSON + CSV).

## 2. Example videos (saved 2026-10-20, faces blurred)

```bash
python -m demo.live --select E01 P03 F FL --headless --no-realtime --ghost --save-mp4 outputs/full/demo/examples/E01_P03_F_FL_reference.mp4
python -m demo.live --select E02 P07 F ML --headless --no-realtime --ghost --save-mp4 outputs/full/demo/examples/E02_P07_F_ML_reference.mp4
python -m demo.live --select E03 P05 F FL --headless --no-realtime --ghost --save-mp4 outputs/full/demo/examples/E03_P05_F_FL_reference.mp4
python -m demo.live --select E04 P10 F FL --headless --no-realtime --ghost --save-mp4 outputs/full/demo/examples/E04_P10_F_FL_reference.mp4
```

## 3. Streamlit app

```bash
streamlit run demo/app.py
```

Sidebar: **dataset video** → E01 / P15 / F / FL (held out by fold 0) → **assess**. Expected: 10 reps, Module III set
score 88.0, model set score 82.8 (MQ 85.9; physiotherapists 76.67); per-rep table; for the chosen rep the patient
against the PT reference at 5 template phases (ghost in cyan, per-joint differences green / red), the angle-vs-template
curves and the embedding plot; **download JSON**. Moving the rep slider keeps the result.

## 4. Automated checks

```bash
python -m pytest -q tests/test_demo.py
```

The online session reproduces the batch own-view reps (E03, E05, E07, E09 videos); the reference player follows the
template for E01–E04; the online aligner tracks a slowed-down template and waits at rest; mirroring; the face blur
(including frames without landmarks). Agreement of the live alignment with Module III's batch DTW on held-out reps
(median |online − batch| template frame): E01 1%, E02 3%, E03 8%, E04 6% of the template.
