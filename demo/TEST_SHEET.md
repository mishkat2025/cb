# Demo test sheet (final scope E01–E04, held-out subjects)

Commands to check the demo by hand. **Open PowerShell (or cmd) in `D:\cb`** and paste the commands as they are; they
call the venv's Python directly (`.venv\Scripts\python.exe`), so nothing needs activating. The demo runs on the **CPU**
(`--device cpu` is the default), so it can run while a training job uses the GPU.

- **Held out:** one model per exercise (T17). `--select` loads the GroupKFold-5 fold whose test split holds the
  subject, so every subject below is **held out** of the model that scores it (the console prints the checkpoint and
  no `WARNING: LEAKAGE` line).
- Videos play at real-time speed in a window. Keys: `q` quit, `r` reset the set, `s` save the report (JSON + CSV to
  `cvdc-physio\outputs\full\demo\`). `--headless --no-realtime` runs without a window as fast as possible.
- Saved `.mp4` files have faces blurred (D10). Outputs are gitignored.
- Expected numbers were checked on 2026-10-20 (21:20–21:40) with the code of that day's commit. "Module III / model"
  = the set scores printed on the last console line; "PT" = the physiotherapists' score of that video. The scores are
  single examples of the within-group results (PROGRESS §6h.9: no model beats the within-group mean), not a validation.
- Decisions: X1–X8 in `cvdc-physio\DESIGN_DECISIONS.md`; the reference display is X7 (E03's rest pose: X8).

## 1. One held-out video per exercise (front view, with the ghost)

| Exercise | Command (from `D:\cb`) | Expected (console) |
|---|---|---|
| E01 abduction | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P03 F FL --ghost` | fold 0; 9 reps (`too_few_reps`, mild); Module III 84.7 / model 80.2 (PT 60.5) |
| E02 adduction | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E02 P07 F ML --ghost` | fold 2; 14 reps; 73.3 / 95.5 (PT 47.6); reps 3, 8, 11 and 13 are half reps (`adduction_range_insufficient`, severe) |
| E03 lateral rotation | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E03 P05 F FL --ghost` | fold 1; 10 reps; 85.5 / 61.4 (PT 69.0); at rest the panel shows the demo's 90/90 hold (labelled) |
| E04 medial rotation | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E04 P10 F FL --ghost` | fold 1; 10 reps; 81.1 / 87.3 (PT 81.9); the patient uses the left arm, so the panel says `mirrored` |

## 2. What to look at in the window

**At rest** (before the first rep, between reps, after the last):
- top left: `reps: N | at rest`;
- right panel: the PT's own skeleton in its **rest pose** (E01, E02: arm hanging nearly straight; E04: the start
  position, forearm forward; E03: the demo's **90/90 hold**, forearm forward, with the cyan label `PT rest pose: its
  90/90 hold (the demo rests with the arm down)`) and the line `patient at rest | PT at rest | not compared`;
- **no** angle labels on the patient, **no** `channel: patient / target` list, **no** ghost.

**During a rep:**
- top left: `in a rep`;
- panel: the PT moves through its real rep with the patient, **rest → peak → rest**; the line
  `patient in a rep | PT frame k/n, template j/m` (during the PT's rest lead-in and return: `not compared`). E03: the
  demo lowers the arm from the top instead of rotating back, so its return to 90/90 is the demo's rotation played back,
  labelled `PT return: its rotation played back (the demo lowers the arm)`; it is short (the patients rotate back fast);
- labels at the joints (`elev +3`, `elbow -8`, ...: patient − target in degrees at that template frame; **green inside
  the band, red outside**) and the same list at the bottom of the panel (`patient / target (diff)`);
- `--ghost` on a front-view video: the PT as a faint cyan skeleton **on** the patient, centred on the mid-hip and
  scaled to the patient's shoulder width (E04: torso length, because its demo is filmed obliquely; labelled
  `ghost: PT reference (oblique view)`).

The first ~10–20% of each rep and its last ~10–20% count as rest (the rep starts once the patient is 25% of the way to
the PT's peak). During a hold at the top, the PT stays at the start of its own hold and catches up on the way down.

## 3. More held-out checks

| Check | Command (from `D:\cb`) | Expected |
|---|---|---|
| E01 non-expert | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P30 F FL --ghost` | fold 4; 2 reps (`too_few_reps`, severe); Module III 67.6 / model 14.2 (PT 64.3) |
| E03 non-expert, one-rep clip | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E03 P30 F FL --ghost` | fold 4; 1 rep (`too_few_reps`, severe); 89.6 / 52.5 (PT 50.0) |
| E01 side view | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P03 L FL --ghost` | fold 0; prints `ghost off: the L view is not the reference's front view`; 9 reps; 90.0 / 60.9 (PT 59.0); panel says to compare the 3D-angle differences, not the picture |
| E02 side view | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E02 P07 R ML --ghost` | fold 2; ghost off; 16 reps; 76.8 / 82.6 (PT 47.6). Side views follow the PT poorly (X7 check) |
| E04 side view | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E04 P10 L FL --ghost` | fold 1; ghost off (`not the reference's oblique view`); 9 reps; 76.1 / 79.6 (PT 81.9) |
| Module III only | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P03 F FL --no-model` | no checkpoint needed (X3); 9 reps; Module III 84.7; no model scores |
| Leakage warning | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P11 F FL --fold 0` | prints `WARNING: LEAKAGE: subject P11 was in the training split of E01 proposed_j1 fold 0 ...` (red line in the window) |
| A video file | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --source "D:\cb\mobiphysio\E01(Abduction)\E01_P03_AF_VFL_GM.mp4" --exercise E01 --view F --ghost` | the same as the first E01 row (the subject is looked up, so the held-out fold is used) |

`--list E03` lists an exercise's dataset videos (subject, group, view, variation, score).

## 4. Example videos and screenshots (saved 2026-10-20, faces blurred)

```
.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P03 F FL --ghost --headless --no-realtime --save-mp4 cvdc-physio\outputs\full\demo\examples_x7\E01_P03_F_FL_reference.mp4
.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E02 P07 F ML --ghost --headless --no-realtime --save-mp4 cvdc-physio\outputs\full\demo\examples_x7\E02_P07_F_ML_reference.mp4
.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E03 P05 F FL --ghost --headless --no-realtime --save-mp4 cvdc-physio\outputs\full\demo\examples_x7\E03_P05_F_FL_reference.mp4
.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E04 P10 F FL --ghost --headless --no-realtime --save-mp4 cvdc-physio\outputs\full\demo\examples_x7\E04_P10_F_FL_reference.mp4
```

Each takes ~35 s. Rest | peak screenshots of these videos are in the same folder:
`E01_P03_AF_VFL_GM_rest_15.0s_peak_5.7s.png`, `E02_P07_AF_VML_GM_rest_21.8s_peak_4.0s.png`,
`E03_P05_AF_VFL_GM_rest_3.2s_peak_3.9s.png`, `E04_P10_AF_VFL_GF_rest_5.0s_peak_3.5s.png`. The earlier version (X6,
arm ~25° out at rest) is kept in `cvdc-physio\outputs\full\demo\examples\` for comparison.

## 5. Streamlit app

```
.venv\Scripts\python.exe -m streamlit run cvdc-physio\demo\app.py
```

Sidebar: **dataset video** → E01 / P15 / F / FL (held out by fold 0) → **assess**. Expected: 10 reps, Module III set
score 88.0, model set score 82.8 (MQ 85.9; PT 76.67); the per-rep table; for the chosen rep, the patient against the
PT reference at 5 template phases (the PT's own keypoints at the matching frame of its rep, in cyan; per-joint
differences green / red), the angle-vs-template curves and the embedding plot; **download JSON**. Moving the rep slider
keeps the result.

## 6. Automated checks

```
.venv\Scripts\python.exe cvdc-physio\scripts\phase6\verify_pt_keypoints.py
.venv\Scripts\python.exe cvdc-physio\scripts\phase6\check_reference_display.py
cd cvdc-physio
..\.venv\Scripts\python.exe -m pytest -q tests\test_demo.py
cd ..
```

- `verify_pt_keypoints.py` (~1 min): the PT demo keypoints re-detected on the PT videos in memory at the rest, peak and
  end of every demo rep. Expected: median error 0–1.8% of the torso in E01 (≤ 6.5% in the oblique E04), and the
  display window from rest to rest (E01 2D arm angle 10° / 178° / 11°; E03 rest pose = frame 232, shoulder 81°,
  forearm 81°, "PT rest pose: its 90/90 hold ...").
- `check_reference_display.py` (~1 min): the live loop replayed on the cached keypoints of the four videos of §1.
  Expected: one rep episode per rep (E02: 14 of 14, one half rep missed and one extra), 91–99.5% of the between-rep
  frames shown at rest.
- `pytest` must run inside `cvdc-physio` (the tests use repo-relative paths). Expected: 16 passed.
