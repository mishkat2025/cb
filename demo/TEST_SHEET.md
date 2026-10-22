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
- Expected numbers were re-checked on 2026-10-22 (22:55) after X11. "Assessment / suppl. model" = the two set scores
  on the last console lines: the **assessment** (Module III's score: with the Proposed model loaded, the completed
  Module III that aligns and compares in the CVDC embedding space; the pipeline's output) and the network's own
  **supplementary** score (unchanged since 2026-10-20); "PT" = the physiotherapists' score of that video. The scores
  are single examples of the within-group results (no score beats the within-group mean), not a validation.
- Decisions: X1–X11 in `cvdc-physio\DESIGN_DECISIONS.md`; the reference display is X7 (E03's rest pose: X8), the
  four-panel export X9, the assessment / supplementary split X11.

## 1. One held-out video per exercise (front view, with the ghost)

| Exercise | Command (from `D:\cb`) | Expected (console) |
|---|---|---|
| E01 abduction | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P03 F FL --ghost` | fold 0; 9 reps (`too_few_reps`, mild); assessment 87.7 / suppl. model 80.2 (PT 60.5) |
| E02 adduction | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E02 P07 F ML --ghost` | fold 2; 14 reps; 78.8 / 95.5 (PT 47.6); reps 3, 8, 11 and 13 are half reps (`adduction_range_insufficient`, severe) |
| E03 lateral rotation | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E03 P05 F FL --ghost` | fold 1; 10 reps; 86.6 / 61.4 (PT 69.0); at rest the panel shows the demo's 90/90 hold (labelled) |
| E04 medial rotation | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E04 P10 F FL --ghost` | fold 1; 10 reps; 84.1 / 87.3 (PT 81.9); the patient uses the left arm, so the panel says `mirrored` |

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
| Module III only | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P03 F FL --no-model` | no checkpoint needed (X3); 9 reps; rule-based Module III 84.7; no supplementary score |
| Leakage warning | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P11 F FL --fold 0` | prints `WARNING: LEAKAGE: subject P11 was in the training split of E01 proposed_j1 fold 0 ...` (red line in the window) |
| A video file | `.venv\Scripts\python.exe cvdc-physio\demo\live.py --source "D:\cb\dataset\videos\E01(Abduction)\E01_P03_AF_VFL_GM.mp4" --exercise E01 --view F --ghost` | the same as the first E01 row (the subject is looked up, so the held-out fold is used) |

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

## 4b. Four-panel export videos (X9; offline, CPU)

```
.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E01 P03 F FL --ghost --export4 cvdc-physio\outputs\full\demo\examples_x7\E01_P03_F_FL_4panel.mp4
.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E02 P07 F ML --ghost --export4 cvdc-physio\outputs\full\demo\examples_x7\E02_P07_F_ML_4panel.mp4
.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E03 P05 F FL --ghost --export4 cvdc-physio\outputs\full\demo\examples_x7\E03_P05_F_FL_4panel.mp4
.venv\Scripts\python.exe cvdc-physio\demo\live.py --select E04 P10 F FL --ghost --export4 cvdc-physio\outputs\full\demo\examples_x7\E04_P10_F_FL_4panel.mp4
```

`--export4` renders without a window, as fast as the CPU allows (~45 s per video; the first run per exercise and fold
also encodes that fold's training reps for the PCA, cached in `cvdc-physio\outputs\full\demo\pca\`), and writes the
video at the source's frame rate, so it plays back at normal speed. Same console output and scores as §1. Layout:
**top-left** the patient video with the skeleton (error joints red, face blurred); **top-right** the PT reference panel
(as in the live window); **bottom-left** "estimated 3D (single camera)": the patient (green) and the PT (cyan, at the
frame the reference panel shows) from MediaPipe's world landmarks, centred on the mid-hip, torso-scaled, with a slowly
turning camera (12°/s); **bottom-right** the Proposed model's embedding of every completed rep (yellow trail) and the
PT reference (cyan star) in the PCA plane of that fold's training reps (dots coloured by score), the distance gauge
(centred cosine to the PT reference) and the rep scores, with a per-rep table. The live window stays two-panel.
Screenshots: `…_4panel_rep4_peak_<t>s.png` in the same folder.

## 5. Streamlit app

```
.venv\Scripts\python.exe -m streamlit run cvdc-physio\demo\app.py
```

**From the dataset** → exercise `E01 · Abduction`, filters **Front** and **Full light** → tick the **P03** row
→ **Assess this video** (X10). Expected (2026-10-21, ~1 min with the four-panel video on): 9 reps (`Too few
repetitions`, mild), assessment score 87.7 (Correct; Module III with the CVDC model, fold 0), PT 60.5
(Partly correct), no rep problems; the four-panel video plays in the page and is saved as
`cvdc-physio\outputs\full\demo\app\E01_P03_AF_VFL_GM__proposed_j1__ghost_4panel.mp4` (H.264, 30 s; a second Assess
reuses it); below: the per-rep table, one rep in detail (problems as sentences, measurements, score breakdown that adds
up to the rep score, angle differences, the network's own outputs marked supplementary), the patient-vs-PT and angle
plots, the **Supplementary** section at the end (the network's set score 80.2, its answers and the embedding map; X11),
the JSON download and the glossary. With **No network** in the sidebar the assessment is the rule-based 84.7.

Problems check (video toggle off, ~30 s): `E02 · Aduction`, **Front**, **Medium light**, **P07** → 14 reps,
73.3 / model 95.5 (PT 47.6); reps 3, 8, 11 and 13 show `Movement too small (arm movement across the body)` (severe) with
`Arm height off` and `Too fast`.

## 6. Automated checks

```
.venv\Scripts\python.exe cvdc-physio\scripts\full\phase6\verify_pt_keypoints.py
.venv\Scripts\python.exe cvdc-physio\scripts\full\phase6\check_reference_display.py
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
