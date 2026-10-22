# CVDC on MobiPhysio: final report, scope E01–E04

Final scope set by the supervisor (D18): **E01 abduction, E02 adduction, E03 lateral rotation, E04 medial rotation**;
E05–E09 appear only in Appendix A of [`final_E01_E04_tables.md`](final_E01_E04_tables.md). Every number below is taken
from generated tables: [`pipeline_tables.md`](pipeline_tables.md) (the pipeline's output, §4),
[`final_E01_E04_tables.md`](final_E01_E04_tables.md) and [`final_E01_E04_loso_tables.md`](final_E01_E04_loso_tables.md)
(CVDC's effect and the supplementary model comparison, GroupKFold-5 and LOSO) and
[`completion_tables.md`](completion_tables.md) (the measures added to complete the proposal). Change log: §10.

**How this report is organised (2026-10-22, X11).** The proposal describes one pipeline (its Fig. 3): 2D pose →
ST-GCN with CVDC → Module III, which compares each repetition with the physiotherapist's template → **an exercise
score and the located errors**. That score is the system's result and is reported first (§4); what CVDC contributes
inside the pipeline follows (§5). The networks' learned score head, which the proposal does not describe, is reported
as a **supplementary analysis** (§6): it trains every network on the physiotherapists' scores so that the networks
can be compared on the same task (B1, B2 and Proposed need some training target), and much of what the project
learned comes from it, but it is not the pipeline's output.

## Summary

**The pipeline's output: Module III's score, with the CVDC representation inside Module III (§4)**

1. **The score agrees weakly and positively with the physiotherapists.** Under LOSO (the final scheme), pooled over
   the E01–E04 experts, Spearman ρ 0.17 [0.07, 0.25] with the total score and 0.20 [0.10, 0.30] with the
   movement-quality score (MQ); non-experts 0.21 [0.02, 0.36]. Best strata: E03 non-experts ρ 0.39 [0.13, 0.59], E02
   experts MQ 0.39 [0.17, 0.56]. **As a predictor of the physiotherapists' score it does not beat the group mean in
   any stratum** (ΔRMSE −0.30 to +0.05, every CI includes 0).
2. **Putting CVDC inside Module III does not change the agreement.** Against the rule-based Module III (the same
   comparison without any network): pooled experts Δρ +0.01 [−0.04, 0.06] (LOSO) and −0.00 [−0.04, 0.03]
   (GroupKFold-5); the only significant change is a small loss in E03 experts under GroupKFold-5 (−0.06 [−0.11, −0.01]).
3. **Errors are located (joint, frames, severity), but they cannot be checked against labels** (the dataset has
   none). The embedding rule flags 86% of the E01 non-experts' reps against 12–16% of the experts'; within the
   experts few rules fire clearly more often in the weaker videos (< 80) than in the best (≥ 95), the clearest being
   E02's arm-height, elbow, shoulder and trunk rules (23 → 41%, 13 → 29%, 3 → 14%, 6 → 20%).
4. **The same rep filmed by three cameras gets a moderately stable score**: ICC 0.41–0.68 (LOSO), 0.26–0.47 without
   the speed part, which the synced cameras share by construction.

**What CVDC contributes inside the pipeline (§5)**

5. **CVDC's clear, measurable effect is view consistency.** Within-group VCI 0.51–0.58 right after CVDC for the
   experts, against ≤ 0.08 for a random encoder and 0.06–0.30 for B1. The distance of a rep to the PT reference (the
   proposal's "vector difference") becomes view-stable: ICC 0.55–0.78 vs 0.13–0.45 for B1 (all CIs above 0); under
   LOSO 0.63–0.78 vs 0.10–0.38 in E01, E02 and E04 (E03 0.36, n.s.).
6. **It holds on a camera never seen in training**: VCI 0.41 vs 0.15 for B1; distance ICC 0.27–0.74 vs 0.00–0.16.
7. **The embedding carries a little depth information, no more.** From the front view alone it predicts the depth
   excursion the side camera measures better than B1 (pooled ΔR² +0.10 [0.03, 0.16]), as well as KTPFormer's explicit
   3D does, but the absolute R² stays ≤ 0.04 pooled.
8. **Every encoder, even an untrained one, separates experts from non-experts** (AUC 0.87–0.92 pooled), because the
   groups differ in clip length and recording set-up; every result is therefore reported within groups.

**Supplementary analysis: the learned score head, used to compare the models (§6)**

9. **No network predicts the physiotherapists' score within a group better than the group mean** (0 of 24
   comparisons better, 13 worse, under both schemes). **Among the networks, Proposed is the most accurate for the
   experts** (LOSO pooled −1.88 [−2.94, −0.86] RMSE vs B1, lower error for 21 of 24 experts; −1.48 vs B2-KTPFormer) and
   **worse than B1 for the non-experts** (+2.27 [0.53, 4.23]). On an unseen camera B1's error rises by 4.68 points,
   Proposed's by 1.37. KTPFormer's 3D input does not beat 2D. B2-DepthAnything: §6.6.
10. **Other measures the proposal lists (§6.4, §7):** the derived quality class is weak for every network (κ ≤ 0.27);
    no recording condition makes the networks significantly worse for the experts; KTPFormer's 3D is the steadiest
    skeleton over time.

11. **Demo:** the pipeline on a video file (or an upload): the patient with the skeleton, the physiotherapist's
    reference from rest to rest with live per-joint differences, and per rep Module III's score and errors; the
    network's own score is shown separately as supplementary (X11). A four-panel export adds the estimated 3D
    skeletons and the CVDC embedding. `demo/TEST_SHEET.md`.

**In one line:** the proposal's pipeline runs end to end on E01–E04, and Module III scores each repetition against
the physiotherapist's demo inside the CVDC representation and locates its errors; its score agrees only weakly with
the physiotherapists and does not predict their score better than the group average, and adding CVDC to Module III
does not change that. CVDC's measurable effect is that the representation, and the comparison with the
demo, become independent of the camera; the supplementary comparison shows it is also the most accurate network for
the experts, but no network beats the group average either.

## 1. Data and scope

| Exercise | experts: videos / subjects / reps (mean score) | non-experts: videos / subjects / reps (mean score) |
|---|---|---|
| E01 abduction | 382 / 24 / 3,347 (80.1) | 175 / 27 / 231 (44.6) |
| E02 adduction | 384 / 24 / 3,516 (74.1) | – |
| E03 lateral rotation | 383 / 24 / 3,284 (75.4) | 167 / 27 / 178 (44.4) |
| E04 medial rotation | 383 / 24 / 3,042 (75.8) | – |

MobiPhysio: three phones (front, left, right), 3 physiotherapist judges × 7 EAAQ items per video (total 0–100; MQ =
100 × mean of PF1, PF2, CF1, CF2, CF3, D13). Experts (recorded 2024) did synchronised multi-view sets of ~10 reps;
non-experts (2022) did single-rep, mostly front-only, unsynchronised clips. Each exercise has one physiotherapist demo
video, used as the reference (Module III templates, the embedding reference and the demo).

## 2. The pipeline (proposal Fig. 3) and the comparison models

- **Phase 1 (2D pose):** MediaPipe Pose (full model, 960 px) → 30 fps grid, interpolation, smoothing, root (mid-hip)
  and torso normalisation; rep segmentation per exercise (P6); views synchronised in software (audio, creation time,
  motion).
- **Encoder: ST-GCN with CVDC (Proposed; one model per exercise, T17):** an ST-GCN on 2D keypoints, pretrained with
  the cross-view contrastive loss (CVDC: the same rep seen by two cameras is a positive pair; synced, audio-lag,
  rep-matched and synthetic-view pairs) and then fine-tuned jointly (supervised + CVDC, λ 1.0).
- **Module III, the assessment (M8–M20):** each rep is aligned with the physiotherapist demo's template **in the
  Proposed model's CVDC embedding space** (M19) and compared with it along that alignment: 3D joint angles, the main
  angle's angular velocity (M20) and the embedding distance, each against per-view targets and the bands of the
  high-scoring experts; rules locate the errors (joint, frames, severity); the parts give a rep score (0–100), and the
  set score is their mean. **This set score is the pipeline's output.** The same Module III without any network
  (rule-based: angles only, aligned on the moving segment, M8–M18) is the comparison.
- **Comparison encoders:** B1 = the same ST-GCN on 2D keypoints without CVDC; B2-KTPFormer = on KTPFormer-lifted 3D;
  B2-DepthAnything = on 2D plus DepthAnything depth at the joints (A11); the frozen Proposed = the CVDC encoder with
  only the heads trained.
- **Supplementary instrument (not part of the pipeline, X11):** a learned score head on every encoder (rep head, and
  a video head for the set score and EAAQ items), trained on the physiotherapists' scores. It gives B1 and B2 a
  training target and puts every network on the same task, so that the networks can be compared (§6). No exercise
  head (one model per exercise).

## 3. Evaluation

GroupKFold-5 per exercise, grouped by subject and balanced for expert / non-expert; CVDC pretraining inside each fold
on training subjects only. **LOSO** (one subject held out at a time, 147 folds; §6.3) is the final scheme for the
networks and for Module III; the CVDC encoder of each LOSO fold is the one of the GroupKFold-5 fold that held the
subject out (C7, leak-checked). Video level; **within groups first** (each exercise × group), pooled over E01–E04
within each group second; 95% CIs from the subject bootstrap (2,000 draws), paired for differences. Baselines:
predict-the-mean, group-only (within-group training mean), duration-only, rep-count-only, set features. Cross-view
protocol: every held-out camera × method × fold (180 runs), the models seeing only the two training cameras in every
stage (including CVDC). VCI = within-group similarity of the same rep across cameras minus similarity to other groups
(centred 256-d embeddings). Distance view stability = ICC(1) of the standardised distance to the PT reference over
synced expert performances.

## 4. Results: the pipeline's output (Module III's score)

Full tables: [`pipeline_tables.md`](pipeline_tables.md).

**Agreement of the set score with the physiotherapists (LOSO).** Spearman ρ [95% CI]; Δ = pipeline − rule-based
Module III on the same draws (bold: CI excludes 0):

| Stratum | total: pipeline | total: rule-based | total: Δ | MQ: pipeline | MQ: rule-based | MQ: Δ |
|---|---|---|---|---|---|---|
| E01 expert | 0.06 [−0.05, 0.17] | 0.06 [−0.06, 0.19] | 0.00 [−0.05, 0.05] | 0.16 [0.02, 0.29] | 0.19 [0.02, 0.34] | −0.03 [−0.09, 0.03] |
| E01 non-expert | 0.08 [−0.14, 0.29] | 0.01 [−0.23, 0.24] | 0.07 [−0.04, 0.17] | 0.08 [−0.14, 0.29] | 0.01 [−0.22, 0.24] | 0.07 [−0.04, 0.17] |
| E02 expert | 0.25 [0.06, 0.43] | 0.18 [−0.01, 0.37] | 0.07 [−0.03, 0.18] | 0.39 [0.17, 0.56] | 0.34 [0.12, 0.52] | 0.05 [−0.03, 0.14] |
| E03 expert | 0.17 [−0.00, 0.35] | 0.22 [0.06, 0.38] | −0.05 [−0.11, 0.01] | 0.11 [−0.08, 0.28] | 0.15 [−0.03, 0.32] | −0.04 [−0.10, 0.01] |
| E03 non-expert | 0.39 [0.13, 0.59] | 0.43 [0.18, 0.61] | −0.03 [−0.12, 0.07] | 0.38 [0.12, 0.58] | 0.42 [0.17, 0.60] | −0.04 [−0.13, 0.06] |
| E04 expert | 0.20 [−0.00, 0.37] | 0.18 [−0.07, 0.39] | 0.03 [−0.08, 0.12] | 0.13 [−0.07, 0.29] | 0.06 [−0.18, 0.27] | 0.07 [−0.02, 0.16] |
| experts, E01–E04 | 0.17 [0.07, 0.25] | 0.15 [0.05, 0.25] | 0.01 [−0.04, 0.06] | 0.20 [0.10, 0.30] | 0.19 [0.08, 0.30] | 0.01 [−0.03, 0.06] |
| non-experts, E01–E04 | 0.21 [0.02, 0.36] | 0.20 [0.00, 0.35] | 0.01 [−0.05, 0.07] | 0.20 [0.01, 0.35] | 0.20 [0.02, 0.35] | −0.00 [−0.06, 0.05] |

(Pooled rows: centred within exercise.) Under GroupKFold-5 the picture is the same (pooled experts 0.14 [0.05, 0.24]
vs 0.15; the only significant difference is E03 experts −0.06 [−0.11, −0.01]).

**As a predictor.** A within-group line from the set score to the total score, fitted on each fold's training
videos, has the group mean's RMSE in every stratum (LOSO: E01 experts 12.49 vs 12.46, E02 16.20 vs 16.24, E03 14.98
vs 14.95, E04 8.98 vs 8.94; E01 non-experts 8.00 vs 7.94, E03 8.88 vs 9.18; all differences' CIs include 0).

**Errors and their rates.** Every rep gets its rule-based errors with joint, frames and severity (M12), now including
`angular_velocity_deviation` and `movement_pattern_deviation`. Without error labels only their rates can be reported
(pipeline_tables.md §5): most rules fire about as often in the experts' best videos (≥ 95) as in their weaker ones
(< 80); E02's posture rules are the exception (arm height 23 → 41%, elbow 13 → 29%, shoulder 3 → 14%, trunk lean
6 → 20%). Against the non-experts the range, speed and embedding rules fire far more often (E01: range 70%, too fast
80%, movement pattern 86% of reps). The E03 range rule was recalibrated to the experts (M18, post-hoc): it no longer
flags the best videos (38% → 7% of their reps).

**View stability and recording conditions.** The same rep filmed by three cameras gets a rep-score ICC of 0.46,
0.56, 0.41 and 0.68 (E01–E04, LOSO). Under occlusion the rule-based Module III's agreement with the experts' scores
falls (ρ −0.15 [−0.33, 0.03] vs 0.25 [0.07, 0.39] in full light, completion_tables.md §1).

## 5. Results: what CVDC contributes inside the pipeline

**View consistency (GroupKFold-5) and on the held-out camera**, experts:

| Exercise | VCI after CVDC / Prop λ1 / B1 | distance ICC Prop λ1 / B1 (Δ) | held-out camera: distance ICC Prop λ1 / B1 (Δ) |
|---|---|---|---|
| E01 | 0.51 / 0.42 / 0.12 | 0.69 / 0.13 (**+0.56**) | 0.53 / 0.01 (**+0.51**) |
| E02 | 0.53 / 0.44 / 0.30 | 0.73 / 0.45 (**+0.28**) | 0.58 / 0.10 (**+0.48**) |
| E03 | 0.52 / 0.41 / 0.06 | 0.55 / 0.14 (**+0.41**) | 0.27 / 0.16 (**+0.11**) |
| E04 | 0.58 / 0.53 / 0.23 | 0.78 / 0.35 (**+0.42**) | 0.74 / 0.00 (**+0.73**) |

(Bold = paired CI excludes 0.) On the held-out camera the embedding stays view-consistent (VCI 0.41 vs 0.15 for B1).
E03 keeps the least view stability on the unseen camera (0.27).

**View consistency under LOSO** (a fold's held-out data is one subject, so the VCI's "other groups" can only be that
subject's other recordings; a stricter contrast than the GroupKFold-5 VCI; "after CVDC" and the frozen variant use the
GroupKFold-5 encoders, C7):

| Stratum | VCI: after CVDC / Prop λ1 / B1 / B2-KTP (LOSO) | VCI same-subject, GroupKFold-5: Prop λ1 / B1 | distance ICC: Prop λ1 / frozen / B1 (Δ λ1 − B1) | distance ICC, GroupKFold-5: Prop λ1 / B1 |
|---|---|---|---|---|
| E01 expert | 0.33 / 0.26 / −0.04 / −0.03 | 0.25 / −0.03 | 0.65 / 0.74 / 0.10 (**+0.55 [0.39, 0.73]**) | 0.69 / 0.13 |
| E02 expert | 0.18 / 0.16 / −0.00 / 0.02 | 0.17 / −0.01 | 0.63 / 0.71 / 0.38 (**+0.25 [0.02, 0.54]**) | 0.73 / 0.45 |
| E03 expert | 0.29 / 0.18 / −0.07 / −0.14 | 0.20 / −0.05 | 0.36 / 0.65 / 0.20 (+0.16 [−0.04, 0.38]) | 0.55 / 0.14 |
| E04 expert | 0.35 / 0.32 / 0.05 / −0.03 | 0.27 / 0.03 | 0.78 / 0.84 / 0.34 (**+0.44 [0.28, 0.62]**) | 0.78 / 0.35 |
| experts, E01–E04 | 0.28 [0.21, 0.37] / 0.23 [0.16, 0.31] / −0.02 / −0.05 | 0.22 / −0.02 | – | – |

Random-init floor: VCI −0.13 to −0.06, distance ICC −0.08 to 0.29. The non-experts have few cross-view pairs (119
same-rep pairs over E01 and E03) and no method is view-consistent there (VCI −0.17 to −0.01).

**Depth.** For the same rep the side camera measures how far the moving hand travels in depth for the front camera.
From the front view alone, the CVDC embeddings predict that depth excursion better than B1 (pooled ΔR² +0.10 [0.03,
0.16] Proposed, +0.11 [0.06, 0.17] frozen CVDC encoder), as well as KTPFormer's explicit 3D (+0.11 [0.05, 0.17]), but
the absolute R² stays small (≤ 0.04 pooled, ≤ 0.16 in E02): "depth-aware" holds only in this weak sense
(completion_tables.md §3).

**Inside Module III** the CVDC alignment and embedding distance leave the agreement with the physiotherapists
unchanged (§4). So CVDC makes the comparison with the demo camera-independent, but on these labels that does not
translate into closer agreement with the physiotherapists.

**Group signal.** The expert / non-expert groups are separable from every encoder's embeddings, including an
untrained one (AUC E01 0.95–0.98, E03 0.72–0.83, pooled 0.87–0.92); Module III's set score separates them too (E01 AUC
0.97, pipeline_tables.md §4). Pooled results would mainly measure the group, which is why every result is reported
within groups.

## 6. Supplementary analysis: the learned score head, used to compare the models

> **Not the pipeline's output.** The proposal's pipeline outputs Module III's score (§4). This section reports the
> networks' learned score head (§2), which the proposal does not describe: it trains every encoder on the
> physiotherapists' scores so that B1, B2 and Proposed are compared on the same task. It is the project's instrument
> for comparing the models, and it produced several of its findings (the label ceiling, the group confound, the
> cross-camera robustness in score terms), but it is not a clinical score.

### 6.1 GroupKFold-5

**Total score, video head, RMSE / ρ** (pooled rows: within-exercise ρ):

| Stratum (subjects) | B1 | B2-KTP | Prop λ1 | Prop frozen | group-only | set-features | noise floor |
|---|---|---|---|---|---|---|---|
| E01 expert (24) | 13.29 / 0.08 | 14.67 / −0.02 | 13.46 / 0.12 | 14.72 / −0.03 | 12.45 | 12.22 / 0.18 | 5.28 |
| E01 non-expert (26) | 11.44 / 0.11 | 15.68 / −0.18 | 15.48 / −0.18 | 14.15 / −0.01 | 7.78 | 7.48 / 0.26 | – |
| E02 expert (24) | 18.89 / 0.13 | 17.52 / 0.16 | 16.78 / 0.27 | 15.77 / 0.21 | 16.01 | 16.96 / −0.03 | 4.38 |
| E03 expert (24) | 20.08 / 0.03 | 20.88 / 0.03 | 16.52 / 0.27 | 15.76 / 0.24 | 14.71 | 15.26 / 0.05 | 3.38 |
| E03 non-expert (25) | 16.53 / −0.12 | 14.77 / −0.09 | 17.92 / −0.07 | 16.22 / −0.10 | 9.32 | 9.23 / 0.07 | – |
| E04 expert (24) | 9.08 / 0.37 | 8.83 / 0.33 | 8.06 / 0.52 | 8.76 / 0.29 | 8.94 | 8.90 / −0.02 | 2.45 |
| experts, E01–E04 (24) | 15.97 / 0.11 | 16.10 / 0.10 | 14.16 / 0.28 | 14.06 / 0.19 | 13.31 | 13.69 / 0.12 | – |
| non-experts, E01–E04 (31) | 14.17 / 0.01 | 15.24 / −0.12 | 16.73 / −0.10 | 15.21 / 0.01 | 8.57 | 8.39 / 0.26 | – |

Against the group mean (total score, paired RMSE difference): none of the 24 network × stratum comparisons is
significantly better; all 8 in the non-expert strata and 5 of the 16 in the expert strata are significantly worse.
The closest is E04 Proposed (−0.88 [−2.08, 0.39]). MQ behaves the same way (§1 of the tables). Pooled over the
experts, Proposed is +0.85 [−0.36, 2.23] against the group mean and B1 +2.67 [0.78, 4.52]; Proposed is −1.82 [−2.78,
−0.82] vs B1 (lower in E02, E03 and E04 individually) and ranks videos best within exercises (centred ρ 0.28 vs 0.11);
for the non-experts it is worse than B1 (+2.55 [0.82, 4.42]).

### 6.2 On a camera never seen in training (score)

Train on two cameras, test on the third: B1 loses accuracy on the camera it never saw (experts: total-score RMSE
+4.68 [3.36, 6.08] over the seen cameras, MQ +1.72), Proposed much less (+1.37 [0.39, 2.39]; MQ +0.26 [−0.03, 0.68]);
the difference of the two gaps is −3.32 [−4.20, −2.37] (MQ −1.46). On the held-out camera Proposed is 5.4 points more
accurate than B1, but still worse than the group mean (+2.64 [1.15, 4.12]). Per exercise (gap = held-out − seen RMSE;
bold = CI excludes 0): E01 +0.79 / **+8.14**, E02 +0.73 / **+5.48**, E03 **+2.81** / **+3.34**, E04 **+0.84** / +0.85
(Proposed / B1). In E04 both gaps are small (narrow expert scores, SD 8.7).

### 6.3 LOSO per exercise

**What was run.** Leave-one-subject-out within each exercise, one model per exercise: every subject with ≥ 1 rep is
the test set once (E01 50 folds: 24 experts / 26 non-experts; E02 24; E03 49: 24 / 25; E04 24 = 147 folds), for B1,
B2-KTPFormer, Proposed (λ 1.0) and the frozen Proposed (588 runs, 19.3 h on one GPU; estimated 19.0 h from a smoke
test). Every setting is the GroupKFold-5 runs' (early stopping on 20% of each group's training subjects, video head,
equal video weights, targets standardised per fold, no exercise head, E03 P02 in score regression); only the split
changes. **CVDC shortcut (C7):** LOSO runs no CVDC pretraining; both Proposed variants of subject s start from the
CVDC encoder of the GroupKFold-5 fold that held s out. Checked for all 147 subjects before training and in all 294
Proposed runs: s was in that encoder's test fold and never among the subjects it trained or early-stopped on, and
every subject the encoder saw is a training subject of the LOSO fold (no leakage). Same metrics, baselines (refitted
on each LOSO fold's training videos) and subject bootstrap as §3, plus per-subject paired tests (each subject's RMSE
over its videos; Wilcoxon signed-rank, Holm over the six within-exercise strata; rank-biserial r). Full tables:
[`final_E01_E04_loso_tables.md`](final_E01_E04_loso_tables.md) (decision E30).

1. **Still no network predicts the score within a group better than the within-group mean.** Total score: 0 of the
   24 network × stratum comparisons are significantly better than group-only, 13 are significantly worse (the same
   counts as GroupKFold-5); MQ: 0 better, 5 worse (GroupKFold-5: 0 and 7). Pooled over the E01–E04 experts, Proposed
   is +0.84 [−0.53, 2.23] RMSE against the group mean (GroupKFold-5 +0.85), B1 +2.72 [0.86, 4.55] (+2.67). The
   closest is the frozen Proposed in E02 (−1.34 [−3.58, 0.54]).
2. **Proposed is still the most accurate network for the experts**: pooled −1.88 [−2.94, −0.86] RMSE vs B1 and
   −1.48 [−2.32, −0.77] vs B2-KTPFormer; per subject it has the lower error for 21 of 24 experts against each
   (Wilcoxon p < 0.001; r −0.80 / −0.87); it ranks videos best within exercises (centred ρ 0.28 vs 0.13 for B1). The
   advantage is clear within E03 (−3.79 vs B1, −2.81 vs B2-KTPFormer; Holm p < 0.001 / 0.003) and not significant
   within E01, E02 and E04 on their own. For MQ it is now also significant pooled (−0.66 [−1.50, −0.11] vs B1;
   GroupKFold-5 −0.29, n.s.). **For the non-experts Proposed is worse than B1** (+2.27 [0.53, 4.23]; higher error for
   23 of 31 subjects, p 0.018).
3. **LOSO vs GroupKFold-5: the conclusions do not change.** On the same videos and bootstrap draws no network's pooled
   RMSE differs between the schemes (Δ −0.70 to +0.20, all CIs include 0); 3 of the 24 per-stratum network RMSEs differ
   significantly (B1 E01 experts +1.01, frozen Proposed E03 experts +1.37, B2-KTPFormer E03 non-experts +2.10). The
   group-only baseline is slightly worse under LOSO (+0.14 [0.05, 0.25] experts pooled), as expected when the
   training mean always excludes the held-out subject.

**Total score, video head: RMSE / ρ (LOSO)**

| Stratum (subjects) | B1 | B2-KTP | Prop λ1 | Prop frozen | group-only | set-features | noise floor |
|---|---|---|---|---|---|---|---|
| E01 expert (24) | 14.29 / −0.01 | 14.38 / 0.05 | 13.14 / 0.19 | 14.65 / 0.03 | 12.46 | 12.36 / 0.13 | 5.28 |
| E01 non-expert (26) | 12.34 / −0.22 | 13.45 / −0.00 | 15.41 / −0.33 | 13.83 / −0.02 | 7.94 | 7.61 / 0.22 | – |
| E02 expert (24) | 17.87 / 0.21 | 17.01 / 0.19 | 16.28 / 0.31 | 14.90 / 0.28 | 16.24 | 16.94 / −0.08 | 4.38 |
| E03 expert (24) | 21.13 / −0.06 | 20.15 / −0.06 | 17.34 / 0.12 | 17.13 / 0.09 | 14.95 | 16.08 / −0.19 | 3.38 |
| E03 non-expert (25) | 15.07 / −0.19 | 16.87 / −0.36 | 16.64 / −0.03 | 15.76 / −0.16 | 9.18 | 9.09 / 0.15 | – |
| E04 expert (24) | 8.64 / 0.41 | 9.56 / 0.18 | 8.79 / 0.43 | 8.43 / 0.39 | 8.94 | 9.02 / −0.05 | 2.45 |
| experts, E01–E04 (24) | 16.17 / 0.13 | 15.77 / 0.11 | 14.29 / 0.28 | 14.16 / 0.21 | 13.45 | 13.97 / 0.07 | – |
| non-experts, E01–E04 (31) | 13.76 / −0.15 | 15.23 / −0.15 | 16.02 / −0.13 | 14.81 / −0.02 | 8.58 | 8.37 / 0.31 | – |

(Pooled rows: within-exercise ρ.) MQ, pooled RMSE: experts B1 12.09, B2-KTP 11.74, Prop λ1 11.43, frozen 11.37,
group-only 11.27, set-features 11.69; non-experts 13.26, 13.48, 13.32, 12.86, 11.77, 11.47.

**Proposed λ 1.0 vs B1 and vs B2-KTPFormer** (total score; paired bootstrap RMSE difference [95% CI]; subjects with
lower / higher error under Proposed; Wilcoxon p, Holm-adjusted within exercises; rank-biserial r):

| Stratum | vs B1: Δ RMSE | subjects | p | r | vs B2-KTP: Δ RMSE | subjects | p | r |
|---|---|---|---|---|---|---|---|---|
| E01 expert | −1.15 [−2.80, 0.23] | 15 / 9 | 0.53 | −0.32 | −1.24 [−3.57, 0.66] | 13 / 11 | 1.00 | −0.14 |
| E01 non-expert | **3.06 [0.26, 5.77]** | 8 / 18 | 0.25 | 0.44 | 1.95 [−0.02, 3.93] | 8 / 18 | 0.63 | 0.32 |
| E02 expert | −1.59 [−3.71, 0.47] | 16 / 8 | 0.25 | −0.44 | −0.73 [−2.44, 0.63] | 15 / 9 | 0.48 | −0.39 |
| E03 expert | **−3.79 [−6.70, −1.50]** | 21 / 3 | **< 0.001** | −0.82 | **−2.81 [−4.79, −1.39]** | 19 / 5 | **0.003** | −0.77 |
| E03 non-expert | **1.56 [0.13, 3.08]** | 9 / 16 | 0.53 | 0.27 | −0.24 [−2.98, 3.17] | 16 / 9 | 1.00 | −0.22 |
| E04 expert | 0.15 [−0.75, 1.06] | 12 / 12 | 0.83 | 0.05 | −0.77 [−2.06, 0.32] | 12 / 12 | 1.00 | −0.23 |
| experts, E01–E04 | **−1.88 [−2.94, −0.86]** | 21 / 3 | **< 0.001** | −0.80 | **−1.48 [−2.32, −0.77]** | 21 / 3 | **< 0.001** | −0.87 |
| non-experts, E01–E04 | **2.27 [0.53, 4.23]** | 8 / 23 | **0.018** | 0.48 | 0.79 [−0.95, 2.78] | 13 / 18 | 0.17 | 0.29 |

(Pooled rows: unadjusted p.) The frozen Proposed behaves alike for the experts (pooled −2.01 [−3.15, −0.92] vs B1,
17 / 7 subjects, p 0.001; −1.61 vs B2-KTPFormer) and is not worse than B1 for the non-experts (+1.06 [−1.80, 3.49]).

**LOSO vs GroupKFold-5 at a glance**

| | GroupKFold-5 | LOSO |
|---|---|---|
| network × stratum comparisons better / worse than group-only (total score; MQ) | 0 / 13; 0 / 7 | 0 / 13; 0 / 5 |
| experts pooled: Prop λ1 − group-only | +0.85 [−0.36, 2.23] | +0.84 [−0.53, 2.23] |
| experts pooled: Prop λ1 − B1 (total; MQ) | **−1.82 [−2.78, −0.82]**; −0.29 [−0.81, 0.05] | **−1.88 [−2.94, −0.86]**; **−0.66 [−1.50, −0.11]** |
| experts pooled: Prop λ1 − B2-KTP | **−1.95 [−2.88, −1.07]** | **−1.48 [−2.32, −0.77]** |
| experts: subjects with lower error, Prop λ1 vs B1 (Wilcoxon p) | 19 / 24 (< 0.001) | 21 / 24 (< 0.001) |
| within-exercise ranking, experts (centred ρ): Prop λ1 / B1 | 0.28 / 0.11 | 0.28 / 0.13 |
| per exercise, Prop λ1 better than B1 (CI < 0) | E02, E03, E04 | E03 |
| non-experts pooled: Prop λ1 − B1 (total; MQ) | **+2.55 [0.82, 4.42]**; **+1.11 [0.01, 2.43]** | **+2.27 [0.53, 4.23]**; +0.06 [−0.66, 0.91] |
| VCI experts pooled, same-subject contrast: Prop λ1 / B1 | 0.22 / −0.02 | 0.23 / −0.02 |
| distance ICC, Prop λ1 > B1 (CI > 0) | E01, E02, E03, E04 | E01, E02, E04 (E03 n.s.) |

### 6.4 The derived quality class (precision, recall, F1, accuracy)

On the D6 classes cut from the score (≥ 80 correct, 50–79 partially correct, < 50 incorrect; not a dataset label),
every network is weak: pooled over the E01–E04 experts under LOSO, Proposed's quality head reaches accuracy 0.60,
macro F1 0.37 [0.34, 0.39], κ 0.14 (B1 0.57 / 0.35 / 0.09; the binned group mean 0.54 / 0.30 / −0.03); for the
non-experts every network's κ is ≤ 0, while the binned set-features baseline reaches 0.61 / 0.46 / 0.08
(completion_tables.md §1).

### 6.5 Recording conditions and rep-to-rep stability

For the experts, no condition (medium or low light, low or high jitter, occlusion) makes any network significantly
less accurate than full light (all ΔRMSE CIs include 0; Proposed under high jitter +2.23 [−0.18, 4.96]). Within one
set, Proposed's predicted rep score varies more from rep to rep than B1's (median SD 2.45 vs 1.40 points; Module
III's rep score 3.02).

### 6.6 B2-DepthAnything

*Pending: filled in when `scripts/full/completion/depth_chain.py` has finished (PROGRESS §0): baseline 2 with
DepthAnything depth at the joints (A11), GroupKFold-5 and LOSO, next to B1, B2-KTPFormer and Proposed.*

## 7. Temporal stability (proposal: "jitter variance, frame-to-frame smoothness")

KTPFormer's lifted 3D is the steadiest skeleton (rest jitter 0.003 torso lengths in the front view, vs 0.005 for the
networks' smoothed 2D input, 0.007 raw 2D and 0.010 raw MediaPipe 3D); high jitter and occlusion raise the raw 2D
jitter about 8× (0.006 → 0.044 / 0.046). Pose detection drops only in the left view under low light (0.88) and high
jitter (0.84) (completion_tables.md §1–2).

## 8. Limitations

- **Four shoulder exercises, one body region**, two of them (E02, E04) with experts only, so the expert / non-expert
  contrast exists only in E01 and E03, and there it is confounded with clip length, views, recording year, location
  and phones (E9–E12).
- **Small samples**: 24 experts per exercise (the same 24 people), 25–26 non-experts at rep level in E01 / E03, with
  the non-experts mostly single-view and unsynchronised (few cross-view pairs for CVDC).
- **Skewed, narrow expert scores**, dominated by the repetition items (PF3 / PF4), with an inter-view label-noise floor
  of 2.5–5.3 points; within-group score prediction is hard for any method.
- **3 views, synchronisation reconstructed in software**; **no error-type labels**, so Module III's errors are judged
  only by their rates and by the agreement of the score; **3D angles from a single camera** are view-dependent (Module
  III's measures and the demo's 3D panel are estimates).
- **PT references**: one demo per exercise; E03's V3 demo exceeds most experts' end range (hence M18) and lowers the
  arm between reps; E04's demo is filmed obliquely and brings the arm down to the side at its peak.
- **The quality classes** are cut from the score, so the classification metrics repeat the score results in another
  form. **The depth check's reference** is the side camera's 2D view (noisy, one depth direction).
- **Post-hoc choices** are labelled as such (M14 and M18 calibrations, X7–X9 demo display). Under LOSO the CVDC
  encoder is the GroupKFold-5 fold's (C7); the cross-view protocol and the group probe are GroupKFold-5 only.
- **Not done, because they need people:** a usability study and a physiotherapist review of the feedback; the
  live-camera test is replaced by recorded videos uploaded to the Streamlit app.

## 9. Reproduction

From `D:\cb\cvdc-physio` with `..\.venv\Scripts\python.exe` (one GPU job at a time):

```
# models (GroupKFold-5) and their analyses
python -m cvdc.train.run_per_exercise --tag perex_gkf5                                   # 6.7 h, GPU
python -m cvdc.eval.cross_view --config configs/per_exercise.yaml --exercises E01 E02 E03 E04   # ~6.5 h, GPU
python scripts/full/phase3/embedding_distance_perex.py
python scripts/full/phase3/crossview_distance_perex.py
python scripts/full/phase3/report_crossview_perex.py
python scripts/full/phase3/group_probe.py --exercises E01 E02 E03 E04 --out outputs/full/results/final_E01_E04/step1
python scripts/full/phase3/report_perex.py --exercises E01 E02 E03 E04 --out outputs/full/results/final_E01_E04/step1
# rule-based Module III (comparison) and the published tables
python -m cvdc.assess.run --config configs/per_exercise.yaml
python scripts/full/phase4/report_module3.py
python scripts/full/phase4/rule_calibration.py
python scripts/full/final/report_final.py
# LOSO (detached, resumable), then its report
explorer.exe D:\cb\cvdc-physio\scripts\full\loso\launch.cmd                             # 19.3 h, GPU
python scripts/full/phase3/embedding_distance_perex.py --config configs/per_exercise_loso.yaml --tag perex_loso --exercises E01 E02 E03 E04 --loso
python scripts/full/loso/report_loso.py
# the pipeline's output: Module III with the CVDC representation (M19, M20)
python scripts/full/completion/encode.py --tag perex_gkf5 --z                           # 3 min, GPU
python scripts/full/completion/encode.py --tag perex_loso --loso                        # 5 min, GPU
python -m cvdc.assess.run --config configs/completion.yaml --exercises E01 E02 E03 E04
python -m cvdc.assess.run --config configs/per_exercise.yaml --exercises E01 E02 E03 E04 --loso --out outputs/full/results/completion/module3_loso_original --no-json
python -m cvdc.assess.run --config configs/completion.yaml --exercises E01 E02 E03 E04 --loso --join perex_loso --out outputs/full/results/completion/module3_loso --no-json
python scripts/full/phase4/report_module3.py --config configs/completion.yaml --dir outputs/full/results/completion/<run> [--loso]
python scripts/full/completion/module3_comparison.py [--loso]
python scripts/full/final/report_pipeline.py                                             # reports/pipeline_tables.md
# measures added for the proposal, DepthAnything, and their tables
python scripts/full/completion/classification_conditions.py
python scripts/full/completion/temporal_stability.py
python scripts/full/completion/depth_check.py
explorer.exe D:\cb\cvdc-physio\scripts\full\completion\run_depth_chain.cmd              # ~8 h, GPU
python scripts/full/completion/depth_baseline_report.py
python scripts/full/completion/report_completion.py                                     # reports/completion_tables.md
```

Design decisions: `DESIGN_DECISIONS.md` (D18 final scope, T17 one model per exercise, E24 cross-view, E30 LOSO, M18
calibration, M19 CVDC in Module III, M20 angular velocity, A11 DepthAnything, E31–E32 added measures, X7–X11 demo).
Progress log: `PROGRESS.md` §0, §6h–§6j.

## 10. Change log

- **2026-10-21:** final scope E01–E04, GroupKFold-5 results, cross-view protocol, rule-based Module III, demo.
- **2026-10-22:** LOSO per exercise for the networks (now §6.3).
- **2026-10-22:** the proposal's remaining items (D19, A11, M19, M20, E31, E32): classification metrics, recording
  conditions, temporal stability, the depth check, Module III with the CVDC representation and the angular velocity
  (GroupKFold-5 and LOSO), DepthAnything (running); file-structure cleanup.
- **2026-10-22 (X11):** the report and the demo reorganised around the proposal's single pipeline: Module III's
  score is the result (§4), the learned score head a labelled supplementary analysis (§6). No number changed; the
  earlier sections were moved, not rewritten.
