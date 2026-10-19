"""Wrong-exercise check from the movement itself (user answer 5, 2026-10-20; D8, D14): do the E05 non-experts whose
mean PF1 is below 0.5 perform a circumduction?

PF1 = 0 alone is no reason to exclude a subject; only a movement that shows a wrong exercise is (as E03 P02).

Revision before the first run on the flagged subjects: the first design measured the circle of the 3D wrist path in
every view. On the Step 0 sample, a high-scoring expert's circumduction (E05 P01) gives 8.8-9.0 turns in the left
view but ~0 turns in the front view: the arm circles in the sagittal plane and MediaPipe's depth from the front view
flattens the circle into a line. The non-experts are mostly front-view only, so two lines of evidence are used:

A. Posture match (every front-view clip). Posture per frame = the moving arm's upper-arm and forearm directions in
   the trunk frame (two unit vectors: outward, up, forward; x 57.3, so distances read roughly as degrees; 3D world
   landmarks, 10 frames/s). For each candidate exercise E01-E05 (the upper-limb exercises), a pool of front-view
   frames from the subjects whose mean PF1 in that exercise is >= 0.5 (E03 P02 left out); a clip's distance to a
   pool = the mean over its frames of the nearest pool frame, with the clip's own subject removed from every pool
   (leave-subject-out). The clip is assigned to the nearest exercise. Calibration: the same assignment for every
   front-view non-expert clip of E01, E03 and E05 whose subject has mean PF1 >= 0.5 (the share assigned to their own
   exercise = the method's accuracy).
B. Circle in the side views (L / R clips, where a subject has them): turns of the 3D wrist path around its centre in
   its best-fit plane, |sum of d theta| / 360 deg; a circumduction rep gives ~1 turn.

Verdict per flagged subject (rule fixed before running on them):
  * 'wrong exercise' if A assigns the majority of its front clips to another exercise and the E05 reference clips
    are assigned to E05 in >= 70% of cases, or if it has side-view clips and the majority turn < 0.5 while >= 80% of
    the reference side-view clips (non-experts with mean PF1 >= 0.5, experts) turn >= 0.5;
  * 'circumduction' if A assigns the majority of its front clips to E05 and B (if available) does not contradict;
  * otherwise 'inconclusive'.
Writes <outputs.phase1>/pf1_motion_check_<EX>{,_subjects,_calibration}.csv and <outputs.checks>/pf1_<EX>_wrist_paths.png
(keypoint trajectories in the image plane only, no frames).

usage: python scripts/phase1_full/pf1_motion_check.py [--config configs/data_full.yaml] [--exercise E05]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvdc.assess.angles import trunk_frame  # noqa: E402
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.skeleton import J  # noqa: E402

CANDIDATES = ["E01", "E02", "E03", "E04", "E05"]
CALIBRATE = ["E01", "E03", "E05"]               # exercises with non-expert front-view clips
SUB = 3                                          # 30 -> 10 frames/s
TURNS_MIN, MAJORITY, A_MIN_ACC, B_MIN_REF = 0.5, 0.5, 0.70, 0.80


def in_trunk(w, v, side):
    lat, up, fwd = trunk_frame(w)
    out = lat if side == "right" else -lat
    return np.stack([np.sum(v * out, -1), np.sum(v * up, -1), np.sum(v * fwd, -1)], -1)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load(pre_path):
    with np.load(pre_path) as z:
        return z["world"].astype(float), z["xy"].astype(float)


def moving_side(w):
    rng = {}
    for s in ("left", "right"):
        r = w[:, J[f"{s}_wrist"]] - w[:, J[f"{s}_shoulder"]]
        rng[s] = float(np.nanmax(np.nanpercentile(r, 95, 0) - np.nanpercentile(r, 5, 0)))
    return max(rng, key=rng.get)


def posture(w, side):
    sh, el, wr = (w[:, J[f"{side}_{j}"]] for j in ("shoulder", "elbow", "wrist"))
    f = np.c_[unit(in_trunk(w, el - sh, side)), unit(in_trunk(w, wr - el, side))] * 57.3
    f = f[::SUB]
    return f[np.isfinite(f).all(1)]


def turns(w, side):
    r = in_trunk(w, w[:, J[f"{side}_wrist"]] - w[:, J[f"{side}_shoulder"]], side)
    r = r[np.isfinite(r).all(1)]
    if len(r) < 10:
        return np.nan
    z = r - r.mean(0)
    p = z @ np.linalg.svd(z, full_matrices=False)[2][:2].T
    th = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
    return float(abs(th[-1] - th[0]) / (2 * np.pi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_full.yaml")
    ap.add_argument("--exercise", default="E05")
    a = ap.parse_args()
    cfg = load_config(a.config)
    ex = a.exercise
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    idx = idx[idx.exercise.isin(CANDIDATES) & idx.pre_path.map(lambda p: resolve(p).exists())].copy()
    idx["subject_PF1"] = idx.groupby(["exercise", "subject"]).PF1.transform("mean")
    excl = {(e, s) for e, subs in (cfg["data"].get("exercise_label_exclude") or {}).items() for s in subs}
    flagged = sorted(idx[(idx.exercise == ex) & (idx.group == "non-expert") & (idx.subject_PF1 < 0.5)].subject.unique())

    feats, meta = {}, {}
    for r in idx.itertuples():
        w, xy = load(resolve(r.pre_path))
        side = moving_side(w)
        meta[r.video_id] = dict(side=side, turns=turns(w, side), xy=xy if r.exercise == ex else None)
        if r.view == "F":
            feats[r.video_id] = posture(w, side)
    pool_rows = idx[(idx.view == "F") & (idx.subject_PF1 >= 0.5)
                    & ~pd.Series([(e, s) in excl for e, s in zip(idx.exercise, idx.subject)], index=idx.index)]

    def assign(r):
        d = {}
        for e in CANDIDATES:
            pr = pool_rows[(pool_rows.exercise == e) & (pool_rows.subject != r.subject)]
            if not len(pr):
                continue
            B = np.vstack([feats[v] for v in pr.video_id if len(feats[v])])
            A = feats[r.video_id]
            d[e] = float(cKDTree(B).query(A)[0].mean()) if len(A) else np.nan
        best = min(d, key=lambda k: d[k] if np.isfinite(d[k]) else np.inf)
        return best, d

    rows = []
    test = idx[(idx.view == "F") & (idx.group == "non-expert") & idx.exercise.isin(CALIBRATE)]
    for r in test.itertuples():
        best, d = assign(r)
        rows.append(dict(video_id=r.video_id, exercise=r.exercise, subject=r.subject, subject_PF1=r.subject_PF1,
                         assigned=best, **{f"d_{e}": v for e, v in d.items()}))
    C = pd.DataFrame(rows)
    out = resolve(cfg["outputs"]["phase1"])
    C.to_csv(out / f"pf1_motion_check_{ex}_calibration.csv", index=False)
    ref = C[C.subject_PF1 >= 0.5]
    conf = pd.crosstab(ref.exercise, ref.assigned, normalize="index").round(2)
    acc = ref.groupby("exercise").apply(lambda g: (g.assigned == g.name).mean(), include_groups=False)

    sides = idx[(idx.exercise == ex) & idx.view.isin(["L", "R"])].copy()
    sides["turns"] = [meta[v]["turns"] for v in sides.video_id]
    side_ref = sides[sides.subject_PF1 >= 0.5]
    side_ref_ok = float((side_ref.turns >= TURNS_MIN).mean())

    D = idx[(idx.exercise == ex) & idx.subject.isin(flagged)].copy()
    D["turns"] = [meta[v]["turns"] for v in D.video_id]
    D = D.merge(C[["video_id", "assigned"] + [c for c in C if c.startswith("d_")]], on="video_id", how="left")
    D[["video_id", "subject", "view", "variation", "score", "PF1", "turns", "assigned"]
      + [c for c in D if c.startswith("d_")]].to_csv(out / f"pf1_motion_check_{ex}.csv", index=False)
    srows = []
    for s, g in D.groupby("subject"):
        f, sv = g[g.view == "F"], g[g.view.isin(["L", "R"])]
        share_other = float((f.assigned != ex).mean()) if len(f) else np.nan
        share_noturn = float((sv.turns < TURNS_MIN).mean()) if len(sv) else np.nan
        wrong = ((len(f) and share_other > MAJORITY and acc.get(ex, 0) >= A_MIN_ACC)
                 or (len(sv) and share_noturn > MAJORITY and side_ref_ok >= B_MIN_REF))
        circ = len(f) and (1 - share_other) > MAJORITY and not (len(sv) and share_noturn > MAJORITY)
        srows.append(dict(subject=s, subject_PF1=g.subject_PF1.iloc[0], videos=len(g), front=len(f), side=len(sv),
                          score_mean=g.score.mean(), front_assigned=f.assigned.value_counts().to_dict(),
                          share_front_other=share_other, side_turns=sv.turns.round(2).tolist(),
                          share_side_no_turn=share_noturn,
                          verdict="wrong exercise" if wrong else "circumduction" if circ else "inconclusive"))
    S = pd.DataFrame(srows)
    S.to_csv(out / f"pf1_motion_check_{ex}_subjects.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(f"{ex}: flagged non-experts (mean PF1 < 0.5): {flagged}")
    print(f"\nA. calibration, front-view non-expert clips of subjects with mean PF1 >= 0.5 ({len(ref)} clips), "
          "share assigned to each exercise (rows = true exercise):")
    print(conf.to_string())
    print("own-exercise accuracy:", acc.round(2).to_dict(), f"(rule needs {ex} >= {A_MIN_ACC})")
    print(f"\nB. side-view clips of {ex} with mean PF1 >= 0.5: {len(side_ref)}; share with >= {TURNS_MIN} turns "
          f"{side_ref_ok:.2f} (rule needs >= {B_MIN_REF}); turns p5/p50/p95 by group:")
    print(side_ref.groupby("group").turns.describe(percentiles=[.05, .5, .95])[["count", "5%", "50%", "95%"]].round(2)
          .to_string())
    print("\nflagged subjects:")
    print(S.round(2).to_string(index=False))

    # figure: 2D wrist path (image plane, torso units) of each flagged subject's clips next to 3 reference subjects
    refs = list(C[(C.exercise == ex) & (C.subject_PF1 >= 0.5)].groupby("subject").size()
                .sort_values(ascending=False).index[:3])
    subs = flagged + refs
    fig, axs = plt.subplots(len(subs), 6, figsize=(14, 2.4 * len(subs)), squeeze=False)
    for i, s in enumerate(subs):
        vids = idx[(idx.exercise == ex) & (idx.subject == s)].sort_values(["view", "variation"]).head(6)
        for j, ax in enumerate(axs[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= len(vids):
                ax.axis("off")
                continue
            v = vids.iloc[j]
            m = meta[v.video_id]
            p = m["xy"][:, J[f"{m['side']}_wrist"]] - (m["xy"][:, J["left_shoulder"]] + m["xy"][:, J["right_shoulder"]]) / 2
            p = p[np.isfinite(p).all(1)]
            ax.plot(p[:, 0], p[:, 1], lw=0.8, color="#c05621" if s in flagged else "#2f855a")
            ax.plot(p[0, 0], p[0, 1], "ko", ms=3)
            ax.plot(0, 0, "k+", ms=6)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(2.2, -2.2)
            ax.set_aspect("equal")
            asg = C.set_index("video_id").assigned.get(v.video_id, "-")
            ax.set_title(f"{v.view} {v.variation} PF1 {v.PF1:.1f} {m['side'][0]}\nnearest {asg}  turns {m['turns']:.2f}",
                         fontsize=7)
        axs[i, 0].set_ylabel(f"{s}\n({'PF1<0.5' if s in flagged else 'reference'})", fontsize=8)
    fig.suptitle(f"{ex} non-experts: moving wrist relative to mid-shoulder (+), image plane, torso units; dot = start. "
                 "Orange = mean PF1 < 0.5, green = reference.", fontsize=9)
    fig.tight_layout()
    chk = resolve(cfg["outputs"]["checks"])
    chk.mkdir(parents=True, exist_ok=True)
    fig.savefig(chk / f"pf1_{ex}_wrist_paths.png", dpi=80)
    print(f"\nwrote {out / f'pf1_motion_check_{ex}_subjects.csv'} and {chk / f'pf1_{ex}_wrist_paths.png'}")


if __name__ == "__main__":
    main()
