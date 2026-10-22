"""Step 0 pose sample for the new exercises: select videos, extract MediaPipe Pose keypoints, preprocess.

Sample per exercise (seeded): up to 6 expert subjects x 2 variations of complete F+L+R groups (variations spread
over FL/ML/LL/LJ/HJ/O; same subject, different variation = separate recordings for the sync controls), extra
front-view expert videos scoring >= 80 from other subjects (pool for the PT-reference comparison), and for
non-experts one F+L+R group plus up to 8 single videos of distinct subjects. Special cases are always added:
E03 P02 (wrong exercise) and E09 P21 (PF1 = 0 on all videos). The PT demos of E03-E09 are extracted too.

Same extractor and settings as Phase 1 (cvdc.data.extract: Tasks PoseLandmarker full model, 960 px, PyAV
timestamps, rotation honoured) and the Phase 1 preprocessing (cvdc.data.preprocess). Keypoints only; no pixels.
Outputs: data/cache/step0_full/{kp,kp_ref,pre,pre_ref}/ and <out>/pose_sample.csv, pose_summary.csv.

usage: python -I scripts/full/step0/pose_sample.py --videos D:/cb/dataset/videos --meta-root D:/cb
       --inv outputs/step0_full/inventory.csv --out outputs/step0_full [--workers 12]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.extract import run_jobs  # noqa: E402
from cvdc.data.preprocess import preprocess_arrays  # noqa: E402
from cvdc.data.skeleton import J  # noqa: E402

EXERCISES = ["E03", "E04", "E05", "E06", "E07", "E08", "E09"]
VARS = ["FL", "ML", "LL", "LJ", "HJ", "O"]
SPECIAL = {"E03": ("P02", "FL"), "E09": ("P21", "FL")}
REFS = {"E03": ["Later Rotation V3.mp4", "Lateral Rotation.mp4", "lateral rotation V2.mp4"],
        "E04": ["Medial Rotation.mp4"], "E05": ["Circumduction.mp4"], "E06": ["Wrist Extension.mp4"],
        "E07": ["Hip Joint Flection.mp4"], "E08": ["Both Leg Flection.mp4"], "E09": ["Back Extension.mp4"]}
CACHE = REPO / "data" / "cache" / "step0_full"
VIS_JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
              "left_index", "right_index", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def select(inv, seed=0):
    picks = []
    for ex in EXERCISES:
        rng = np.random.default_rng(seed + int(ex[1:]))
        g = inv[inv.exercise == ex]
        e = g[g.group == "expert"]
        full = e[e.n_views_group == 3]
        subj = sorted(full.subject.unique(), key=lambda s: int(s[1:]))
        chosen = list(rng.choice(subj, size=min(6, len(subj)), replace=False))
        order = list(rng.permutation(VARS)) * 2
        k = 0
        for s in chosen:
            have = set(full[full.subject == s].variation)
            got = []
            for _ in range(len(order)):
                v = order[k % len(order)]
                k += 1
                if v in have and v not in got:
                    got.append(v)
                if len(got) == 2:
                    break
            for v in got:
                picks += [(r.video_id, "expert_group") for r in full[(full.subject == s) & (full.variation == v)].itertuples()]
        if ex in SPECIAL:
            s, v = SPECIAL[ex]
            picks += [(r.video_id, "special") for r in e[(e.subject == s) & (e.variation == v)].itertuples()]
        extra = e[(e.view == "F") & (e.score >= 80) & ~e.subject.isin(chosen)]
        extra = extra.sample(frac=1, random_state=seed).drop_duplicates("subject").head(8)
        picks += [(v, "expert_front_hi") for v in extra.video_id]
        n = g[g.group == "non-expert"]
        if len(n):
            nf = n[n.n_views_group == 3]
            if len(nf):
                gid = rng.choice(sorted(nf.group_id.unique()))
                picks += [(r.video_id, "nonexpert_group") for r in nf[nf.group_id == gid].itertuples()]
            ns = n[n.view == "F"].sample(frac=1, random_state=seed).drop_duplicates("subject").head(8)
            picks += [(v, "nonexpert_single") for v in ns.video_id]
    sel = pd.DataFrame(picks, columns=["video_id", "reason"]).drop_duplicates("video_id")
    return inv.merge(sel, on="video_id")


def lying_angle(xy, vis):
    """Median angle (deg) of the 2D trunk axis (mid-hip -> mid-shoulder) from image vertical: ~0 upright, ~90 lying."""
    sh = (xy[:, J["left_shoulder"]] + xy[:, J["right_shoulder"]]) / 2
    hp = (xy[:, J["left_hip"]] + xy[:, J["right_hip"]]) / 2
    d = sh - hp
    return float(np.nanmedian(np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1]))))


def summarize(kp_path, pre_path, pc):
    with np.load(kp_path) as z:
        lm, det, t, w, hw = z["lm2d"], z["detected"], z["t"], z["world"], z["frame_hw"]
    r = dict(n_frames=len(t), duration_s=float(t[-1]), det_rate=float(det.mean()))
    vis = np.nan_to_num(lm[:, :, 3])
    for n in VIS_JOINTS:
        r[f"vis_{n}"] = float((vis[:, J[n]] >= 0.5).mean())
    # longest run without a detected pose (s)
    runs, cur = [], 0
    for d in det:
        cur = 0 if d else cur + 1
        runs.append(cur)
    dt = np.median(np.diff(t)) if len(t) > 1 else 1 / 30
    r["max_miss_s"] = float(max(runs) * dt)
    if r["det_rate"] >= pc["min_detection_rate"]:
        d = preprocess_arrays(lm, w, t, det, hw, pc)
        Path(pre_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pre_path, **d)
        r["lying_deg"] = lying_angle(d["xy"], d["vis"])
        r["preprocessed"] = True
    else:
        Path(pre_path).unlink(missing_ok=True)
        r["preprocessed"] = False
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--meta-root", required=True)
    ap.add_argument("--inv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    p, pc = cfg["pose"], cfg["preprocess"]
    out = resolve(a.out)
    inv = pd.read_csv(a.inv)
    sel = select(inv, a.seed)
    sel.to_csv(out / "pose_sample.csv", index=False)
    print(sel.groupby(["exercise", "reason"]).size().unstack(fill_value=0).to_string())
    print(f"{len(sel)} videos, ~{int((sel.size_mb).sum())} MB")
    conf = {k: p[k] for k in ("min_pose_detection_confidence", "min_pose_presence_confidence", "min_tracking_confidence")}
    jobs = [(r.video_id, str(Path(a.videos) / r.folder / r.file), str(CACHE / "kp" / f"{r.video_id}.npz"))
            for r in sel.itertuples()]
    refs = [(ex, f) for ex, fs in REFS.items() for f in fs]
    ref_root = Path(a.meta_root) / cfg["data"]["reference_dir"]
    stem = lambda f: f.replace(".mp4", "").strip().replace(" ", "_")  # noqa: E731
    jobs += [(stem(f), str(ref_root / f), str(CACHE / "kp_ref" / f"{stem(f)}.npz")) for _, f in refs]
    if not a.summary_only:
        run_jobs(jobs, resolve(p["model"]), p["max_side"], p["pad_square"], conf, a.workers)
    rows = []
    for r in sel.itertuples():
        rows.append(dict(video_id=r.video_id, exercise=r.exercise, subject=r.subject, group=r.group, view=r.view,
                         variation=r.variation, score=r.score, reason=r.reason,
                         **summarize(CACHE / "kp" / f"{r.video_id}.npz", CACHE / "pre" / f"{r.video_id}.npz", pc)))
    for ex, f in refs:
        rows.append(dict(video_id=stem(f), exercise=ex, group="pt_demo", view="?", reason="pt_demo",
                         **summarize(CACHE / "kp_ref" / f"{stem(f)}.npz", CACHE / "pre_ref" / f"{stem(f)}.npz", pc)))
    S = pd.DataFrame(rows)
    S.to_csv(out / "pose_summary.csv", index=False)
    pd.set_option("display.width", 250)
    d = S[S.group != "pt_demo"]
    print("\n=== detection rate (mean over videos) by exercise x view")
    print(d.pivot_table(index="exercise", columns="view", values="det_rate", aggfunc="mean").round(3).to_string())
    print("\n=== detection rate by exercise x variation")
    print(d.pivot_table(index="exercise", columns="variation", values="det_rate", aggfunc="mean").round(3).to_string())
    print("\n=== videos with detection < 90% / < 50% by exercise:",
          d.groupby("exercise").det_rate.agg(lambda s: f"{(s < .9).sum()}/{(s < .5).sum()} of {len(s)}").to_dict())
    print("\n=== joint visibility >= 0.5 (share of frames), mean by exercise x view")
    cols = [c for c in S.columns if c.startswith("vis_")]
    print(d.groupby(["exercise", "view"])[cols].mean().round(2).rename(columns=lambda c: c[4:]).to_string())
    print("\n=== trunk axis from image vertical (deg; ~0 upright, ~90 lying), median by exercise x view")
    print(d.pivot_table(index="exercise", columns="view", values="lying_deg", aggfunc="median").round(0).to_string())
    print("\n=== PT demos")
    print(S[S.group == "pt_demo"][["video_id", "exercise", "n_frames", "duration_s", "det_rate", "lying_deg"]
                                  + cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
