"""Phase 2 check of the 3D sources for B2: MediaPipe world landmarks vs KTPFormer lifted 3D (A10).

No 3D ground truth exists, so the sources are compared on what synchronized views make measurable:
  view consistency  the same rep seen from F, L and R at the same aligned instants (synced groups share rep
                    boundaries, P6b) should give the same 3D joint angle: mean |angle_a(t) - angle_b(t)| over the
                    rep frames and view pairs. Angles: E01 shoulder elevation, E02 horizontal adduction (Module III
                    angles, M7a) and the elbow angle, on the group's moving arm.
  bone stability    coefficient of variation over time of the upper-arm and forearm lengths
  jitter            mean |second difference| of the wrists (torso lengths per frame^2)
  body pitch        median lean of the mid-ankle -> mid-shoulder axis in depth (camera pitch, A8)
Labels are not used, so the check can choose the lifter's input scale (``lifter.input_torso``) without leakage.

usage: python scripts/pilot/phase2/lifter_check.py [--groups 20] [--scales 0.15 0.22 0.3 0.45] [--seed 0]
       python scripts/pilot/phase2/lifter_check.py --all     (every synced 3-view group, the cached full lift)
Writes outputs/phase2/lifter_check_*.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.assess.angles import horizontal_adduction, joint_angle, shoulder_elevation  # noqa: E402
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.preprocess import resample_fixed  # noqa: E402
from cvdc.data.skeleton import JOINT_SETS  # noqa: E402
from cvdc.models.lifter import KTPFormerLifter  # noqa: E402

COCO = JOINT_SETS["coco17"]
ANGLE = {"E01": ("shoulder_elevation", shoulder_elevation), "E02": ("horizontal_adduction", horizontal_adduction)}


def to33(P):
    full = np.full(P.shape[:-2] + (33, P.shape[-1]), np.nan)
    full[..., COCO, :] = P
    return full


def angles(P, ex, side):
    F = to33(P)
    return {ANGLE[ex][0]: ANGLE[ex][1](F, side),
            "elbow": joint_angle(F, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")}


def source_metrics(load, groups, members, reps):
    """load(video_id) -> (T, 17, 3) coco17 normalised 3D. Returns (per-view-pair angle diffs, per-video stats)."""
    rows, vrows, cache = [], [], {}

    def get(v):
        if v not in cache:
            cache[v] = load(v)
        return cache[v]

    for gid in groups:
        m = members[(members.group_id == gid) & (members.n_members == 3)]
        for sid, s in m.groupby("set_id"):
            per_view = {}
            for r in s.itertuples():
                rr = reps.iloc[r.item]
                P = get(r.video_id)[rr.start:rr.end + 1]
                per_view[r.view] = {k: resample_fixed(v, 64) for k, v in angles(P, r.exercise, rr.side).items()}
            for a, b in (("F", "L"), ("F", "R"), ("L", "R")):
                if a in per_view and b in per_view:
                    for k in per_view[a]:  # each view resampled to the same 64-step rep grid as the packs
                        d = np.abs(per_view[a][k] - per_view[b][k])
                        rows.append(dict(group_id=gid, exercise=s.exercise.iloc[0], set_id=sid, views=a + b,
                                         angle=k, mean_abs_diff_deg=np.nanmean(d) if np.isfinite(d).any() else np.nan))
    for v, P in cache.items():
        ex = v[:3]
        side = reps[reps.video_id == v].side.iloc[0]
        s, e, w = (5, 7, 9) if side == "left" else (6, 8, 10)
        ua = np.linalg.norm(P[:, s] - P[:, e], axis=-1)
        fa = np.linalg.norm(P[:, e] - P[:, w], axis=-1)
        acc = np.linalg.norm(P[2:, [9, 10]] - 2 * P[1:-1, [9, 10]] + P[:-2, [9, 10]], axis=-1)
        up = (P[:, 5] + P[:, 6]) / 2 - (P[:, 15] + P[:, 16]) / 2
        vrows.append(dict(video_id=v, exercise=ex, view=v.split("_")[2][1], upper_arm_cv=ua.std() / ua.mean(),
                          forearm_cv=fa.std() / fa.mean(), wrist_jitter=acc.mean(),
                          body_pitch_deg=np.degrees(np.median(np.arctan2(up[:, 2], -up[:, 1])))))
    return pd.DataFrame(rows), pd.DataFrame(vrows)


def summarize(name, D, V):
    out = {"source": name}
    for (ex, k), g in D.groupby(["exercise", "angle"]):
        out[f"{ex}_{k}_viewdiff_deg"] = g.mean_abs_diff_deg.median()
    for (ex, vv), g in D[D.angle != "elbow"].groupby(["exercise", "views"]):
        out[f"{ex}_main_{vv}_deg"] = g.mean_abs_diff_deg.median()
    out["upper_arm_cv"] = V.upper_arm_cv.median()
    out["forearm_cv"] = V.forearm_cv.median()
    out["wrist_jitter"] = V.wrist_jitter.median()
    for vw, g in V.groupby("view"):
        out[f"body_pitch_{vw}_deg"] = g.body_pitch_deg.median()
    out["rep_sets"] = D.set_id.nunique()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, default=20, help="synced 3-view groups per exercise")
    ap.add_argument("--scales", type=float, nargs="*", default=[0.15, 0.22, 0.3, 0.45])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="all synced 3-view groups; KTPFormer from the cached full lift")
    a = ap.parse_args()
    cfg = load_config()
    out = resolve("outputs/phase2")
    out.mkdir(parents=True, exist_ok=True)
    reps = pd.read_csv(resolve("data/reps.csv"))
    members = pd.read_csv(resolve(cfg["cvdc"]["pairs"]["members_csv"]))
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"])).set_index("video_id")
    sr = pd.read_csv(resolve(cfg["sync"]["report_csv"]))
    ok = sr[(sr.cls == "synced") & (sr.aligned_views == "FLR")]
    rng = np.random.default_rng(a.seed)
    if a.all:
        groups = list(ok.group_id)
    else:
        groups = [g for ex, d in ok.groupby("exercise") for g in rng.choice(sorted(d.group_id), a.groups, replace=False)]
    vids = sorted(members[members.group_id.isin(groups)].video_id.unique())
    print(f"{len(groups)} synced 3-view groups, {len(vids)} videos")

    def mp_load(v):
        with np.load(resolve(idx.loc[v, "pre_path"])) as z:
            return z["world"][:, COCO].astype(float)

    results, tables = [], []
    D, V = source_metrics(mp_load, groups, members, reps)
    results.append(summarize("mediapipe_world", D, V))
    tables.append(D.assign(source="mediapipe_world"))
    scales = [cfg["lifter"]["input_torso"]] if a.all else a.scales
    lifter = None if a.all else KTPFormerLifter(cfg["lifter"])
    for sc in scales:
        d = resolve(cfg["lifter"]["out_dir"]) if a.all else resolve(f"data/cache/lift_check/ktp_{sc}")
        if not a.all:
            d.mkdir(parents=True, exist_ok=True)
            lifter.lc = {**cfg["lifter"], "input_torso": sc}
            t0, n = time.time(), 0
            for v in vids:
                with np.load(resolve(idx.loc[v, "pre_path"])) as z:
                    xy = z["xy"][:, COCO].astype(float)
                np.savez_compressed(d / f"{v}.npz", **lifter.lift_coco(xy, cfg["preprocess"]["fps"]))
                n += len(xy)
            print(f"  lifted {len(vids)} videos at input_torso {sc}: {n / (time.time() - t0):.0f} frames/s", flush=True)

        def kt_load(v, d=d):
            with np.load(d / f"{v}.npz") as z:
                return z["world"].astype(float)

        D, V = source_metrics(kt_load, groups, members, reps)
        results.append(summarize(f"ktpformer_torso{sc}", D, V))
        tables.append(D.assign(source=f"ktpformer_torso{sc}"))
    R = pd.DataFrame(results)
    tag = "all" if a.all else "scales"
    R.to_csv(out / f"lifter_check_{tag}.csv", index=False)
    pd.concat(tables).to_csv(out / f"lifter_check_{tag}_pairs.csv", index=False)
    print(R.set_index("source").T.round(3).to_string())


if __name__ == "__main__":
    main()
