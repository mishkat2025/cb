"""Step 0 analysis of the pose sample (E03-E09): rep signals, cross-view rep agreement, motion sync test,
and the PT-reference comparison. Reads data/cache/step0_full/pre*/ (pose_sample.py) and av_pairs.csv (sync_av.py).

1. Candidate signals per video (3D = MediaPipe world landmarks after preprocessing; 2D = image keypoints), both
   sides: shoulder elevation, elbow flexion, forearm elevation / azimuth in the trunk frame, wrist PC1 and 2D wrist
   height (upper limb); wrist deviation from the forearm line from the Pose hand points (E06); hip and knee flexion,
   2D knee height (lower limb); trunk-leg angle, 2D trunk inclination and shoulder height (trunk); PC1 of all
   2D joints. The moving side = the side with the larger p5-p95 range.
2. Rep counts with the Phase 1 segmenter (cvdc.data.reps.segment_reps) and per-video adaptive thresholds
   (prominence 0.4 x and minimum amplitude 0.3 x the p5-p95 range; oriented so the first second is the rest
   level). Feasibility = cross-view agreement in synced expert F+L+R groups (all three counts within +-1), as in P6.
3. Motion sync: per same-group view pair, the best candidate-signal lag (same signal in both views) within
   +-0.3 s of the device-corrected audio lag (+-5 s without one) and the Phase 1 rep-timing test
   (cvdc.data.sync.timing_test: residual < 42 ms with >= 4 matched crossings). Controls = same subject and view
   set, different variation (separate recordings), searched over +-5 s.
4. PT demos: posture (lying angle), moving limb, rep count and duration, and the Chamfer distance in a per-exercise
   3D angle space to the frames of high-scoring (>= 80) expert videos of the sample.

usage: python -I scripts/step0_full/pose_analysis.py --out outputs/step0_full
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.assess.angles import angle_between, shoulder_elevation, trunk_frame, wrist_path_pc1  # noqa: E402
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.reps import segment_reps  # noqa: E402
from cvdc.data.skeleton import JOINT_SETS, J  # noqa: E402
from cvdc.data.sync import timing_test, xcorr_lags  # noqa: E402

CACHE = REPO / "data" / "cache" / "step0_full"
FPS = 30
SIDES = ("left", "right")
REGION = {"E03": "upper", "E04": "upper", "E05": "upper", "E06": "wrist", "E07": "lower", "E08": "lower",
          "E09": "trunk"}
DEVICE_OFFSET = {("F", "L"): 0.093, ("F", "R"): 0.057, ("L", "R"): -0.044}   # S5 (Phase 1 estimates)
# per-exercise angle space for the PT-reference comparison (moving side first)
REF_SPACE = {"upper": ["el", "elb", "fa_up", "fa_az"], "wrist": ["wdev3", "elb", "el"],
             "lower": ["hip_m", "hip_o", "knee_m", "knee_o"], "trunk": ["trunk_leg", "hip_m", "knee_m"]}


def P(pre, k, n):
    return pre[k][:, J[n]]


def signals(pre):
    """All candidate signals on the uniform grid; keys '<name>_<l|r>' for sided ones."""
    w, xy = pre["world"].astype(float), pre["xy"].astype(float)
    lat, up, fwd = trunk_frame(w)
    s = {}
    for side in SIDES:
        c = side[0]
        sh, el, wr = (P(pre, "world", f"{side}_{j}") for j in ("shoulder", "elbow", "wrist"))
        ua, fa = el - sh, wr - el
        s[f"el_{c}"] = shoulder_elevation(w, side)
        s[f"elb_{c}"] = 180 - angle_between(-ua, fa)
        s[f"fa_up_{c}"] = angle_between(fa, up)                        # 0 = forearm up along the trunk axis
        out = lat if side == "right" else -lat
        s[f"fa_az_{c}"] = np.degrees(np.arctan2(np.sum(fa * fwd, -1), np.sum(fa * out, -1)))
        s[f"wpc1_{c}"] = wrist_path_pc1(w, side)
        msh = (P(pre, "xy", "left_shoulder") + P(pre, "xy", "right_shoulder")) / 2
        s[f"wy_{c}"] = -(P(pre, "xy", f"{side}_wrist")[:, 1] - msh[:, 1])
        hand3 = (P(pre, "world", f"{side}_index") + P(pre, "world", f"{side}_pinky")) / 2 - wr
        s[f"wdev3_{c}"] = angle_between(fa, hand3)                     # 0 = hand in line with the forearm
        fa2 = P(pre, "xy", f"{side}_wrist") - P(pre, "xy", f"{side}_elbow")
        h2 = (P(pre, "xy", f"{side}_index") + P(pre, "xy", f"{side}_pinky")) / 2 - P(pre, "xy", f"{side}_wrist")
        s[f"wdev2_{c}"] = np.degrees(np.arctan2(fa2[:, 0] * h2[:, 1] - fa2[:, 1] * h2[:, 0], np.sum(fa2 * h2, -1)))
        hip, kn, an = (P(pre, "world", f"{side}_{j}") for j in ("hip", "knee", "ankle"))
        s[f"hip_{c}"] = 180 - angle_between(sh - hip, kn - hip)        # 0 = trunk and thigh in line
        s[f"knee_{c}"] = 180 - angle_between(hip - kn, an - kn)        # 0 = straight knee
        mhip = (P(pre, "xy", "left_hip") + P(pre, "xy", "right_hip")) / 2
        s[f"ky_{c}"] = -(P(pre, "xy", f"{side}_knee")[:, 1] - mhip[:, 1])
    msh3 = (P(pre, "world", "left_shoulder") + P(pre, "world", "right_shoulder")) / 2
    mhip3 = (P(pre, "world", "left_hip") + P(pre, "world", "right_hip")) / 2
    mkn3 = (P(pre, "world", "left_knee") + P(pre, "world", "right_knee")) / 2
    s["trunk_leg"] = 180 - angle_between(msh3 - mhip3, mkn3 - mhip3)
    msh, mhip = ((P(pre, "xy", f"left_{j}") + P(pre, "xy", f"right_{j}")) / 2 for j in ("shoulder", "hip"))
    d = msh - mhip
    s["trunk_incl2"] = np.degrees(np.arctan2(-d[:, 1], np.abs(d[:, 0])))   # 2D trunk angle above image horizontal
    s["shy"] = -(msh[:, 1] - mhip[:, 1])
    X = xy[:, JOINT_SETS["coco17"]].reshape(len(xy), -1)
    if np.isfinite(X).all() and len(X) > 3:
        z = X - X.mean(0)
        s["pc1_all"] = z @ np.linalg.svd(z, full_matrices=False)[2][0]
    return s


def moving(s, base):
    """Moving side for a sided signal family: the side with the larger p5-p95 range."""
    rng = {c: np.nanpercentile(s[f"{base}_{c}"], 95) - np.nanpercentile(s[f"{base}_{c}"], 5)
           if np.isfinite(s[f"{base}_{c}"]).sum() > 10 else -1 for c in "lr"}
    return max(rng, key=rng.get)


def candidate(s, region):
    """Rep-signal candidates of the exercise region: {name: signal} with sided ones on the moving side."""
    fam = {"upper": ["el", "elb", "fa_up", "fa_az", "wpc1", "wy"], "wrist": ["wdev3", "wdev2", "elb", "fa_up", "wy"],
           "lower": ["hip", "knee", "ky"], "trunk": ["hip", "knee"]}[region]
    out = {}
    for b in fam:
        c = moving(s, b)
        out[b] = s[f"{b}_{c}"]
        out[f"{b}@side"] = c
    if region in ("lower", "trunk"):
        out["hip_sum"] = s["hip_l"] + s["hip_r"]
        out["knee_sum"] = s["knee_l"] + s["knee_r"]
    if region == "trunk":
        out.update(trunk_leg=s["trunk_leg"], trunk_incl2=s["trunk_incl2"], shy=s["shy"])
    if "pc1_all" in s:
        out["pc1_all"] = s["pc1_all"]
    return out


def count_reps(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < FPS * 2:
        return dict(n=np.nan)
    x = x.copy()
    x[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), x[ok])
    m = median_filter(x, size=9, mode="nearest")
    lo, hi = np.percentile(m, [5, 95])
    rng = hi - lo
    if rng <= 1e-9:
        return dict(n=0)
    start = np.median(m[:FPS])
    pol = "peak" if abs(start - lo) <= abs(start - hi) else "valley"
    reps = segment_reps(x, FPS, prominence=0.4 * rng, min_distance_s=1.0, min_amplitude=0.3 * rng,
                        polarity=pol, max_rep_s=20.0)
    dur = [r["duration_s"] for r in reps]
    return dict(n=len(reps), polarity=pol, range=rng, rep_s=float(np.median(dur)) if dur else np.nan)


def load(pre_path):
    with np.load(pre_path) as z:
        return {k: z[k] for k in ("world", "xy", "vis", "t")}


def motion_pair(sa, sb, center, window, mc):
    lags = np.arange(int(np.floor((center - window) * FPS)), int(np.ceil((center + window) * FPS)) + 1)
    best = None
    for k in sa:
        if k not in sb or k.endswith("@side") or not (np.isfinite(sa[k]).all() and np.isfinite(sb[k]).all()):
            continue
        if sa[k].std() < 1e-6 or sb[k].std() < 1e-6:
            continue
        rs = xcorr_lags(sa[k], sb[k], lags, mc["min_overlap"])
        if np.isnan(rs).all():
            continue
        sc = np.abs(rs) if k.startswith(("pc1", "wpc1", "fa_az", "wdev2")) else rs   # sign-free signals
        i = int(np.nanargmax(sc))
        if best is None or sc[i] > best[0]:
            best = (sc[i], i, rs, k)
    if best is None:
        return dict(motion_r=np.nan, motion_confirmed=False)
    sc, i, rs, k = best
    sign = int(np.sign(rs[i])) or 1
    tt = timing_test(sa[k], sb[k], lags[i] / FPS, sign, FPS, mc)
    edge = i in (0, len(rs) - 1)
    ok = tt["matched"] >= mc["min_matched"] and tt["resid_sd_lin_ms"] < mc["resid_threshold_ms"] and not edge
    return dict(motion_r=float(sc), motion_sig=k, motion_lag_s=lags[i] / FPS + (tt["mean_resid_s"] if np.isfinite(
        tt["mean_resid_s"]) else 0.0), motion_edge=edge, motion_confirmed=bool(ok), **tt)


def chamfer(A, B):
    A, B = A[np.isfinite(A).all(1)], B[np.isfinite(B).all(1)]
    if len(A) < 5 or len(B) < 5:
        return np.nan, np.nan
    return float(cKDTree(B).query(A)[0].mean()), float(cKDTree(A).query(B)[0].mean())


def ref_feats(s, region):
    """Angle-space matrix (frames x dims) for the PT-reference comparison, moving side first."""
    if region == "upper":
        c = moving(s, "wpc1")   # wrist path range: works for rotations, where the elevation barely changes
        return np.c_[s[f"el_{c}"], s[f"elb_{c}"], s[f"fa_up_{c}"], s[f"fa_az_{c}"]], c
    if region == "wrist":
        c = moving(s, "wdev3")
        return np.c_[s[f"wdev3_{c}"], s[f"elb_{c}"], s[f"el_{c}"]], c
    c = moving(s, "hip")
    o = "l" if c == "r" else "r"
    if region == "lower":
        return np.c_[s[f"hip_{c}"], s[f"hip_{o}"], s[f"knee_{c}"], s[f"knee_{o}"]], c
    return np.c_[s["trunk_leg"], (s["hip_l"] + s["hip_r"]) / 2, (s["knee_l"] + s["knee_r"]) / 2], c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/step0_full")
    a = ap.parse_args()
    cfg = load_config()
    mc = cfg["sync"]["motion"]
    out = resolve(a.out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    S = pd.read_csv(out / "pose_summary.csv")
    vids = S[(S.group != "pt_demo") & S.preprocessed.astype(bool)].copy()
    inv = pd.read_csv(out / "inventory.csv").set_index("video_id")
    vids["group_id"] = inv.loc[vids.video_id, "group_id"].values
    vids["PF4"] = inv.loc[vids.video_id, "PF4"].values
    sig = {v: signals(load(CACHE / "pre" / f"{v}.npz")) for v in vids.video_id}

    # ---- 1-2. rep counts per candidate signal
    rows = []
    for r in vids.itertuples():
        reg = REGION[r.exercise]
        cand = candidate(sig[r.video_id], reg)
        for k, x in cand.items():
            if k.endswith("@side"):
                continue
            c = count_reps(x)
            rows.append(dict(video_id=r.video_id, exercise=r.exercise, subject=r.subject, group=r.group, view=r.view,
                             variation=r.variation, group_id=r.group_id, score=r.score, PF4=r.PF4, signal=k,
                             side=cand.get(f"{k}@side", ""), **c))
    R = pd.DataFrame(rows)
    R.to_csv(out / "rep_counts_sample.csv", index=False)

    # sync classes for the agreement test (filled below from the classification)
    av = pd.read_csv(out / "av_pairs.csv")
    same = av[av.kind == "same_group"].copy()
    same["group_id"] = same.ex + "_" + same.pid + "_V" + same.var_a
    cls = pd.read_csv(out / "sync_groups_classified.csv").set_index("group_id").cls
    meta_sync = {gid: c == "synced" for gid, c in cls.items()}
    # subjects whose every video of the exercise has PF1 ~ 0 (wrong exercise, e.g. E03 P02) are left out here
    pf1 = pd.read_csv(out / "inventory.csv").groupby(["exercise", "subject"]).PF1.mean()
    wrong = {k for k, v in pf1.items() if v < 0.5}
    print("left out of the agreement test (mean PF1 < 0.5):",
          sorted(f"{e}_{s}" for e, s in wrong if f"{e}_{s}" in set(R.exercise + "_" + R.subject)))
    agree = []
    Rx = R[[(e, s) not in wrong for e, s in zip(R.exercise, R.subject)]]
    for (ex, sg), g in Rx[Rx.group == "expert"].groupby(["exercise", "signal"]):
        per = g.groupby("group_id").agg(nv=("view", "nunique"), lo=("n", "min"), hi=("n", "max"), med=("n", "median"))
        per = per[(per.nv == 3) & per.index.map(lambda x: bool(meta_sync.get(x, False)))]
        if not len(per):
            continue
        agree.append(dict(exercise=ex, signal=sg, groups=len(per), within1=float(((per.hi - per.lo) <= 1).mean()),
                          exact=float((per.hi == per.lo).mean()), median_count=float(g.n.median()),
                          count_p10=float(g.n.quantile(.1)), count_p90=float(g.n.quantile(.9)),
                          rep_s=float(g.rep_s.median()),
                          zero_rep_videos=int((g.n == 0).sum()), videos=len(g)))
    A = pd.DataFrame(agree).sort_values(["exercise", "within1", "exact"], ascending=[True, False, False])
    A.to_csv(out / "rep_signal_agreement.csv", index=False)
    print("=== rep-signal feasibility: synced expert F+L+R groups of the sample, rep counts per view")
    print(A.round(2).to_string(index=False))
    best = A.groupby("exercise").head(1).set_index("exercise")
    print("\n=== best signal per exercise: counts by view and group")
    for ex, b in best.iterrows():
        g = R[(R.exercise == ex) & (R.signal == b.signal)]
        print(f"{ex} {b.signal}: by view median", g[g.group == "expert"].groupby("view").n.median().to_dict(),
              " experts PF4=1 videos median count", g[(g.group == "expert") & (g.PF4 >= .999)].n.median(),
              " non-expert counts", g[g.group == "non-expert"].n.value_counts().sort_index().to_dict(),
              " sides", g.side.value_counts().to_dict(),
              " PF1~0 subjects' counts", {f"{s}{v}": int(n) for s, v, n in zip(g.subject, g.view, g.n)
                                          if (ex, s) in wrong and np.isfinite(n)})

    # ---- 3. motion sync on same-group pairs and controls
    vids_by = vids.set_index("video_id")
    aud = {(r.ex, r.pid, r.var_a, r.view_a, r.view_b): r for r in same.itertuples()}
    mrows = []
    for gid, g in vids.groupby("group_id"):
        g = g.sort_values("view")
        for x, y in combinations(g.itertuples(), 2):
            ex, pid, var = x.exercise, x.subject, inv.loc[x.video_id, "variation_raw"]
            ar = aud.get((ex, pid, var, x.view, y.view))
            lag3 = getattr(ar, "audio_lag3_s", np.nan) if ar is not None else np.nan
            usable = np.isfinite(lag3) and abs(getattr(ar, "audio_lag3_raw_s", 0)) < 2.9
            center = lag3 + DEVICE_OFFSET.get((x.view, y.view), 0.0) if usable else 0.0
            win = mc["refine_window_s"] if usable else mc["global_window_s"]
            reg = REGION[ex]
            sa, sb = candidate(sig[x.video_id], reg), candidate(sig[y.video_id], reg)
            res = motion_pair(sa, sb, center, win, mc)
            mrows.append(dict(kind="same_group", exercise=ex, group=x.group, group_id=gid, view_a=x.view, view_b=y.view,
                              video_a=x.video_id, video_b=y.video_id, audio_lag3_s=lag3, center_s=center, window_s=win,
                              d_creation_s=getattr(ar, "d_creation_s", np.nan) if ar is not None else np.nan, **res))
    # controls: same subject, same view pair, different variation (both in the sample)
    for (ex, subj), g in vids.groupby(["exercise", "subject"]):
        gids = sorted(g.group_id.unique())
        for g1, g2 in combinations(gids, 2):
            for x in g[g.group_id == g1].itertuples():
                for y in g[g.group_id == g2].itertuples():
                    reg = REGION[ex]
                    res = motion_pair(candidate(sig[x.video_id], reg), candidate(sig[y.video_id], reg), 0.0,
                                      mc["global_window_s"], mc)
                    mrows.append(dict(kind="control", exercise=ex, group=x.group, group_id=f"{g1}|{g2}",
                                      view_a=x.view, view_b=y.view, video_a=x.video_id, video_b=y.video_id, **res))
    M = pd.DataFrame(mrows)
    M.to_csv(out / "motion_pairs_sample.csv", index=False)
    print("\n=== motion test (sample): confirmed share by exercise, pair kind and view pair")
    M["vp"] = M.view_a + "-" + M.view_b
    print(M.groupby(["exercise", "kind"]).agg(pairs=("motion_confirmed", "size"),
                                              confirmed=("motion_confirmed", "mean"),
                                              resid_med_ms=("resid_sd_lin_ms", "median")).round(3).to_string())
    print(M[M.kind == "same_group"].groupby(["exercise", "vp"]).motion_confirmed.mean().round(2).unstack().to_string())
    c = M[(M.kind == "control") & (M.matched >= mc["min_matched"])]
    print(f"controls testable: {len(c)}, falsely confirmed: {int(c.motion_confirmed.sum())}; "
          f"resid p1/p5/median {c.resid_sd_lin_ms.quantile(.01):.0f}/{c.resid_sd_lin_ms.quantile(.05):.0f}/"
          f"{c.resid_sd_lin_ms.median():.0f} ms")
    ok = M[(M.kind == "same_group") & M.motion_confirmed & M.audio_lag3_s.notna()]
    print("motion-confirmed pairs: video lag - raw audio lag (ms) by view pair:",
          ok.assign(d=1000 * (ok.motion_lag_s - ok.audio_lag3_s)).groupby("vp").d.agg(["count", "median", "std"])
          .round(0).to_dict("index"))

    # ---- 4. PT demos vs high-scoring expert frames
    refs = S[S.group == "pt_demo"]
    rrows = []
    for r in refs.itertuples():
        if not bool(r.preprocessed):
            rrows.append(dict(ref=r.video_id, exercise=r.exercise, det_rate=r.det_rate))
            continue
        reg = REGION[r.exercise]
        s = signals(load(CACHE / "pre_ref" / f"{r.video_id}.npz"))
        X, side = ref_feats(s, reg)
        cand = candidate(s, reg)
        b = best.signal.get(r.exercise)
        rc = count_reps(cand[b]) if b in cand else dict(n=np.nan)
        row = dict(ref=r.video_id, exercise=r.exercise, det_rate=r.det_rate, lying_deg=r.lying_deg,
                   duration_s=r.duration_s, moving_side=side, rep_signal=b, reps=rc.get("n"), rep_s=rc.get("rep_s"))
        for k, j in zip(REF_SPACE[reg], range(X.shape[1])):
            row[f"{k}_p5"], row[f"{k}_p50"], row[f"{k}_p95"] = np.nanpercentile(X[:, j], [5, 50, 95]).round(0)
        pool = vids[(vids.exercise == r.exercise) & (vids.group == "expert") & (vids.score >= 80)]
        for vset, name in ((["F"], "front"), (["F", "L", "R"], "all")):
            pp = pool[pool.view.isin(vset)]
            if not len(pp):
                continue
            D = np.vstack([ref_feats(sig[v], reg)[0] for v in pp.video_id])
            d1, d2 = chamfer(X, D)
            row[f"chamfer_{name}"] = round(0.5 * (d1 + d2), 1)
            row[f"pool_{name}"] = len(pp)
        rrows.append(row)
        # dataset percentiles of the same space (front, high-scoring)
    Rf = pd.DataFrame(rrows)
    Rf.to_csv(out / "ref_compare.csv", index=False)
    print("\n=== PT demos vs high-scoring (>= 80) expert sample frames (3D angles, deg; Chamfer = mean of both directions)")
    print(Rf.to_string(index=False))
    print("\n=== dataset (expert >= 80, front) angle percentiles p5/p50/p95 for the same spaces")
    for ex in sorted(vids.exercise.unique()):
        reg = REGION[ex]
        pp = vids[(vids.exercise == ex) & (vids.group == "expert") & (vids.score >= 80) & (vids.view == "F")]
        if not len(pp):
            continue
        D = np.vstack([ref_feats(sig[v], reg)[0] for v in pp.video_id])
        print(ex, f"({len(pp)} videos)", {k: np.nanpercentile(D[:, j], [5, 50, 95]).round(0).tolist()
                                          for j, k in enumerate(REF_SPACE[reg])})


if __name__ == "__main__":
    main()
