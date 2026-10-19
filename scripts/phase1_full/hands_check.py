"""E06 MediaPipe Hands check (user answer 2, P1c): is the wrist-extension angle from the hand landmarks usable?

1. Detection: share of frames with a hand per body side, by view and variation; distance of the hand's wrist landmark
   to the Pose wrist (2D forearm lengths); share of frames where MediaPipe's handedness label is mirrored as expected.
2. The angle on the moving side (the side of the group's rep signal, Phase 1 reps): wrist extension = forearm (Pose
   world elbow -> wrist) vs hand (Hands world wrist -> middle MCP), signed with the back-of-hand normal
   (cvdc.assess.angles.wrist_extension), per decoded frame, then onto the 30 fps grid (gaps <= 0.5 s interpolated,
   longer gaps NaN, as P5). Compared with the Pose-point wrist deviation (forearm vs mid(index, pinky) - wrist, 3D
   world; Step 0's 'wdev3', cross-view rep agreement 25%).
3. In synced expert F+L+R groups (Phase 1 sync, the same performance seen by three phones): (a) rep counts per view
   with the adaptive segmenter (P6c), the share of groups whose three counts lie within +-1 (Step 0 feasibility
   criterion; the Pose wrist height that segments E06 reached 75% on the sample); (b) the angle's agreement across
   views at aligned instants (offsets from the sync report): median |difference| and Pearson r per view pair;
   (c) the same on frames where both views detected the hand only (no gap filling beyond 0.1 s), after a 0.3 s
   median filter, Spearman, for the angle variants (Hands signed / unsigned 3D, Hands 2D, Pose points 2D) and the Pose
   wrist height (E06's rep signal) as the reference; with ``--compare-dir`` also for hands from other settings
   (``python -m cvdc.data.extract --only ... --hand-dir DIR --hands-full-res``: crops from full-resolution frames).
4. Skeleton-and-hand renders (keypoints only, no frames) of one group: <outputs.checks>/hands_E06_*.png.

usage: python scripts/phase1_full/hands_check.py [--config configs/data_full.yaml]
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvdc.assess.angles import angle_between, wrist_extension  # noqa: E402
from cvdc.config import exercise_cfg, load_config, resolve  # noqa: E402
from cvdc.data.extract import summarize_hands  # noqa: E402
from cvdc.data.reps import segment_signal  # noqa: E402
from cvdc.data.skeleton import J  # noqa: E402

FPS = 30
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
BODY = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24), (0, 11), (0, 12)]


def to_grid(t, a, n, max_gap_s=0.5):
    """Values a at times t -> the uniform grid (n frames at FPS); grid points farther than max_gap_s from a finite
    sample are NaN."""
    ok = np.isfinite(a)
    g = np.arange(n) / FPS
    out = np.full(n, np.nan)
    if ok.sum() < 2:
        return out
    tv, av = t[ok], a[ok]
    out = np.interp(g, tv, av)
    k = np.clip(np.searchsorted(tv, g), 1, len(tv) - 1)
    gap = np.minimum(np.abs(g - tv[k - 1]), np.abs(g - tv[k]))
    out[gap > max_gap_s] = np.nan
    return out


def series(kp_path, hand_path, side, n):
    with np.load(kp_path) as z:
        w, t = z["world"].astype(float), z["t"]
    with np.load(hand_path) as z:
        hw = z["hand_world"][:, 0 if side == "left" else 1].astype(float)
    hands = wrist_extension(w, hw, side)
    fa = w[:, J[f"{side}_wrist"]] - w[:, J[f"{side}_elbow"]]
    hp = (w[:, J[f"{side}_index"]] + w[:, J[f"{side}_pinky"]]) / 2 - w[:, J[f"{side}_wrist"]]
    posepts = angle_between(fa, hp)
    return to_grid(t, hands, n), to_grid(t, posepts, n)


VARIANTS = ("hands_3d_signed", "hands_3d", "hands_2d", "pose_points_2d", "pose_wrist_height_2d")


def variants(r, side, n, hand_dir=None):
    """Wrist-angle variants on the grid, only at frames with a detected hand (gaps <= 0.1 s), 0.3 s median filter:
    the Hands angle signed / unsigned in 3D and unsigned in 2D (image), the Pose-point angle (2D), and the Pose
    wrist height above mid-shoulder (2D, the E06 rep signal) for reference. ``mask`` = 1 where the hand was found."""
    with np.load(resolve(r.kp_path)) as z:
        w, t, lm, (H, W) = z["world"].astype(float), z["t"], z["lm2d"].astype(float), z["frame_hw"]
    hp = (Path(hand_dir) / Path(r.hand_path).name) if hand_dir is not None else resolve(r.hand_path)
    with np.load(hp) as z:
        k = 0 if side == "left" else 1
        hw, h2 = z["hand_world"][:, k].astype(float), z["hand2d"][:, k].astype(float)
    fa = w[:, J[f"{side}_wrist"]] - w[:, J[f"{side}_elbow"]]
    P2, Q2 = lm[..., :2] * (W, H), h2[..., :2] * (W, H)
    fa2 = P2[:, J[f"{side}_wrist"]] - P2[:, J[f"{side}_elbow"]]
    hpt = (P2[:, J[f"{side}_index"]] + P2[:, J[f"{side}_pinky"]]) / 2 - P2[:, J[f"{side}_wrist"]]
    msh = (P2[:, J["left_shoulder"]] + P2[:, J["right_shoulder"]]) / 2
    raw = dict(hands_3d_signed=wrist_extension(w, hw, side), hands_3d=angle_between(fa, hw[:, 9] - hw[:, 0]),
               hands_2d=angle_between(fa2, Q2[:, 9] - Q2[:, 0]), pose_points_2d=angle_between(fa2, hpt),
               pose_wrist_height_2d=-(P2[:, J[f"{side}_wrist"], 1] - msh[:, 1]) / H)
    found = np.isfinite(hw).all(-1).all(-1)
    out = {}
    for k, v in raw.items():
        x = to_grid(t, np.where(found, v, np.nan), n, max_gap_s=0.1)
        ok = np.isfinite(x)
        if ok.sum() > 9:
            x[ok] = median_filter(x[ok], size=9, mode="nearest")
        out[k] = x
    out["mask"] = np.nan_to_num(to_grid(t, np.where(found, 1.0, np.nan), n, max_gap_s=0.1))
    return out


def count(sig, rc):
    if np.isfinite(sig).sum() < 2 * FPS:
        return np.nan
    return len(segment_signal(sig, FPS, rc)[0])


def render(group, idx, cfg, out):
    """F / L / R skeleton + hand landmarks at 4 instants each (keypoints only)."""
    vids = idx[idx.group_id == group].sort_values("view")
    fig, axs = plt.subplots(len(vids), 5, figsize=(14, 3.2 * len(vids)), squeeze=False)
    for i, r in enumerate(vids.itertuples()):
        with np.load(resolve(r.kp_path)) as z:
            lm, (H, W), t = z["lm2d"], z["frame_hw"], z["t"]
        with np.load(resolve(r.hand_path)) as z:
            h2 = z["hand2d"]
        for j, f in enumerate(np.linspace(0.15, 0.85, 5) * (len(t) - 1)):
            f = int(f)
            ax = axs[i, j]
            P = lm[f, :, :2] * (W, H)
            for a, b in BODY:
                ax.plot(P[[a, b], 0], P[[a, b], 1], "k-", lw=1)
            for s, c in ((0, "#2b6cb0"), (1, "#c05621")):
                if np.isfinite(h2[f, s]).all():
                    Q = h2[f, s, :, :2] * (W, H)
                    for a, b in HAND_EDGES:
                        ax.plot(Q[[a, b], 0], Q[[a, b], 1], "-", color=c, lw=0.7)
            sel = P[[0, 11, 12, 13, 14, 15, 16, 23, 24]]
            fin = sel[np.isfinite(sel).all(1)]
            if len(fin):
                pad = 0.25 * (fin.max(0) - fin.min(0)) + 20
                lo, hi = fin.min(0) - pad, fin.max(0) + pad
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(hi[1], lo[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{r.view}  t={t[f]:.1f}s", fontsize=8)
    fig.suptitle(f"{group}: Pose skeleton + MediaPipe Hands (blue = body left, orange = body right); keypoints only",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_full.yaml")
    ap.add_argument("--compare-dir", help="hands of the same videos from other settings (e.g. full-resolution crops)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out, chk = resolve(cfg["outputs"]["phase1"]), resolve(cfg["outputs"]["checks"])
    chk.mkdir(parents=True, exist_ok=True)
    rc = exercise_cfg(cfg, "E06")["reps"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    idx = idx[(idx.exercise == "E06") & idx.hand_path.fillna("").ne("")]
    V = pd.read_csv(out / "reps_per_video.csv").set_index("video_id")
    G = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    pd.set_option("display.width", 220)

    # 1. detection
    rows = []
    for r in idx.itertuples():
        if resolve(r.hand_path).exists():
            rows.append(dict(video_id=r.video_id, view=r.view, variation=r.variation,
                             **summarize_hands(resolve(r.hand_path))))
    Dt = pd.DataFrame(rows)
    Dt.to_csv(out / "hands_detection_E06.csv", index=False)
    print(f"=== 1. hands found: {len(Dt)}/{len(idx)} E06 videos have a hands file")
    cols = ["hand_det_left", "hand_det_right", "hand_both", "wrist_dist_left_p50", "wrist_dist_right_p50",
            "label_mirrored_left", "label_mirrored_right"]
    print(Dt.groupby("view")[cols].mean().round(3).to_string())
    print(Dt.groupby("variation")[["hand_det_left", "hand_det_right", "hand_both"]].mean().round(3).T.to_string())

    # 2. the angle on the moving side, per video
    sig, rows = {}, []
    for r in idx.itertuples():
        if r.video_id not in V.index or not resolve(r.hand_path).exists():
            continue
        side, n = V.loc[r.video_id, "side"], int(V.loc[r.video_id, "n_frames_30"])
        h, p = series(resolve(r.kp_path), resolve(r.hand_path), side, n)
        o = "left" if side == "right" else "right"
        ho, _ = series(resolve(r.kp_path), resolve(r.hand_path), o, n)
        sig[r.video_id] = (h, p)
        rows.append(dict(video_id=r.video_id, group_id=r.group_id, view=r.view, variation=r.variation, side=side,
                         defined=float(np.isfinite(h).mean()), defined_other=float(np.isfinite(ho).mean()),
                         ext_p5=np.nanpercentile(h, 5) if np.isfinite(h).any() else np.nan,
                         ext_p95=np.nanpercentile(h, 95) if np.isfinite(h).any() else np.nan,
                         reps_hands=count(h, rc), reps_posepts=count(p, rc), reps_pipeline=V.loc[r.video_id, "n_reps_own"]))
    A = pd.DataFrame(rows)
    A["range_hands"] = A.ext_p95 - A.ext_p5
    A.to_csv(out / "hands_angle_E06.csv", index=False)
    print("\n=== 2. wrist extension on the moving side (Hands, 3D signed): share of frames defined, p5 / p95 (deg)")
    print(A.groupby("view")[["defined", "defined_other", "ext_p5", "ext_p95", "range_hands"]].median().round(2).to_string())
    ref = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    for r in ref[ref.exercise == "E06"].drop_duplicates("ref_id").itertuples():
        if isinstance(r.hand_path, str) and r.hand_path and resolve(r.hand_path).exists():
            with np.load(resolve(r.pre_path)) as z:
                n = len(z["t"])
            for side in ("left", "right"):
                h, _ = series(resolve(r.kp_path), resolve(r.hand_path), side, n)
                print(f"PT demo {r.ref_id} {side}: defined {np.isfinite(h).mean():.2f}, p5 / p50 / p95 "
                      f"{np.nanpercentile(h, [5, 50, 95]).round(1) if np.isfinite(h).any() else 'n/a'}, "
                      f"reps {count(h, rc)}")

    # 3. synced expert F+L+R groups
    syn = [g for g in A.group_id.unique() if g in G.index and G.loc[g, "cls"] == "synced"
           and G.loc[g, "group"] == "expert" and (A.group_id == g).sum() == 3]
    agree, diffs = [], []
    for g in syn:
        a = A[A.group_id == g].set_index("view")
        rowg = dict(group_id=g)
        for col in ("reps_hands", "reps_posepts", "reps_pipeline"):
            c = a[col].values.astype(float)
            rowg[col + "_within1"] = bool(np.isfinite(c).all() and c.max() - c.min() <= 1) if np.isfinite(c).all() else np.nan
        agree.append(rowg)
        off = {v: G.loc[g, f"offset_{v}_s"] for v in "FLR"}
        for v1, v2 in combinations("FLR", 2):
            if not (np.isfinite(off[v1]) and np.isfinite(off[v2])):
                continue
            for k, name in ((0, "hands"), (1, "pose_points")):
                s1, s2 = sig[a.loc[v1, "video_id"]][k], sig[a.loc[v2, "video_id"]][k]
                ta = np.arange(len(s1)) / FPS - off[v1]          # aligned clock
                s2a = np.interp(ta + off[v2], np.arange(len(s2)) / FPS, s2, left=np.nan, right=np.nan)
                ok = np.isfinite(s1) & np.isfinite(s2a)
                if ok.sum() < 3 * FPS:
                    continue
                diffs.append(dict(group_id=g, pair=f"{v1}-{v2}", source=name, frames=int(ok.sum()),
                                  mean_abs_diff=float(np.mean(np.abs(s1[ok] - s2a[ok]))),
                                  pearson=float(np.corrcoef(s1[ok], s2a[ok])[0, 1])))
    Ag, Df = pd.DataFrame(agree), pd.DataFrame(diffs)
    Ag.to_csv(out / "hands_rep_agreement_E06.csv", index=False)
    Df.to_csv(out / "hands_crossview_E06.csv", index=False)
    print(f"\n=== 3a. cross-view rep agreement, synced expert F+L+R groups ({len(Ag)}): share within +-1")
    print(Ag.drop(columns="group_id").mean().round(3).to_string())
    print("\n=== 3b. same instants across views (aligned clock): median over view pairs and groups")
    print(Df.groupby(["source", "pair"])[["mean_abs_diff", "pearson"]].median().round(3).unstack(0).to_string())
    print(Df.groupby("source")[["mean_abs_diff", "pearson"]].median().round(3).to_string())

    # 3c. fairer: only frames where both views detected the hand, 0.3 s median filter, Spearman; angle variants
    rows = []
    hand_dirs = [("crops_960", None)] + ([("crops_fullres", resolve(args.compare_dir))] if args.compare_dir else [])
    for crops, hdir in hand_dirs:
        for g in syn:
            vids = A[A.group_id == g].set_index("view").video_id
            if hdir is not None and not all((hdir / f"{v}.npz").exists() for v in vids):
                continue
            side = V.loc[vids.iloc[0], "side"]
            S = {v: variants(idx.set_index("video_id").loc[vids[v]], side, int(V.loc[vids[v], "n_frames_30"]), hdir)
                 for v in "FLR"}
            off = {v: G.loc[g, f"offset_{v}_s"] for v in "FLR"}
            for v1, v2 in combinations("FLR", 2):
                ta = np.arange(len(S[v1]["mask"])) / FPS - off[v1]
                m2 = np.interp(ta + off[v2], np.arange(len(S[v2]["mask"])) / FPS, S[v2]["mask"], left=0, right=0)
                both = (S[v1]["mask"] > 0.99) & (m2 > 0.99)
                for k in VARIANTS:
                    s1, s2 = S[v1][k], S[v2][k]
                    s2a = np.interp(ta + off[v2], np.arange(len(s2)) / FPS, s2, left=np.nan, right=np.nan)
                    ok = np.isfinite(s1) & np.isfinite(s2a) & both
                    if ok.sum() > 3 * FPS:
                        rows.append(dict(crops=crops, group_id=g, pair=f"{v1}-{v2}", signal=k, frames=int(ok.sum()),
                                         spearman=float(spearmanr(s1[ok], s2a[ok]).statistic),
                                         mean_abs_diff=float(np.mean(np.abs(s1[ok] - s2a[ok])))))
    Dv = pd.DataFrame(rows)
    Dv.to_csv(out / "hands_crossview_detected_E06.csv", index=False)
    print("\n=== 3c. frames with the hand detected in both views, 0.3 s median: Spearman / mean |diff| across views "
          "(median over view pairs and groups)")
    print(Dv.groupby(["crops", "signal"]).agg(groups=("group_id", "nunique"), spearman=("spearman", "median"),
                                              mean_abs_diff=("mean_abs_diff", "median"),
                                              frames=("frames", "median")).round(3).to_string())

    # 3d. each side camera sees its near hand only (L -> the body's right hand, R -> the left; P5 visibility), so compare
    # the front view with the side view that sees the same hand
    rows = []
    for g in syn:
        vids = A[A.group_id == g].set_index("view").video_id
        off = {v: G.loc[g, f"offset_{v}_s"] for v in "FLR"}
        for side, v2 in (("right", "L"), ("left", "R")):
            S = {v: variants(idx.set_index("video_id").loc[vids[v]], side, int(V.loc[vids[v], "n_frames_30"]))
                 for v in ("F", v2)}
            ta = np.arange(len(S["F"]["mask"])) / FPS - off["F"]
            m2 = np.interp(ta + off[v2], np.arange(len(S[v2]["mask"])) / FPS, S[v2]["mask"], left=0, right=0)
            both = (S["F"]["mask"] > 0.99) & (m2 > 0.99)
            for k in VARIANTS:
                s1, s2 = S["F"][k], S[v2][k]
                s2a = np.interp(ta + off[v2], np.arange(len(s2)) / FPS, s2, left=np.nan, right=np.nan)
                ok = np.isfinite(s1) & np.isfinite(s2a) & both
                if ok.sum() > 3 * FPS:
                    rows.append(dict(group_id=g, pair=f"F-{v2} ({side} hand)", signal=k, frames=int(ok.sum()),
                                     spearman=float(spearmanr(s1[ok], s2a[ok]).statistic),
                                     mean_abs_diff=float(np.mean(np.abs(s1[ok] - s2a[ok])))))
    Dn = pd.DataFrame(rows)
    Dn.to_csv(out / "hands_crossview_nearhand_E06.csv", index=False)
    print("\n=== 3d. as 3c, the front view vs the side view that sees the same hand (median over groups)")
    print(Dn.groupby(["signal", "pair"])[["spearman", "mean_abs_diff", "frames"]].median().round(3).to_string())

    # 4. renders: the group with the median hand detection, and the one with the worst
    if len(syn):
        det = A[A.group_id.isin(syn)].groupby("group_id").defined.mean().sort_values()
        for tag, g in (("median", det.index[len(det) // 2]), ("worst", det.index[0])):
            render(g, idx, cfg, chk / f"hands_E06_{tag}_{g}.png")
        print(f"\nrenders: hands_E06_median_{det.index[len(det) // 2]}.png, hands_E06_worst_{det.index[0]}.png")


if __name__ == "__main__":
    main()
