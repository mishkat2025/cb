"""X7 check: the live reference display (demo/reference.py LiveReference) replayed on the cached Phase 1 keypoints of
dataset videos, frame by frame exactly like demo/live.py (session update every 0.5 s), without video or MediaPipe.

Per video: the PT display window (rest -> peak -> rest), and against the batch own-view reps of the final pass:
  coverage   share of the frames of the batch reps (start..end, 30 fps grid -> video frames) shown as "in rep"
  rest ok    share of the frames >= 0.5 s away from every batch rep shown "at rest" (reference at rest, no flags)
  gap rest   share of the frames outside every batch rep (before, between, after) shown "at rest"
  missed     batch reps with no "in rep" frame
  extra      "in rep" episodes that overlap no batch rep
  compared   share of the batch-rep frames with a live comparison (in a rep and the PT frame inside the template)
  align err  median |live template frame - Module III's batch DTW template frame| over the compared frames, as % of
             the template (the scoring alignment, which also sees the rest of the rep)
  target gap the same in degrees of the main channel's target (what it changes in the displayed comparison)
Keypoint-derived numbers only.

usage: python scripts/phase6/check_reference_display.py [VIDEO_ID ...]   (default: the E01-E04 demo videos)
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.skeleton import J  # noqa: E402
from demo.pipeline import OnlineSession  # noqa: E402
from demo.reference import LiveReference  # noqa: E402

DEFAULT = ["E01_P03_AF_VFL_GM", "E02_P07_AF_VML_GM", "E03_P05_AF_VFL_GM", "E04_P10_AF_VFL_GF"]


def arm2d(xy, side):
    down = (xy[J["left_hip"]] + xy[J["right_hip"]]) / 2 - (xy[J["left_shoulder"]] + xy[J["right_shoulder"]]) / 2
    u = xy[J[f"{side}_elbow"]] - xy[J[f"{side}_shoulder"]]
    return float(np.degrees(np.arccos(np.clip(u @ down / (np.linalg.norm(u) * np.linalg.norm(down)), -1, 1))))


def replay(cfg, video_id, idx):
    r = idx.loc[video_id]
    with np.load(resolve(r.kp_path)) as z:
        t, lm, w, det, hw = z["t"] - z["t"][0], z["lm2d"], z["world"], z["detected"], tuple(z["frame_hw"])
    pose = types.SimpleNamespace(t=[], world=[], lm2d=[], detected=[], frame_hw=hw)
    pose.arrays = lambda: (np.asarray(pose.t), np.stack(pose.lm2d), np.stack(pose.world), np.asarray(pose.detected),
                           np.full((len(pose.t), 2, 21, 3), np.nan, np.float32))
    sess = OnlineSession(cfg, r.exercise, r.view)
    live = LiveReference(cfg, sess, r.view)
    out, last = [], -1.0
    for i in range(len(t)):
        pose.t.append(float(t[i]))
        pose.world.append(w[i] if det[i] else np.full((33, 3), np.nan, np.float32))
        pose.lm2d.append(lm[i])
        pose.detected.append(bool(det[i]))
        if t[i] - last >= 0.5:
            last = t[i]
            sess.update(pose)
            live.after_update(sess, pose)
        out.append(live.frame(pose))
    sess.update(pose, final=True)
    return sess, live, np.asarray(t), out


def main():
    cfg = load_config(REPO / "configs" / "per_exercise.yaml")
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"])).set_index("video_id")
    rows = []
    for vid in sys.argv[1:] or DEFAULT:
        sess, live, t, out = replay(cfg, vid, idx)
        pl, fps = live.player, cfg["preprocess"]["fps"]
        inr = np.array([o["in_rep"] for o in out])
        js = np.array([o["j"] for o in out])
        flagged = np.array([bool(o["diffs"]) for o in out])
        in_any = np.zeros(len(t), bool)
        near = np.zeros(len(t), bool)
        missed, err, derr, n_cmp, n_rep = 0, [], [], 0, 0
        tgt = live.targets[sess.ref["main"]]
        for rp, dbg in zip(sess.reps, sess.debug):
            a, b = rp["start"] / fps, rp["end"] / fps
            m = (t >= a) & (t <= b)
            in_any |= m
            near |= (t >= a - 0.5) & (t <= b + 0.5)
            missed += int(not inr[m].any())
            path = dbg["aligned"]["path"]
            for i in np.flatnonzero(m):
                g = int(round(t[i] * fps)) - rp["start"]
                jb = path[path[:, 0] == g, 1]
                n_rep += 1
                if js[i] >= 0 and len(jb):
                    n_cmp += 1
                    err.append(abs(js[i] - np.median(jb)) / pl.n_template)
                    derr.append(abs(tgt[js[i]] - tgt[int(np.median(jb))]))
        ep = np.flatnonzero(np.diff(np.r_[0, inr.astype(int), 0]) == 1)
        ee = np.flatnonzero(np.diff(np.r_[0, inr.astype(int), 0]) == -1)
        extra = sum(not in_any[s:e].any() for s, e in zip(ep, ee))
        rest = ~near
        side = pl.side
        rows.append(dict(
            video=vid, reps=len(sess.reps), episodes=len(ep), coverage=round(float(inr[in_any].mean()), 3),
            rest_ok=round(float((~inr[rest]).mean()), 3) if rest.any() else np.nan,
            gap_rest=round(float((~inr[~in_any]).mean()), 3) if (~in_any).any() else np.nan,
            flags_at_rest=int(flagged[~inr].sum()), missed=missed, extra=int(extra),
            compared=round(n_cmp / max(n_rep, 1), 3),
            align_err_pct=round(100 * float(np.median(err)), 1) if err else np.nan,
            target_gap_deg=round(float(np.nanmedian(derr)), 1) if derr else np.nan,
            pt_frames=f"{pl.seq[0]}..{pl.seq[-1]} rest {pl.seq[pl.rest_k]} (rep {pl.rep[0]}-{pl.rep[2]}, "
                      f"peak {pl.rep[1]})",
            pt_arm2d_rest_peak_end=f"{arm2d(pl.xy[pl.rest_k], side):.0f}/{arm2d(pl.xy[pl.peak], side):.0f}/"
                                   f"{arm2d(pl.xy[-1], side):.0f}"))
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
