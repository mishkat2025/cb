"""X7 check: the PT demo keypoints against the PT video, in memory (no frame is written anywhere).

For the primary PT demo of each exercise and each of its reps, at rest (rep start), peak and end (30 fps grid frames
of the reps table), plus the video's first and last frame: the frame at that time is decoded (Phase 1 decoder) and
MediaPipe Pose (IMAGE mode, the Phase 1 model) re-run on it; the stored keypoints (raw and preprocessed, mapped back to
pixels) are compared with the re-detection (median distance over 8 upper-body joints, % of the torso length). Also
printed: the 2D upper-arm angle from the trunk (stored and re-detected), the 3D shoulder and forearm elevations, the
template's first / last frame, and the display window the demo now uses (rest -> peak -> rest, X7).

usage: python scripts/phase6/verify_pt_keypoints.py [E01 E02 E03 E04]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

from cvdc.assess.angles import ASSESS_ANGLES  # noqa: E402
from cvdc.assess.module3 import assess_config  # noqa: E402
from cvdc.assess.template import load_library  # noqa: E402
from cvdc.config import data_path, data_root, exercise_cfg, load_config, resolve  # noqa: E402
from cvdc.data.reps import video_info  # noqa: E402
from cvdc.data.skeleton import J  # noqa: E402
from cvdc.data.video import iter_frames  # noqa: E402
from demo.reference import ReferencePlayer  # noqa: E402

UPPER = [J[n] for n in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                        "left_hip", "right_hip")]


def arm2d(xy, side):
    """2D angle (deg) of the upper arm (shoulder -> elbow) from the trunk's down direction (mid-shoulder -> mid-hip)."""
    down = (xy[J["left_hip"]] + xy[J["right_hip"]]) / 2 - (xy[J["left_shoulder"]] + xy[J["right_shoulder"]]) / 2
    u = xy[J[f"{side}_elbow"]] - xy[J[f"{side}_shoulder"]]
    return float(np.degrees(np.arccos(np.clip(u @ down / (np.linalg.norm(u) * np.linalg.norm(down)), -1, 1))))


def main():
    cfg = load_config(REPO / "configs" / "per_exercise.yaml")
    ref_idx = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    rr = pd.read_csv(data_path(cfg, "reps_ref_csv"))
    pc = cfg["pose"]
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(resolve(pc["model"]))), running_mode=vision.RunningMode.IMAGE,
        num_poses=1, output_segmentation_masks=False))
    for ex in sys.argv[1:] or ["E01", "E02", "E03", "E04"]:
        row = ref_idx[(ref_idx.exercise == ex) & (ref_idx.role == "primary")].iloc[0]
        kp, pre = np.load(resolve(row.kp_path)), np.load(resolve(row.pre_path))
        H, W = [int(v) for v in pre["frame_hw"]]
        reps = rr[(rr.ref_id == row.ref_id) & (rr.exercise == ex)]
        side = reps.side.iloc[0]
        ang, info = video_info({k: pre[k] for k in pre.files}, exercise_cfg(cfg, ex))
        sig = ang[side]
        print(f"\n===== {ex} {row.file} ({W}x{H}, moving side {side}; rep signal {info['primary_angle']})")
        want = {0: "video start", len(pre["t"]) - 1: "video end"}
        for r in reps.itertuples():
            for tag, f in (("rest", r.start), ("peak", r.peak), ("end", r.end)):
                want[int(f)] = f"rep {r.rep} {tag}"
        frames = {}
        for t, rgb, _ in iter_frames(data_root(cfg) / row.rel_path, pc["max_side"]):
            for f in want:
                if f not in frames or abs(t - pre["t"][f]) < abs(frames[f][0] - pre["t"][f]):
                    frames[f] = (t, rgb)
        w3 = pre["world"].astype(float)
        elev = ASSESS_ANGLES["shoulder_elevation"](w3, side)
        fore = ASSESS_ANGLES["forearm_elevation"](w3, side)
        tp = float(pre["torso_px"])
        for f in sorted(want):
            t, rgb = frames[f]
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if not r.pose_landmarks:
                print(f"  {want[f]:<12} frame {f:3d}: no re-detection")
                continue
            redet = np.array([[p.x * W, p.y * H] for p in r.pose_landmarks[0]])
            k = int(np.argmin(np.abs(kp["t"] - t)))
            raw = kp["lm2d"][k, :, :2].astype(float) * [W, H]
            prepx = pre["root_px"][f] + pre["xy"][f].astype(float) * tp
            e_raw = np.median(np.linalg.norm(raw[UPPER] - redet[UPPER], axis=1)) / tp
            e_pre = np.median(np.linalg.norm(prepx[UPPER] - redet[UPPER], axis=1)) / tp
            print(f"  {want[f]:<12} frame {f:3d} ({pre['t'][f]:5.2f} s) | signal {sig[f]:8.2f} | arm 2D stored "
                  f"{arm2d(pre['xy'][f].astype(float), side):5.1f}, re-detected {arm2d(redet, side):5.1f} | 3D elev "
                  f"{elev[f]:5.1f}, forearm {fore[f]:5.1f} | error raw {100 * e_raw:3.1f}%, preprocessed "
                  f"{100 * e_pre:3.1f}%")
        ac = assess_config(cfg, ex)
        ref, _, _ = load_library(resolve(cfg["templates"]["library_dir"]) / ex)
        pl = ReferencePlayer(cfg, ex, ref, ac)
        print(f"  template (scoring): {ref['n']} frames, main {ref['main']} first / last "
              f"{ref['curves'][ref['main']][0]:.1f} / {ref['curves'][ref['main']][-1]:.1f}")
        s, kr = pl.seq, pl.rest_k
        print(f"  display (X7, X8): PT frames {s[0]}..{s[-1]}, rest pose {s[kr]} (rep {pl.rep[0]}-{pl.rep[2]}, peak "
              f"{pl.rep[1]}); arm 2D at rest / peak / end {arm2d(pl.xy[kr], side):.1f} / "
              f"{arm2d(pl.xy[pl.peak], side):.1f} / {arm2d(pl.xy[-1], side):.1f}; 3D shoulder elevation "
              f"{elev[s[kr]]:.1f} / {elev[s[pl.peak]]:.1f} / {elev[s[-1]]:.1f}; forearm {fore[s[kr]]:.1f} / "
              f"{fore[s[pl.peak]]:.1f} / {fore[s[-1]]:.1f}" + (f"; {pl.rest_label}" if pl.rest_label else ""))
    det.close()


if __name__ == "__main__":
    main()
