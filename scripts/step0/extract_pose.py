"""Run MediaPipe PoseLandmarker (Tasks API, VIDEO mode) on a list of videos.

Saves keypoints only (no frames): image landmarks (T,33,4: x,y,z,visibility),
world landmarks (T,33,3), timestamps and a detected mask, one .npz per video.
usage: extract_pose.py <model.task> <out_dir> <max_side> <workers> <video> [<video> ...]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def run(args):
    model, out_dir, max_side, path = args
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    out = Path(out_dir) / (Path(path).stem + ".npz")
    if out.exists():
        return path, "cached", 0.0
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    t0 = time.time()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img, world, ts, det = [], [], [], []
    shape = None
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            shape = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_ms = int(round(i * 1000.0 / fps))
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
            if r.pose_landmarks:
                img.append([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]])
                world.append([[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]])
                det.append(True)
            else:
                img.append(np.full((33, 4), np.nan))
                world.append(np.full((33, 3), np.nan))
                det.append(False)
            ts.append(i / fps)
            i += 1
    cap.release()
    np.savez_compressed(out, img=np.asarray(img, np.float32), world=np.asarray(world, np.float32),
                        t=np.asarray(ts), detected=np.asarray(det), fps=fps,
                        frame_hw=np.asarray(shape if shape else (0, 0)))
    return path, f"{len(ts)} frames, det {np.mean(det) if det else 0:.1%}", time.time() - t0


if __name__ == "__main__":
    model, out_dir, max_side, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    vids = sys.argv[5:]
    if len(vids) == 1 and vids[0].endswith(".txt"):
        vids = [l.strip() for l in open(vids[0], encoding="utf-8") if l.strip()]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for path, msg, dt in ex.map(run, [(model, out_dir, max_side, v) for v in vids]):
            print(f"{Path(path).name:32s} {msg:28s} {dt:6.1f}s", flush=True)
    print(f"total {time.time() - t0:.1f}s for {len(vids)} videos")
