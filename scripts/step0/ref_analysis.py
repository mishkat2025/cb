"""Characterise the PT reference videos from keypoints only: posture, apparent view, moving limb, reps."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

KP = Path(sys.argv[1])
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
NAMES = {L_WR: "L_wrist", R_WR: "R_wrist", L_EL: "L_elbow", R_EL: "R_elbow", L_KN: "L_knee", R_KN: "R_knee",
         L_AN: "L_ankle", R_AN: "R_ankle"}
rows = []
for f in sorted(KP.glob("*.npz")):
    d = np.load(f)
    img, w3 = d["img"], d["world"]
    h, w = d["frame_hw"]
    xy = img[..., :2] * np.array([w, h])
    msh, mhip = (xy[:, L_SH] + xy[:, R_SH]) / 2, (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    v = msh - mhip
    torso = np.nanmedian(np.linalg.norm(v, axis=1))
    tilt = np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))  # 0 = upright, 90 = horizontal
    sh_ratio = np.nanmedian(np.linalg.norm(xy[:, L_SH] - xy[:, R_SH], axis=1)) / torso
    # facing direction from world landmarks: z of shoulders (MediaPipe z: smaller = closer to camera)
    dz = np.nanmedian(w3[:, L_SH, 2] - w3[:, R_SH, 2])
    # most-moving distal joint (relative to mid-shoulder / mid-hip), torso-normalised
    rel = {j: (xy[:, j] - (msh if j in (L_WR, R_WR, L_EL, R_EL) else mhip)) / torso for j in NAMES}
    mov = {NAMES[j]: float(np.nanstd(r, axis=0).sum()) for j, r in rel.items()}
    top = sorted(mov.items(), key=lambda kv: -kv[1])[:2]
    j = [k for k, n in NAMES.items() if n == top[0][0]][0]
    r = rel[j] - np.nanmean(rel[j], 0)
    _, _, vt = np.linalg.svd(np.nan_to_num(r), full_matrices=False)
    s = np.nan_to_num(r) @ vt[0]
    fps = float(d["fps"])
    s = savgol_filter(s, 9, 2) if len(s) > 11 else s
    z = (s - s.mean()) / (s.std() + 1e-9)
    n_pk = max(len(find_peaks(z, prominence=1.0, distance=int(0.8 * fps))[0]),
               len(find_peaks(-z, prominence=1.0, distance=int(0.8 * fps))[0]))
    rows.append(dict(video=f.stem, dur_s=round(len(xy) / fps, 1), fps=round(fps, 1), det=float(d["detected"].mean()),
                     torso_tilt_deg=round(float(np.nanmedian(tilt)), 0), shoulder_ratio=round(float(sh_ratio), 2),
                     L_minus_R_shoulder_depth=round(float(dz), 2),
                     moving=f"{top[0][0]} ({top[0][1]:.2f}), {top[1][0]} ({top[1][1]:.2f})", approx_reps=n_pk))
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).to_string(index=False))
