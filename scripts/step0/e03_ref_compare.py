"""Compare the three E03 (lateral rotation) PT reference versions with E03 dataset executions,
using MediaPipe 3D world landmarks (metres, hip-centred, y down) so the view does not matter."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

KP, KP_REF, SCORES = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SH, EL, WR, HIP = {"L": 11, "R": 12}, {"L": 13, "R": 14}, {"L": 15, "R": 16}, (23, 24)


def ang(u, v):
    c = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def feats(path):
    d = np.load(path)
    w, fps = d["world"].astype(float), float(d["fps"])
    ok = np.isfinite(w).all((1, 2))
    w = w[ok]
    mid_sh = (w[:, 11] + w[:, 12]) / 2
    mid_hip = (w[:, HIP[0]] + w[:, HIP[1]]) / 2
    down = mid_hip - mid_sh
    out = {}
    motion = {s: np.linalg.norm(np.std(w[:, WR[s]] - w[:, SH[s]], 0)) for s in "LR"}
    out["wrist_motion_L_m"], out["wrist_motion_R_m"] = motion["L"], motion["R"]
    s = max(motion, key=motion.get)
    out["moving"] = "both" if min(motion.values()) > 0.6 * max(motion.values()) else s
    ua = w[:, EL[s]] - w[:, SH[s]]
    fa = w[:, WR[s]] - w[:, EL[s]]
    abd = ang(ua, down)                       # 0 = arm at side, 90 = horizontal
    elbow = 180 - ang(-ua, fa)                # 0 = straight, 90 = right angle
    fa_elev = 90 - ang(fa, np.array([0, -1.0, 0]))  # +90 = forearm pointing up, 0 = horizontal
    fa_az = np.degrees(np.arctan2(fa[:, 0], -fa[:, 2]))  # horizontal-plane direction of forearm
    for name, x in (("abd", abd), ("elbow", elbow), ("fa_elev", fa_elev), ("fa_az", fa_az)):
        out[f"{name}_med"] = np.median(x)
        out[f"{name}_rom"] = np.percentile(x, 95) - np.percentile(x, 5)
    main = fa_elev if out["fa_elev_rom"] >= out["fa_az_rom"] else fa_az
    out["main_signal"] = "forearm elevation" if out["fa_elev_rom"] >= out["fa_az_rom"] else "forearm azimuth"
    z = savgol_filter(main, 9, 2) if len(main) > 11 else main
    z = (z - z.mean()) / (z.std() + 1e-9)
    pk, _ = find_peaks(z, prominence=1.0, distance=int(0.8 * fps))
    out["reps"] = len(pk)
    out["period_s"] = np.median(np.diff(pk)) / fps if len(pk) > 1 else np.nan
    return out


rows = []
for f in sorted(KP_REF.glob("*.npz")):
    if re.search(r"later|lateral", f.stem, re.I):
        rows.append(dict(src="REF", video=f.stem, **feats(f)))
sc = pd.read_csv(SCORES)
sc["video"] = sc.file.str.replace(".mp4", "", regex=False)
score = dict(zip(sc.video.str.upper(), sc.avg_round))
for f in sorted(KP.glob("E03_*.npz")):
    rows.append(dict(src="E03", video=f.stem, score=score.get(f.stem.upper()), **feats(f)))
df = pd.DataFrame(rows)
pd.set_option("display.width", 230)
cols = ["src", "video", "score", "moving", "wrist_motion_L_m", "wrist_motion_R_m", "abd_med", "abd_rom", "elbow_med",
        "fa_elev_med", "fa_elev_rom", "fa_az_med", "fa_az_rom", "main_signal", "reps", "period_s"]
print(df[cols].round(2).to_string(index=False))
e = df[df.src == "E03"]
print("\nE03 dataset sample summary (n=%d videos):" % len(e))
print("  moving arm:", e.moving.value_counts().to_dict(), "  main signal:", e.main_signal.value_counts().to_dict())
num = ["abd_med", "abd_rom", "elbow_med", "fa_elev_med", "fa_elev_rom", "fa_az_rom", "period_s", "reps"]
print(e[num].describe().loc[["25%", "50%", "75%"]].round(1).to_string())
hi = e[e.score >= 80]
print("  high-scoring (>=80) medians:", hi[num].median().round(1).to_dict())
df.to_csv(KP.parent / "e03_ref_compare.csv", index=False)
