import sys, re, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
KP, KPR, SC = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def ang(u, v):
    c = (u*v).sum(-1) / (np.linalg.norm(u, axis=-1)*np.linalg.norm(v, axis=-1) + 1e-9); return np.degrees(np.arccos(np.clip(c, -1, 1)))
def arm_series(path, side):
    w = np.load(path)["world"].astype(float); w = w[np.isfinite(w).all((1, 2))]
    sh, el, wr = {"L": (11, 13, 15), "R": (12, 14, 16)}[side]
    down = (w[:, 23] + w[:, 24]) / 2 - (w[:, 11] + w[:, 12]) / 2
    ua, fa = w[:, el] - w[:, sh], w[:, wr] - w[:, el]
    if side == "L": pass
    return np.c_[ang(ua, down), 180 - ang(-ua, fa), 90 - ang(fa, np.array([0, -1.0, 0]))]  # abd, elbow flex, forearm elev
sc = pd.read_csv(SC); score = dict(zip(sc.file.str.replace(".mp4", "", regex=False).str.upper(), sc.avg_round))
def moving_side(path):
    w = np.load(path)["world"].astype(float); w = w[np.isfinite(w).all((1, 2))]
    m = {s: np.linalg.norm(np.std(w[:, wr] - w[:, sh], 0)) for s, (sh, wr) in {"L": (11, 15), "R": (12, 16)}.items()}
    return max(m, key=m.get)
pool = {}
for f in sorted(KP.glob("E03_*_AF_*.npz")):
    if score.get(f.stem.upper(), 0) >= 80:
        pool[f.stem] = arm_series(f, moving_side(f))
P = np.vstack(list(pool.values())); tree = cKDTree(P)
print(f"dataset pool: {len(pool)} front-view E03 videos scoring >= 80, {len(P)} frames (moving arm, mirrored to one side)")
print("dataset percentiles [abd, elbow_flex, forearm_elev] p5/p50/p95:", np.percentile(P, [5, 50, 95], 0).round(0).tolist())
for f in sorted(KPR.glob("*.npz")):
    if not re.search(r"later|lateral", f.stem, re.I): continue
    for side in "RL":
        X = arm_series(f, side)
        d_ref_to_data = tree.query(X)[0].mean(); d_data_to_ref = cKDTree(X).query(P)[0].mean()
        print(f"{f.stem:22s} arm {side}: p5/p50/p95 {np.percentile(X, [5, 50, 95], 0).round(0).tolist()}  "
              f"chamfer ref->data {d_ref_to_data:5.1f}  data->ref {d_data_to_ref:5.1f}  mean {0.5*(d_ref_to_data+d_data_to_ref):5.1f} deg")
