"""P6 check: which primary-angle signal and thresholds segment reps consistently?

Expert F/L/R groups are simultaneous recordings of one performance (PROGRESS §3.5), so every view must
give the same rep count. Criteria per candidate: share of expert multi-view groups whose views agree
exactly / within 1 rep, share of expert videos with 0 reps, and Spearman of the count with 10 x PF4
(PF4 = "Repetition: minimum 10 times", 0.1 steps; mapping to a count not confirmed).
usage: python scripts/phase1/rep_signal_check.py E02 [E01 ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvdc.assess.angles import arm_frontal_angle, shoulder_elevation, wrist_path_pc1  # noqa: E402
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.reps import moving_side, segment_reps  # noqa: E402
from cvdc.data.skeleton import J  # noqa: E402

def elevation_2d_height(xy, side):
    """View-robust 2D proxy of arm elevation: arccos(-wrist height above the shoulder / arm length)."""
    d = xy[:, J[f"{side}_wrist"]] - xy[:, J[f"{side}_shoulder"]]
    L = np.nanpercentile(np.linalg.norm(d, axis=1), 99)
    return np.degrees(np.arccos(np.clip(d[:, 1] / L, -1, 1)))  # image y points down: hanging arm -> 0 deg


SIGNALS = {  # name -> (source, function)
    "elevation": ("world", lambda w, s: shoulder_elevation(w, s)),
    "frontal_wrist": ("world", lambda w, s: arm_frontal_angle(w, s, "wrist")),
    "elevation_2d_height": ("xy", elevation_2d_height),
    "wrist_path_pc1": ("world", wrist_path_pc1),
}
GRID = {
    "E01": [(n, "peak", p, a) for n in ("elevation", "elevation_2d_height") for p, a in [(30, 30), (40, 40), (60, 60)]],
    "E02": [("elevation", "peak", p, a) for p, a in [(8, 8), (10, 10), (12, 12), (15, 15), (20, 20)]]
    + [("frontal_wrist", "valley", p, a) for p, a in [(10, 10), (15, 15), (20, 20), (25, 25)]]
    + [("elevation_2d_height", "peak", p, a) for p, a in [(10, 10), (15, 15), (20, 20)]]
    + [("wrist_path_pc1", "peak", p, a) for p, a in [(0.15, 0.15), (0.25, 0.25), (0.35, 0.35)]],
}


def main():
    cfg = load_config()
    fps = cfg["preprocess"]["fps"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    rows = []
    for ex in sys.argv[1:]:
        g = idx[(idx.exercise == ex) & idx.pre_path.map(lambda p: resolve(p).exists())]
        sig = {}
        for r in g.itertuples():
            with np.load(resolve(r.pre_path)) as z:
                src = {"world": z["world"].astype(float), "xy": z["xy"].astype(float)}
            sig[r.video_id] = {n: {s: f(src[k], s) for s in ("right", "left")} for n, (k, f) in SIGNALS.items()}
        for name, pol, prom, amp in GRID[ex]:
            counts = {}
            for v, d in sig.items():
                a = d[name]
                side, _ = moving_side({s: (-x if pol == "valley" else x) for s, x in a.items()})
                counts[v] = len(segment_reps(a[side], fps, **{**cfg["reps"], "prominence": prom,
                                                              "min_amplitude": amp, "polarity": pol}))
            c = g.assign(n=g.video_id.map(counts))
            e = c[c.group == "expert"]
            mv = e[e.n_views_group >= 2].groupby("group_id").n.agg(["min", "max"])
            rho = spearmanr(e.n, 10 * e.PF4).statistic
            rows.append(dict(exercise=ex, signal=name, polarity=pol, prominence=prom, min_amp=amp,
                             expert_median=e.n.median(), expert_zero=(e.n == 0).mean(),
                             groups=len(mv), views_equal=(mv["min"] == mv["max"]).mean(),
                             views_within1=((mv["max"] - mv["min"]) <= 1).mean(),
                             rho_PF4=rho, nonexpert_median=c[c.group == "non-expert"].n.median()))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows).round(3)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    R.to_csv(resolve(cfg["outputs"]["phase1"]) / f"rep_signal_check_{'_'.join(sys.argv[1:])}.csv", index=False)


if __name__ == "__main__":
    main()
