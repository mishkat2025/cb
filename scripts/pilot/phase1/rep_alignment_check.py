"""P6 check at rep level: in audio-synced groups (lags known to ~tens of ms), do rep peaks coincide across views
after alignment? Per view pair: share of peaks with a peak of the other view within +-0.4 s (matched / max count),
for candidate rep signals. A signal whose sign flips with the view puts peaks half a cycle apart.
usage: python scripts/pilot/phase1/rep_alignment_check.py E02 [E01]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cvdc.assess.angles import shoulder_elevation, trunk_frame, wrist_path_pc1  # noqa: E402
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.data.reps import moving_side, segment_reps  # noqa: E402
from cvdc.data.skeleton import J  # noqa: E402


def wrist_rel(w, side):
    lat, up, fwd = trunk_frame(w)
    d = w[:, J[f"{side}_wrist"]] - w[:, J[f"{side}_shoulder"]]
    out = lat if side == "right" else -lat
    return np.stack([np.sum(d * out, -1), np.sum(d * up, -1), np.sum(d * fwd, -1)], -1)  # outward, up, forward


def pc1_tail(w, side):
    """Earlier orientation rule (kept for the record): the side of the longer tail is the movement."""
    s = wrist_path_pc1(w, side)
    return s if np.nanmean(s) >= np.nanmedian(s) else -s


SIGNALS = {
    "E02": {"pc1_tail": (pc1_tail, 0.25), "pc1_medial": (wrist_path_pc1, 0.25),
            "medial_disp": (lambda w, s: -wrist_rel(w, s)[:, 0], 0.15)},
    "E01": {"elevation": (shoulder_elevation, 30)},
}


def main():
    cfg = load_config()
    fps = cfg["preprocess"]["fps"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    G = pd.read_csv(resolve(cfg["sync"]["report_csv"]))
    rows = []
    for ex in sys.argv[1:]:
        groups = G[(G.exercise == ex) & (G.method == "audio")]
        for name, (fn, prom) in SIGNALS[ex].items():
            stats = []
            for g in groups.itertuples():
                vids = idx[idx.group_id == g.group_id].set_index("view")
                peaks = {}
                for v, r in vids.iterrows():
                    p = resolve(r.pre_path)
                    if not p.exists():
                        continue
                    w = np.load(p)["world"].astype(float)
                    sig = {s: fn(w, s) for s in ("right", "left")}
                    side, _ = moving_side(sig)
                    reps = segment_reps(sig[side], fps, **{**cfg["reps"], "prominence": prom, "min_amplitude": prom})
                    peaks[v] = np.array([rp["peak"] / fps - getattr(g, f"offset_{v}_s") for rp in reps])
                for a, b in (("F", "L"), ("F", "R"), ("L", "R")):
                    if a in peaks and b in peaks and len(peaks[a]) and len(peaks[b]):
                        m = sum(np.min(np.abs(peaks[b] - t)) <= 0.4 for t in peaks[a])
                        stats.append(dict(pair=f"{a}-{b}", match=m / max(len(peaks[a]), len(peaks[b])),
                                          count_diff=abs(len(peaks[a]) - len(peaks[b]))))
            S = pd.DataFrame(stats)
            for pair, s in S.groupby("pair"):
                rows.append(dict(exercise=ex, signal=name, pair=pair, groups=len(s), peak_match=s.match.mean(),
                                 count_within1=(s.count_diff <= 1).mean()))
    R = pd.DataFrame(rows).round(3)
    pd.set_option("display.width", 200)
    print(R.pivot_table(index=["exercise", "signal"], columns="pair", values=["peak_match", "count_within1"]).round(3))
    R.to_csv(resolve(cfg["outputs"]["phase1"]) / f"rep_alignment_check_{'_'.join(sys.argv[1:])}.csv", index=False)


if __name__ == "__main__":
    main()
