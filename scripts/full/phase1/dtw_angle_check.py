"""Do the per-exercise DTW angles (user, 2026-10-20; A5) tell matching reps apart better than the pilot's upper-limb
angles? Test on synced expert groups, where the matching rep is known (shared rep_uid, P6b): for a rep in one view, the
DTW cost to the same rep in another view vs to the neighbouring rep (i + 1) in that view. A useful angle set gives the
same rep the lower cost. Also the pilot's known-warp test (A5): the other view's rep replayed slow-then-fast (t^1.5),
re-aligned by DTW on the angles; error = mean |recovered - true| position in frames (vs the error without DTW).
Rep-matched pairs (not-synced groups) are paired by index and aligned by DTW, so the angles
decide how well their time axes are matched. Reported per exercise: the share of rep pairs where the same rep is
cheaper, and the median costs (mean per-step Euclidean distance over the angle channels, degrees).

usage: python scripts/full/phase1/dtw_angle_check.py [--config configs/data_full.yaml] [--groups 40]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cvdc.assess.dtw import dtw, sample_at, warp_positions  # noqa: E402
from cvdc.config import data_path, load_config, resolve  # noqa: E402
from cvdc.data.pairs import angle_features, dtw_angle_names  # noqa: E402

WARP = 1.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_full.yaml")
    ap.add_argument("--groups", type=int, default=40, help="synced groups sampled per exercise (seed 0)")
    ap.add_argument("--extra-set", action="append", help="another angle set to test, EX=angle,angle,... (repeatable)")
    ap.add_argument("--only", nargs="*", help="exercises")
    a = ap.parse_args()
    cfg = load_config(a.config)
    reps = pd.read_csv(data_path(cfg, "reps_csv"))
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"])).set_index("video_id")
    G = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    default = cfg["cvdc"]["pairs"]["dtw_angles"]
    rng = np.random.default_rng(0)
    rows = []
    for ex in sorted(set(reps.exercise) & set(a.only or reps.exercise)):
        sets = {"pilot_upper_limb": default, "per_exercise": dtw_angle_names(cfg, ex)}
        if sets["per_exercise"] == default:
            continue
        for spec in a.extra_set or []:          # EX=angle,angle,...
            e, angs = spec.split("=")
            if e == ex:
                sets["+".join(angs.split(","))] = angs.split(",")
        r = reps[(reps.exercise == ex) & reps.rep_source.isin(["best_view", "transfer"]) & (reps.group == "expert")]
        gids = [g for g in r.group_id.unique() if G.loc[g, "cls"] == "synced"]
        for gid in rng.choice(gids, min(a.groups, len(gids)), replace=False):
            rg = r[r.group_id == gid]
            views = sorted(rg.view.unique())
            if len(views) < 2:
                continue
            world = {}
            for v in views:
                with np.load(resolve(idx.loc[rg[rg.view == v].video_id.iloc[0], "pre_path"])) as z:
                    world[v] = z["world"]
            va, vb = views[0], views[1]
            ra, rb = rg[rg.view == va].sort_values("rep"), rg[rg.view == vb].sort_values("rep").set_index("rep_uid")
            uids = list(rb.index)
            for x in ra.itertuples():
                if x.rep_uid not in rb.index:
                    continue
                k = uids.index(x.rep_uid)
                if k + 1 >= len(uids):
                    continue
                same, nxt = rb.loc[x.rep_uid], rb.loc[uids[k + 1]]
                for name, angs in sets.items():
                    fa = angle_features(world[va][x.start:x.end + 1], angs)
                    fs = angle_features(world[vb][same.start:same.end + 1], angs)
                    fn = angle_features(world[vb][nxt.start:nxt.end + 1], angs)
                    cs, ps = dtw(fa, fs)
                    cn, pn = dtw(fa, fn)
                    # known warp (A5): view b's rep replayed slow-then-fast, q(t) = t^1.5; the views share aligned
                    # boundaries, so frame i of view a belongs at (i / (n-1))^(1/1.5) (m-1) of the warped copy
                    n, m = len(fa), len(fs)
                    t = np.linspace(0, 1, m)
                    fw = sample_at(fs, t ** WARP * (m - 1))
                    _, pw = dtw(fa, fw)
                    true = (np.arange(n) / (n - 1)) ** (1 / WARP) * (m - 1)
                    rows.append(dict(exercise=ex, angles=name, group_id=gid, same=cs / len(ps), next=cn / len(pn),
                                     warp_err=float(np.abs(warp_positions(pw, n) - true).mean()),
                                     linear_err=float(np.abs(np.arange(n) * (m - 1) / (n - 1) - true).mean())))
    D = pd.DataFrame(rows)
    D["same_cheaper"] = D.same < D.next
    out = resolve(cfg["outputs"]["phase1"])
    D.to_csv(out / f"dtw_angle_check{'_' + '_'.join(a.only) if a.only else ''}.csv", index=False)
    T = D.groupby(["exercise", "angles"]).agg(rep_pairs=("same", "size"), same_cheaper=("same_cheaper", "mean"),
                                              warp_err_frames=("warp_err", "median"), no_dtw_err=("linear_err", "median"),
                                              cost_same=("same", "median"), cost_next=("next", "median")).round(3)
    pd.set_option("display.width", 200)
    print(T.to_string())


if __name__ == "__main__":
    main()
