"""Phase 4 item 6: recording-noise test. The views of one expert performance (a synced group: three phones filming
at once, S1-S3) get different PF3 / PF4 (repetition items) in 176 of 258 groups (E3). Do these differences follow how
many reps each view's video actually contains (the phones start and stop up to ~1-2 s apart, P6b)?

Per synced group with >= 2 aligned views: the reps are taken from the best view's own segmentation (all reps it
recorded, P6b) and placed on the group's aligned clock (aligned time = t - offset_view, outputs/sync_report.csv). For
each view, a rep counts as
  contained   its whole span lies inside the view's recording (0.1 s tolerance, reps.transfer_tol_s), and
  half        at least half of its span lies inside.
The view's own segmentation count (n_reps_own, Phase 1) is reported too (noisier in side views, §5.6).
Tests, over view pairs of the same performance (ordered so that PF4 is higher in the first view):
  - share of pairs with different PF4 whose contained count is also higher in the first view, lower, or equal;
    against a within-group permutation null (PF4 labels shuffled across the group's views; exact for 2-3 views);
  - within-group centred correlations of PF4 (and the total score) with the contained count;
  - PF4 against min(1, count / 10) (the EAAQ's "minimum 10 times") and PF3 = 1 against count >= 10.

usage: python scripts/pilot/phase4/recording_noise.py
Writes outputs/phase4/recording_noise_views.csv (one row per view), recording_noise_pairs.csv, recording_noise.json.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cvdc.config import load_config, resolve  # noqa: E402
from cvdc.eval.recording import counts_per_view  # noqa: E402


def _agree(pf4, cnt, pairs):
    """Share of view pairs with different PF4 whose count is higher in the view with the higher PF4."""
    hit = tot = 0
    for p, c, (i, j) in zip(pf4, cnt, pairs):
        if not len(i):
            continue
        dp, dc = p[i] - p[j], c[i] - c[j]
        m = dp != 0
        tot += m.sum()
        hit += (np.sign(dc[m]) == np.sign(dp[m])).sum()
    return hit / tot if tot else np.nan


def perm_null(groups, col="contained", n=5000):
    """Within-group permutation null for _agree: PF4 labels shuffled across each group's views. Returns (observed,
    null mean, one-sided p = P(null >= observed))."""
    rng = np.random.default_rng(0)
    pf4 = [g.PF4.to_numpy(float) for g in groups]
    cnt = [g[col].to_numpy(float) for g in groups]
    pairs = [np.triu_indices(len(g), 1) for g in groups]
    obs = _agree(pf4, cnt, pairs)
    sims = np.array([_agree([rng.permutation(p) for p in pf4], cnt, pairs) for _ in range(n)])
    return float(obs), float(sims.mean()), float((sims >= obs - 1e-12).mean())


def pair_rows(groups):
    rows = []
    for g in groups:
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            if a.PF4 == b.PF4:
                continue
            hi, lo = (a, b) if a.PF4 > b.PF4 else (b, a)
            rows.append(dict(group_id=hi.group_id, exercise=hi.exercise, view_hi=hi.view, view_lo=lo.view,
                             dPF4=hi.PF4 - lo.PF4, dPF3=hi.PF3 - lo.PF3, d_contained=hi.contained - lo.contained,
                             d_half=hi.half - lo.half, d_own=hi.n_reps_own - lo.n_reps_own,
                             d_dur=hi.dur_s - lo.dur_s, d_score=hi.score - lo.score))
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    fps, tol = cfg["preprocess"]["fps"], cfg["reps"]["transfer_tol_s"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"])).set_index("video_id")
    rpv = pd.read_csv(resolve(cfg["outputs"]["phase1"]) / "reps_per_video.csv").set_index("video_id")
    own = pd.read_csv(resolve("data/reps_perview.csv"))
    sync = pd.read_csv(resolve(cfg["sync"]["report_csv"])).set_index("group_id")
    out = resolve("outputs/phase4")
    rows = []
    for gid, d in rpv[rpv.group == "expert"].groupby("group_id"):
        if gid not in sync.index or sync.loc[gid, "cls"] != "synced" or d.exercise.iloc[0] not in cfg["data"]["include_exercises"]:
            continue
        off = {v: sync.loc[gid, f"offset_{v}_s"] for v in d.view}
        d = d[[np.isfinite(off[v]) for v in d.view]]
        if len(d) < 2 or not d.is_best.any():
            continue
        best = d[d.is_best].iloc[0]
        ob = own[own.video_id == best.name]
        views = {r.view: (off[r.view], int(r.n_frames_30)) for r in d.itertuples()}
        cnt = counts_per_view(ob, off[best.view], views, fps, tol)
        for vid, r in d.iterrows():
            rows.append(dict(video_id=vid, group_id=gid, exercise=r.exercise, subject=r.subject, view=r.view,
                             best_view=best.view, contained=cnt[r.view][0], half=cnt[r.view][1],
                             n_reps_own=int(r.n_reps_own), n_reps_set=int(r.n_reps_set), dur_s=(r.n_frames_30 - 1) / fps,
                             start_aligned_s=-off[r.view], end_aligned_s=(r.n_frames_30 - 1) / fps - off[r.view],
                             PF3=idx.loc[vid, "PF3"], PF4=idx.loc[vid, "PF4"], score=idx.loc[vid, "score"]))
    V = pd.DataFrame(rows)
    V.to_csv(out / "recording_noise_views.csv", index=False)
    res = {}
    for ex in ["all", *cfg["data"]["include_exercises"]]:
        Vx = V if ex == "all" else V[V.exercise == ex]
        groups = [g.reset_index(drop=True) for _, g in Vx.groupby("group_id")]
        differ = [g for g in groups if g.PF4.nunique() > 1]
        P = pair_rows(groups)
        P.to_csv(out / f"recording_noise_pairs_{ex}.csv", index=False)
        c = Vx.copy()
        for k in ("PF4", "PF3", "score", "contained", "half", "n_reps_own", "dur_s"):
            c[k + "_c"] = c[k] - c.groupby("group_id")[k].transform("mean")
        cc = c[c.groupby("group_id").PF4.transform("nunique") > 1]  # groups whose views differ in PF4
        r = {"groups": len(groups), "views": len(Vx), "groups_PF4_differs": len(differ),
             "groups_PF3_differs": int(sum(g.PF3.nunique() > 1 for g in groups)),
             "groups_count_differs": int(sum(g.contained.nunique() > 1 for g in groups)),
             "groups_both_differ": int(sum(g.contained.nunique() > 1 and g.PF4.nunique() > 1 for g in groups)),
             "pairs_PF4_differs": len(P)}
        for col in ("contained", "half", "own", "dur"):
            r[f"pairs_{col}_higher"] = float((P[f"d_{col}"] > 0).mean()) if len(P) else None
            r[f"pairs_{col}_equal"] = float((P[f"d_{col}"] == 0).mean()) if len(P) else None
            r[f"pairs_{col}_lower"] = float((P[f"d_{col}"] < 0).mean()) if len(P) else None
        for col in ("contained", "half", "n_reps_own"):
            obs, null, p = perm_null(groups, col)
            r.update({f"perm_{col}_obs": obs, f"perm_{col}_null": null, f"perm_{col}_p": p})
        for k in ("contained", "half", "n_reps_own", "dur_s"):
            ok = cc[k + "_c"].abs() + cc["PF4_c"].abs() > 0
            r[f"spearman_PF4_{k}_within"] = float(spearmanr(cc.loc[ok, "PF4_c"], cc.loc[ok, k + "_c"])[0])
            r[f"pearson_score_{k}_within"] = float(pearsonr(c["score_c"], c[k + "_c"])[0])
        r["r2_score_on_contained_within"] = r["pearson_score_contained_within"] ** 2
        pred = np.minimum(1.0, Vx.contained / 10.0)
        r["mae_PF4_vs_count10"] = float(np.abs(Vx.PF4 - pred).mean())
        r["mae_PF4_vs_groupmean"] = float(np.abs(Vx.PF4 - Vx.groupby("group_id").PF4.transform("mean")).mean())
        r["PF3_1_given_count_ge10"] = float(Vx[Vx.contained >= 10].PF3.mean()) if (Vx.contained >= 10).any() else None
        r["PF3_1_given_count_lt10"] = float(Vx[Vx.contained < 10].PF3.mean()) if (Vx.contained < 10).any() else None
        r["n_count_ge10"], r["n_count_lt10"] = int((Vx.contained >= 10).sum()), int((Vx.contained < 10).sum())
        r["PF4_mean_by_contained"] = {int(k): [round(float(x.PF4.mean()), 3), len(x)] for k, x in Vx.groupby("contained")}
        res[ex] = r
        print(ex, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, indent=0))
    (out / "recording_noise.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
