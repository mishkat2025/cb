"""Cross-view positive pairs for CVDC at rep level (C1, A5), built on the Phase 1 caches.

Items are the rows of data/reps.csv, i.e. the rows of the rep pack (packs/reps_<joint set>_T<T>.npz).
Edges between two items of different views of one (exercise, subject, variation) group:
  sync / audio  the two reps share a ``rep_uid``: synced groups have identical, time-aligned rep boundaries in
                every aligned view (P6b). The source is the view pair's ``pair_source`` in outputs/sync_pairs.csv:
                ``sync`` (audio-proven, or motion-confirmed lags) or ``audio`` (device-corrected audio lag only).
  rep           view pairs whose source is ``rep`` (not-synced or unknown groups): rep i of one view is paired
                with rep i of the other (``cvdc.pairs.rep_pairing: index``) and time-aligned by DTW on 3D joint
                angles (``cvdc.pairs.dtw_angles``, world landmarks).
A rep *set* is a connected component of these edges: one repetition with all its views. Sets of one item
(single-view videos, edge reps of unpaired views) still get positives from synthetic views (``synthetic``,
generated on the fly, cvdc/data/synthetic.py). Pair weights are read from ``sync.pair_weights`` when a batch is
built, so changing a weight needs no rebuild; ``cvdc.sources`` selects the sources (C5 ablations).

DTW alignment: in a set without a shared ``rep_uid``, every member is warped onto the time axis of the set's
reference member (first view in ``cvdc.pairs.view_preference``) and resampled to T; these aligned copies are
saved to packs/reps_<joint set>_T<T>_aligned.npz and replace the plain packs in CVDC batches.

usage: python -m cvdc.data.pairs [--config C] [--model-config M]
Writes data/cvdc_pairs.csv (one row per edge), data/cvdc_members.csv (one row per rep item, with its set),
the aligned pack and outputs/phase2/table_pair_counts.csv.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from ..assess.angles import joint_angle, shoulder_elevation
from ..assess.dtw import dtw, sample_at, warp_positions
from ..config import load_config, resolve
from .preprocess import resample_fixed
from .reps import fill_nan
from .skeleton import JOINT_SETS

ANGLES = {
    "shoulder_elevation_left": lambda P: shoulder_elevation(P, "left"),
    "shoulder_elevation_right": lambda P: shoulder_elevation(P, "right"),
    "elbow_left": lambda P: joint_angle(P, "left_shoulder", "left_elbow", "left_wrist"),
    "elbow_right": lambda P: joint_angle(P, "right_shoulder", "right_elbow", "right_wrist"),
}
REAL_SOURCES = ("sync", "audio", "rep")


def angle_features(world, names):
    """(n, 33, 3) world landmarks -> (n, len(names)) joint angles in degrees, NaNs interpolated (0 if none)."""
    cols = []
    for nm in names:
        a = fill_nan(ANGLES[nm](world))
        cols.append(np.zeros(len(world)) if a is None else a)
    return np.stack(cols, 1)


def build_edges(reps, view_pairs, rep_pairing="index"):
    """reps: data/reps.csv with an ``item`` column; view_pairs: outputs/sync_pairs.csv.
    Returns a DataFrame of edges (item_a, item_b, pair_source)."""
    src = {}
    for r in view_pairs.itertuples():
        src[(r.video_a, r.video_b)] = src[(r.video_b, r.video_a)] = r.pair_source
    edges = []
    for _, g in reps.groupby("rep_uid"):
        for a, b in itertools.combinations(g.itertuples(), 2):
            s = src.get((a.video_id, b.video_id))
            if s in ("sync", "audio"):
                edges.append((a.item, b.item, s))
    if rep_pairing != "index":
        raise ValueError(f"rep_pairing {rep_pairing!r}")
    by_video = {v: g.sort_values("rep") for v, g in reps.groupby("video_id")}
    for r in view_pairs[view_pairs.pair_source == "rep"].itertuples():
        if r.video_a not in by_video or r.video_b not in by_video:
            continue
        ga, gb = by_video[r.video_a], by_video[r.video_b]
        for ia, ib in zip(ga.item, gb.item):  # rep i <-> rep i
            edges.append((ia, ib, "rep"))
    E = pd.DataFrame(edges, columns=["item_a", "item_b", "pair_source"])
    E[["item_a", "item_b"]] = np.sort(E[["item_a", "item_b"]].to_numpy(), axis=1)
    return E.drop_duplicates(["item_a", "item_b"]).reset_index(drop=True)


def components(n, edges):
    """Union-find over n items -> set id per item (0..k-1, ordered by first item)."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(edges.item_a, edges.item_b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, sid = np.unique(roots, return_inverse=True)
    return sid


def choose_ref(members, view_preference):
    order = {v: i for i, v in enumerate(view_preference)}
    return min(members.itertuples(), key=lambda r: (order.get(r.view, len(order)), r.item)).item


def build(cfg, verbose=True):
    pc, cc = cfg["preprocess"], cfg["cvdc"]["pairs"]
    joints = JOINT_SETS[pc["joint_set"]]
    T = pc["T_rep"]
    name = f"reps_{pc['joint_set']}_T{T}"
    reps = pd.read_csv(resolve("data/reps.csv"))
    pack_csv = pd.read_csv(resolve(pc["pack_dir"]) / f"{name}.csv")
    if not ((pack_csv.video_id.values == reps.video_id.values).all() and (pack_csv.rep.values == reps.rep.values).all()):
        raise RuntimeError("data/reps.csv and the rep pack are out of sync: rerun python -m cvdc.data.pack")
    reps["item"] = np.arange(len(reps))
    vp = pd.read_csv(resolve(cfg["sync"]["pairs_csv"]))
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"])).set_index("video_id")

    E = build_edges(reps, vp, cc["rep_pairing"])
    reps["set_id"] = components(len(reps), E)
    if reps.groupby("set_id").video_id.apply(lambda v: v.duplicated().any()).any():
        raise RuntimeError("a rep set holds two reps of the same video")
    E["set_id"] = reps.set_id.values[E.item_a]

    # reference member and DTW alignment of sets without a shared rep_uid
    names = cc["dtw_angles"]
    pre_cache = {}

    def segment(item):
        r = reps.iloc[item]
        if r.video_id not in pre_cache:
            with np.load(resolve(idx.loc[r.video_id, "pre_path"])) as z:
                pre_cache[r.video_id] = {k: z[k] for k in ("xy", "vis", "world")}
        d = pre_cache[r.video_id]
        return {k: v[r.start:r.end + 1] for k, v in d.items()}

    def feats(item):
        return angle_features(segment(item)["world"], names)

    reps["is_ref"], reps["aligned"], reps["dtw_cost_deg"] = False, "none", np.nan
    al_items, al = [], {"xy": [], "vis": [], "world": []}
    for sid, m in reps.groupby("set_id"):
        if len(m) == 1:
            continue
        ref = choose_ref(m, cc["view_preference"])
        reps.loc[ref, "is_ref"] = True
        ref_uid = reps.at[ref, "rep_uid"]
        fr = feats(ref)
        for it in m.item:
            if it == ref:
                continue
            if reps.at[it, "rep_uid"] == ref_uid:
                reps.loc[it, "aligned"] = "shared_uid"
                continue
            cost, path = dtw(fr, feats(it))
            pos = warp_positions(path, len(fr))
            seg = segment(it)
            for k in al:
                al[k].append(resample_fixed(sample_at(seg[k][:, joints], pos), T))
            al_items.append(it)
            reps.loc[it, ["aligned", "dtw_cost_deg"]] = "dtw", cost / len(path)
    pairs_dtw = []
    for e in E.itertuples():
        if e.pair_source == "rep":
            cost, path = dtw(feats(e.item_a), feats(e.item_b))
            pairs_dtw.append(cost / len(path))
        else:
            pairs_dtw.append(np.nan)
    E["dtw_cost_deg"] = pairs_dtw  # mean per-step Euclidean distance over the angle channels, degrees

    # outputs
    w = cfg["sync"]["pair_weights"]
    cols = ["video_id", "rep", "rep_uid", "view", "group_id", "exercise", "subject", "group", "variation"]
    ra, rb = reps.iloc[E.item_a][cols].reset_index(drop=True), reps.iloc[E.item_b][cols].reset_index(drop=True)
    pairs = pd.DataFrame({"set_id": E.set_id, "item_a": E.item_a, "item_b": E.item_b,
                          "video_a": ra.video_id, "video_b": rb.video_id, "view_a": ra.view, "view_b": rb.view,
                          "rep_a": ra.rep, "rep_b": rb.rep, "pair_source": E.pair_source,
                          "pair_weight": E.pair_source.map(w), "dtw_cost_deg": E.dtw_cost_deg,
                          "group_id": ra.group_id, "exercise": ra.exercise, "subject": ra.subject,
                          "group": ra.group, "variation": ra.variation})
    srcs = pairs.groupby("set_id").pair_source.agg(lambda s: "+".join(sorted(set(s))))
    reps["n_members"] = reps.groupby("set_id").item.transform("size")
    reps["set_sources"] = reps.set_id.map(srcs).fillna("single")
    members = reps[["item", "set_id", "n_members", "set_sources", "is_ref", "aligned", "dtw_cost_deg", *cols]]
    pairs.to_csv(resolve(cc["pairs_csv"]), index=False)
    members.to_csv(resolve(cc["members_csv"]), index=False)
    out = resolve(pc["pack_dir"]) / f"{name}{cc['aligned_suffix']}.npz"
    np.savez_compressed(out, item=np.asarray(al_items, np.int64),
                        **{k: np.asarray(v, np.float32).reshape((-1, T, len(joints)) + ((2,) if k == "xy" else (3,) if k == "world" else ()))
                           for k, v in al.items()})
    summary = pair_summary(pairs, members)
    sp = resolve(cc["summary_csv"])
    sp.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(sp, index=False)
    if verbose:
        print(summary.to_string(index=False))
        print(f"{len(pairs)} edges, {members.set_id.nunique()} rep sets, {len(al_items)} DTW-aligned members -> {out.name}")
    return pairs, members


def pair_summary(pairs, members):
    """Rep-level pair counts per exercise, expert/non-expert group and source; sets by size; synthetic-only
    sets (one real view)."""
    rows = []
    for (ex, grp), m in members.groupby(["exercise", "group"]):
        p = pairs[(pairs.exercise == ex) & (pairs.group == grp)]
        sets = m.drop_duplicates("set_id")
        row = dict(exercise=ex, group=grp, reps=len(m), sets=len(sets),
                   sets_3_views=int((sets.n_members == 3).sum()), sets_2_views=int((sets.n_members == 2).sum()),
                   sets_1_view=int((sets.n_members == 1).sum()))
        for s in REAL_SOURCES:
            row[f"pairs_{s}"] = int((p.pair_source == s).sum())
        row["dtw_cost_deg_median_rep"] = float(p.loc[p.pair_source == "rep", "dtw_cost_deg"].median()) if (p.pair_source == "rep").any() else np.nan
        rows.append(row)
    S = pd.DataFrame(rows)
    tot = S.drop(columns=["exercise", "group", "dtw_cost_deg_median_rep"]).sum()
    S.loc[len(S)] = dict(exercise="all", group="all", **tot.to_dict(),
                         dtw_cost_deg_median_rep=float(pairs.loc[pairs.pair_source == "rep", "dtw_cost_deg"].median()))
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model-config")
    a = ap.parse_args()
    build(load_config(a.config, a.model_config))


if __name__ == "__main__":
    main()
