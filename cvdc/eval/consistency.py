"""Viewpoint Consistency Index (VCI, E13): mean cosine similarity between embeddings of the same performance seen
from different views, minus the mean cosine similarity to other groups' embeddings.

``key`` defines "same performance": the rep set at rep level (default) or the (exercise, subject, variation)
group. Three variants are reported (supervisor, 2026-10-17):
  vci_centred  PRIMARY: the 256-d embedding z minus the training fold's mean embedding (what the task heads and
               the embedding-to-reference distance use; centring removes the common offset of the non-negative
               pooled ST-GCN features, which otherwise makes every cosine high)
  vci_raw      the 256-d embedding as is
  vci_proj     the 128-d L2-normalised projection (what the CVDC loss optimises)
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def viewpoint_consistency(z, key, view, group, center=None):
    """z: (N, D) embeddings; key: (N,) same-performance ids; view: (N,) view ids; group: (N,) group ids;
    center: (D,) vector subtracted first (the training fold's mean embedding) or None.
    Returns dict(vci, same, other, n_same_pairs)."""
    z = z.float()
    if center is not None:
        z = z - center.float()
    z = F.normalize(z, dim=-1)
    sim = z @ z.T
    same = (key[:, None] == key[None, :]) & (view[:, None] != view[None, :])
    other = group[:, None] != group[None, :]
    s, o = sim[same].mean().item(), sim[other].mean().item()
    return {"vci": s - o, "same": s, "other": o, "n_same_pairs": int(same.sum().item()) // 2}


@torch.no_grad()
def vci_report(z, proj, key, view, group, z_train_mean):
    """The three VCI variants (E13). Returns a flat dict: vci_centred (primary), vci_raw, vci_proj, plus the
    same / other similarities of each."""
    out = {}
    for name, e, c in (("centred", z, z_train_mean), ("raw", z, None), ("proj", proj, None)):
        v = viewpoint_consistency(e, key, view, group, c)
        out.update({f"vci_{name}": v["vci"], f"same_{name}": v["same"], f"other_{name}": v["other"]})
    out["n_same_view_pairs"] = v["n_same_pairs"]
    return out


def pooled_vci(snaps, members, gcode, strata, n_boot, seed, ci, pair_filter=None):
    """Within-stratum VCI (E13, E19) pooled over folds, with subject-bootstrap CIs (E17).

    snaps: iterable of (key, fold, npz) or (key, fold, npz, ctx) with npz an embedding snapshot
    (cvdc.train.trainer.embedding_snapshot: item, z, proj, z_train_mean); key: any hashable label of the pooled
    result (e.g. (method, stage)); snapshots with the same key are pooled. members: data/cvdc_members.csv (one row per
    rep item: set_id, view, subject, exercise, group); gcode: (N,) (exercise, subject, variation) code per item;
    strata: [(name, exercise or None, group or None)]; pair_filter(ctx, mm) -> (n, n) bool or None, the pairs that
    count among a snapshot's items of one stratum (e.g. one member in the snapshot's held-out view, E24).
    Folds are pooled by adding their per-subject pair sums (no cross-fold pairs). Returns (rows, per-fold rows);
    every row carries its bootstrap replicates (``boot``) and the bootstrapped subjects (``present``), so two rows with
    the same ``present`` have the same subject draws (paired differences)."""
    import numpy as np
    import pandas as pd

    from .metrics import add_sums, interval, subject_draws, vci_from_sums, vci_sums

    subjects = sorted(members.subject.unique())
    spos = {s: i for i, s in enumerate(subjects)}
    sgroup = members.drop_duplicates("subject").set_index("subject").group
    sums, fold_rows = {}, []
    for snap in snaps:
        key, fold, npz = snap[:3]
        ctx = snap[3] if len(snap) > 3 else None
        with np.load(npz) as z:
            it, center = z["item"], z["z_train_mean"]
            emb = {"centred": z["z"], "raw": z["z"], "proj": z["proj"]}
        mm = members.iloc[it]
        for sname, ex, grp in strata:
            sel = np.ones(len(mm), bool)
            if ex:
                sel &= (mm.exercise == ex).to_numpy()
            if grp:
                sel &= (mm.group == grp).to_numpy()
            ok = pair_filter(ctx, mm[sel]) if pair_filter is not None else None
            for var, e in emb.items():
                s = vci_sums(e[sel], mm.set_id.to_numpy()[sel], mm.view.to_numpy()[sel], gcode[it][sel],
                             np.array([spos[x] for x in mm.subject.to_numpy()[sel]], int), len(subjects),
                             center if var == "centred" else None, ok)
                sums[(key, sname, var)] = add_sums(sums.get((key, sname, var)), s)
                fold_rows.append(dict(key=key, stratum=sname, variant=var, fold=fold, **vci_from_sums(s)))
    fv = pd.DataFrame(fold_rows)
    rows = []
    for (key, sname, var), s in sums.items():
        present = np.flatnonzero((s["same_cnt"] > 0) | (s["other_cnt"].sum(1) > 0))
        M = np.zeros((n_boot, len(subjects)))
        M[:, present] = subject_draws(present, [sgroup[subjects[i]] for i in present], n_boot, seed)
        pt, b = vci_from_sums(s), vci_from_sums(s, M)
        lo, hi = interval(b["vci"], ci)
        x = fv[(fv.key == key) & (fv.stratum == sname) & (fv.variant == var)].vci
        rows.append(dict(key=key, stratum=sname, variant=var, vci=pt["vci"], vci_lo=lo, vci_hi=hi, same=pt["same"],
                         other=pt["other"], same_pairs=int(s["same_cnt"].sum() // 2), fold_mean=x.mean(),
                         fold_sd=x.std(ddof=1), folds=len(x), boot=b["vci"],
                         present=tuple(present)))
    return rows, fv
