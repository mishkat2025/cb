"""CVDC batches: a sampler over rep sets that always loads all available views of a set together, plus synthetic
views, augmentations and the pairwise positive-weight matrix for the weighted InfoNCE (cvdc/losses/cvdc.py).

Per rep set (data/cvdc_members.csv, built by cvdc/data/pairs.py), restricted to the allowed items (training
subjects): real views are joined by edges of the enabled sources (``cvdc.sources``); the set's largest connected
component is loaded (ties, e.g. all single views in the synthetic-only ablation, are drawn at random per batch).
Rep-matched members use their DTW-aligned copies. ``synthetic.views_per_set`` synthetic views are added per set,
each projected from a random loaded member. A set with fewer than 2 items after this is skipped by the CVDC
sampler (it still serves supervised training).

Positive weights (A7): real pair = its source weight (``sync.pair_weights``); synthetic view vs its own source =
w_synthetic; synthetic view vs another member, or two synthetic views of different sources = min(w_synthetic,
weight of the two sources' real pair). A pair of loaded members without an enabled edge gets 0 (neutral).
"""
import numpy as np
import pandas as pd
import torch

from ..config import resolve
from .augment import augment
from .dataset import to_input
from .synthetic import synthetic_views

REAL = ("sync", "audio", "rep")


def positive_weights(src, set_of, n_real, edge_w, w_syn):
    """(n, n) positive weights (A7). src: source real item of each batch item; set_of: set index of each item;
    items >= n_real are synthetic; edge_w: {(item_a, item_b): weight} for enabled real edges (both orders)."""
    n = len(src)
    W = np.zeros((n, n), np.float32)
    by_set = {}
    for i, s in enumerate(set_of):
        by_set.setdefault(s, []).append(i)
    for ii in by_set.values():
        for a in ii:
            for b in ii:
                if a == b:
                    continue
                syn = a >= n_real or b >= n_real
                if src[a] == src[b]:
                    W[a, b] = w_syn if syn else 0.0
                else:
                    wr = edge_w.get((src[a], src[b]), 0.0)
                    W[a, b] = min(w_syn, wr) if syn else wr
    return W


class CVDCBatcher:
    def __init__(self, data, cfg, allowed=None, seed=0):
        """data: PackData (rep level); allowed: (N,) bool tensor/array of items that may be used (training
        subjects), None = all."""
        cc, pc = cfg["cvdc"], cfg["preprocess"]
        self.data, self.cc, self.mode = data, cc, cfg["model"]["input"]
        if self.mode not in ("xy", "xyv", "xyz"):
            raise NotImplementedError(f"CVDC batches for input {self.mode!r} (B2 with a lifter is trained without CVDC)")
        self.coord_channels = 3 if self.mode == "xyz" else 2
        self.sources = set(cc["sources"])
        bad = self.sources - set(REAL) - {"synthetic"}
        if bad:
            raise ValueError(f"unknown cvdc.sources {sorted(bad)}")
        self.w = {k: float(v) for k, v in cfg["sync"]["pair_weights"].items()}
        self.use_syn = "synthetic" in self.sources
        self.n_syn = int(cc["synthetic"]["views_per_set"]) if self.use_syn else 0
        dev = data.device
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        members = pd.read_csv(resolve(cc["pairs"]["members_csv"]))
        pairs = pd.read_csv(resolve(cc["pairs"]["pairs_csv"]))
        if not (members.item.to_numpy() == np.arange(len(data))).all():
            raise RuntimeError("cvdc_members.csv does not match the rep pack: rerun python -m cvdc.data.pairs")
        allowed = np.ones(len(data), bool) if allowed is None else np.asarray(
            allowed.cpu() if torch.is_tensor(allowed) else allowed, bool)

        # DTW-aligned copies of rep-matched members replace the plain pack in CVDC batches
        name = f"reps_{pc['joint_set']}_T{pc['T_rep']}{cc['pairs']['aligned_suffix']}"
        with np.load(resolve(pc["pack_dir"]) / f"{name}.npz") as z:
            self.xy, self.vis, self.world = data.xy.clone(), data.vis.clone(), data.world.clone()
            if len(z["item"]):
                it = torch.from_numpy(z["item"]).to(dev)
                self.xy[it], self.vis[it], self.world[it] = (torch.from_numpy(z[k]).to(dev) for k in ("xy", "vis", "world"))

        pairs = pairs[pairs.pair_source.isin(self.sources) & allowed[pairs.item_a] & allowed[pairs.item_b]]
        edge_w = {}
        for a, b, s in zip(pairs.item_a, pairs.item_b, pairs.pair_source):
            edge_w[(a, b)] = edge_w[(b, a)] = self.w[s]
        self.edge_w = edge_w
        self.sets = []  # per set: (list of candidate components, group code, set id)
        self.source_pairs = pairs.pair_source.value_counts().to_dict()
        gcode = data.group_code.cpu().numpy()
        for sid, m in members[allowed].groupby("set_id"):
            items = m.item.to_list()
            comps = self._components(items)
            big = max(len(c) for c in comps)
            comps = [c for c in comps if len(c) == big]
            if big + self.n_syn < 2:
                continue
            self.sets.append((comps, int(gcode[items[0]]), int(sid)))

    def _components(self, items):
        left, comps = set(items), []
        while left:
            stack, comp = [left.pop()], []
            while stack:
                a = stack.pop()
                comp.append(a)
                for b in list(left):
                    if (a, b) in self.edge_w:
                        left.remove(b)
                        stack.append(b)
            comps.append(sorted(comp))
        return comps

    def __len__(self):
        return int(np.ceil(len(self.sets) / self.cc["sampler"]["batch_sets"]))

    def order(self):
        """One epoch's set order. With ``one_set_per_group``: sets are ranked at random within their group and
        groups shuffled, then sorted by rank, so consecutive sets (a batch) rarely share a group."""
        n = len(self.sets)
        if not self.cc["sampler"].get("one_set_per_group", True):
            return self.rng.permutation(n)
        groups = np.array([s[1] for s in self.sets])
        gperm = {g: r for r, g in enumerate(self.rng.permutation(np.unique(groups)))}
        rank = np.zeros(n, int)
        for g in np.unique(groups):
            ii = np.flatnonzero(groups == g)
            rank[ii] = self.rng.permutation(len(ii))
        return np.lexsort((np.array([gperm[g] for g in groups]), rank))

    def __iter__(self):
        o, B = self.order(), self.cc["sampler"]["batch_sets"]
        for i in range(0, len(o), B):
            yield self.batch([self.sets[j] for j in o[i:i + B]])

    def batch(self, sets, augment_views=True):
        real, set_of, grp = [], [], []
        for k, (comps, g, _) in enumerate(sets):
            comp = comps[self.rng.integers(len(comps))] if len(comps) > 1 else comps[0]
            real += comp
            set_of += [k] * len(comp)
            grp += [g] * len(comp)
        dev = self.data.device
        real_t = torch.tensor(real, device=dev)
        src = list(real)  # source real item of every batch item
        syn_src, syn_set = [], []
        if self.n_syn:
            for k, (comps, g, _) in enumerate(sets):
                loaded = [r for r, s in zip(real, set_of) if s == k]
                for _ in range(self.n_syn):
                    syn_src.append(loaded[self.rng.integers(len(loaded))])
                    syn_set.append(k)
        x = to_input(self.xy[real_t], self.vis[real_t], self.world[real_t], self.mode)
        if syn_src:
            st = torch.tensor(syn_src, device=dev)
            sv = synthetic_views(self.world[st], self.cc["synthetic"], self.gen)
            xs = {"xy": sv["xy"], "xyv": torch.cat([sv["xy"], self.vis[st][..., None]], -1), "xyz": sv["world"]}[self.mode]
            x = torch.cat([x, xs], 0)
            src += syn_src
            set_of += syn_set
            grp += [sets[k][1] for k in syn_set]
        n_real = len(real)
        set_ids = torch.tensor(set_of, device=dev)
        W = self.weights(src, set_of, n_real)
        if augment_views:
            x = augment(x, self.cc["augment"], self.coord_channels, set_ids, self.gen)
        items = torch.tensor(src, device=dev)
        return {"x": x.permute(0, 3, 1, 2).contiguous(), "set_ids": set_ids, "W": W,
                "group_ids": torch.tensor(grp, device=dev), "exercise": self.data.exercise[items],
                "is_syn": torch.arange(len(src), device=dev) >= n_real, "item": items}

    def weights(self, src, set_of, n_real):
        return torch.from_numpy(positive_weights(src, set_of, n_real, self.edge_w, self.w["synthetic"])).to(
            self.data.device)

    def summary(self):
        n_items = sum(len(c[0]) for c, _, _ in self.sets)
        return {"sources": sorted(self.sources), "sets": len(self.sets), "real_items": n_items,
                "synthetic_per_set": self.n_syn, "real_pairs_by_source": self.source_pairs}
