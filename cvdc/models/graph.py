"""Skeleton graph and partition strategies of ST-GCN (Yan et al. 2018), following the reference implementation
(github.com/yysijie/st-gcn, net/utils/graph.py).

The adjacency includes self-loops (hop 0) and is column-normalised (each column sums to 1), so the graph
convolution ``einsum('nkctv,kvw->nctw')`` averages over each joint's neighbourhood. Partition strategies:
  uniform   one subset (the whole 1-hop neighbourhood);
  distance  hop 0 (the joint itself) and hop 1 (its neighbours);
  spatial   root (the joint itself, and neighbours at the same distance from the centre), centripetal
            (neighbours closer to the centre) and centrifugal (neighbours further away). Default.
Distances to the centre joint are full graph distances. (The reference code measures them with the same
``max_hop = 1`` cut-off as the neighbourhood, so every joint more than one hop from the centre gets an
infinite distance and its neighbours all fall into the root subset; here the partition follows the paper.)
"""
import numpy as np

from ..data.skeleton import COCO17_CENTER, COCO17_EDGES, JOINT_SETS

LAYOUTS = {"coco17": (len(JOINT_SETS["coco17"]), COCO17_EDGES, COCO17_CENTER)}


def hop_distance(num_node, edges, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    hop = np.full((num_node, num_node), np.inf)
    reach = np.stack([np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]) > 0
    for d in range(max_hop, -1, -1):
        hop[reach[d]] = d
    return hop


def normalize_digraph(A):
    """A @ D^-1 with D the column sums: every column with an edge sums to 1."""
    col = A.sum(0)
    Dn = np.diag(np.where(col > 0, 1.0 / np.where(col > 0, col, 1), 0.0))
    return A @ Dn


class Graph:
    """``A``: (K, V, V) float32 partitioned adjacency; K = 1 (uniform), max_hop + 1 (distance) or 3 (spatial)."""

    def __init__(self, layout="coco17", strategy="spatial", max_hop=1):
        if layout not in LAYOUTS:
            raise NotImplementedError(f"graph layout {layout!r}; defined: {sorted(LAYOUTS)}")
        self.num_node, self.edges, self.center = LAYOUTS[layout]
        self.hop = hop_distance(self.num_node, self.edges, max_hop)
        self.center_dist = hop_distance(self.num_node, self.edges, self.num_node)[:, self.center]
        self.A = self._partition(strategy, max_hop).astype(np.float32)

    def _partition(self, strategy, max_hop):
        V, hop, dc = self.num_node, self.hop, self.center_dist
        valid = range(max_hop + 1)
        adj = np.zeros((V, V))
        for h in valid:
            adj[hop == h] = 1
        norm = normalize_digraph(adj)
        if strategy == "uniform":
            return norm[None]
        if strategy == "distance":
            A = np.zeros((len(valid), V, V))
            for k, h in enumerate(valid):
                A[k][hop == h] = norm[hop == h]
            return A
        if strategy == "spatial":
            A = []
            for h in valid:
                root, close, far = np.zeros((V, V)), np.zeros((V, V)), np.zeros((V, V))
                for i in range(V):
                    for j in range(V):
                        if hop[j, i] != h:
                            continue
                        if dc[j] == dc[i]:
                            root[j, i] = norm[j, i]
                        elif dc[j] > dc[i]:
                            close[j, i] = norm[j, i]
                        else:
                            far[j, i] = norm[j, i]
                A += [root] if h == 0 else [root + close, far]
            return np.stack(A)
        raise ValueError(f"unknown partition strategy {strategy!r}")
