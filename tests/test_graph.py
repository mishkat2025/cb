import numpy as np
import pytest

from cvdc.data.skeleton import COCO17_CENTER, COCO17_EDGES
from cvdc.models.graph import Graph, hop_distance

V = 17


def test_adjacency_has_self_loops_and_edges_and_is_symmetric():
    hop = hop_distance(V, COCO17_EDGES, 1)
    assert (np.diag(hop) == 0).all()
    adj = hop <= 1
    assert (adj == adj.T).all()
    assert adj.sum() == V + 2 * len(COCO17_EDGES)  # self-loops + both directions of 16 edges
    assert len(COCO17_EDGES) == 16
    for a, b in COCO17_EDGES:
        assert adj[a, b] and adj[b, a]


def test_skeleton_is_one_tree_rooted_at_the_nose():
    d = hop_distance(V, COCO17_EDGES, V)[:, COCO17_CENTER]
    assert np.isfinite(d).all()  # connected
    assert len(COCO17_EDGES) == V - 1  # a tree
    for child, parent in COCO17_EDGES:  # edges are (child, parent): parents are one hop closer to the centre
        assert d[child] == d[parent] + 1


@pytest.mark.parametrize("strategy,k", [("uniform", 1), ("distance", 2), ("spatial", 3)])
def test_partitions_sum_to_the_column_normalised_adjacency(strategy, k):
    A = Graph("coco17", strategy).A
    assert A.shape == (k, V, V)
    total = A.sum(0)
    np.testing.assert_allclose(total.sum(0), 1.0, atol=1e-6)  # every column sums to 1
    np.testing.assert_array_equal(total > 0, hop_distance(V, COCO17_EDGES, 1) <= 1)
    assert (A >= 0).all()


def test_spatial_partition_is_self_inward_outward():
    A = Graph("coco17", "spatial").A
    root, a1, a2 = (a > 0 for a in A)
    np.testing.assert_array_equal(root, np.eye(V, dtype=bool))  # a tree has no same-distance neighbours
    for child, parent in COCO17_EDGES:
        # A[k, j, i]: message from j to i. Subset 1: j further from the centre than i; subset 2: j closer.
        assert a1[child, parent] and not a1[parent, child]
        assert a2[parent, child] and not a2[child, parent]
    assert a1.sum() == a2.sum() == len(COCO17_EDGES)


def test_unknown_layout_and_strategy_fail_loudly():
    with pytest.raises(NotImplementedError):
        Graph("mp33")
    with pytest.raises(ValueError):
        Graph("coco17", "nope")
