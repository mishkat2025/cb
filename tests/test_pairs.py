import numpy as np
import pandas as pd

from cvdc.data.cvdc_batch import positive_weights
from cvdc.data.pairs import build_edges, choose_ref, components


def toy_reps():
    """Group A (synced): F, L, R share rep_uids A_r00, A_r01; R lacks A_r01 (edge rep dropped in that view).
    Group B (not synced): F has 2 reps, L has 1, own rep_uids. Video C: single view."""
    rows = [("A_F", 0, "A_r00", "F"), ("A_F", 1, "A_r01", "F"), ("A_L", 0, "A_r00", "L"), ("A_L", 1, "A_r01", "L"),
            ("A_R", 0, "A_r00", "R"),
            ("B_F", 0, "B_F_r00", "F"), ("B_F", 1, "B_F_r01", "F"), ("B_L", 0, "B_L_r00", "L"),
            ("C_F", 0, "C_F_r00", "F")]
    r = pd.DataFrame(rows, columns=["video_id", "rep", "rep_uid", "view"])
    r["item"] = np.arange(len(r))
    return r


def toy_view_pairs():
    return pd.DataFrame([("A_F", "A_L", "sync"), ("A_F", "A_R", "audio"), ("A_L", "A_R", "audio"),
                         ("B_F", "B_L", "rep")], columns=["video_a", "video_b", "pair_source"])


def edge_set(E):
    return {(a, b, s) for a, b, s in zip(E.item_a, E.item_b, E.pair_source)}


def test_edges_follow_shared_rep_ids_and_view_pair_sources():
    E = build_edges(toy_reps(), toy_view_pairs())
    assert edge_set(E) == {
        (0, 2, "sync"), (0, 4, "audio"), (2, 4, "audio"),  # A rep 0 in F, L, R
        (1, 3, "sync"),                                     # A rep 1 in F, L (R did not record it)
        (5, 7, "rep"),                                      # B: rep 0 <-> rep 0; B_F rep 1 has no partner
    }


def test_sets_are_connected_components():
    r = toy_reps()
    sid = components(len(r), build_edges(r, toy_view_pairs()))
    assert sid[0] == sid[2] == sid[4]
    assert sid[1] == sid[3] != sid[0]
    assert sid[5] == sid[7]
    assert len({sid[6], sid[8], sid[0], sid[1], sid[5]}) == 5  # unpaired reps are their own sets
    assert sid.max() + 1 == 5


def test_reference_member_follows_view_preference():
    r = toy_reps()
    assert choose_ref(r.iloc[[2, 4, 0]], ["F", "R", "L"]) == 0
    assert choose_ref(r.iloc[[2, 4]], ["F", "R", "L"]) == 4


def test_positive_weight_rules():
    # one set: real items 10 (F), 11 (L), 12 (R); synthetic items from sources 10 and 12
    src = [10, 11, 12, 10, 12]
    set_of = [0, 0, 0, 0, 0]
    edge_w = {(10, 11): 1.0, (11, 10): 1.0, (10, 12): 0.8, (12, 10): 0.8}  # no enabled L-R edge
    W = positive_weights(src, set_of, 3, edge_w, w_syn=1.0)
    assert np.allclose(W, W.T)
    assert W[0, 1] == 1.0 and W[0, 2] == 0.8 and W[1, 2] == 0.0  # real pairs: source weights, missing -> 0
    assert W[0, 3] == 1.0 and W[2, 4] == 1.0  # synthetic vs its own source: w_syn
    assert W[1, 3] == 1.0 and W[1, 4] == 0.0  # synthetic vs another member: min(w_syn, real pair)
    assert W[3, 4] == 0.8  # two synthetic views of different sources: min(w_syn, real pair)
    assert (np.diag(W) == 0).all()
    W2 = positive_weights([1, 2], [0, 1], 2, {(1, 2): 1.0, (2, 1): 1.0}, 1.0)
    assert (W2 == 0).all()  # different sets never get a positive weight
