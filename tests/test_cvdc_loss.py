import math

import pytest
import torch
import torch.nn.functional as F

from cvdc.losses.cvdc import cvdc_loss, cvdc_stats, pair_masks


def toy_batch(n_sets=6, views=3, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = F.normalize(torch.randn(n_sets * views, dim, generator=g), dim=-1)
    set_ids = torch.arange(n_sets).repeat_interleave(views)
    W = (set_ids[:, None] == set_ids[None, :]).float()
    W.fill_diagonal_(0)
    return z, set_ids, W


def test_matches_a_hand_computed_value():
    z = F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.2]]), dim=-1)
    set_ids = torch.tensor([0, 0, 1, 1])
    W = torch.tensor([[0, 0.5, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
    tau = 0.1
    s = z @ z.T / tau
    expected = 0.0
    for i, j, w in [(0, 1, 0.5), (1, 0, 0.5), (2, 3, 1.0), (3, 2, 1.0)]:
        den = torch.logsumexp(torch.stack([s[i, k] for k in range(4) if k != i]), 0)
        expected += w * -(s[i, j] - den)
    expected /= 4
    torch.testing.assert_close(cvdc_loss(z, set_ids, W, temperature=tau), expected)


def test_invariant_to_batch_order():
    z, set_ids, W = toy_batch()
    W = W * torch.rand(W.shape, generator=torch.Generator().manual_seed(1)).triu(1).add(0.5)  # varied weights
    W = torch.maximum(W, W.T)
    groups = set_ids // 2
    ex = set_ids % 2
    base = cvdc_loss(z, set_ids, W, groups, ex)
    for seed in range(5):
        p = torch.randperm(len(z), generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(cvdc_loss(z[p], set_ids[p], W[p][:, p], groups[p], ex[p]), base)
        torch.testing.assert_close(cvdc_loss(z[p], set_ids[p], W[p][:, p], groups[p], ex[p],
                                             negatives="different_exercise_only"),
                                   cvdc_loss(z, set_ids, W, groups, ex, negatives="different_exercise_only"))


def test_optimising_pulls_positives_together():
    z0, set_ids, W = toy_batch(n_sets=8, views=3, dim=8, seed=3)
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.SGD([z], lr=0.5)
    before = cvdc_stats(z0, set_ids, W)
    first = cvdc_loss(F.normalize(z, dim=-1), set_ids, W).item()
    for _ in range(100):
        opt.zero_grad()
        cvdc_loss(F.normalize(z, dim=-1), set_ids, W).backward()
        opt.step()
    after = cvdc_stats(F.normalize(z.detach(), dim=-1), set_ids, W)
    assert cvdc_loss(F.normalize(z.detach(), dim=-1), set_ids, W).item() < first
    assert after["pos_sim"] > before["pos_sim"] + 0.3
    assert after["pos_sim"] - after["neg_sim"] > 0.8
    assert after["align"] < before["align"]


def test_perfectly_clustered_embeddings_give_a_small_loss():
    # each set on its own orthogonal axis: positives similarity 1, negatives 0
    set_ids = torch.arange(4).repeat_interleave(2)
    z = F.one_hot(set_ids, 4).float()
    W = (set_ids[:, None] == set_ids[None, :]).float().fill_diagonal_(0)
    loss = cvdc_loss(z, set_ids, W, temperature=0.1)
    expected = -(10 - math.log(math.exp(10) + 6))  # 1 positive at 1/tau, 6 negatives at 0
    assert loss.item() == pytest.approx(expected, abs=1e-5)  # float32
    assert loss.item() < 0.01


def test_weights_scale_the_positive_terms():
    z, set_ids, W = toy_batch()
    torch.testing.assert_close(cvdc_loss(z, set_ids, 0.5 * W), 0.5 * cvdc_loss(z, set_ids, W))


def test_zero_weight_pairs_are_neutral():
    """A same-set pair with weight 0 is neither a positive nor a negative: removing its items' mutual pair is
    the same as never having it in the denominator."""
    z, set_ids, W = toy_batch(n_sets=3, views=3)
    W2 = W.clone()
    W2[0, 1] = W2[1, 0] = 0.0
    pos, neg = pair_masks(set_ids)
    assert pos[0, 1] and not neg[0, 1]
    # anchor 0 by hand: positives {2}, denominator = {2} + 6 negatives (1 is excluded)
    s = z @ z.T / 0.1
    den0 = torch.logsumexp(s[0, [2, 3, 4, 5, 6, 7, 8]], 0)
    l0 = -(s[0, 2] - den0)
    full = cvdc_loss(z, set_ids, W2)
    # recompute the mean over anchors with anchor 0 replaced by the hand value
    per = []
    for i in range(9):
        p = [j for j in range(9) if set_ids[j] == set_ids[i] and j != i and W2[i, j] > 0]
        d = [j for j in range(9) if j != i and (set_ids[j] != set_ids[i] or W2[i, j] > 0)]
        den = torch.logsumexp(s[i, d], 0)
        per.append(sum(W2[i, j] * -(s[i, j] - den) for j in p) / len(p))
    torch.testing.assert_close(per[0], l0)
    torch.testing.assert_close(full, torch.stack(per).mean())


def test_negative_modes():
    set_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    groups = torch.tensor([0, 0, 0, 0, 1, 1])  # sets 0 and 1: two reps of the same group
    ex = torch.tensor([0, 0, 0, 0, 1, 1])
    _, neg = pair_masks(set_ids, groups, ex)
    assert not neg[0, 2] and neg[0, 4]  # same group, other rep: neutral
    _, neg = pair_masks(set_ids, groups, ex, same_group_other_rep="negative")
    assert neg[0, 2]
    _, neg = pair_masks(set_ids, None, ex, negatives="different_exercise_only")
    assert not neg[0, 2] and neg[0, 4]


def test_anchors_without_positives_are_skipped():
    z, set_ids, W = toy_batch(n_sets=4, views=2)
    single = F.normalize(torch.randn(1, z.shape[1]), dim=-1)
    z2 = torch.cat([z, single])
    s2 = torch.cat([set_ids, torch.tensor([99])])
    W2 = torch.zeros(len(z2), len(z2))
    W2[:-1, :-1] = W
    loss = cvdc_loss(z2, s2, W2)
    assert torch.isfinite(loss)
    assert cvdc_loss(z[:1], set_ids[:1], W[:1, :1]).item() == 0.0  # no positives at all


def test_collapse_shows_in_the_stats():
    z, set_ids, W = toy_batch()
    collapsed = F.normalize(torch.ones_like(z) + 1e-4 * torch.randn_like(z), dim=-1)
    assert cvdc_stats(collapsed, set_ids, W)["proj_std"] < 1e-3
    assert cvdc_stats(z, set_ids, W)["proj_std"] > 0.1
