"""CVDC loss: weighted multi-positive InfoNCE (SupCon-style, Khosla et al. 2020, "L_out"), C1 and C3.

A batch holds rep *sets*: each set is one repetition seen from its available views (real camera views,
time-aligned by the pair builder, plus synthetic rotated views). Items of the same set are positives with a
per-pair weight w_ij from the pair source (sync 1.0, audio 0.8, rep 0.5, synthetic 1.0; ``sync.pair_weights``).
For anchor i with positives P(i) (same set, w_ij > 0) and denominator D(i) = P(i) + negatives N(i):

    l_ij = -log( exp(s_ij / tau) / sum_{k in D(i)} exp(s_ik / tau) ),   s = cosine similarity
    L_i  = (1 / |P(i)|) * sum_{j in P(i)} w_ij * l_ij
    L    = mean of L_i over the anchors that have at least one positive

With every w_ij = 1 this is SupCon's L_out. The weights scale each positive pair's term, so less certain pairs
(e.g. rep-matched, 0.5) pull with less force than sync pairs; normalising by |P(i)| (not by the weight sum)
keeps that effect for anchors whose positives all share one weight (A6).

Negatives (``negatives``): ``all_other_groups`` = every item of another (exercise, subject, variation) group;
``different_exercise_only`` = only items of another exercise (the proposal's "different exercises are
separated"). Items of the same group but another rep set are neutral by default (``same_group_other_rep``):
they are neither time-aligned positives nor other groups (A4). Same-set pairs with weight 0 (a disabled
source) are neutral too. Neutral pairs are left out of the numerator and the denominator.
"""
import torch


def pair_masks(set_ids, group_ids=None, exercise=None, negatives="all_other_groups",
               same_group_other_rep="neutral"):
    """Boolean (N, N) masks: same_set (excluding the diagonal) and negatives."""
    n = set_ids.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=set_ids.device)
    same = set_ids[:, None] == set_ids[None, :]
    neg = ~same
    if group_ids is not None and same_group_other_rep == "neutral":
        neg &= group_ids[:, None] != group_ids[None, :]
    elif same_group_other_rep not in ("neutral", "negative"):
        raise ValueError(same_group_other_rep)
    if negatives == "different_exercise_only":
        if exercise is None:
            raise ValueError("negatives=different_exercise_only needs exercise ids")
        neg &= exercise[:, None] != exercise[None, :]
    elif negatives != "all_other_groups":
        raise ValueError(negatives)
    return same & ~eye, neg


def cvdc_loss(proj, set_ids, weights, group_ids=None, exercise=None, temperature=0.1,
              negatives="all_other_groups", same_group_other_rep="neutral"):
    """proj: (N, D) L2-normalised projections; set_ids: (N,) ints, equal ids = views of one rep (positives);
    weights: (N, N) positive-pair weights (read only where the set ids match; 0 = neutral); group_ids,
    exercise: (N,) ints for the negative masks. Returns the scalar loss (0 with a gradient path when no anchor
    has a positive)."""
    z = proj.float()
    sim = z @ z.T / temperature
    same, neg = pair_masks(set_ids, group_ids, exercise, negatives, same_group_other_rep)
    w = weights.to(sim.dtype)
    pos = same & (w > 0)
    den = pos | neg
    log_den = torch.logsumexp(sim.masked_fill(~den, float("-inf")), dim=1, keepdim=True)
    log_prob = torch.where(pos, sim - log_den, torch.zeros_like(sim))
    n_pos = pos.sum(1)
    anchors = n_pos > 0
    if not anchors.any():
        return sim.sum() * 0.0
    per_anchor = -(torch.where(pos, w, torch.zeros_like(w)) * log_prob).sum(1) / n_pos.clamp(min=1)
    return per_anchor[anchors].mean()


@torch.no_grad()
def cvdc_stats(proj, set_ids, weights, group_ids=None, exercise=None, negatives="all_other_groups",
               same_group_other_rep="neutral", z=None) -> dict:
    """Monitoring (spec Phase 2 'Log'): mean cosine similarity of positive and negative pairs; per-dimension
    std of the L2-normalised projections (collapse detector: ~1/sqrt(D) when spread, -> 0 on collapse) and of
    the raw embedding z; alignment (mean squared distance of positive pairs) and uniformity
    (log mean exp(-2 * squared distance) over all pairs), Wang & Isola 2020."""
    p = proj.float()
    sim = p @ p.T
    same, neg = pair_masks(set_ids, group_ids, exercise, negatives, same_group_other_rep)
    pos = same & (weights > 0)
    off = ~torch.eye(len(p), dtype=torch.bool, device=p.device)
    sq = (2 - 2 * sim).clamp(min=0)
    out = {"pos_sim": sim[pos].mean().item() if pos.any() else float("nan"),
           "neg_sim": sim[neg].mean().item() if neg.any() else float("nan"),
           "proj_std": p.std(0).mean().item(),
           "align": sq[pos].mean().item() if pos.any() else float("nan"),
           "uniform": torch.log(torch.exp(-2 * sq[off]).mean()).item(),
           "n_pos_pairs": int(pos.sum().item()) // 2, "n_anchors": int((pos.sum(1) > 0).sum().item())}
    if z is not None:
        out["z_std"] = z.float().std(0).mean().item()
    return out
