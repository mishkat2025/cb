"""Evaluation metrics with subject-level bootstrap CIs (E1, E9, E17).

Every metric takes item weights W of shape (B, n): a row of ones gives the point estimate, and a matrix of
subject multiplicities gives B bootstrap replicates in one call (an item's weight = how often its subject was
drawn). Resampling subjects keeps all reps, views and videos of a subject together, as the splits do, and the
same draws are used for every method, so differences between methods are paired.

  subject_draws    (B, S) subject multiplicities, drawn with replacement within each stratum (e.g. expert /
                   non-expert, so a pooled resample keeps the group sizes)
  regression       RMSE, MAE, Pearson r, Spearman rho (Pearson on weighted mid-ranks)
  classification   accuracy, macro F1 and Cohen's kappa (unweighted) from weighted confusion matrices
  vci_sums, vci_from_sums   the Viewpoint Consistency Index (E13) from per-subject sums of cosine similarities,
                   so it can be pooled over folds and bootstrapped over subjects without recomputing similarities
  interval         percentile interval of the replicates
  icc_weighted     one-way random-effects ICC(1) over sets (e.g. one performance seen from several views), with
                   set multiplicities, for the view stability of a measure (Phase 4)
  mean_abs_diff_weighted   weighted mean |x_i - x_j| over pairs (Gini mean difference)
"""
import numpy as np


def subject_draws(subjects, strata, n_boot, seed=0):
    """subjects: (S,) unique subject keys; strata: (S,) stratum of each subject. Returns (n_boot, S) counts."""
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    M = np.zeros((n_boot, len(subjects)))
    for s in sorted(set(strata)):
        idx = np.flatnonzero(strata == s)
        k = len(idx)
        d = rng.integers(0, k, (n_boot, k)) + k * np.arange(n_boot)[:, None]
        M[:, idx] = np.bincount(d.ravel(), minlength=n_boot * k).reshape(n_boot, k)
    return M


def item_weights(item_subjects, subjects, M):
    """Expand (B, S) subject multiplicities to (B, n) item weights."""
    pos = {s: i for i, s in enumerate(subjects)}
    return M[:, [pos[s] for s in item_subjects]]


def interval(rep, ci=0.95):
    rep = np.asarray(rep, float)
    rep = rep[np.isfinite(rep)]
    if not len(rep):
        return np.nan, np.nan
    a = (1 - ci) / 2
    return float(np.quantile(rep, a)), float(np.quantile(rep, 1 - a))


def _wmean(x, W):
    return (W * x).sum(-1) / W.sum(-1)


def wcorr(x, y, W):
    """Weighted Pearson correlation per row of W; x, y: (n,) or (B, n). NaN where either side is constant."""
    dx = x - _wmean(x, W)[:, None]
    dy = y - _wmean(y, W)[:, None]
    vx, vy = (W * dx * dx).sum(-1), (W * dy * dy).sum(-1)
    den = np.sqrt(vx * vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (W * dx * dy).sum(-1) / den
    tiny = 1e-12 * np.maximum(W.sum(-1), 1) ** 2
    r[~(den > tiny)] = np.nan
    return r


def wranks(x, W):
    """Weighted mid-ranks per row of W: tied values share the mean rank of their block; an item of weight w counts
    as w copies. x: (n,) without NaN; returns (B, n)."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    new = np.r_[True, xs[1:] != xs[:-1]]
    starts, gid = np.flatnonzero(new), np.cumsum(new) - 1
    gs = np.add.reduceat(W[:, order], starts, axis=1)
    grank = np.cumsum(gs, 1) - gs + (gs + 1) / 2
    R = np.empty(W.shape)
    R[:, order] = grank[:, gid]
    return R


def regression(y, p, W, chunk=500):
    """y, p: (n,) true and predicted values; W: (B, n). Returns dict of (B,) arrays: rmse, mae, pearson, spearman."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {k: [] for k in ("rmse", "mae", "pearson", "spearman")}
    for i in range(0, len(W), chunk):
        w = W[i:i + chunk]
        e = p - y
        out["rmse"].append(np.sqrt(_wmean(e * e, w)))
        out["mae"].append(_wmean(np.abs(e), w))
        out["pearson"].append(wcorr(y, p, w))
        out["spearman"].append(wcorr(wranks(y, w), wranks(p, w), w))
    return {k: np.concatenate(v) for k, v in out.items()}


def classification(t, p, W, k, labels=None):
    """t, p: (n,) int classes in [0, k); W: (B, n). Macro F1 over ``labels`` (default: the classes present in t or
    p), F1 = 0 for a class with no true or predicted item in a replicate. Returns dict of (B,) arrays."""
    t, p = np.asarray(t, int), np.asarray(p, int)
    labels = sorted(set(t) | set(p)) if labels is None else [int(x) for x in labels]
    onehot = np.zeros((len(t), k * k))
    onehot[np.arange(len(t)), t * k + p] = 1
    C = (W @ onehot).reshape(len(W), k, k)  # rows: true class, columns: predicted
    tp = np.diagonal(C, axis1=1, axis2=2)
    fp, fn = C.sum(1) - tp, C.sum(2) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        n = C.sum((1, 2))
        po = tp.sum(1) / n
        pe = (C.sum(2) * C.sum(1)).sum(1) / n ** 2
        kappa = np.where(pe < 1 - 1e-12, (po - pe) / (1 - pe), np.nan)
    return {"accuracy": po, "macro_f1": f1[:, labels].mean(1), "kappa": kappa}


def vci_sums(z, key, view, group, subj, n_subj, center=None, pair_ok=None):
    """Per-subject sums for the VCI (E13; cvdc/eval/consistency.py defines it on one set of embeddings).
    z: (n, D) embeddings; key: same-performance id (rep set); view, group ((exercise, subject, variation) code):
    (n,) ids; subj: (n,) subject index in [0, n_subj); center: vector subtracted first (vci_centred) or None;
    pair_ok: (n, n) bool, only these pairs count (both same and other pairs; e.g. one member in a held-out view and
    the other in a seen view, E24), None = all.
    Same pairs: same key, different view (always within one subject). Other pairs: different groups. Ordered pairs,
    as in viewpoint_consistency. Returns dict(same_sum (S,), same_cnt (S,), other_sum (S, S), other_cnt (S, S))."""
    z = np.asarray(z, np.float64)
    if center is not None:
        z = z - np.asarray(center, np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    sim = z @ z.T
    key, view, group, subj = (np.asarray(a) for a in (key, view, group, subj))
    same = (key[:, None] == key[None, :]) & (view[:, None] != view[None, :])
    if (subj[:, None] != subj[None, :])[same].any():
        raise ValueError("a same-performance pair spans two subjects")
    other = group[:, None] != group[None, :]
    if pair_ok is not None:
        same, other = same & pair_ok, other & pair_ok
    S = np.zeros((len(z), n_subj))
    S[np.arange(len(z)), subj] = 1
    return {"same_sum": S.T @ (sim * same).sum(1), "same_cnt": S.T @ same.sum(1).astype(float),
            "other_sum": S.T @ (sim * other) @ S, "other_cnt": S.T @ other.astype(float) @ S}


def add_sums(a, b):
    return {k: a[k] + b[k] for k in a} if a else dict(b)


def vci_from_sums(s, M=None):
    """VCI = mean same-pair cosine - mean other-pair cosine, for subject multiplicities M (B, S) (an ordered pair of
    subjects drawn m_s and m_t times counts m_s * m_t times). M None = point estimate. Returns dict of (B,) arrays
    (or floats): vci, same, other."""
    point = M is None
    M = np.ones((1, len(s["same_cnt"]))) if point else M
    M2 = M * M
    with np.errstate(invalid="ignore", divide="ignore"):
        same = (M2 @ s["same_sum"]) / (M2 @ s["same_cnt"])
        other = np.einsum("bs,st,bt->b", M, s["other_sum"], M) / np.einsum("bs,st,bt->b", M, s["other_cnt"], M)
    out = {"vci": same - other, "same": same, "other": other}
    return {k: float(v[0]) for k, v in out.items()} if point else out


def icc_weighted(x, set_code, Wset):
    """One-way random-effects ICC(1) over sets (performances) with set multiplicities Wset (B, S); x: (n,) values,
    set_code: (n,) in [0, S). Unbalanced ANOVA estimator with k0."""
    S = Wset.shape[1]
    n_s = np.bincount(set_code, minlength=S).astype(float)
    s1 = np.bincount(set_code, weights=x, minlength=S)
    s2 = np.bincount(set_code, weights=x * x, minlength=S)
    N = Wset @ n_s
    a = Wset.sum(1)
    grand = (Wset @ s1) / N
    ssb = Wset @ (s1 ** 2 / n_s) - N * grand ** 2
    ssw = Wset @ (s2 - s1 ** 2 / n_s)
    msb, msw = ssb / (a - 1), ssw / (N - a)
    k0 = (N - (Wset @ n_s ** 2) / N) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def mean_abs_diff_weighted(x, w):
    """Weighted mean |x_i - x_j| over ordered pairs i != j (rows of w: (B, n)), via sorting."""
    o = np.argsort(x)
    xs, ws = x[o], w[:, o]
    C = np.cumsum(ws, 1)
    tot = C[:, -1:]
    below = C - ws
    above = tot - C
    s = 2 * (ws * xs * (below - above)).sum(1)
    pairs = tot[:, 0] ** 2 - (ws ** 2).sum(1)
    return s / pairs
