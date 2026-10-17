"""Dynamic time warping on multichannel sequences (Euclidean local cost), with the warping path.

Used by the CVDC pair builder to time-align rep-matched pairs on 3D joint angles (C1 c) and, in Phase 4, by
Module III to align a rep to its template.
"""
import numpy as np


def dtw(a, b, band=None):
    """a: (n, d), b: (m, d) (1-D inputs are treated as d = 1). Returns (cost, path): the accumulated cost of
    the optimal alignment and its path as an (L, 2) int array of (i, j) pairs from (0, 0) to (n-1, m-1).
    ``band``: optional Sakoe-Chiba half-width in frames (on the diagonal scaled to the two lengths)."""
    a = np.asarray(a, float).reshape(len(a), -1)
    b = np.asarray(b, float).reshape(len(b), -1)
    n, m = len(a), len(b)
    C = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        if band is None:
            lo, hi = 1, m
        else:
            c = (i - 1) * (m - 1) / max(n - 1, 1)
            lo, hi = max(1, int(np.floor(c - band)) + 1), min(m, int(np.ceil(c + band)) + 1)
        for j in range(lo, hi + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, [(n - 1, m - 1)]
    while (i, j) != (1, 1):
        steps = ((i - 1, j - 1), (i - 1, j), (i, j - 1))
        i, j = min(steps, key=lambda s: D[s])
        path.append((i - 1, j - 1))
    return float(D[n, m]), np.asarray(path[::-1])


def warp_positions(path, n):
    """For each index 0..n-1 of the first sequence, the mean matched index of the second sequence: the
    fractional positions at which to sample the second sequence to put it on the first one's time axis."""
    path = np.asarray(path)
    pos = np.zeros(n)
    np.add.at(pos, path[:, 0], path[:, 1])
    return pos / np.bincount(path[:, 0], minlength=n)


def sample_at(X, pos):
    """Linear interpolation of X (m, ...) along axis 0 at fractional positions pos (k,)."""
    X = np.asarray(X)
    pos = np.clip(np.asarray(pos, float), 0, len(X) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(X) - 1)
    w = (pos - i0).reshape((-1,) + (1,) * (X.ndim - 1))
    return (1 - w) * X[i0] + w * X[i1]
