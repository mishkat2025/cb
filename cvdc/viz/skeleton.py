"""Skeleton-only drawing (no video pixels are ever used: D10)."""
import numpy as np

from ..data.skeleton import LEFT, MP_DRAW_EDGES, RIGHT

COL_L, COL_R, COL_C = "#2b6cb0", "#dd6b20", "#4a5568"


def draw_skeleton(ax, xy, gap=None, lw=2.0, ms=3.5, alpha=1.0):
    """Draw one frame of 2D keypoints (33,2) in image orientation (y down). Interpolated joints are hollow."""
    for a, b in MP_DRAW_EDGES:
        if np.isfinite(xy[[a, b]]).all():
            c = COL_L if a in LEFT and b in LEFT else COL_R if a in RIGHT and b in RIGHT else COL_C
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], "-", color=c, lw=lw, alpha=alpha, solid_capstyle="round")
    pts = sorted({j for e in MP_DRAW_EDGES for j in e})
    for j in pts:
        if not np.isfinite(xy[j]).all():
            continue
        c = COL_L if j in LEFT else COL_R if j in RIGHT else COL_C
        hollow = gap is not None and gap[j] > 0
        ax.plot(*xy[j], "o", ms=ms, mfc="white" if hollow else c, mec=c, alpha=alpha)


def montage(fig, xy_seq, frames, gap_seq=None, titles=None, row=0, nrows=1):
    """One row of skeleton panels sharing limits; xy_seq (T,33,2) in root-centred torso units."""
    sel = xy_seq[frames]
    pts = sorted({j for e in MP_DRAW_EDGES for j in e})
    finite = sel[:, pts][np.isfinite(sel[:, pts]).all(-1)]
    lo, hi = finite.min(0) - 0.2, finite.max(0) + 0.2
    axes = []
    for k, f in enumerate(frames):
        ax = fig.add_subplot(nrows, len(frames), row * len(frames) + k + 1)
        draw_skeleton(ax, xy_seq[f], None if gap_seq is None else gap_seq[f])
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(hi[1], lo[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if titles:
            ax.set_title(titles[k], fontsize=8)
        axes.append(ax)
    return axes
