"""Phase 6 (X6, user 2026-10-20): the patient against the PT reference, live.

  ReferencePlayer  the exercise's primary PT demo as a skeleton on Module III's template time axis: the demo rep
                   (on the demo's own side) whose DTW alignment to the template (the same alignment feature as
                   Module III, M8 / M17) is cheapest, its 2D keypoints (root-centred, torso-normalised, P5) and 3D world
                   landmarks averaged over the demo frames matched to every template frame; ``frame(j, side)`` mirrors
                   it (x -> -x, left <-> right) when the patient moves the other side (P8)
  OnlineAligner    open-end DTW of the live frames since the last finished rep against the template's alignment feature:
                   the template frame that matches the newest frame, kept monotonic within a rep; reset when the session
                   reports a new rep (and caught up on the frames after it) or after a hold at the template's end
  live_channels    Module III's channels for the newest frames (median over the last few frames, the raw world
                   landmarks), so they can be compared with the per-view targets and bands of the library (M8, M9,
                   M14) at the aligned template frame: the live angle difference patient - target, in or out of band
CPU only (a 120-frame DTW row per frame).
"""
import numpy as np
import pandas as pd

from cvdc.assess.dtw import dtw
from cvdc.assess.template import align_feature, channel_curves
from cvdc.config import data_path, exercise_cfg, resolve
from cvdc.data.skeleton import MP33

MIRROR33 = [MP33.index(n.replace("left", "@").replace("right", "left").replace("@", "right")) for n in MP33]

# where to write each channel's live difference: (joint name pattern, offset rows); {s} = moving side, {o} = other
CHANNEL_JOINT = {"shoulder_elevation": ("{s}_shoulder", 0), "horizontal_adduction": ("{s}_elbow", 1),
                 "elbow_extension": ("{s}_elbow", 0), "contralateral_elevation": ("{o}_shoulder", 0),
                 "trunk_lateral_lean": ("mid_hip", 0), "shoulder_asymmetry": ("{s}_shoulder", 1),
                 "forearm_elevation": ("{s}_wrist", 0), "hip_flexion": ("{s}_hip", 0), "knee_flexion": ("{s}_knee", 0),
                 "contralateral_hip_flexion": ("{o}_hip", 0), "contralateral_knee_flexion": ("{o}_knee", 0),
                 "trunk_thigh_angle": ("mid_hip", 0), "knee_flexion_both": ("{s}_knee", 1),
                 "hip_flexion_asymmetry": ("mid_hip", 1), "knee_flexion_asymmetry": ("{o}_knee", 1)}


SHORTC = {"shoulder_elevation": "elev", "horizontal_adduction": "h-add", "elbow_extension": "elbow",
          "contralateral_elevation": "other arm", "trunk_lateral_lean": "lean", "shoulder_asymmetry": "shrug",
          "forearm_elevation": "forearm", "hip_flexion": "hip", "knee_flexion": "knee",
          "contralateral_hip_flexion": "other hip", "contralateral_knee_flexion": "other knee",
          "trunk_thigh_angle": "trunk", "knee_flexion_both": "knees", "hip_flexion_asymmetry": "hip L-R",
          "knee_flexion_asymmetry": "knee L-R"}


def mirror33(X):
    """Mirror image of root-centred MediaPipe-33 keypoints (..., 33, C): x -> -x, left / right swapped."""
    Y = np.asarray(X, float)[..., MIRROR33, :].copy()
    Y[..., 0] = -Y[..., 0]
    return Y


def _on_template_nd(X, path, n_ref):
    """(n_ref, ...) mean of the rep frames matched to every template frame (path rows (rep frame, template frame))."""
    i, j = path[:, 0], path[:, 1]
    s = np.zeros((n_ref,) + X.shape[1:])
    c = np.zeros(n_ref)
    np.add.at(s, j, X[i])
    np.add.at(c, j, 1)
    return s / np.maximum(c, 1)[(slice(None),) + (None,) * (X.ndim - 1)]


class ReferencePlayer:
    def __init__(self, cfg, exercise, ref, ac):
        ref_idx = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
        rr = pd.read_csv(data_path(cfg, "reps_ref_csv"))
        row = ref_idx[(ref_idx.exercise == exercise) & (ref_idx.role == "primary")].iloc[0]
        with np.load(resolve(row.pre_path)) as z:
            xy, world = z["xy"].astype(float), z["world"].astype(float)
        reps = rr[(rr.ref_id == row.ref_id) & (rr.exercise == exercise)]
        self.side = reps.side.iloc[0]
        self.file = row.file
        self.view = (exercise_cfg(cfg, exercise).get("references") or {}).get("primary_view", "front")
        self.n = ref["n"]
        best = None
        for r in reps.itertuples():
            P = world[r.start:r.end + 1]
            ch = channel_curves(P, self.side, [ac["angle"]] if ac["angle"] in ac["channels"] else [], ac.get(
                "angle_min_proj", 0.25)) if ac.get("align") == "main" else {}
            cost, path = dtw(align_feature(P, self.side, ac, ch), ref["feat"])
            score = cost / len(path)
            if best is None or score < best[0]:
                best = (score, r.start, path)
        _, s0, path = best
        rows = path[:, 0] + s0
        p2 = np.c_[rows, path[:, 1]]
        self.xy = _on_template_nd(xy, p2, self.n)          # (n_ref, 33, 2)
        self.world = _on_template_nd(world, p2, self.n)    # (n_ref, 33, 3)

    def frame(self, j, side):
        """(xy (33, 2), mirrored?) of template frame j for a patient moving ``side``."""
        j = int(np.clip(j, 0, self.n - 1))
        mir = side is not None and side != self.side
        return (mirror33(self.xy[j]) if mir else self.xy[j]), mir


class OnlineAligner:
    def __init__(self, ref, ac, hold_s=0.6, fps=30.0, max_step=5, rest_frac=0.1):
        self.F = np.asarray(ref["feat"], float)
        self.ac = ac
        self.hold = int(round(hold_s * fps))
        self.max_step = int(max_step)
        # 'at rest': closer to the template's first frame than rest_frac x the template's largest excursion from it
        self.rest_thr = rest_frac * float(np.linalg.norm(self.F - self.F[0], axis=1).max())
        self.reset()

    def reset(self):
        self.D, self.i, self.j, self.at_end = None, 0, 0, 0

    def feature(self, world_frame, side):
        """The alignment feature of one raw frame (None if the frame has no pose)."""
        P = np.asarray(world_frame, float)[None]
        if not np.isfinite(P).all():
            return None
        ch = channel_curves(P, side, [self.ac["angle"]], self.ac.get("angle_min_proj", 0.25)) \
            if self.ac.get("align") == "main" else {}
        f = align_feature(P, side, self.ac, ch)[0]
        return f if np.isfinite(f).all() else None

    def step(self, f):
        """One new frame's feature -> the matching template frame (monotonic within the current rep)."""
        if f is None:
            return self.j
        c = np.linalg.norm(self.F - f, axis=1)
        n = len(c)
        if self.D is None:
            D = np.cumsum(c)
        else:
            prev = self.D
            D = np.empty(n)
            D[0] = prev[0] + c[0]
            diag = np.minimum(prev[1:], prev[:-1]) + c[1:]
            for k in range(1, n):  # D[k] = c[k] + min(prev[k], D[k-1], prev[k-1])
                D[k] = min(diag[k - 1], D[k - 1] + c[k])
        self.D, self.i = D, self.i + 1
        # the template starts and ends at rest, so a resting patient matches both ends: while the patient is still
        # near the template's start (and the alignment is in its first 15%), stay anchored there; otherwise the
        # length-normalised open-end DTW position, monotonic, advancing at most max_step template frames per frame
        if self.j < 0.15 * n and c[0] < self.rest_thr:
            self.reset()
            return 0
        jj = int(np.argmin(D / (self.i + np.arange(n) + 1)))
        self.j = int(np.clip(jj, self.j, self.j + self.max_step))
        self.at_end = self.at_end + 1 if self.j >= n - 2 else 0
        if self.at_end > self.hold:  # the rep ended and the patient holds or rests: start the next one
            self.reset()
        return self.j


def live_channels(world_frames, side, ac, k=5):
    """{channel: median over the last k raw frames} of Module III's channels (degrees)."""
    from cvdc.assess.angles import ASSESS_ANGLES
    P = np.asarray(world_frames[-k:], float)
    P = P[np.isfinite(P).all(axis=(1, 2))]
    if not len(P):
        return {}
    chans = [c for c in ac["channels"] if c in ASSESS_ANGLES]  # E06's Hands wrist angle is not drawn live
    ch = channel_curves(P, side, chans, ac.get("angle_min_proj", 0.25))
    return {c: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan for c, v in ch.items()}


def compare_figure(sess, player, k, view):
    """X6: the patient's rep k against the PT reference at 5 phases of the template (DTW-aligned as in Module III):
    patient skeleton (black) with the reference as a ghost (cyan; mirrored for the other side), both root-centred and
    torso-normalised, and each channel's difference (patient - target, deg) at its joint, green inside the band."""
    import matplotlib.pyplot as plt

    from cvdc.data.skeleton import MP_DRAW_EDGES, J
    d, r = sess.debug[k], sess.reps[k]
    al, cmp_ = d["aligned"], d["cmp"]
    n = sess.ref["n"]
    side = sess.side or "right"
    o = "left" if side == "right" else "right"
    band = sess.ac["band_k"]
    phases = np.linspace(0, n - 1, 5).round().astype(int)
    fig, axes = plt.subplots(1, len(phases), figsize=(3.0 * len(phases), 3.6), squeeze=False)
    for ax, j in zip(axes[0], phases):
        pf = r["start"] + int(al["span"][j, 0])
        P = sess.pre["xy"][pf]
        R, mir = player.frame(j, side)
        for X, col, lw, a in ((R, "#2bb5d8", 4, 0.45), (P, "k", 2, 1.0)):
            for e0, e1 in MP_DRAW_EDGES:
                ax.plot(X[[e0, e1], 0], X[[e0, e1], 1], color=col, lw=lw, alpha=a, solid_capstyle="round")
        for c, (jn, row) in CHANNEL_JOINT.items():
            if c not in cmp_["dev"]:
                continue
            dv, s = cmp_["dev"][c][j], sess.lib[view]["sigma"][c][j] * band
            if not (np.isfinite(dv) and np.isfinite(s)):
                continue
            name = jn.format(s=side, o=o)
            p = (P[J["left_hip"]] + P[J["right_hip"]]) / 2 if name == "mid_hip" else P[J[name]]
            ax.text(p[0] + 0.08, p[1] + 0.22 * row, f"{SHORTC.get(c, c)} {dv:+.0f}", fontsize=7,
                    color="green" if abs(dv) <= s else "red")
        ax.set_title(f"template {100 * j / (n - 1):.0f}%" + (" (ref. mirrored)" if mir else ""), fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    note = f"cyan = PT reference ({player.file}, {player.view} view)"
    if view != "F" or player.view != "front":
        note += f"; this video is the {view} view: compare the view-corrected 3D-angle differences, not the picture"
    fig.suptitle(note, fontsize=8)
    fig.tight_layout()
    return fig
