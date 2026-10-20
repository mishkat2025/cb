"""Phase 6 (X6, X7, user 2026-10-20): the patient against the PT reference, live.

  ReferencePlayer  what the panel and the ghost show (X7): the PT's own 2D keypoints (root-centred, torso-normalised,
                   P5) of one real rep of the exercise's primary PT demo, the rep whose DTW alignment to Module III's
                   template is cheapest, **extended to the PT's rest on both sides** (the segmented rep starts and ends
                   at 10% of its amplitude, P6, so E01's demo rep starts and ends with the arm still ~20 deg out while
                   the PT's arm hangs at ~9 deg at rest): from the rep's start back, and from its end forward, to the
                   nearest frame where the moving arm is still. The averaged 3D template stays for scoring; ``tmap``
                   gives, per displayed frame, the template frame it matches (DTW of the frames the template covers:
                   the rep, E03 its 90/90 working phase, M3), -1 in the rest lead-in and return. ``frame(k, side)``
                   mirrors it (x -> -x, left <-> right) when the patient moves the other side (P8). E03 (X8,
                   ``references.display``): the rest pose is the demo's 90/90 hold and the return its rotation played
                   back, since the demo rests with the arm at the side while the patients hold 90/90
  RepTracker       the patient's state: at rest or in a rep, from the distance of the live alignment feature (Module
                   III's, M8 / M17) to the patient's own rest feature, with hysteresis; during a rep the displayed PT
                   frame by DTW of the frames since the patient left rest against the PT rep (open begin before the PT's
                   peak, open end, monotonic, at most a few PT frames per video frame). At rest the PT rest pose is
                   shown and nothing is compared
  live_channels    Module III's channels for the newest frames (median over the last few frames, the raw world
                   landmarks), so they can be compared with the per-view targets and bands of the library (M8, M9,
                   M14) at the matched template frame: the live angle difference patient - target, in or out of band
CPU only (a DTW row of ~150-250 PT frames per video frame).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from cvdc.assess.dtw import dtw
from cvdc.assess.template import align_feature, channel_curves
from cvdc.config import data_path, exercise_cfg, resolve
from cvdc.data.skeleton import MP33, J

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


def still_frames(xy, side, thr=0.01, k=5):
    """(n,) True where the ``side`` arm (elbow and wrist, 2D, root-centred torso units) is still: the k-frame running
    median of its mean per-frame displacement is below ``thr`` (PT demos: < 0.006 at rest, > 0.013 while moving)."""
    limb = np.asarray(xy, float)[:, [J[f"{side}_elbow"], J[f"{side}_wrist"]]]
    sp = np.r_[0.0, np.linalg.norm(np.diff(limb, axis=0), axis=2).mean(1)]
    return median_filter(np.nan_to_num(sp, nan=np.inf), size=k, mode="nearest") < thr


def rest_window(still, start, end, lo, hi):
    """The rep start..end extended to rest: back from ``start`` to the nearest still frame (not before ``lo``) and
    forward from ``end`` to the nearest still frame (not after ``hi``)."""
    a, b = int(start), int(end)
    while a > lo and not still[a]:
        a -= 1
    while b < hi and not still[b]:
        b += 1
    return a, b


def hold_frame(still, signal, lo, hi):
    """X8: the last frame of the still run in lo..hi-1 at the lowest rep-signal level (the hold at the movement's
    rest, e.g. E03's 90/90 hold before the rotation; a hold at the top has a high signal)."""
    s = np.asarray(signal, float)
    idx = np.arange(int(lo), int(hi))
    runs = [g for g in np.split(idx, np.flatnonzero(~still[idx])) if len(g)]
    runs = [g[1:] if not still[g[0]] else g for g in runs]   # each split starts at a moving frame
    runs = [g for g in runs if len(g) >= 5] or [g for g in runs if len(g)]
    if not runs:
        return int(lo)
    return int(min(runs, key=lambda g: np.nanmean(s[g]))[-1])


def unit_blocks(f, ac):
    """Direction features back to unit vectors per 3-d segment block (a mean of unit vectors is shorter)."""
    f = np.asarray(f, float)
    if ac.get("align", "arm") == "main":
        return f
    g = f.reshape(f.shape[:-1] + (-1, 3))
    return (g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)).reshape(f.shape)


def features(P, side, ac):
    """Module III's alignment feature (M8, M17) of world-landmark frames P (n, 33, 3)."""
    P = np.asarray(P, float)
    ch = channel_curves(P, side, [ac["angle"]], ac.get("angle_min_proj", 0.25)) if ac.get("align") == "main" else {}
    return align_feature(P, side, ac, ch)


def live_feature(world_frames, side, ac, k=3):
    """The alignment feature of the newest frame, from the median of the last k raw world frames (None if none has a
    pose)."""
    P = np.asarray(world_frames[-k:], float)
    P = P[np.isfinite(P).all(axis=(1, 2))]
    if not len(P):
        return None
    f = features(np.median(P, 0)[None], side, ac)[0]
    return f if np.isfinite(f).all() else None


class ReferencePlayer:
    def __init__(self, cfg, exercise, ref, ac, still_thr=0.01):
        from cvdc.assess.run import working_phase
        from cvdc.data.reps import video_info
        ref_idx = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
        rr = pd.read_csv(data_path(cfg, "reps_ref_csv"))
        row = ref_idx[(ref_idx.exercise == exercise) & (ref_idx.role == "primary")].iloc[0]
        with np.load(resolve(row.pre_path)) as z:
            pre = {k: z[k] for k in z.files}
        xy, world = pre["xy"].astype(float), pre["world"].astype(float)
        reps = rr[(rr.ref_id == row.ref_id) & (rr.exercise == exercise)].sort_values("start")
        self.side = side = reps.side.iloc[0]
        self.file = row.file
        refs = exercise_cfg(cfg, exercise).get("references") or {}
        self.view = refs.get("primary_view", "front")
        disp = refs.get("display") or {}                # X8: per-exercise display choices (E03)
        self.rest_label, self.return_label = disp.get("rest_label"), disp.get("return_label")
        self.fps = float(cfg["preprocess"]["fps"])
        best = None
        for r in reps.itertuples():  # the demo rep whose alignment to the template is cheapest
            cost, path = dtw(features(world[r.start:r.end + 1], side, ac), ref["feat"])
            if best is None or cost / len(path) < best[0]:
                best = (cost / len(path), r)
        r = best[1]
        self.rep = (int(r.start), int(r.peak), int(r.end))
        # the frames the template covers (the rep; E03 its 90/90 working phase, M3)
        ws, we = r.start, r.end
        if ac.get("template_working_phase"):
            ws, we = working_phase(world, r.start, r.end, side, ac["template_working_phase"]) or (ws, we)
        still = still_frames(xy, side, still_thr)
        if disp.get("rest") == "hold":
            # X8: rest pose = the demo's still hold at the movement's rest level inside the covered frames (E03: the
            # 90/90 hold before the rotation); the return to it = the rotation played back from the frame that
            # matches the template's end (the demo itself lowers the arm from the top)
            ang, _ = video_info(pre, exercise_cfg(cfg, exercise))
            h = hold_frame(still, ang[side], ws, int(r.peak))
            up = np.arange(h, int(r.peak) + 1)
            f_ret = int(up[np.argmin(np.linalg.norm(features(world[up], side, ac) - ref["feat"][-1], axis=1))])
            back = np.arange(f_ret, h - 1, -1) if disp.get("return") == "reversed" else np.arange(0)
            self.seq = np.r_[np.arange(ws, we + 1), back].astype(int)
            self.covered, self.rest_k = (0, we - ws), h - ws
        else:
            i = list(reps.start).index(r.start)
            lo = int(reps.end.iloc[i - 1]) + 1 if i > 0 else 0
            hi = int(reps.start.iloc[i + 1]) - 1 if i + 1 < len(reps) else len(xy) - 1
            k0, k1 = rest_window(still, r.start, r.end, lo, hi)
            self.seq = np.arange(k0, k1 + 1)
            self.covered, self.rest_k = (ws - k0, we - k0), 0
        self.xy = xy[self.seq]                         # (m, 33, 2): the PT's own keypoints, rest -> peak -> rest
        self.world = world[self.seq]                   # (m, 33, 3): its 3D world landmarks (root-centred, torso units)
        self.feat = features(world[self.seq], side, ac)
        self.n, self.n_template = len(self.xy), int(ref["n"])
        self.peak = int(np.flatnonzero(self.seq == r.peak)[0])
        self.tail = (self.covered[1] + 1, self.n)      # display frames of the return (not compared)
        # covered frames -> template frame by DTW
        _, path = dtw(features(world[ws:we + 1], side, ac), ref["feat"])
        self.tmap = np.full(self.n, -1, int)
        for f in range(we - ws + 1):
            self.tmap[self.covered[0] + f] = int(np.median(path[path[:, 0] == f, 1]))
        sw = np.linalg.norm(self.xy[:, J["left_shoulder"]] - self.xy[:, J["right_shoulder"]], axis=1)
        self.shoulder_w = float(np.median(sw))           # torso units (the ghost's scale, X7)
        # the live time axis: the PT's rest lead-in, Module III's template (so the live phase is the scoring
        # alignment, M8), the PT's return to rest; per axis frame the displayed PT frame and the template frame
        L, tail = self.covered[0], np.arange(*self.tail)
        self.axis_feat = np.concatenate([self.feat[:L], np.asarray(ref["feat"], float), self.feat[tail]])
        self.axis_display = np.r_[np.arange(L), [self.frame_at_template(j) for j in range(self.n_template)],
                                  tail].astype(int)
        self.axis_template = np.r_[np.full(L, -1), np.arange(self.n_template), np.full(len(tail), -1)].astype(int)
        pf = np.sort(np.asarray(ref.get("peak_frames", []), int))
        runs = np.split(pf, np.flatnonzero(np.diff(pf) > 1) + 1) if len(pf) else []
        self.template_peak = max(runs, key=len) if runs else np.array([self.n_template // 2])  # longest peak run
        self.axis_begin = L + int(self.template_peak[0])

    def frame(self, k, side):
        """(xy (33, 2), mirrored?) of displayed PT frame k for a patient moving ``side``."""
        k = int(np.clip(k, 0, self.n - 1))
        mir = side is not None and side != self.side
        return (mirror33(self.xy[k]) if mir else self.xy[k]), mir

    def template_frame(self, k):
        """The template frame displayed PT frame k matches (-1: rest lead-in / return, not compared)."""
        return int(self.tmap[int(np.clip(k, 0, self.n - 1))])

    def frame_at_template(self, j):
        """The displayed PT frame that matches template frame j (the middle one if several)."""
        ks = np.flatnonzero(self.tmap == j)
        if not len(ks):
            ok = np.flatnonzero(self.tmap >= 0)
            ks = ok[[int(np.argmin(np.abs(self.tmap[ok] - j)))]]
        return int(ks[len(ks) // 2])


def rest_feature(world, signal, side, ac, polarity="peak", frac=0.1, recent=300):
    """The patient's rest feature: the mean alignment feature of the frames in the rest band of the session's rep
    signal (within ``frac`` of the p5-p95 range from the rest end, the segmentation's own rest level, P6), from the
    last ``recent`` frames when 5 of them are in the band (a patient's rest pose drifts over a set). None before there
    are 5 such frames."""
    s = np.asarray(signal, float)
    ok = np.isfinite(s)
    if ok.sum() < 5:
        return None
    lo, hi = np.percentile(s[ok], [5, 95])
    band = ok & ((s <= lo + frac * (hi - lo)) if polarity != "valley" else (s >= hi - frac * (hi - lo)))
    late = band & (np.arange(len(s)) >= len(s) - recent)
    band = late if late.sum() >= 5 else band
    f = features(np.asarray(world, float)[band], side, ac)
    f = f[np.isfinite(f).all(1)]
    return unit_blocks(f.mean(0), ac) if len(f) >= 5 else None


class RepTracker:
    """X7: at rest or in a rep, and the frame of the live axis (``ReferencePlayer.axis_*``: the PT's rest lead-in,
    Module III's template, the PT's return). ``push(feature)`` once per video frame -> (in_rep, axis frame).

    At rest / in a rep: the progress p of the live feature from the patient's rest feature (``set_rest``, from the
    session's rep signal; until the session has one, the mean feature of the first ``first_n`` frames with a pose)
    toward the template's peak feature, p = (f - rest) . (peak - rest) / |peak - rest|^2 (0 at rest, 1 at the PT's
    peak; a distance alone also counts moving away from the peak, e.g. an arm dropped below its usual rest): a rep
    starts after ``enter_n`` frames with p > ``enter`` and ends after ``exit_n`` frames with p < ``exit``. During a
    rep the axis frame comes from a DTW of the frames since the patient last was at rest (replayed
    when the rep starts) against the axis features: open begin anywhere before the template's peak (a patient may rest
    elsewhere than the PT, e.g. E03's 90/90 hold vs the demo's arm at the side), open end, length-normalised cost,
    monotonic, at most ``max_step`` axis frames per video frame; after ``hold_s`` at the axis end the alignment
    restarts (a next rep without a rest in between)."""

    def __init__(self, player, ref_feat, fps=30.0, enter=0.25, exit=0.18, enter_n=2, exit_n=6, max_step=12,
                 hold_s=0.6, replay_s=4.0, first_n=10, ac=None):
        self.F = np.asarray(player.axis_feat, float)
        self.n = len(self.F)
        self.k_begin = max(1, int(player.axis_begin))
        self.ac = ac or {}
        self.peak_f = unit_blocks(np.asarray(ref_feat, float)[player.template_peak].mean(0), self.ac)
        self.enter, self.exit, self.enter_n, self.exit_n = enter, exit, int(enter_n), int(exit_n)
        self.max_step, self.hold, self.replay = int(max_step), int(round(hold_s * fps)), int(round(replay_s * fps))
        self.first_n, self.fps = int(first_n), float(fps)
        self.reset()

    def reset(self):
        self.rest_f, self.hist, self.from_session = None, [], False
        self.in_rep, self.cnt, self.last_rest = False, 0, -1
        self._dtw_reset()

    def _dtw_reset(self):
        self.D = self.O = None
        self.i, self.k, self.at_end = 0, 0, 0

    def set_rest(self, rest_f):
        if rest_f is not None:
            self.rest_f, self.from_session = np.asarray(rest_f, float), True

    def _provisional_rest(self):
        """Before the session's first estimate: the mean feature of the first frames with a pose (the segmentation
        also takes the first second as rest, P6c)."""
        f = [g for g in self.hist if g is not None][:self.first_n]
        if len(f) >= self.first_n:
            self.rest_f = unit_blocks(np.mean(f, 0), self.ac)

    def progress(self, f):
        """0 at the patient's rest, 1 at the template's peak (projection on the rest -> peak direction)."""
        u = self.peak_f - self.rest_f
        return float((np.asarray(f, float) - self.rest_f) @ u / max(float(u @ u), 1e-9))

    def rebuild(self, feats, rest_f=None, replay_s=10.0):
        """The moving side is known (or changed): the history's features recomputed for it, and its last
        ``replay_s`` seconds replayed through the state machine (a rep under way when the side becomes known)."""
        feats = list(feats)
        self.reset()
        self.set_rest(rest_f)
        n_old = max(0, len(feats) - int(round(replay_s * self.fps)))
        self.hist, self.last_rest = feats[:n_old], n_old - 1
        for f in feats[n_old:]:
            self.push(f)

    def push(self, f):
        self.hist.append(f)
        idx = len(self.hist) - 1
        if not self.from_session and self.rest_f is None:
            self._provisional_rest()
        if self.rest_f is None:
            return False, 0
        if f is None:  # no pose: keep the state
            return self.in_rep, (self.k if self.in_rep else 0)
        p = self.progress(f)
        if not self.in_rep:
            if p < self.exit:
                self.last_rest = idx
            self.cnt = self.cnt + 1 if p > self.enter else 0
            if self.cnt >= self.enter_n:
                self.in_rep, self.cnt = True, 0
                self._dtw_reset()
                for g in self.hist[max(self.last_rest + 1, idx - self.replay):idx + 1]:
                    self._dtw_step(g)
            return self.in_rep, (self.k if self.in_rep else 0)
        self.cnt = self.cnt + 1 if p < self.exit else 0
        if self.cnt >= self.exit_n:
            self.in_rep, self.cnt, self.last_rest = False, 0, idx
            self._dtw_reset()
            return False, 0
        self._dtw_step(f)
        return True, self.k

    def restart(self, from_idx):
        """A rep has just been reported while the patient is in a rep: realign from the frame after its end (only if
        the patient went straight on into this rep; a rep started after a rest is already aligned from that rest)."""
        if self.in_rep and self.last_rest + 1 < int(from_idx):
            self._dtw_reset()
            for g in self.hist[max(int(from_idx), len(self.hist) - self.replay):]:
                self._dtw_step(g)

    def _dtw_step(self, f):
        if f is None:
            return
        c = np.linalg.norm(self.F - f, axis=1)
        n = self.n
        if self.D is None:  # open begin: any PT frame before the peak
            D = np.where(np.arange(n) < self.k_begin, c, np.inf)
            O = np.arange(n)
            self.i, self.k = 1, int(np.argmin(D))
        else:
            prev, po = self.D, self.O
            D, O = np.empty(n), np.empty(n, int)
            D[0], O[0] = prev[0] + c[0], po[0]
            for k in range(1, n):  # D[k] = c[k] + min(prev[k], D[k-1], prev[k-1]), carrying the path's origin
                m, o = prev[k], po[k]
                if D[k - 1] < m:
                    m, o = D[k - 1], O[k - 1]
                if prev[k - 1] < m:
                    m, o = prev[k - 1], po[k - 1]
                D[k], O[k] = m + c[k], o
            self.i += 1
            jj = int(np.argmin(D / (self.i + np.arange(n) - O + 1)))
            self.k = int(np.clip(jj, self.k, self.k + self.max_step))
        self.D, self.O = D, O
        self.at_end = self.at_end + 1 if self.k >= n - 2 else 0
        if self.at_end > self.hold:
            self._dtw_reset()


def angle_diffs(live, targets, sigma, j, band_k):
    """{channel: (patient - target at template frame j, within the band?, patient, target)} (X6)."""
    out = {}
    for c, v in live.items():
        if c not in targets or not np.isfinite(v):
            continue
        t, s = targets[c][j], sigma[c][j] * band_k
        if np.isfinite(t) and np.isfinite(s):
            out[c] = (v - t, abs(v - t) <= s, v, t)
    return out


class LiveReference:
    """The live loop's reference logic (X7), shared by demo/live.py and the checks. Per video frame, after
    ``PoseStream.process`` (and, every 0.5 s, ``OnlineSession.update`` followed by ``after_update``), ``frame``
    returns {in_rep, k (displayed PT frame), j (template frame, -1 = not compared), diffs, side}. Until the session's
    first update (2 s) gives the moving side, the side is provisional: the arm whose alignment feature has moved
    further from its first frames."""

    def __init__(self, cfg, sess, view):
        from cvdc.assess.module3 import channel_targets
        self.player = ReferencePlayer(cfg, sess.ex, sess.ref, sess.ac)
        self.tracker = RepTracker(self.player, sess.ref["feat"], cfg["preprocess"]["fps"], ac=sess.ac)
        self.targets, self.sigma = channel_targets(sess.ref, sess.lib[view], sess.ac), sess.lib[view]["sigma"]
        self.ac = sess.ac
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.side, self.locked, self.n_seen = None, False, 0
        self._pf, self._f0, self._exc = {"right": [], "left": []}, {}, {"right": 0.0, "left": 0.0}

    def _provisional_side(self, pose):
        for s in self._pf:
            self._pf[s].append(live_feature(pose.world[-3:], s, self.ac))
            ok = [f for f in self._pf[s] if f is not None]
            if s not in self._f0 and len(ok) >= self.tracker.first_n:
                self._f0[s] = unit_blocks(np.mean(ok[:self.tracker.first_n], 0), self.ac)
            if s in self._f0 and self._pf[s][-1] is not None:
                self._exc[s] = max(self._exc[s], float(np.linalg.norm(self._pf[s][-1] - self._f0[s])))
        return max(self._exc, key=self._exc.get) if len(self._f0) == 2 else None

    def after_update(self, sess, pose):
        """After a session update: the moving side (history rebuilt when it differs from the provisional one) and the
        patient's rest feature; a newly reported rep restarts the alignment after its end."""
        if sess.side is None or sess.pre is None:
            return
        rest = rest_feature(sess.pre["world"], sess.signal, sess.side, self.ac, getattr(sess, "polarity", "peak"))
        self.locked = True
        if sess.side != self.side:
            self.side = sess.side
            self.tracker.rebuild([live_feature(pose.world[max(0, i - 2):i + 1], self.side, self.ac)
                                  for i in range(len(pose.world) - 1)], rest)  # newest frame: frame()
            self.n_seen = len(sess.entries)
            return
        self.tracker.set_rest(rest)
        if len(sess.entries) > self.n_seen:
            self.n_seen = len(sess.entries)
            t_end = sess.entries[-1]["time_s"][1]
            self.tracker.restart(int(np.searchsorted(np.asarray(pose.t[:-1]), t_end, side="right")))

    def frame(self, pose):
        rest = {"in_rep": False, "k": self.player.rest_k, "j": -1, "diffs": {}, "side": self.side}
        if not self.locked:
            side = self._provisional_side(pose)
            if side is None:
                return rest
            if side != self.side:
                self.side = side
                self.tracker.rebuild(self._pf[side][:-1])
            f = self._pf[side][-1]
        else:
            f = live_feature(pose.world[-3:], self.side, self.ac)
        in_rep, e = self.tracker.push(f)
        if not in_rep:
            return {**rest, "side": self.side}
        k, j = int(self.player.axis_display[e]), int(self.player.axis_template[e])
        diffs = angle_diffs(live_channels(pose.world, self.side, self.ac), self.targets, self.sigma, j,
                            self.ac["band_k"]) if j >= 0 else {}
        return {"in_rep": True, "k": k, "j": j, "diffs": diffs, "side": self.side}


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
    patient skeleton (black) with the PT's own keypoints at the matching frame of its rep as a ghost (cyan; mirrored for
    the other side, X7), both root-centred and torso-normalised, and each channel's difference (patient - target, deg)
    at its joint, green inside the band."""
    import matplotlib.pyplot as plt

    from cvdc.data.skeleton import MP_DRAW_EDGES
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
        R, mir = player.frame(player.frame_at_template(j), side)
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
