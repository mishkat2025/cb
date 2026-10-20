"""Module III steps 3-4 (proposal Fig. 2): the Reference Template Library and similarity / alignment.

The physiotherapist's demo is the reference ("truth", M1): per exercise, its reps' 3D joint-angle curves on the 30 fps
grid (P5), averaged after DTW alignment (one DBA step), give the reference curve PT_c(t) of every channel c and the
reference speed. Left-arm performances are compared as their mirror image: every channel is computed for the moving
arm (``side``), so a left-arm rep and the right-arm demo are measured the same way (P8, M1).

High-scoring expert reps (video score >= ``templates.min_score``, M2) add what one demo cannot give, per camera view
(F / L / R), because 3D angles read the same movement differently from each camera (M7a, M8):
  view bias   b_v,c(t) = mu_v,c(t) - mu_F,c(t): how view v reads the movement differently from the front camera that
              filmed the demo (mu = mean of the experts' aligned curves); the target in view v is PT_c(t) + b_v,c(t)
  band        sigma_v,c(t): the experts' per-frame SD around their own mean (floored at ``sigma_floor_deg``); the
              tolerance band is target +- k sigma (M9)
  scalars     per view, mean and SD of the rep ROM of the main angle and of the log2 speed ratio (M10, M11)
The library is fitted on training subjects only (per GroupKFold fold) for evaluation; ``fit_library`` on all experts
gives the deployable library (demo).

Alignment (M8): DTW on the direction of the moving upper arm in the trunk frame (unit vector: outward, up, forward).
It is defined in every frame (unlike E02's horizontal adduction, undefined while the arm hangs) and both E01's and
E02's main angles are functions of it. Every channel is then read along the same path. Full scope (M17):
``assess.align`` picks the segment per exercise: arm (E01, E02, E05), forearm (E03, E04, E06), thigh of the moving
leg (E07), both thighs (E08) or the main angle itself (E09, whose trunk has no limb segment); ``assess.rom_direction``
says which end of the main angle is the end range (max, or min for E03, where the forearm rises as its elevation from
the trunk's up axis falls); channels can come from outside the Pose landmarks (``extra``: E06's wrist angle from
MediaPipe Hands, side views only).
"""
import warnings

import numpy as np

from .angles import ALIGN_SEGMENTS, ASSESS_ANGLES, horizontal_adduction, other, segment_direction, trunk_frame
from .dtw import dtw
from ..data.skeleton import J


def arm_direction(P, side):
    """(n, 3) unit vector of the moving upper arm (shoulder -> elbow) in the trunk frame: (outward, up, forward).
    For side 'left' outward = -lat, so a left arm is expressed as its mirror image."""
    lat, up, fwd = trunk_frame(P)
    u = P[..., J[f"{side}_elbow"], :] - P[..., J[f"{side}_shoulder"], :]
    out = lat if side == "right" else -lat
    v = np.stack([np.sum(u * out, -1), np.sum(u * up, -1), np.sum(u * fwd, -1)], -1)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def channel_curves(P, side, channels, min_proj=0.25, extra=None):
    """{channel: (n,) degrees} for the moving limb ``side`` (horizontal adduction is NaN while the arm hangs, M7a);
    channels in ``extra`` ({name: (n,) values for these frames}) are taken from there (E06's Hands wrist angle)."""
    out = {}
    for c in channels:
        if extra is not None and c in extra:
            out[c] = np.asarray(extra[c], float)
        elif c == "horizontal_adduction":
            out[c] = horizontal_adduction(P, side, min_proj)
        else:
            out[c] = ASSESS_ANGLES[c](P, side)
    return out


def align_feature(P, side, ex_assess, ch):
    """The DTW alignment feature (M8, M17): the direction of the configured segment in the trunk frame, both thighs
    side by side, or the main angle (scaled to ~1)."""
    a = ex_assess.get("align", "arm")
    if a in ALIGN_SEGMENTS:
        return segment_direction(P, side, *ALIGN_SEGMENTS[a])
    if a == "thighs":
        return np.concatenate([segment_direction(P, side, "hip", "knee"),
                               segment_direction(P, other(side), "hip", "knee")], 1)
    if a == "main":
        x = np.asarray(ch[ex_assess["angle"]], float)
        x = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
        return (x / 90.0)[:, None]
    raise ValueError(f"assess.align {a!r}: arm | forearm | thigh | thighs | main")


def wrist_path_measures(P, side):
    """E05 path measures (M16): the wrist path relative to the shoulder in the trunk frame over the rep: amplitude =
    the SD along its first two principal components (torso units, root-centred world coordinates), circularity =
    their ratio (1 = a circle, 0 = a line)."""
    lat, up, fwd = trunk_frame(P)
    d = P[..., J[f"{side}_wrist"], :] - P[..., J[f"{side}_shoulder"], :]
    out = lat if side == "right" else -lat
    r = np.stack([np.sum(d * out, -1), np.sum(d * up, -1), np.sum(d * fwd, -1)], -1)
    r = r[np.isfinite(r).all(1)]
    if len(r) < 5:
        return {"path_amplitude": np.nan, "path_circularity": np.nan}
    sv = np.linalg.svd(r - r.mean(0), compute_uv=False) / np.sqrt(len(r))
    return {"path_amplitude": float(np.hypot(sv[0], sv[1])), "path_circularity": float(sv[1] / max(sv[0], 1e-9))}


def angle_range(a, name):
    """ROM (deg) of the defined values of one rep: max - min; horizontal adduction (an arctan2 angle) as the
    circular range, 360 - the largest gap between the sorted angles. NaN with fewer than 2 defined values."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return np.nan
    if name != "horizontal_adduction":
        return float(a.max() - a.min())
    s = np.sort(np.mod(a, 360.0))
    return float(360.0 - np.diff(np.r_[s, s[0] + 360.0]).max())


def rep_signals(world, start, end, side, ex_assess, fps, extra=None):
    """Signals of one rep (frames start..end of the uniform grid): alignment feature, channel curves, timing.
    ``extra``: {channel: (n_video,) values on the same grid} for channels not computed from the Pose landmarks."""
    P = np.asarray(world[start:end + 1], float)
    ex = {c: np.asarray(v, float)[start:end + 1] for c, v in (extra or {}).items()}
    ch = channel_curves(P, side, ex_assess["channels"], ex_assess.get("angle_min_proj", 0.25), ex)
    main = ex_assess["angle"]
    sig = {"feat": align_feature(P, side, ex_assess, ch), "ch": ch, "n": len(P), "start": int(start),
           "end": int(end), "side": side, "duration_s": (len(P) - 1) / fps,
           "rom": end_range(ch[main], direction=ex_assess.get("rom_direction", "max")),
           "excursion": angle_range(ch[main], main)}
    if "wrist_path" in (ex_assess.get("path_measures") or []):
        sig["path"] = wrist_path_measures(P, side)
    return sig


def end_range(a, k=3, direction="max"):
    """ROM as in clinical goniometry (M10): the end-range angle reached, measured from the anatomical zero position:
    the maximum of the main angle after a k-frame running median (removes single-frame 3D glitches; a sharp peak
    without a hold loses at most one frame's change), or its minimum with ``direction='min'`` (E03, M17). NaN-aware."""
    a = np.asarray(a, float)
    if direction == "min":
        return -end_range(-a, k)
    if not np.isfinite(a).any():
        return np.nan
    if len(a) >= k:
        from scipy.ndimage import median_filter

        ok = np.isfinite(a)
        f = median_filter(np.where(ok, a, -np.inf), size=k, mode="nearest")
        f = np.where(ok & np.isfinite(f), f, np.nan)
        if np.isfinite(f).any():
            return float(np.nanmax(f))
    return float(np.nanmax(a))


def align(feat_rep, feat_ref, band=None):
    """DTW path (L, 2) of (rep frame, template frame) pairs on the arm-direction feature."""
    return dtw(feat_rep, feat_ref, band)[1]


def on_template(x, path, n_ref):
    """(n_ref,): for every template frame, the mean of the rep values x matched to it (NaN-aware: NaN where every
    matched value is NaN)."""
    x = np.asarray(x, float)
    i, j = path[:, 0], path[:, 1]
    v = x[i]
    ok = np.isfinite(v)
    s = np.zeros(n_ref)
    c = np.zeros(n_ref)
    np.add.at(s, j[ok], v[ok])
    np.add.at(c, j[ok], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def matched_span(path, n_ref):
    """(n_ref, 2): first and last rep frame matched to every template frame."""
    lo = np.full(n_ref, np.iinfo(np.int64).max)
    hi = np.full(n_ref, -1)
    np.minimum.at(lo, path[:, 1], path[:, 0])
    np.maximum.at(hi, path[:, 1], path[:, 0])
    return np.stack([lo, hi], 1)


def resample(x, n):
    """Linear resampling of (m,) or (m, d) to n points (NaN-aware for 1-D: NaN stays NaN where a neighbour is)."""
    x = np.asarray(x, float)
    t_old, t_new = np.linspace(0, 1, len(x)), np.linspace(0, 1, n)
    if x.ndim == 1:
        return np.interp(t_new, t_old, x)
    return np.stack([np.interp(t_new, t_old, x[:, k]) for k in range(x.shape[1])], 1)


def build_reference(pt_reps, ex_assess, fps, meta=None):
    """The reference template from the PT demo's reps (list of rep_signals dicts). One DBA step: the rep whose length
    is closest to the median is the axis; every rep is DTW-aligned onto it on the arm direction and the channel curves
    and arm directions are averaged; the result is resampled to the mean rep length, so the template keeps the demo's
    mean duration. Returns a dict (curves, feat, n, duration_s, rom, peak, channels, main, ...)."""
    if not pt_reps:
        raise ValueError("no PT reps")
    lens = np.array([r["n"] for r in pt_reps])
    ax = pt_reps[int(np.argmin(np.abs(lens - np.median(lens))))]
    n_ax = ax["n"]
    chans = list(ax["ch"])
    acc = {c: [] for c in chans}
    feats = []
    for r in pt_reps:
        path = align(r["feat"], ax["feat"])
        for c in chans:
            acc[c].append(on_template(r["ch"][c], path, n_ax))
        feats.append(np.stack([on_template(r["feat"][:, k], path, n_ax) for k in range(r["feat"].shape[1])], 1))
    n = int(round(lens.mean()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames where every PT rep is undefined stay NaN
        curves = {c: np.nanmean(np.stack(acc[c]), 0) for c in chans}
    curves = {c: _resample_nan(v, n) for c, v in curves.items()}
    f = resample(np.mean(feats, 0), n)
    if ex_assess.get("align", "arm") != "main":  # direction features: back to unit vectors, per 3-d segment block
        f = f.reshape(n, -1, 3)
        f = (f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-9)).reshape(n, -1)
    main = ex_assess["angle"]
    pk = curves[main]
    lo, hi = np.nanmin(pk), np.nanmax(pk)
    if ex_assess.get("rom_direction", "max") == "min":  # the end range is the minimum (E03, M17)
        peak_frames, hi = np.flatnonzero(pk <= lo + 0.1 * (hi - lo)), lo
    else:
        peak_frames = np.flatnonzero(pk >= lo + 0.9 * (hi - lo))  # the template's peak region (range errors, M12)
    return {"curves": curves, "feat": f, "n": n, "duration_s": (n - 1) / fps, "channels": chans, "main": main,
            "rom": float(np.mean([r["rom"] for r in pt_reps])),
            "excursion": float(np.mean([r["excursion"] for r in pt_reps])), "peak": float(hi), "peak_frames": peak_frames, "n_pt_reps": len(pt_reps),
            "pt_durations_s": [float(r["duration_s"]) for r in pt_reps], **(meta or {})}


def _resample_nan(v, n):
    ok = np.isfinite(v)
    if ok.all():
        return resample(v, n)
    out = resample(np.where(ok, v, 0.0), n)
    mask = resample(ok.astype(float), n)
    out[mask < 0.5] = np.nan
    return out


def aligned_rep(sig, ref):
    """A rep on the template's time axis: its DTW path, every channel's curve sampled at the template frames, the
    main angle's angular velocity (deg/s, in the rep's own time) and the speed ratio (template / rep duration)."""
    path = align(sig["feat"], ref["feat"])
    n = ref["n"]
    curves = {c: on_template(sig["ch"][c], path, n) for c in ref["channels"]}
    fps = (sig["n"] - 1) / sig["duration_s"] if sig["duration_s"] > 0 else 30.0
    with np.errstate(invalid="ignore"):
        w = np.gradient(sig["ch"][ref["main"]]) * fps if sig["n"] > 1 else np.full(sig["n"], np.nan)
    return {"path": path, "curves": curves, "omega": on_template(w, path, n), "span": matched_span(path, n),
            "log2_speed": float(np.log2(ref["duration_s"] / sig["duration_s"])) if sig["duration_s"] > 0 else np.nan}


def fit_library(ref, aligned_by_view, scalars_by_view, sigma_floor, min_reps=10, views=("F", "L", "R")):
    """Fit the per-view view bias, band SD and scalar tolerances from high-scoring expert reps.

    aligned_by_view: {view: list of aligned_rep()['curves'] dicts}; scalars_by_view: {view: DataFrame-like dict of
    arrays 'rom', 'log2_speed', 'omega_dev'}. Frames with fewer than ``min_reps`` defined expert values get NaN (not
    assessed in that view). Returns {view: {'bias': {c: (n,)}, 'sigma': {c: (n,)}, 'count': {c: (n,)},
    'rom_mu', 'rom_sd', 'speed_mu', 'speed_sd', 'n_reps'}}."""
    lib = {}
    mu = {}
    for v in views:
        reps = aligned_by_view.get(v, [])
        entry = {"bias": {}, "sigma": {}, "count": {}, "mean": {}, "n_reps": len(reps)}
        for c in ref["channels"]:
            if reps:
                X = np.stack([r[c] for r in reps])
                cnt = np.isfinite(X).sum(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # undefined frames stay NaN
                    m = np.nanmean(np.where(cnt >= 1, X, np.nan), 0) if cnt.any() else np.full(ref["n"], np.nan)
                    sd = np.nanstd(X, 0, ddof=1) if (cnt >= 2).any() else np.full(ref["n"], np.nan)
                ok = cnt >= min_reps
                m, sd = np.where(ok, m, np.nan), np.where(ok, sd, np.nan)
            else:
                cnt, m, sd = np.zeros(ref["n"], int), np.full(ref["n"], np.nan), np.full(ref["n"], np.nan)
            entry["count"][c], entry["mean"][c] = cnt, m
            entry["sigma"][c] = np.maximum(sd, sigma_floor.get(c, sigma_floor.get("default", 3.0)))
            mu.setdefault(c, {})[v] = m
        s = scalars_by_view.get(v) or {}
        for k, key in (("rom", "rom"), ("speed", "log2_speed")):
            x = np.asarray(s.get(key, []), float)
            x = x[np.isfinite(x)]
            entry[f"{k}_mu"] = float(x.mean()) if len(x) >= min_reps else np.nan
            entry[f"{k}_sd"] = float(x.std(ddof=1)) if len(x) >= min_reps else np.nan
        entry["rom_sd"] = max(entry["rom_sd"], sigma_floor.get("rom", 3.0)) if np.isfinite(entry["rom_sd"]) else np.nan
        lib[v] = entry
    for v in views:  # view bias relative to the front camera that filmed the demo (0 for F)
        for c in ref["channels"]:
            lib[v]["bias"][c] = np.zeros(ref["n"]) if v == "F" else mu[c][v] - mu[c]["F"]
        lib[v]["rom_bias"] = 0.0 if v == "F" else lib[v]["rom_mu"] - lib["F"]["rom_mu"]
    return lib


def save_library(path, ref, lib, meta=None):
    """Write a template + library to <path>.npz (arrays, no pickle) and <path>.json (scalars)."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = {"ref__feat": ref["feat"], "ref__peak_frames": ref["peak_frames"]}
    arr.update({f"ref__curves__{c}": v for c, v in ref["curves"].items()})
    scal = {"ref": {k: v for k, v in ref.items() if k not in ("feat", "peak_frames", "curves")}, "lib": {},
            "meta": meta or {}}
    for v, e in lib.items():
        for part in ("bias", "sigma", "count", "mean"):
            arr.update({f"lib__{v}__{part}__{c}": x for c, x in e[part].items()})
        scal["lib"][v] = {k: x for k, x in e.items() if k not in ("bias", "sigma", "count", "mean")}
    np.savez_compressed(path.with_suffix(".npz"), **arr)
    path.with_suffix(".json").write_text(json.dumps(scal, indent=1, default=float), encoding="utf-8")


def load_library(path):
    """Inverse of save_library: (ref, lib, meta)."""
    import json
    from pathlib import Path

    path = Path(path)
    scal = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    ref = dict(scal["ref"], curves={})
    lib = {v: dict(e, bias={}, sigma={}, count={}, mean={}) for v, e in scal["lib"].items()}
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        for k in z.files:
            parts = k.split("__")
            if parts[0] == "ref":
                if parts[1] == "curves":
                    ref["curves"][parts[2]] = z[k]
                else:
                    ref[parts[1]] = z[k]
            else:
                lib[parts[1]][parts[2]][parts[3]] = z[k]
    return ref, lib, scal["meta"]
