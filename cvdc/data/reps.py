"""Repetition segmentation from each exercise's primary joint angle (P6).

A rep is rest -> peak -> rest of the primary angle (``polarity: peak``; ``valley`` flips the signal). With
``reps.adaptive: true`` (full scope, E03-E09; the Step 0 method of PROGRESS §3.7.5) the prominence and the minimum
amplitude are fractions of each video's own p5-p95 signal range and the polarity follows the rest level of the first
second (``adaptive_params``). Rep signals that are not joint angles (wrist-path PC1, all-joint PC1, 2D wrist height)
are used for segmentation only (M16).
The angle is median-filtered (``median_s``); peaks need a minimum prominence and spacing; each rep starts where the angle last leaves its rest level
before the peak and ends where it first returns after the peak (rest level = within ``rest_frac`` of the
amplitude above the neighbouring minimum), padded by ``pad_s`` without crossing that minimum. Reps with an
amplitude below ``min_amplitude`` on either side (e.g. a video that ends with the arm still raised)
or a duration outside [min_rep_s, max_rep_s] are dropped.

Synced groups (outputs/sync_report.csv, cls = synced) get identical rep boundaries in all aligned views:
the reps are segmented on the aligned view with the best signal quality (share of frames with the trunk
and moving-arm joints visible; ties broken by ``reps.view_preference``) and transferred to the other views
with the group's offsets. The group's moving arm is the best view's. Reps share ``rep_uid`` across views.
The phones start and stop up to ~1 s apart, so an edge rep may be missing from some view's recording (beyond
``reps.transfer_tol_s``): with ``reps.transfer_keep: common`` (default) such a rep is dropped from every view,
so all aligned views hold identical rep sets; ``covered`` drops it from that view only. ``n_reps`` counts
the kept (training) reps; ``n_reps_set`` is the set's rep count, taken from the best view's own
segmentation (all reps it recorded), and is the count to compare with PF4 / the EAAQ. Views of not-synced
groups, unaligned views and single-view videos are segmented on their own.

Times are frame indices on the uniform preprocessing grid (cfg preprocess.fps). Run after cvdc.data.sync.

usage: python -m cvdc.data.reps [--config C] [--workers N]
Writes data.reps_csv (one row per rep; pilot: data/reps.csv), data.reps_perview_csv (every view segmented on its own,
for comparison), data.reps_ref_csv (PT demos) and <outputs.phase1>/reps_per_video.csv.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from ..assess.angles import UNITS, primary_angle
from ..config import data_path, exercise_cfg, load_config, resolve
from .skeleton import J


def fill_nan(x):
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    if not ok.any():
        return None
    x[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), x[ok])
    return x


def segment_reps(angle, fps, prominence=40, min_distance_s=1.0, min_amplitude=40, rest_frac=0.1,
                 pad_s=0.1, min_rep_s=0.8, max_rep_s=15.0, polarity="peak", median_s=0.3, **_):
    """Return a list of reps: dict(start, peak, end, ...) with frame indices into ``angle``. ``prominence`` and
    ``min_amplitude`` are in the signal's unit (degrees for angles, torso lengths for wrist_path_pc1).
    A running median (``median_s``) first removes short pose glitches (1-3 frame spikes or dips)."""
    s = fill_nan(angle)
    if s is None or len(s) < 3:
        return []
    k = int(round(median_s * fps)) // 2 * 2 + 1
    if k > 1:
        s = median_filter(s, size=k, mode="nearest")
    if polarity == "valley":
        s = -s
    peaks, _ = find_peaks(s, prominence=prominence, distance=max(1, int(round(min_distance_s * fps))))
    pad = int(round(pad_s * fps))
    reps = []
    for i, p in enumerate(peaks):
        lo = peaks[i - 1] if i > 0 else 0
        hi = peaks[i + 1] if i + 1 < len(peaks) else len(s) - 1
        vl = lo + int(np.argmin(s[lo:p + 1]))
        vr = p + int(np.argmin(s[p:hi + 1]))
        amp_l, amp_r = s[p] - s[vl], s[p] - s[vr]
        if min(amp_l, amp_r) < min_amplitude:
            continue
        rest_l, rest_r = s[vl] + rest_frac * amp_l, s[vr] + rest_frac * amp_r
        start = vl + int(np.flatnonzero(s[vl:p + 1] <= rest_l)[-1])
        end = p + int(np.flatnonzero(s[p:vr + 1] <= rest_r)[0])
        start, end = max(vl, start - pad), min(vr, end + pad)
        dur = (end - start) / fps
        if not (min_rep_s <= dur <= max_rep_s):
            continue
        top_l, top_r = s[vl] + 0.9 * amp_l, s[vr] + 0.9 * amp_r
        up90 = start + int(np.flatnonzero(s[start:p + 1] >= top_l)[0])
        down90 = p + int(np.flatnonzero(s[p:end + 1] >= top_r)[-1])
        sign = -1 if polarity == "valley" else 1
        reps.append(dict(start=start, peak=int(p), end=end, duration_s=dur,
                         peak_val=sign * s[p], rest_start_val=sign * s[vl], rest_end_val=sign * s[vr],
                         amplitude=float(min(amp_l, amp_r)),
                         raise_s=(up90 - start) / fps, hold_s=(down90 - up90) / fps, lower_s=(end - down90) / fps))
    return reps


def adaptive_params(signal, fps, rc):
    """Per-video thresholds (``reps.adaptive``, full-scope Step 0 method, PROGRESS §3.7.5): prominence and minimum
    amplitude as fractions of the signal's p5-p95 range (after a 0.3 s running median), and the polarity from the
    rest level: 'peak' when the median of the first second lies nearer the p5 than the p95, else 'valley'. Returns
    the segment_reps keyword arguments (prominence, min_amplitude, polarity) or None for a flat / empty signal."""
    s = fill_nan(signal)
    if s is None or len(s) < 3:
        return None
    k = int(round(rc.get("median_s", 0.3) * fps)) // 2 * 2 + 1
    m = median_filter(s, size=k, mode="nearest") if k > 1 else s
    lo, hi = np.percentile(m, [5, 95])
    rng = hi - lo
    if rng <= 1e-9:
        return None
    pol = rc.get("polarity", "auto")
    if pol == "auto":
        start = np.median(m[:max(1, int(round(rc.get("rest_window_s", 1.0) * fps)))])
        pol = "peak" if abs(start - lo) <= abs(start - hi) else "valley"
    return dict(prominence=rc["prominence_frac"] * rng, min_amplitude=rc["min_amplitude_frac"] * rng, polarity=pol)


def segment_signal(signal, fps, rc):
    """segment_reps with the exercise's settings; with ``adaptive: true`` the thresholds and polarity come from the
    video's own signal (adaptive_params). Returns (reps, polarity)."""
    if rc.get("adaptive"):
        ap = adaptive_params(signal, fps, rc)
        if ap is None:
            return [], "peak"
        return segment_reps(signal, fps, **{**rc, **ap}), ap["polarity"]
    return segment_reps(signal, fps, **rc), rc.get("polarity", "peak")


def moving_side(angles: dict, side_cfg="auto"):
    """'auto' picks the side whose primary angle has the larger p5-p95 range."""
    rng = {s: float(np.nanpercentile(a, 95) - np.nanpercentile(a, 5)) if np.isfinite(a).any() else 0.0
           for s, a in angles.items()}
    side = max(rng, key=rng.get) if side_cfg == "auto" else side_cfg
    return side, rng


def _points(pre, source):
    return pre["world"] if source == "world" else pre["xy"]


def orient(sig, ref, sign=1):
    """Flip a signal of arbitrary sign (a principal component) so that it correlates with ``sign`` x ``ref``."""
    ok = np.isfinite(sig) & np.isfinite(ref)
    if ok.sum() < 3 or sig[ok].std() < 1e-9 or ref[ok].std() < 1e-9:
        return sig
    return -sig if sign * np.corrcoef(sig[ok], ref[ok])[0, 1] < 0 else sig


def video_angles(pre, pa_cfg):
    """The rep signal per side. With ``orient_by`` (P6c) a signal of arbitrary sign is flipped per side so that it
    rises with ``sign`` x the named joint angle of that side (e.g. E07: all-joint PC1 rising with hip flexion), so the
    rep is always the same phase of the movement."""
    P = _points(pre, pa_cfg.get("source", "world"))
    out = {s: primary_angle(P, pa_cfg["name"], s) for s in ("right", "left")}
    ob = pa_cfg.get("orient_by")
    if ob:
        Q = _points(pre, ob.get("source", "world"))
        out = {s: orient(v, primary_angle(Q, ob["name"], s), ob.get("sign", 1)) for s, v in out.items()}
    return out


SIGNAL_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def load_pre(path):
    with np.load(path) as z:
        return {k: z[k] for k in ("world", "xy", "vis", "t")}


def view_quality(pre, side, vis_thr=0.5, limb=("elbow", "wrist")):
    """Share of frames where the trunk joints and the moving limb's joints (``reps.quality_joints``; the arm's elbow
    and wrist by default) are all visible."""
    j = [J[n] for n in SIGNAL_JOINTS + tuple(f"{side}_{x}" for x in limb)]
    return float((pre["vis"][:, j] >= vis_thr).all(axis=1).mean())


def video_info(pre, ex_cfg, side=None):
    """Rep signal per side and the moving side. A signal without sides (e.g. all_joint_pc1) takes the moving side from
    ``primary_angle.side_from`` (another signal's larger p5-p95 range), so 'side' still names the moving limb."""
    pa = ex_cfg["primary_angle"]
    ang = video_angles(pre, pa)
    side_from = pa.get("side_from")
    if side_from:
        rng_sig = {s: primary_angle(_points(pre, side_from.get("source", "world")), side_from["name"], s)
                   for s in ("right", "left")}
        auto_side, rng = moving_side(rng_sig, pa.get("side", "auto"))
    else:
        auto_side, rng = moving_side(ang, pa.get("side", "auto"))
    side = side or auto_side
    other = "left" if side == "right" else "right"
    limb = tuple(ex_cfg["reps"].get("quality_joints") or ("elbow", "wrist"))
    return ang, dict(side=side, side_own=auto_side, range_moving=rng[side], range_other=rng[other],
                     range_signal=side_from["name"] if side_from else pa["name"],   # the ranges' signal (and unit)
                     unit=UNITS[side_from["name"] if side_from else pa["name"]], primary_angle=pa["name"],
                     n_frames_30=len(pre["t"]),
                     angle_source="3d_mediapipe_world" if pa.get("source", "world") == "world" else "2d_image",
                     quality=view_quality(pre, side, limb=limb))


def transfer_reps(reps, off_src, off_dst, s_dst, fps, tol_s, polarity="peak"):
    """Map reps of the source view onto another view of the same synced group (aligned time = t - offset).
    Reps not covered by the other view's recording (beyond ``tol_s``) are dropped; values are re-read in the
    destination view's own signal (amplitude in the source view's polarity), timings are kept."""
    n, tol, out = len(s_dst), int(round(tol_s * fps)), []
    for i, r in enumerate(reps):
        k = {f: int(round((r[f] / fps - off_src + off_dst) * fps)) for f in ("start", "peak", "end")}
        if k["start"] < -tol or k["end"] > n - 1 + tol:
            continue
        k["start"], k["end"] = max(k["start"], 0), min(k["end"], n - 1)
        k["peak"] = min(max(k["peak"], k["start"]), k["end"])
        pv, a, b = s_dst[k["peak"]], s_dst[k["start"]], s_dst[k["end"]]
        sg = -1.0 if polarity == "valley" else 1.0
        out.append((i, dict(r, **k, peak_val=pv, rest_start_val=a, rest_end_val=b,
                            amplitude=float(min(sg * (pv - a), sg * (pv - b))))))
    return out  # [(index of the source rep, rep)]


def group_job(args):
    """All views of one (exercise, subject, variation) group. Synced groups with >= 2 aligned views: reps are
    segmented on the aligned view with the best signal quality (ties: view preference) and transferred to the
    other aligned views; every other view is segmented on its own."""
    gid, rows, offsets, ex_cfg, fps, rc = args
    try:
        pres = {r["view"]: load_pre(r["pre_path"]) for r in rows}
        info, sig, own, pol = {}, {}, {}, {}
        for v, pre in pres.items():
            ang, info[v] = video_info(pre, ex_cfg)
            sig[v] = ang
            own[v], pol[v] = segment_signal(ang[info[v]["side"]], fps, ex_cfg["reps"])
            info[v]["polarity"] = pol[v]
        aligned = [v for v in pres if np.isfinite(offsets.get(v, np.nan))]
        best = None
        if len(aligned) >= 2:
            pref = {v: i for i, v in enumerate(rc["view_preference"])}
            best = max(aligned, key=lambda v: (round(info[v]["quality"], 2), -pref.get(v, 9)))
        trans = {}
        if best is not None:
            side_g = info[best]["side"]
            for v in aligned:
                trans[v] = dict(enumerate(own[best])) if v == best else dict(
                    transfer_reps(own[best], offsets[best], offsets[v], sig[v][side_g], fps, rc["transfer_tol_s"],
                                  pol[best]))
            if rc["transfer_keep"] == "common":  # identical rep sets: drop reps some aligned view did not record
                keep = set.intersection(*(set(t) for t in trans.values()))
                trans = {v: {i: x for i, x in t.items() if i in keep} for v, t in trans.items()}
        out_reps, out_info = [], []
        for r in rows:
            v = r["view"]
            if best is not None and v in aligned:
                side = side_g
                src = "best_view" if v == best else "transfer"
                if side != info[v]["side"]:
                    info[v] = dict(video_info(pres[v], ex_cfg, side)[1], polarity=pol[v])
                reps = [trans[v][i] for i in sorted(trans[v])]
                rep_uid = [f"{gid}_r{i:02d}" for i in sorted(trans[v])]  # shared across the group's views
                n_set = len(own[best])
            else:
                reps, side = own[v], info[v]["side"]
                src = "single_view" if len(rows) == 1 else "own_unaligned" if offsets else "own_unsynced"
                rep_uid = [f"{r['video_id']}_r{i:02d}" for i in range(len(reps))]
                n_set = len(reps)
            out_info.append(dict(video_id=r["video_id"], **info[v], n_reps=len(reps), n_reps_set=n_set,
                                 n_reps_own=len(own[v]),
                                 rep_source=src, best_view=best or "", is_best=(v == best)))
            out_reps += [dict(video_id=r["video_id"], rep=i, rep_uid=u, side=side, rep_source=src,
                              source_video=next(x["video_id"] for x in rows if x["view"] == (best if src == "transfer" else v)),  # noqa: E501
                              **x) for i, (x, u) in enumerate(zip(reps, rep_uid))]
        own_rows = [dict(video_id=r["video_id"], rep=i, side=info[r["view"]]["side_own"], **x)
                    for r in rows for i, x in enumerate(own[r["view"]])]
        return gid, out_info, out_reps, own_rows, None
    except Exception as e:  # noqa: BLE001
        return gid, [], [], [], repr(e)


def ref_job(args):
    ref_id, ex, pre_path, ex_cfg, fps = args
    pre = load_pre(pre_path)
    ang, info = video_info(pre, ex_cfg)
    reps, info["polarity"] = segment_signal(ang[info["side"]], fps, ex_cfg["reps"])
    return [dict(ref_id=ref_id, exercise=ex, rep=i, side=info["side"], **r) for i, r in enumerate(reps)], info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cfg = load_config(a.config)
    fps, rc = cfg["preprocess"]["fps"], cfg["reps"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    idx = idx[idx.pre_path.map(lambda p: resolve(p).exists())]
    sync_path = resolve(cfg["sync"]["report_csv"])
    G = pd.read_csv(sync_path).set_index("group_id") if sync_path.exists() else None
    if G is None:
        print("WARNING: no sync report; every view is segmented on its own (run cvdc.data.sync first)")
    jobs = []
    for gid, g in idx.groupby("group_id"):
        offsets = {}
        if G is not None and gid in G.index and G.loc[gid, "cls"] == "synced":
            offsets = {v: G.loc[gid, f"offset_{v}_s"] for v in g.view}
        rows = [dict(video_id=r.video_id, view=r.view, pre_path=resolve(r.pre_path)) for r in g.itertuples()]
        jobs.append((gid, rows, offsets, exercise_cfg(cfg, g.exercise.iloc[0]), fps, rc))
    info, reps, own = [], [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for gid, i, r, o, err in pool.map(group_job, jobs, chunksize=2):
            if err:
                print("ERROR", gid, err)
            info += i
            reps += r
            own += o
    meta = idx[["video_id", "exercise", "subject", "group", "group_id", "view", "variation", "score", "PF3", "PF4"]]
    V = pd.DataFrame(info).merge(meta, on="video_id")
    R = pd.DataFrame(reps).merge(meta.drop(columns=["score", "PF3", "PF4"]), on="video_id")
    O = pd.DataFrame(own).merge(meta.drop(columns=["score", "PF3", "PF4"]), on="video_id")
    R.to_csv(data_path(cfg, "reps_csv"), index=False)
    O.to_csv(data_path(cfg, "reps_perview_csv"), index=False)
    out = resolve(cfg["outputs"]["phase1"])
    V.to_csv(out / "reps_per_video.csv", index=False)

    ref = pd.read_csv(resolve(cfg["data"]["ref_index_csv"]))
    ref_rows = []
    for r in ref.itertuples():
        if resolve(r.pre_path).exists():
            rows, inf = ref_job((r.ref_id, r.exercise, resolve(r.pre_path), exercise_cfg(cfg, r.exercise), fps))
            ref_rows += rows
            print(f"reference {r.ref_id} ({r.exercise} config): side {inf['side']}, {len(rows)} reps, "
                  f"range {inf['range_moving']:.2f} {inf['unit']} (other arm {inf['range_other']:.2f})")
    pd.DataFrame(ref_rows).to_csv(data_path(cfg, "reps_ref_csv"), index=False)

    print(f"\n{len(R)} reps in {len(V)} videos (per-view segmentation: {len(O)}); videos with 0 reps: "
          f"{(V.n_reps == 0).sum()}")
    print("rep source per video:", V.groupby(["exercise", "rep_source"]).size().unstack(fill_value=0).to_dict("index"))
    b = V[V.is_best]
    print("best view of synced groups:", b.groupby("exercise").view.value_counts().unstack(fill_value=0).to_dict("index"))


if __name__ == "__main__":
    main()
