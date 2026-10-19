"""Module III steps 5-7 (proposal Fig. 2): deviations, error detection and localisation, the reference-based score and
the per-video report.

For one rep (already on the template's time axis, ``template.aligned_rep``) in camera view v:
  target_c(t) = PT_c(t) + b_v,c(t)               the demo as camera v reads it (M8)
  dev_c(t)    = rep_c(t) - target_c(t)           angle deviation (deg), per channel and template frame
  u_c(t)      = |dev_c(t)| / sigma_v,c(t)        in units of the high-scoring experts' spread (M9)
Measures (``measures``): per channel the mean |dev| (deg), the signed mean dev (deg) and the mean u (D_c); the ROM of
the main angle against the target ROM (shortfall, deg); the speed ratio (demo / rep duration) and the main angle's
angular-velocity deviation mean |omega_rep - omega_target| (deg/s); the left/right symmetry channel.

Score (M11): four groups, each a normalised distance from the reference in expert-SD units, X_g >= 0:
  angle      mean of D_c over the posture channels (``assess.score.angle_channels``)
  rom        max(0, target ROM - rep ROM) / sd_ROM
  speed      |log2 speed ratio - expert mean| / expert SD   (speed centre from expert data, M5a)
  symmetry   D_c of the symmetry channel
  p_g = min(1, X_g / d_zero);  rep score = 100 x (1 - sum_g w_g p_g / sum_g w_g) over the groups that are defined.
Set (video) score = mean of the rep scores.

Errors (M12): a rule fires where the deviation leaves the band target +- k sigma (``assess.band_k``) for a run of
at least ``min_run_frac`` of the template (and >= ``min_run_frames``); the worst run is reported with its joint, video
frame range (on the 30 fps grid), measured / target / deviation in degrees and a severity from the excess beyond the
band: <= 1 SD mild, <= 2 SD moderate, > 2 SD severe. ROM and speed rules use their scalar z in the same way.
"""
import numpy as np

from .angles import other

SCHEMA_VERSION = 1


def _runs(mask):
    """[(a, b)] inclusive index runs where mask is True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def severity(excess, cuts=(1.0, 2.0)):
    return "mild" if excess <= cuts[0] else "moderate" if excess <= cuts[1] else "severe"


def joint_name(joint, side):
    return {"moving_shoulder": f"{side}_shoulder", "moving_elbow": f"{side}_elbow",
            "other_shoulder": f"{other(side)}_shoulder", "trunk": "trunk"}.get(joint, joint)


def compare(sig, al, ref, lib_v, ac):
    """Deviation curves and measures of one rep (signals ``sig``, aligned ``al``) against the template in its view's
    library entry."""
    dev, u, tgt = {}, {}, {}
    m = {}
    min_def = ac["min_defined_frac"]
    for c in ref["channels"]:
        t = ref["curves"][c] + lib_v["bias"][c]
        d = al["curves"][c] - t
        s = lib_v["sigma"][c]
        ok = np.isfinite(d) & np.isfinite(s)
        tgt[c], dev[c] = t, np.where(ok, d, np.nan)
        u[c] = np.where(ok, np.abs(d) / s, np.nan)
        good = ok.mean() >= min_def
        m[f"dev_abs_{c}"] = float(np.nanmean(np.abs(d[ok]))) if good else np.nan
        m[f"dev_mean_{c}"] = float(np.nanmean(d[ok])) if good else np.nan
        m[f"D_{c}"] = float(np.nanmean(u[c][ok])) if good else np.nan
        m[f"defined_{c}"] = float(ok.mean())
    main = ref["main"]
    m["rom_target"] = ref["rom"] + lib_v["rom_bias"]
    m["rom_shortfall"] = m["rom_target"] - sig["rom"]
    m["z_rom"] = m["rom_shortfall"] / lib_v["rom_sd"]
    m["log2_speed"] = al["log2_speed"]
    m["z_speed"] = (al["log2_speed"] - lib_v["speed_mu"]) / lib_v["speed_sd"]
    w_t = np.gradient(tgt[main]) * ac["fps"]
    dw = al["omega"] - w_t
    m["omega_dev"] = float(np.nanmean(np.abs(dw))) if np.isfinite(dw).mean() >= min_def else np.nan
    return {"dev": dev, "u": u, "target": tgt, "measures": m}


def groups(meas, ac):
    sc = ac["score"]
    ang = [meas[f"D_{c}"] for c in sc["angle_channels"] if np.isfinite(meas.get(f"D_{c}", np.nan))]
    return {"angle": float(np.mean(ang)) if ang else np.nan,
            "rom": max(0.0, meas["z_rom"]) if np.isfinite(meas["z_rom"]) else np.nan,
            "speed": abs(meas["z_speed"]) if np.isfinite(meas["z_speed"]) else np.nan,
            "symmetry": meas.get(f"D_{sc['symmetry_channel']}", np.nan)}


def score(meas, ac):
    """Reference-based rep score 0-100 (M11) and the group distances X_g."""
    sc = ac["score"]
    X = groups(meas, ac)
    num = den = 0.0
    for g, x in X.items():
        if np.isfinite(x):
            w = sc["weights"][g]
            num += w * min(1.0, x / sc["d_zero"])
            den += w
    return (100.0 * (1 - num / den) if den > 0 else np.nan), X


def quality(s, bins, names):
    return None if not np.isfinite(s) else names[int(np.digitize(s, bins))]


def detect_errors(cmp, al, ref, lib_v, sig, ac, rules):
    """Rule-based errors of one rep (M12): list of dicts (type, joint, frames, measured / target / deviation,
    severity, excess_sd)."""
    k, cuts = ac["band_k"], tuple(ac["severity_sd"])
    n = ref["n"]
    min_len = max(ac["min_run_frames"], int(np.ceil(ac["min_run_frac"] * n)))
    side, start = sig["side"], sig["start"]
    out = []
    for r in rules:
        kind = r.get("measure", "channel")
        if kind == "channel":
            c = r["channel"]
            d, u = cmp["dev"][c], cmp["u"][c]
            sign = {"below": d < 0, "above": d > 0, "both": np.isfinite(d)}[r["direction"]]
            mask = np.isfinite(u) & (u > k) & sign
            best = None
            for a, b in _runs(mask):
                if b - a + 1 < min_len:
                    continue
                ex = float(np.nansum(u[a:b + 1] - k))
                if best is None or ex > best[0]:
                    best = (ex, a, b)
            if best is None:
                continue
            _, a, b = best
            w = a + int(np.nanargmax(u[a:b + 1]))
            fr = al["span"][a:b + 1]
            out.append({"type": r["type"], "joint": joint_name(r["joint"], side),
                        "measured_deg": round(float(al["curves"][c][w]), 1),
                        "target_deg": round(float(cmp["target"][c][w]), 1),
                        "deviation_deg": round(float(d[w]), 1),
                        "frames": [int(start + fr[:, 0].min()), int(start + fr[:, 1].max())],
                        "severity": severity(float(u[w] - k), cuts), "excess_sd": round(float(u[w] - k), 2)})
        elif kind == "rom_shortfall":
            z = cmp["measures"]["z_rom"]
            if not (np.isfinite(z) and z > k):
                continue
            fr = al["span"][ref["peak_frames"]]
            m = cmp["measures"]
            out.append({"type": r["type"], "joint": joint_name(r["joint"], side),
                        "measured_deg": round(float(sig["rom"]), 1), "target_deg": round(float(m["rom_target"]), 1),
                        "deviation_deg": round(float(-m["rom_shortfall"]), 1),
                        "frames": [int(start + fr[:, 0].min()), int(start + fr[:, 1].max())],
                        "severity": severity(float(z - k), cuts), "excess_sd": round(float(z - k), 2)})
        elif kind == "speed":
            z = cmp["measures"]["z_speed"]
            want = {"fast": z > k, "slow": z < -k}[r["direction"]] if np.isfinite(z) else False
            if not want:
                continue
            target_s = ref["duration_s"] / 2 ** lib_v["speed_mu"]
            out.append({"type": r["type"], "joint": joint_name(r.get("joint", "moving_shoulder"), side),
                        "measured_s": round(float(sig["duration_s"]), 2), "target_s": round(float(target_s), 2),
                        "deviation_s": round(float(sig["duration_s"] - target_s), 2),
                        "frames": [int(sig["start"]), int(sig["end"])],
                        "severity": severity(float(abs(z) - k), cuts), "excess_sd": round(float(abs(z) - k), 2)})
    return out


def set_errors(n_reps_set, expected, cuts=(1, 3)):
    """Set-level rep-count rule: ``too_few_reps`` when the set has fewer reps than the EAAQ's minimum; severity by
    the number missing (<= 1 mild, <= 3 moderate, more severe)."""
    if expected is None or n_reps_set >= expected:
        return []
    miss = expected - n_reps_set
    return [{"type": "too_few_reps", "measured_reps": int(n_reps_set), "target_reps": int(expected),
             "deviation_reps": int(-miss), "severity": "mild" if miss <= cuts[0] else "moderate" if miss <= cuts[1]
             else "severe"}]


def video_report(meta, reps, set_errs, ac):
    """The per-video JSON (spec §8 shape): exercise, set score, rep entries, angle source."""
    scores = [r["score"] for r in reps if r["score"] is not None and np.isfinite(r["score"])]
    s = float(np.mean(scores)) if scores else None
    return {"schema": SCHEMA_VERSION, **meta, "set_score": None if s is None else round(s, 1),
            "set_quality": None if s is None else quality(s, ac["quality_bins"], ac["quality_names"]),
            "n_reps": len(reps), "set_errors": set_errs, "reps": reps}


def _num(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def rep_entry(i, sig, s, meas, X, errors, ac, embedding=None):
    e = {"rep": int(i), "frames": [sig["start"], sig["end"]], "score": _num(s, 1),
         "quality": quality(s, ac["quality_bins"], ac["quality_names"]) if np.isfinite(s) else None,
         "measures": {"rom_deg": _num(sig["rom"], 1), "rom_target_deg": _num(meas["rom_target"], 1),
                      "excursion_deg": _num(sig.get("excursion"), 1),
                      "rom_shortfall_deg": _num(meas["rom_shortfall"], 1), "duration_s": _num(sig["duration_s"]),
                      "speed_ratio": _num(2 ** meas["log2_speed"]) if np.isfinite(meas["log2_speed"]) else None,
                      "angular_velocity_dev_deg_s": _num(meas["omega_dev"], 1),
                      "angle_dev_deg": {k[8:]: _num(v, 1) for k, v in meas.items() if k.startswith("dev_abs_")},
                      "distance_sd": {g: _num(x) for g, x in X.items()}},
         "errors": errors}
    if embedding is not None:
        e["embedding_distance"] = embedding
    return e


REQUIRED = {"schema": int, "exercise": str, "video_id": str, "view": str, "angle_source": str, "template": dict,
            "n_reps": int, "set_errors": list, "reps": list}
REP_REQUIRED = {"rep": int, "frames": list, "errors": list, "measures": dict}
ERR_REQUIRED = {"type": str, "severity": str}


def validate_report(d):
    """Raise ValueError if a video report does not have the agreed shape (spec §8 JSON, M13)."""
    def need(obj, spec, where):
        for k, t in spec.items():
            if k not in obj:
                raise ValueError(f"{where}: missing {k!r}")
            if not isinstance(obj[k], t):
                raise ValueError(f"{where}: {k!r} is {type(obj[k]).__name__}, expected {t.__name__}")

    need(d, REQUIRED, "report")
    if d["set_score"] is not None and not 0 <= d["set_score"] <= 100:
        raise ValueError("set_score outside 0-100")
    if d["n_reps"] != len(d["reps"]):
        raise ValueError("n_reps != len(reps)")
    for r in d["reps"]:
        need(r, REP_REQUIRED, f"rep {r.get('rep')}")
        if r.get("score") is not None and not 0 <= r["score"] <= 100:
            raise ValueError("rep score outside 0-100")
        if len(r["frames"]) != 2 or r["frames"][0] > r["frames"][1]:
            raise ValueError("bad rep frames")
        for e in r["errors"]:
            need(e, ERR_REQUIRED, f"rep {r['rep']} error")
            if e["severity"] not in ("mild", "moderate", "severe"):
                raise ValueError(f"severity {e['severity']!r}")
            if "frames" in e and not (r["frames"][0] <= e["frames"][0] <= e["frames"][1] <= r["frames"][1]):
                raise ValueError(f"error frames {e['frames']} outside rep {r['frames']}")
            if "joint" not in e:
                raise ValueError("rep error without joint")
    for e in d["set_errors"]:
        need(e, ERR_REQUIRED, "set error")
    return True


def assess_config(cfg, ex):
    """Global ``assess`` settings merged with the exercise's ``assess`` block (``score`` merged key by key)."""
    g, e = cfg["assess"], (cfg["exercises"][ex].get("assess") or {})
    ac = {**g, **{k: v for k, v in e.items() if k != "score"}}
    ac["score"] = {**g.get("score", {}), **e.get("score", {})}
    return ac


def evaluate(sig, al, ref, lib_v, ac):
    """Compare, score and detect errors for one rep that is already aligned: (score, X, measures, errors, cmp)."""
    cmp = compare(sig, al, ref, lib_v, ac)
    s, X = score(cmp["measures"], ac)
    errs = detect_errors(cmp, al, ref, lib_v, sig, ac, ac.get("rules") or [])
    return s, X, cmp["measures"], errs, cmp


class Assessor:
    """Module III for one exercise with a fixed template and library: rep-by-rep assessment (batch runs and the live
    demo). ``view`` selects the library entry (F / L / R; the demo assumes the front view unless told otherwise)."""

    def __init__(self, ref, lib, ac):
        from .template import aligned_rep, rep_signals

        self.ref, self.lib, self.ac = ref, lib, ac
        self._sig, self._align = rep_signals, aligned_rep

    def rep(self, world, start, end, side, view="F", index=0, embedding=None):
        sig = self._sig(world, start, end, side, self.ac, self.ac["fps"])
        al = self._align(sig, self.ref)
        s, X, meas, errs, cmp = evaluate(sig, al, self.ref, self.lib[view], self.ac)
        return rep_entry(index, sig, s, meas, X, errs, self.ac, embedding), {"sig": sig, "aligned": al, "cmp": cmp}
