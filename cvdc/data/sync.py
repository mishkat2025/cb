"""Multi-view synchronisation for every (exercise, subject, variation) group with >= 2 views (S1-S5, C2).

Evidence, in the approved order:
 1. audio     GCC-PHAT lag per view pair (8 kHz mono, band-passed, +-3 s). A F+L+R group whose three lags
              are additive (|lag_FL + lag_LR - lag_FR| <= 20 ms, none at the window edge) is synced (high).
 2. metadata  container creation_time. Recording windows that cannot overlap -> not-synced (high);
              every pair |dC| <= 3 s -> synced (medium).
 3. motion    rep-timing test on keypoint signals (wrist height, 3D shoulder elevation, wrist PC1): best
              cross-correlation lag, then the SD of matched mid-level-crossing times after removing offset
              and linear drift must be < 42 ms with >= 4 matched crossings. The search is +-0.3 s around the
              device-corrected audio lag, or +-5 s when there is no usable audio lag.
Per-device audio-to-video offsets (video lag - audio lag per view pair) are re-estimated on all
motion-confirmed pairs and the refinement is run again with them (S5).

Lags: ``lag_ab > 0`` means an event appears ``lag_ab`` seconds later in view b than in view a, on each
video's own timeline (0 = its first frame). Per-view offsets are relative to the reference view (F if
present): aligned time = t_view - offset_view.

CVDC pair source per view pair (C1, C2, supervisor answer 3), weights in ``sync.pair_weights``:
  'sync'  (w_sync)  audio groups: every pair; creation-time/motion groups: views joined by motion-confirmed pairs
  'audio' (w_audio) creation-time groups: views aligned only through device-corrected audio lags
  'rep'   (w_rep)   not-synced / unknown groups, or views with no usable lag: reps paired by index later
Per-view offsets come from a weighted least-squares fit of the pair lags (motion lags weight 1, audio lags
``sync.audio_lag_weight``), so motion-confirmed lags dominate where they exist.

usage: python -m cvdc.data.sync [--config C] [--workers N]
Writes outputs/sync_report.csv (per group), outputs/sync_pairs.csv (per pair), and controls/offsets tables.
Audio is decoded in memory only and never written.
"""
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..assess.angles import shoulder_elevation
from ..config import data_root, load_config, resolve
from .skeleton import J

# --------------------------------------------------------------------------------------------- audio


def gcc_phat(x, y, fs, band, window_s):
    """GCC-PHAT lag of y relative to x within +-window_s. Returns (lag_s, peak / second-peak ratio)."""
    n = len(x) + len(y)
    nfft = 1 << (n - 1).bit_length()
    X, Y = np.fft.rfft(x - x.mean(), nfft), np.fft.rfft(y - y.mean(), nfft)
    R = np.conj(X) * Y
    f = np.fft.rfftfreq(nfft, 1 / fs)
    R[(f < band[0]) | (f > band[1])] = 0
    R /= np.abs(R) + 1e-12
    full = np.fft.irfft(R, nfft)
    k = int(min(window_s, 0.8 * min(len(x), len(y)) / fs) * fs)
    cc = np.abs(np.concatenate([full[-k:], full[:k + 1]]))
    i = int(np.argmax(cc))
    w = int(0.02 * fs)
    rest = np.concatenate([cc[:max(0, i - w)], cc[i + w + 1:]])
    return (i - k) / fs, float(cc[i] / rest.max()) if len(rest) and rest.max() > 0 else np.nan


def audio_subject_job(args):
    """All same-group pairs of one (exercise, subject): creation-time difference and audio lag."""
    from .video import container_info, read_audio_mono
    rows, ac, root = args
    info, audio = {}, {}
    for r in rows:
        info[r["video_id"]] = container_info(root / r["rel_path"])
        x, a_start = read_audio_mono(root / r["rel_path"], ac["fs"])
        info[r["video_id"]]["a_start"] = a_start
        with np.load(resolve(r["kp_path"])) as z:
            info[r["video_id"]]["t0"] = float(z["t0"])
        audio[r["video_id"]] = x
    out = []
    for a, b in pair_list(rows):
        ia, ib = info[a["video_id"]], info[b["video_id"]]
        d = dict(video_a=a["video_id"], video_b=b["video_id"], d_creation_s=ib["creation"] - ia["creation"],
                 dur_a=ia["duration"], dur_b=ib["duration"], audio_lag_s=np.nan, audio_peak_ratio=np.nan)
        xa, xb = audio[a["video_id"]], audio[b["video_id"]]
        if xa is not None and xb is not None and min(len(xa), len(xb)) > ac["fs"]:
            raw, ratio = gcc_phat(xa, xb, ac["fs"], ac["band_hz"], ac["window_s"])
            # audio-array lag -> lag on the videos' own timelines (0 = first frame)
            d["audio_lag_s"] = raw + (ib["a_start"] - ib["t0"]) - (ia["a_start"] - ia["t0"])
            d["audio_peak_ratio"] = ratio
            d["audio_edge"] = abs(raw) > ac["edge_s"]
        out.append(d)
    return out


def pair_list(rows):
    """Same-group view pairs, ordered by view (F < L < R)."""
    order = {"F": 0, "L": 1, "R": 2}
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group_id"], []).append(r)
    pairs = []
    for g in by_group.values():
        g = sorted(g, key=lambda r: order.get(r["view"], 9))
        pairs += list(itertools.combinations(g, 2))
    return pairs


def audio_stage(idx, cfg, workers):
    """Audio/creation-time results for every same-group pair, cached in data/cache/sync/audio_pairs.csv."""
    cache = resolve(cfg["sync"]["cache_dir"]) / "audio_pairs.csv"
    old = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["video_a", "video_b"])
    multi = idx[idx.n_views_group >= 2]
    need = []
    have = set(zip(old.video_a, old.video_b))
    root = data_root(cfg)
    for (ex, subj), g in multi.groupby(["exercise", "subject"]):
        rows = g.to_dict("records")
        if any((a["video_id"], b["video_id"]) not in have for a, b in pair_list(rows)):
            need.append((rows, cfg["sync"]["audio"], root))
    if need:
        print(f"audio: decoding {sum(len(n[0]) for n in need)} videos in {len(need)} subject jobs", flush=True)
        with ProcessPoolExecutor(workers) as pool:
            new = [r for res in pool.map(audio_subject_job, need) for r in res]
        old = pd.concat([old[~old.set_index(["video_a", "video_b"]).index.isin(
            [(r["video_a"], r["video_b"]) for r in new])], pd.DataFrame(new)], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        old.to_csv(cache, index=False)
    return old

# -------------------------------------------------------------------------------------------- motion


def motion_signals(pre_path):
    """Candidate 1D motion signals on the uniform grid: wrist height above mid-shoulder (2D, up positive),
    3D shoulder elevation per side, and PC1 of both wrists relative to mid-shoulder."""
    with np.load(pre_path) as z:
        xy, w3 = z["xy"].astype(np.float64), z["world"].astype(np.float64)
    msh = (xy[:, J["left_shoulder"]] + xy[:, J["right_shoulder"]]) / 2
    sig = {}
    for side in ("left", "right"):
        sig[f"wy_{side[0]}"] = -(xy[:, J[f"{side}_wrist"], 1] - msh[:, 1])
        sig[f"el_{side[0]}"] = shoulder_elevation(w3, side)
    rel = np.concatenate([xy[:, J["left_wrist"]] - msh, xy[:, J["right_wrist"]] - msh], 1)
    if np.isfinite(rel).all() and len(rel) > 2:
        z0 = rel - rel.mean(0)
        sig["pc1"] = z0 @ np.linalg.svd(z0, full_matrices=False)[2][0]
    return {k: v for k, v in sig.items() if np.isfinite(v).all() and v.std() > 1e-6}


CAND_PAIRS = [("wy_r", "wy_r"), ("wy_l", "wy_l"), ("wy_r", "wy_l"), ("wy_l", "wy_r"), ("el_r", "el_r"),
              ("el_l", "el_l"), ("el_r", "el_l"), ("el_l", "el_r"), ("pc1", "pc1")]


def xcorr_lags(a, b, lags, min_overlap):
    """Pearson r of a[i] vs b[i + lag] for each integer lag (NaN when the overlap is too short)."""
    need = min_overlap * min(len(a), len(b))
    rs = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        a0, b0 = max(0, -lag), max(0, lag)
        k = min(len(a) - a0, len(b) - b0)
        if k < max(need, 10):
            continue
        x, y = a[a0:a0 + k], b[b0:b0 + k]
        sx, sy = x.std(), y.std()
        if sx > 1e-9 and sy > 1e-9:
            rs[i] = np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)
    return rs


def crossings(s, fps, hyst):
    """Times (s) where the z-scored signal crosses its median, with hysteresis +-hyst; (time, direction)."""
    z = (s - np.median(s)) / (s.std() + 1e-9)
    ev, state = [], None
    for i in range(1, len(z)):
        new = "hi" if z[i] > hyst else "lo" if z[i] < -hyst else None
        if new is None:
            continue
        if state is not None and new != state:
            j = i
            while j > 0 and np.sign(z[j]) == np.sign(z[i]):
                j -= 1
            t0 = j + (0 - z[j]) / (z[j + 1] - z[j]) if z[j + 1] != z[j] else j
            ev.append((t0 / fps, 1 if new == "hi" else -1))
        state = new
    return ev


def timing_test(a, b, lag_s, sign, fps, mc):
    """Match crossings of a and (b shifted back by lag); residual SDs (ms) and the mean residual (s)."""
    ea = crossings(a, fps, mc["hysteresis"])
    eb = [(t - lag_s, d * sign) for t, d in crossings(b, fps, mc["hysteresis"])]
    res, ts = [], []
    for t, d in ea:
        c = [tb for tb, db in eb if db == d and abs(tb - t) < mc["match_tol_s"]]
        if c:
            res.append(min(c, key=lambda x: abs(x - t)) - t)
            ts.append(t)
    out = dict(matched=len(res), cross_a=len(ea), cross_b=len(eb), resid_sd_ms=np.nan, resid_sd_lin_ms=np.nan,
               mean_resid_s=np.nan)
    if len(res) >= 3:
        res, ts = np.array(res), np.array(ts)
        p = np.polyfit(ts, res, 1)
        out.update(resid_sd_ms=res.std() * 1000, resid_sd_lin_ms=(res - np.polyval(p, ts)).std() * 1000,
                   mean_resid_s=float(res.mean()))
    return out


def motion_pair(sa, sb, center_s, window_s, fps, mc):
    """Best candidate-signal lag within center +- window, then the timing test."""
    lags = np.arange(int(np.floor((center_s - window_s) * fps)), int(np.ceil((center_s + window_s) * fps)) + 1)
    best = None
    for na, nb in CAND_PAIRS:
        if na not in sa or nb not in sb:
            continue
        rs = xcorr_lags(sa[na], sb[nb], lags, mc["min_overlap"])
        if np.isnan(rs).all():
            continue
        score = np.abs(rs) if na == "pc1" else rs
        i = int(np.nanargmax(score))
        if best is None or score[i] > best[0]:
            best = (score[i], i, rs, na, nb)
    if best is None:
        return dict(motion_r=np.nan)
    sc, i, rs, na, nb = best
    sign = int(np.sign(rs[i])) if na == "pc1" else 1
    frac = 0.0
    if 0 < i < len(rs) - 1 and np.isfinite(rs[i - 1:i + 2]).all():
        y0, y1, y2 = sign * rs[i - 1], sign * rs[i], sign * rs[i + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    lag_s = (lags[i] + frac) / fps
    tt = timing_test(sa[na], sb[nb], lag_s, sign, fps, mc)
    at_edge = i in (0, len(rs) - 1)
    confirmed = (tt["matched"] >= mc["min_matched"] and tt["resid_sd_lin_ms"] < mc["resid_threshold_ms"]
                 and not at_edge)
    refined = lag_s + (tt["mean_resid_s"] if np.isfinite(tt["mean_resid_s"]) else 0.0)
    return dict(motion_r=float(sc), motion_cand=f"{na}/{nb}", motion_lag_s=refined, motion_at_edge=at_edge,
                motion_confirmed=bool(confirmed), **tt)


def motion_job(args):
    """args: list of (key, pre_a, pre_b, center_s, window_s); returns [(key, result)]."""
    items, fps, mc = args
    cache = {}
    out = []
    for key, pa, pb, center, window in items:
        for p in (pa, pb):
            if p not in cache:
                cache[p] = motion_signals(p) if Path(p).exists() else {}  # excluded video: no signals
        out.append((key, motion_pair(cache[pa], cache[pb], center, window, fps, mc)))
    return out


def run_motion(items, fps, mc, workers):
    """Group items by subject so each worker loads a subject's signals once."""
    by = {}
    for it in items:
        by.setdefault(it[0][0], []).append(it)
    with ProcessPoolExecutor(workers) as pool:
        res = [r for part in pool.map(motion_job, [(v, fps, mc) for v in by.values()]) for r in part]
    return dict(res)

# ------------------------------------------------------------------------------------ classification


def is_true(x) -> bool:
    """True only for a real True (False for False, NaN and pd.NA)."""
    return bool(x) if pd.notna(x) else False


def device_offset(offsets, va, vb):
    """Video-minus-audio lag offset for a view pair; L-R is derived from F-L and F-R when not estimated."""
    if f"{va}-{vb}" in offsets:
        return offsets[f"{va}-{vb}"]
    if f"F-{va}" in offsets and f"F-{vb}" in offsets:
        return offsets[f"F-{vb}"] - offsets[f"F-{va}"]
    return 0.0


def solve_offsets(views, ref, lags, weights=None):
    """Weighted least-squares per-view offsets from pair lags {(va, vb): lag} (weights {(va, vb): w},
    default 1). Views not connected to ref -> NaN. Returns (offsets, RMS residual of the pair lags in ms)."""
    conn, changed = {ref}, True
    while changed:
        changed = False
        for (a, b) in lags:
            if (a in conn) != (b in conn):
                conn |= {a, b}
                changed = True
    unk = [v for v in views if v != ref and v in conn]
    off = {v: np.nan for v in views}
    off[ref] = 0.0
    if not unk:
        return off, np.nan
    A, y, w = [], [], []
    for (a, b), lag in lags.items():
        if a in conn and b in conn:
            row = np.zeros(len(unk))
            if a != ref:
                row[unk.index(a)] -= 1
            if b != ref:
                row[unk.index(b)] += 1
            A.append(row)
            y.append(lag)
            w.append((weights or {}).get((a, b), 1.0))
    A, y, sw = np.array(A), np.array(y), np.sqrt(np.array(w))
    sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    for v, o in zip(unk, sol):
        off[v] = float(o)
    resid = y - A @ sol
    return off, float(np.sqrt(np.mean(resid ** 2)) * 1000) if len(y) > len(unk) else np.nan


def motion_components(views, edges):
    """Connected components of views over motion-confirmed edges: {view: component id}."""
    comp = {v: v for v in views}

    def find(v):
        while comp[v] != v:
            v = comp[v]
        return v
    for a, b in edges:
        comp[find(a)] = find(b)
    return {v: find(v) for v in views}


def classify(idx, P, sc):
    """P: per-pair table with audio, creation-time and motion columns. Returns (groups, pairs)."""
    meta = idx.set_index("video_id")
    rows, pair_rows = [], []
    for gid, g in P.groupby("group_id"):
        views = sorted(set(g.view_a) | set(g.view_b), key="FLR".index)
        ref = "F" if "F" in views else views[0]
        first = meta.loc[g.video_a.iloc[0]]
        lag = {(a, b): l for a, b, l in zip(g.view_a, g.view_b, g.audio_lag_s)}
        tri = np.nan
        if views == ["F", "L", "R"] and np.isfinite(list(lag.values())).all() and len(lag) == 3:
            tri = 1000 * abs(lag[("F", "L")] + lag[("L", "R")] - lag[("F", "R")])
        edge = g.audio_edge.fillna(True).astype(bool).any()
        meta_apart = (g.d_creation_s.abs() > (g.dur_a + g.dur_b)).any()
        meta_close = (g.d_creation_s.abs() <= sc["meta"]["max_dc_s"]).all()
        conf_all = g.motion_confirmed.fillna(False).astype(bool).all()
        if tri <= sc["audio"]["additivity_ms"] and not edge:
            cls, method, confd = "synced", "audio", "high"
        elif meta_apart:
            cls, method, confd = "not-synced", "metadata", "high"
        elif meta_close:
            cls, method, confd = "synced", "metadata", "medium"
        elif conf_all:
            cls, method, confd = "synced", "motion", "medium"
        else:
            cls, method, confd = "unknown", "-", "-"
        # lags for alignment: motion-confirmed lags (weight 1) and device-corrected audio lags (weight
        # sync.audio_lag_weight); audio groups use both; creation-time groups too (supervisor answer 3), but
        # only views joined by motion-confirmed pairs count as 'sync', the rest as 'audio'
        mot = {(r.view_a, r.view_b): r.motion_lag_s for r in g.itertuples() if is_true(r.motion_confirmed)}
        aud = {(r.view_a, r.view_b): r.audio_lag_dev_s for r in g.itertuples()
               if np.isfinite(r.audio_lag_dev_s) and pd.notna(r.audio_edge) and not r.audio_edge}
        lags = {**aud, **mot}
        wts = {k: (1.0 if k in mot else sc["audio_lag_weight"]) for k in lags}
        if cls == "synced":
            off, ls_ms = solve_offsets(views, ref, lags, wts)
        else:
            off, ls_ms = {v: np.nan for v in views}, np.nan
        aligned = [v for v in views if np.isfinite(off[v])]
        if len(aligned) < 2:
            aligned = []
        comp = motion_components(views, mot)
        for r in g.itertuples():
            both = r.view_a in aligned and r.view_b in aligned
            if not both:
                src, lag_src = "rep", "-"
            elif method in ("audio", "motion") or comp[r.view_a] == comp[r.view_b]:
                src = "sync"
                lag_src = "motion" if (r.view_a, r.view_b) in mot else "audio+device_offset" if (
                    r.view_a, r.view_b) in aud else "least_squares"
            else:
                src, lag_src = "audio", "audio+device_offset" if (r.view_a, r.view_b) in aud else "least_squares"
            final = off[r.view_b] - off[r.view_a] if both else np.nan
            pair_rows.append(dict(r._asdict(), final_lag_s=final, lag_source=lag_src, pair_source=src,
                                  pair_weight=sc["pair_weights"][src], group_cls=cls))
        via = {}
        for v in views:
            via[v] = ("-" if v not in aligned else "ref" if v == ref else "audio" if method == "audio"
                      else "motion" if comp[v] == comp[ref] else "audio")
        rows.append(dict(group_id=gid, exercise=first.exercise, subject=first.subject, variation=first.variation,
                         group=first.group, views="".join(views), n_views=len(views), cls=cls, method=method,
                         confidence=confd, max_abs_dC_s=float(g.d_creation_s.abs().max()),
                         audio_triangle_err_ms=tri, motion_confirmed_pairs=int(g.motion_confirmed.fillna(False).sum()),
                         n_pairs=len(g), ref_view=ref, aligned_views="".join(aligned), ls_resid_ms=ls_ms,
                         aligned_by=",".join(f"{v}:{via[v]}" for v in views),
                         **{f"offset_{v}_s": off.get(v, np.nan) for v in ("F", "L", "R")}))
    return pd.DataFrame(rows), pd.DataFrame(pair_rows).drop(columns=["Index"], errors="ignore")


def control_items(idx, gw):
    """Same subject, adjacent variations (certainly separate recordings), every view combination."""
    items = []
    for (ex, subj), g in idx.groupby(["exercise", "subject"]):
        vars_ = sorted(g.variation.unique())
        for v1, v2 in zip(vars_, vars_[1:]):
            for a in g[g.variation == v1].itertuples():
                for b in g[g.variation == v2].itertuples():
                    items.append(((f"{ex}_{subj}", "control", a.video_id, b.video_id),
                                  resolve(a.pre_path), resolve(b.pre_path), 0.0, gw))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cfg = load_config(a.config)
    sc, mc, fps = cfg["sync"], cfg["sync"]["motion"], cfg["preprocess"]["fps"]
    idx = pd.read_csv(resolve(cfg["data"]["index_csv"]))
    out = resolve(cfg["outputs"]["phase1"])
    out.mkdir(parents=True, exist_ok=True)

    A = audio_stage(idx, cfg, a.workers)
    meta = idx.set_index("video_id")
    keep = set(idx.video_id)
    P = A[A.video_a.isin(keep) & A.video_b.isin(keep)].copy()
    P["group_id"] = meta.loc[P.video_a, "group_id"].values
    P["view_a"], P["view_b"] = meta.loc[P.video_a, "view"].values, meta.loc[P.video_b, "view"].values
    P["subject_key"] = (meta.loc[P.video_a, "exercise"] + "_" + meta.loc[P.video_a, "subject"]).values
    P["audio_edge"] = P.audio_edge.astype("boolean")

    # motion: pass 1 with prior device offsets, re-estimate offsets, pass 2
    offsets = {k: v / 1000 for k, v in sc["device_offsets_ms"].items()}
    est = None
    for it in (1, 2):
        items = []
        for r in P.itertuples():
            usable = np.isfinite(r.audio_lag_s) and pd.notna(r.audio_edge) and not r.audio_edge
            center = r.audio_lag_s + device_offset(offsets, r.view_a, r.view_b) if usable else 0.0
            window = mc["refine_window_s"] if usable else mc["global_window_s"]
            items.append(((r.subject_key, r.video_a, r.video_b), resolve(meta.loc[r.video_a, "pre_path"]),
                          resolve(meta.loc[r.video_b, "pre_path"]), center, window))
        res = run_motion(items, fps, mc, a.workers)
        M = pd.DataFrame([dict(video_a=k[1], video_b=k[2], motion_center_s=c, motion_window_s=w, **res[k])
                          for k, _, _, c, w in items])
        Q = P.merge(M, on=["video_a", "video_b"], how="left")
        # offsets only from simultaneous recordings (|dC| <= 3 s), where the audio lag is meaningful
        ok = Q[Q.motion_confirmed.fillna(False).astype(bool) & Q.audio_lag_s.notna() & ~Q.audio_edge.fillna(True)
               & (Q.d_creation_s.abs() <= sc["meta"]["max_dc_s"])]
        est = (ok.assign(vp=ok.view_a + "-" + ok.view_b, d_ms=1000 * (ok.motion_lag_s - ok.audio_lag_s))
               .groupby("vp").d_ms.agg(["count", "median", "mean", "std"]).round(1))
        print(f"\nmotion pass {it}: confirmed {int(Q.motion_confirmed.fillna(False).sum())}/{len(Q)} pairs; "
              "video lag - audio lag (ms) on confirmed pairs:")
        print(est.to_string())
        new = {vp: row["median"] / 1000 for vp, row in est.iterrows() if row["count"] >= 5}
        offsets = {**offsets, **new}
    est.to_csv(out / "sync_device_offsets.csv")
    Q["audio_lag_dev_s"] = [r.audio_lag_s + device_offset(offsets, r.view_a, r.view_b)
                            if np.isfinite(r.audio_lag_s) else np.nan for r in Q.itertuples()]
    Q["motion_confirmed"] = Q.motion_confirmed.astype("boolean")

    G, Pairs = classify(idx, Q, sc)
    G.to_csv(resolve(sc["report_csv"]), index=False)
    Pairs.to_csv(resolve(sc["pairs_csv"]), index=False)

    # controls: false-positive rate of the motion test (global search, the most permissive setting)
    C = run_motion(control_items(idx, mc["global_window_s"]), fps, mc, a.workers)
    Ct = pd.DataFrame([dict(video_a=k[2], video_b=k[3], **v) for k, v in C.items()])
    Ct.to_csv(out / "sync_controls.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== groups:")
    print(G.groupby(["exercise", "group", "cls", "method"]).size().unstack(["cls", "method"], fill_value=0).to_string())
    print("\n=== pair sources for CVDC:")
    Pairs["exercise"] = Pairs.group_id.str[:3]
    print(Pairs.groupby(["exercise", "pair_source", "lag_source"]).size().unstack(fill_value=0).to_string())
    sg = Pairs[Pairs.group_cls == "synced"]
    print("\nmotion confirmation within synced groups, by view pair:",
          sg.assign(c=sg.motion_confirmed.fillna(False).astype(bool)).groupby(sg.view_a + "-" + sg.view_b)
          .c.mean().round(3).to_dict())
    tested = Ct[Ct.matched >= mc["min_matched"]]
    print(f"\n=== controls (separate recordings): {len(Ct)} pairs, {len(tested)} testable; "
          f"falsely confirmed: {int(tested.motion_confirmed.sum())} "
          f"({tested.motion_confirmed.mean():.1%}); resid_sd_lin_ms p1/p5/median: "
          f"{tested.resid_sd_lin_ms.quantile(.01):.0f}/{tested.resid_sd_lin_ms.quantile(.05):.0f}/"
          f"{tested.resid_sd_lin_ms.median():.0f}")


if __name__ == "__main__":
    main()
