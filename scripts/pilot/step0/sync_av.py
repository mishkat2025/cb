"""Audio + container-metadata synchronization evidence for multi-view groups.

Per (exercise, participant): decode each video's audio track to 8 kHz mono (kept in memory only, never
written), read creation_time / duration / stream start times (GPS tags are ignored), then for
  * same-group pairs (same exercise, participant, variation; different views)
  * control pairs (same exercise & participant, adjacent variations -> certainly separate recordings)
compute GCC-PHAT audio cross-correlation (lag, peak/second-peak ratio, robust z) and the
creation_time difference.
Lag convention: lag > 0  <=>  an event appears `lag` seconds LATER in video b than in video a.
usage: sync_av.py <inventory.csv> <out_dir> <workers> [EX ...]
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FS = 8000
# dataset root: env MOBIPHYSIO_ROOT, else the folder that contains the cvdc-physio repo
DATA_ROOT = Path(os.environ.get("MOBIPHYSIO_ROOT", Path(__file__).resolve().parents[4] / "dataset" / "videos"))


def load(path):
    import av
    c = av.open(str(path))
    md = c.metadata
    ct = md.get("creation_time")
    info = dict(creation=datetime.fromisoformat(ct.replace("Z", "+00:00")).timestamp() if ct else np.nan,
                duration=c.duration / 1e6 if c.duration else np.nan,
                model=md.get("com.android.model", ""), android=md.get("com.android.version", ""),
                capture_fps=md.get("com.android.capture.fps", ""), has_audio=bool(c.streams.audio))
    vs = c.streams.video[0] if c.streams.video else None
    info["v_start"] = float(vs.start_time * vs.time_base) if vs is not None and vs.start_time is not None else 0.0
    info["v_codec"] = vs.codec_context.name if vs is not None else ""
    audio = None
    if c.streams.audio:
        a = c.streams.audio[0]
        info["a_start"] = float(a.start_time * a.time_base) if a.start_time is not None else 0.0
        info["a_rate"] = a.rate
        res = av.AudioResampler(format="flt", layout="mono", rate=FS)
        chunks = []
        for fr in c.decode(a):
            for o in res.resample(fr):
                chunks.append(o.to_ndarray().ravel())
        for o in res.resample(None):
            chunks.append(o.to_ndarray().ravel())
        audio = np.concatenate(chunks) if chunks else None
        if audio is not None:
            info["audio_rms_db"] = float(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12))
    c.close()
    return info, audio


def gcc_phat_multi(x, y, windows):
    """Return [(lag_s, peak_ratio, peak_z) for each max-lag window] from one FFT."""
    n = len(x) + len(y)
    nfft = 1 << (n - 1).bit_length()
    X, Y = np.fft.rfft(x - x.mean(), nfft), np.fft.rfft(y - y.mean(), nfft)
    R = np.conj(X) * Y
    f = np.fft.rfftfreq(nfft, 1 / FS)
    R[(f < 100) | (f > 3800)] = 0
    R /= np.abs(R) + 1e-12
    full = np.fft.irfft(R, nfft)
    out = []
    for max_lag_s in windows:
        k = int(max_lag_s * FS)
        cc = np.abs(np.concatenate([full[-k:], full[:k + 1]]))
        lags = np.arange(-k, k + 1)
        i = int(np.argmax(cc))
        w = int(0.02 * FS)
        rest = np.concatenate([cc[:max(0, i - w)], cc[i + w + 1:]])
        second = rest.max() if len(rest) else np.nan
        med = np.median(cc)
        mad = np.median(np.abs(cc - med)) * 1.4826 + 1e-12
        out.append((lags[i] / FS, cc[i] / second, (cc[i] - med) / mad))
    return out




def work(args):
    ex, pid, rows = args
    infos, audio = {}, {}
    for r in rows:
        try:
            infos[(r["var"], r["angle"])], audio[(r["var"], r["angle"])] = load(r["path"])
        except Exception as e:  # noqa: BLE001
            infos[(r["var"], r["angle"])], audio[(r["var"], r["angle"])] = dict(error=str(e)), None
        infos[(r["var"], r["angle"])].update(ex=ex, pid=pid, var=r["var"], angle=r["angle"], file=r["file"])
    keys = sorted(infos)
    pairs = [("same_group", a, b) for a, b in itertools.combinations(keys, 2) if a[0] == b[0]]
    vars_ = sorted({k[0] for k in keys})
    for v1, v2 in zip(vars_, vars_[1:]):
        pairs += [("control", a, b) for a in keys if a[0] == v1 for b in keys if b[0] == v2]
    # every LR video vs every other variation of the same view (is LR a re-encoded copy?)
    pairs += [("lr_check", a, b) for a in keys if a[0] == "LR" for b in keys if b[0] != "LR" and b[1] == a[1]]
    out = []
    for kind, a, b in pairs:
        ia, ib = infos[a], infos[b]
        r = dict(kind=kind, ex=ex, pid=pid, var_a=a[0], var_b=b[0], view_a=a[1], view_b=b[1],
                 dur_a=ia.get("duration"), dur_b=ib.get("duration"),
                 d_creation_s=ib.get("creation", np.nan) - ia.get("creation", np.nan),
                 model_a=ia.get("model", ""), model_b=ib.get("model", ""))
        xa, xb = audio[a], audio[b]
        if xa is not None and xb is not None and len(xa) > FS and len(xb) > FS:
            max_lag = min(15.0, 0.8 * min(len(xa), len(xb)) / FS)
            (lag, ratio, z), (lag3, ratio3, z3) = gcc_phat_multi(xa, xb, [max_lag, min(3.0, max_lag)])
            r.update(audio_lag3_raw_s=lag3, audio_peak_ratio3=ratio3, audio_peak_z3=z3)
            # audio lag -> video-timeline lag (correct for per-file audio/video start offsets)
            lag_v = lag - (ib.get("a_start", 0) - ib.get("v_start", 0)) + (ia.get("a_start", 0) - ia.get("v_start", 0))
            off = - (ib.get("a_start", 0) - ib.get("v_start", 0)) + (ia.get("a_start", 0) - ia.get("v_start", 0))
            r.update(audio_lag_s=lag_v, audio_peak_ratio=ratio, audio_peak_z=z, audio_lag3_s=lag3 + off)
        out.append(r)
    return list(infos.values()), out


if __name__ == "__main__":
    inv = pd.read_csv(sys.argv[1])
    out_dir, workers = Path(sys.argv[2]), int(sys.argv[3])
    exs = sys.argv[4:] or sorted(inv.ex.unique())
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = inv[inv.ex.isin(exs)].copy()
    inv["path"] = [str(DATA_ROOT / f / n) for f, n in zip(inv.folder, inv.file)]
    jobs = [(ex, pid, g.to_dict("records")) for (ex, pid), g in inv.groupby(["ex", "pid"])]
    vids, pairs = [], []
    with ProcessPoolExecutor(workers) as pool:
        for i, (v, p) in enumerate(pool.map(work, jobs), 1):
            vids += v
            pairs += p
            if i % 10 == 0:
                print(f"{i}/{len(jobs)} participants done", flush=True)
    pd.DataFrame(vids).to_csv(out_dir / "av_videos.csv", index=False)
    pd.DataFrame(pairs).to_csv(out_dir / "av_pairs.csv", index=False)
    print("videos:", len(vids), "pairs:", len(pairs))

