"""Video access with PyAV: real presentation timestamps and the rotation flag honoured (D9).

Frames and audio are held in memory only; nothing here writes pixels or audio to disk (D10). Container
metadata is read key by key (``creation_time`` only); location/GPS tags are never accessed.
"""
from datetime import datetime

import av
import numpy as np


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def iter_frames(path, max_side=None, pad_square=False):
    """Yield ``(t, rgb, geom)`` per decoded frame.

    ``t``     presentation time in seconds on the container clock.
    ``rgb``   HxWx3 uint8 upright image, downscaled so the long side is <= ``max_side``; if ``pad_square``
              it is centred on a black square canvas.
    ``geom``  dict(orig_hw, proc_hw, pad=(oy, ox, S)) to map normalised landmarks back to the frame.
    """
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        dt = 1.0 / float(vs.average_rate or 30)
        for i, fr in enumerate(c.decode(vs)):
            t = float(fr.time) if fr.time is not None else i * dt
            W, H = fr.width, fr.height
            s = min(1.0, max_side / max(W, H)) if max_side else 1.0
            w, h = (_even(W * s), _even(H * s)) if s < 1 else (W, H)
            img = fr.to_ndarray(width=w, height=h, format="rgb24", interpolation="AREA")
            k = (int(fr.rotation or 0) // 90) % 4
            # always contiguous: rows of rgb24 frames can carry FFmpeg alignment padding, and MediaPipe
            # reads the buffer as packed rows (a padded 540-px-wide frame gave zero detections)
            img = np.ascontiguousarray(np.rot90(img, k) if k else img)
            orig_hw = (W, H) if k % 2 else (H, W)
            ph, pw = img.shape[:2]
            pad = (0, 0, 0)
            if pad_square and ph != pw:
                S = max(ph, pw)
                oy, ox = (S - ph) // 2, (S - pw) // 2
                canvas = np.zeros((S, S, 3), np.uint8)
                canvas[oy:oy + ph, ox:ox + pw] = img
                img, pad = canvas, (oy, ox, S)
            yield t, img, dict(orig_hw=orig_hw, proc_hw=(ph, pw), pad=pad)


def container_info(path) -> dict:
    """creation_time (unix s), duration, stream start times, audio presence. No other metadata is read."""
    with av.open(str(path)) as c:
        ct = c.metadata.get("creation_time")
        vs = c.streams.video[0]
        info = dict(creation=datetime.fromisoformat(ct.replace("Z", "+00:00")).timestamp() if ct else np.nan,
                    duration=c.duration / 1e6 if c.duration else np.nan,
                    v_start=float(vs.start_time * vs.time_base) if vs.start_time is not None else 0.0,
                    has_audio=bool(c.streams.audio), a_start=np.nan)
        if c.streams.audio:
            a = c.streams.audio[0]
            info["a_start"] = float(a.start_time * a.time_base) if a.start_time is not None else 0.0
    return info


def read_audio_mono(path, fs=8000):
    """Decode the first audio stream to mono float32 at ``fs`` Hz (memory only). Returns (x, a_start) or (None, nan)."""
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None, np.nan
        a = c.streams.audio[0]
        a_start = float(a.start_time * a.time_base) if a.start_time is not None else 0.0
        res = av.AudioResampler(format="flt", layout="mono", rate=fs)
        chunks = []
        for fr in c.decode(a):
            chunks += [o.to_ndarray().ravel() for o in res.resample(fr)]
        chunks += [o.to_ndarray().ravel() for o in res.resample(None)]
    return (np.concatenate(chunks).astype(np.float32) if chunks else None), a_start
