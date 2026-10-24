"""Background four-panel exports for the Streamlit app (X9, X13): one export per video at a time, each run writing its
own files, and the finished video decoded in full before it is shown.

Why (2026-10-24): a second click on "Assess this video" while the video of the same file was being made restarted the
page. The first export kept running, a second one started, and both wrote the same `<video>.part.mp4`; the saved video's
first 16.7 s could not be decoded, so it played from 16.7 s. Now:

- `Registry` (one per Streamlit server; app.py keeps it in `st.cache_resource`) maps each output video to its export.
  A rerun, a second click or another tab asking for a video that is being made waits for that export instead of
  starting another.
- Every run writes its own temporary files, `<video>.<run id>.part.mp4` and `<video>.<run id>.log`, so no two processes
  ever write the same file. Leftovers of runs the registry does not know (an earlier server) are removed when a new run
  of the same video starts.
- `check_video` decodes every frame; the video replaces `<video>.mp4` only if it decodes from its first frame (at 0 s)
  to its last. A damaged one is deleted and reported. app.py runs the same check before reusing a saved video.
"""
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import av


def check_video(path):
    """(ok, message): every packet of the video stream decodes to a frame, and the first frame is at the start."""
    n_packets = n_frames = 0
    first = t = None
    try:
        with av.open(str(path)) as c:
            vs = c.streams.video[0]  # default (non-frame) threading, so an error is reported at the damaged frame
            for pk in c.demux(vs):
                if pk.size:
                    n_packets += 1
                    t = float(pk.pts * vs.time_base) if pk.pts is not None else t
                for fr in pk.decode():  # the last, empty packet flushes the decoder
                    n_frames += 1
                    if first is None:
                        first = float(fr.pts * vs.time_base) if fr.pts is not None else 0.0
            rate = float(vs.average_rate or 30)
    except (av.error.FFmpegError, IndexError) as e:
        where = f" at {t:.2f} s" if t is not None else ""
        return False, f"cannot be decoded{where}: {type(e).__name__}"
    if n_frames == 0:
        return False, "no frame decodes"
    if first > 1.5 / rate:
        return False, f"the first frame that decodes is at {first:.2f} s"
    if n_frames != n_packets:
        return False, f"{n_frames} of {n_packets} frames decode"
    return True, f"all {n_frames} frames decode"


def leftovers(out):
    """Temporary files of earlier runs of `out`: `<stem>.<run id>.part.mp4` / `.log`, and the old `<stem>.part.mp4`."""
    out = Path(out)
    pat = re.compile(re.escape(out.stem) + r"(\.\d{8}_\d{6}_[0-9a-f]{6}\.(part\.mp4|log)|\.part\.mp4)$")
    return sorted(p for p in out.parent.glob("*") if pat.fullmatch(p.name)) if out.parent.exists() else []


@dataclass
class Job:
    out: Path
    part: Path
    log: Path
    proc: subprocess.Popen
    log_file: object
    t0: float
    result: tuple = None  # (video or None, error or None), set once the export is finished and checked
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class Registry:
    """The exports of one Streamlit server, by output video."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}

    def running(self, out):
        """The export of `out` that is running or not yet checked, else None."""
        with self._lock:
            job = self._jobs.get(Path(out))
            return job if job is not None and job.result is None else None

    def start(self, out, cmd, cwd=None, env=None):
        """(job, started): the export of `out` already under way, or a new one. `cmd` gets `--export4 <run's .part>`."""
        out = Path(out)
        with self._lock:
            job = self._jobs.get(out)
            if job is not None and job.result is None:
                return job, False
            for p in leftovers(out):
                try:
                    p.unlink()
                except OSError:  # still open in a process this server does not track; it keeps its own files anyway
                    pass
            rid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            part, log = out.with_name(f"{out.stem}.{rid}.part.mp4"), out.with_name(f"{out.stem}.{rid}.log")
            out.parent.mkdir(parents=True, exist_ok=True)
            f = open(log, "w", encoding="utf-8")
            try:
                proc = subprocess.Popen([*map(str, cmd), "--export4", str(part)], cwd=cwd, env=env, stdout=f,
                                        stderr=subprocess.STDOUT)
            except Exception:
                f.close()
                log.unlink(missing_ok=True)
                raise
            job = self._jobs[out] = Job(out, part, log, proc, f, time.time())
            return job, True

    def finish(self, job, on_wait=None, on_check=None, poll_s=1.0):
        """Wait for the export, check its video and put it in place, once, whoever asks first: (video, error)."""
        while job.proc.poll() is None:
            if on_wait is not None:
                on_wait(job)
            time.sleep(poll_s)
        with job.lock:
            if job.result is None:
                if on_check is not None:
                    on_check(job)
                job.result = self._finalise(job)
        return job.result

    @staticmethod
    def _finalise(job):
        job.log_file.close()
        log_out = job.out.with_suffix(".log")
        error = None
        if job.proc.returncode == 0 and job.part.exists():
            ok, msg = check_video(job.part)
            if ok:
                job.part.replace(job.out)
                job.log.replace(log_out)
                return job.out, None
            error = f"The video was made but is damaged ({msg}), so it was deleted."
        job.part.unlink(missing_ok=True)
        tail = job.log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        job.log.replace(log_out)
        return None, "\n".join(([error] if error else []) + tail)
