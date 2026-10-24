"""X13: background four-panel exports: one per video at a time, own files per run, damaged videos never shown."""
import sys
import threading
from fractions import Fraction

import av
import numpy as np
import pytest

from demo.export_jobs import Registry, check_video, leftovers


def make_video(path, n=30):
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as c:
        s = c.add_stream("libx264", rate=Fraction(30))
        s.width = s.height = 64
        s.pix_fmt = "yuv420p"
        for i in range(n):
            img = np.zeros((64, 64, 3), np.uint8)
            img[:, :, 0] = i * 8
            img[20:40, i:i + 10] = 255
            for pk in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return path


def damage(src, dst):
    """Overwrite the start of the frame data, as when two writers shared one file (2026-10-24)."""
    data = bytearray(src.read_bytes())
    m = data.find(b"mdat") + 4
    data[m:m + 1000] = bytes(1000)
    dst.write_bytes(data)
    return dst


@pytest.fixture
def videos(tmp_path):
    good = make_video(tmp_path / "good.mp4")
    return good, damage(good, tmp_path / "bad.mp4")


def fake_export(src, sleep=0.3, code=0):
    """A stand-in for demo/live.py --export4: copies `src` to the run's .part file (the last argument)."""
    body = (f"import shutil, sys, time\ntime.sleep({sleep})\nprint('rep 1', flush=True)\n"
            f"shutil.copy({str(src)!r}, sys.argv[-1])\nsys.exit({code})\n")
    return [sys.executable, "-c", body]


def test_check_video(videos, tmp_path):
    good, bad = videos
    assert check_video(good) == (True, "all 30 frames decode")
    ok, msg = check_video(bad)
    assert not ok and "cannot be decoded at 0.00 s" in msg
    (tmp_path / "text.mp4").write_text("not a video")
    assert not check_video(tmp_path / "text.mp4")[0]


def test_one_export_per_video(videos, tmp_path):
    good, _ = videos
    reg, out = Registry(), tmp_path / "out" / "E01_P01__x.mp4"
    job, started = reg.start(out, fake_export(good))
    job2, started2 = reg.start(out, fake_export(good))  # a second click or a rerun while it runs
    assert started and not started2 and job2 is job and reg.running(out) is job
    assert job.part.name.startswith("E01_P01__x.") and job.part.name.endswith(".part.mp4")
    video, error = reg.finish(job, poll_s=0.05)
    assert (video, error) == (out, None) and check_video(out)[0]
    assert not job.part.exists() and out.with_suffix(".log").read_text().startswith("rep 1")
    assert reg.running(out) is None and leftovers(out) == []


def test_damaged_video_is_deleted(videos, tmp_path):
    _, bad = videos
    reg, out = Registry(), tmp_path / "v.mp4"
    job, _ = reg.start(out, fake_export(bad))
    video, error = reg.finish(job, poll_s=0.05)
    assert video is None and error.startswith("The video was made but is damaged")
    assert not out.exists() and not job.part.exists()


def test_failed_export_reports_its_log(videos, tmp_path):
    good, _ = videos
    reg, out = Registry(), tmp_path / "v.mp4"
    job, _ = reg.start(out, fake_export(good, code=1))
    video, error = reg.finish(job, poll_s=0.05)
    assert video is None and "rep 1" in error and not out.exists() and not job.part.exists()


def test_two_waiters_get_one_result(videos, tmp_path):
    good, _ = videos
    reg, out = Registry(), tmp_path / "v.mp4"
    job, _ = reg.start(out, fake_export(good, sleep=0.5))
    results = []
    threads = [threading.Thread(target=lambda: results.append(reg.finish(job, poll_s=0.05))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert results == [(out, None), (out, None)]


def test_new_run_has_its_own_files_and_clears_leftovers(videos, tmp_path):
    good, _ = videos
    reg, out = Registry(), tmp_path / "v.mp4"
    old = [tmp_path / "v.part.mp4", tmp_path / "v.20260101_000000_abcdef.part.mp4",
           tmp_path / "v.20260101_000000_abcdef.log"]
    for p in old:
        p.write_text("left by an earlier server")
    (tmp_path / "v2.part.mp4").write_text("another video's file")
    assert leftovers(out) == sorted(old)
    job, _ = reg.start(out, fake_export(good))
    assert not any(p.exists() for p in old) and (tmp_path / "v2.part.mp4").exists()
    reg.finish(job, poll_s=0.05)
    job2, started = reg.start(out, fake_export(good))  # a fresh run once the first is done
    assert started and job2.part != job.part
    reg.finish(job2, poll_s=0.05)
