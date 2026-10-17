import av
import numpy as np
import pytest

from cvdc.data.video import iter_frames


def write_video(path, w, h, n=3, fps=20):
    """Tiny synthetic video (random pixels, no people) with rotation flag 0."""
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        rng = np.random.default_rng(0)
        for _ in range(n):
            fr = av.VideoFrame.from_ndarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), format="rgb24")
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


@pytest.mark.parametrize("pad", [False, True])
def test_frames_are_contiguous_and_timestamped(tmp_path, pad):
    """Regression: a 1080x1920 upright video scaled to 540 px wide gave padded (non-contiguous) rgb rows,
    which MediaPipe read as garbage (0% detection on two PT demos)."""
    f = tmp_path / "portrait.mp4"
    write_video(f, 1080, 1920)
    frames = list(iter_frames(f, max_side=960, pad_square=pad))
    assert len(frames) == 3
    for t, img, geom in frames:
        assert img.flags["C_CONTIGUOUS"] and img.dtype == np.uint8
        assert geom["proc_hw"] == (960, 540) and geom["orig_hw"] == (1920, 1080)
        assert img.shape[:2] == ((960, 960) if pad else (960, 540))
    assert np.allclose(np.diff([t for t, _, _ in frames]), 1 / 20, atol=1e-3)
