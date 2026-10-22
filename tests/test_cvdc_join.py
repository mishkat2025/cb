"""M19: the CVDC representation inside Module III (cvdc/assess/cvdc_join.py) and STGCN.sequence."""
import numpy as np
import torch

from cvdc.assess.cvdc_join import align_steps, frame_distance, frame_path, step_frames
from cvdc.models.stgcn import STGCN


def test_sequence_mean_is_the_embedding():
    torch.manual_seed(0)
    for embed in (256, 64):  # identity and linear output layer
        m = STGCN(in_channels=2, embed_dim=embed).eval()
        x = torch.randn(3, 2, 64, 17)
        with torch.no_grad():
            s, z = m.sequence(x), m(x)
        assert s.shape == (3, 16, embed)
        assert torch.allclose(s.mean(1), z, atol=1e-5)


def test_frame_path_is_monotone_and_covers_every_frame():
    rng = np.random.default_rng(0)
    for n_rep, n_ref in ((40, 112), (150, 68), (64, 64), (5, 30)):
        a, b = rng.normal(size=(16, 8)), rng.normal(size=(16, 8))
        sp, _ = align_steps(a, b)
        p = frame_path(sp, n_rep, n_ref, 16, 64)
        assert tuple(p[0]) == (0, 0) and tuple(p[-1]) == (n_rep - 1, n_ref - 1)
        d = np.diff(p, axis=0)
        assert (d >= 0).all() and (d <= 1).all() and (d.sum(1) >= 1).all()
        assert set(p[:, 0]) == set(range(n_rep)) and set(p[:, 1]) == set(range(n_ref))


def test_identical_sequences_align_on_the_diagonal_with_zero_distance():
    a = np.random.default_rng(1).normal(size=(16, 32))
    sp, d = align_steps(a, a * 3.0)  # cosine geometry: scale does not matter
    assert (sp[:, 0] == sp[:, 1]).all() and np.allclose(d, 0, atol=1e-9)
    p = frame_path(sp, 64, 64, 16, 64)
    assert (p[:, 0] == p[:, 1]).all()
    fd = frame_distance(sp, d, 50, 16, 64)
    assert fd.shape == (50,) and np.allclose(fd, 0, atol=1e-9)


def test_a_shifted_sequence_is_found_by_the_alignment():
    t = np.linspace(0, 1, 16)
    base = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), t], 1)
    late = np.stack([np.sin(2 * np.pi * np.clip(t - 0.25, 0, 1)), np.cos(2 * np.pi * np.clip(t - 0.25, 0, 1)),
                     np.clip(t - 0.25, 0, 1)], 1)
    sp, _ = align_steps(late, base)
    # the late rep's step 8 (t ~ 0.53) shows what the reference shows at t ~ 0.28 (step ~4)
    j = sp[sp[:, 0] == 8, 1]
    assert 3 <= j.min() and j.max() <= 6


def test_step_frames_span_the_rep():
    f = step_frames(101, 16, 64)
    assert f[0] == 0 and 90 < f[-1] < 100 and np.all(np.diff(f) > 0)
