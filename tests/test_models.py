import copy

import pytest
import torch

from cvdc.config import load_config
from cvdc.models.model import CVDCModel
from cvdc.models.stgcn import STGCN

CFG = load_config()


@pytest.mark.parametrize("mode,c", [("xy", 2), ("xyv", 3), ("xyz", 3)])
@pytest.mark.parametrize("T", [64, 300])
def test_shapes_for_every_input_mode_and_length(mode, c, T):
    cfg = copy.deepcopy(CFG)
    cfg["model"]["input"] = mode
    m = CVDCModel(cfg).eval()
    out = m(torch.randn(3, c, T, 17), exercise=torch.tensor([0, 1, 0]), project=True)
    assert out["z"].shape == (3, 256)
    assert out["proj"].shape == (3, 128)
    torch.testing.assert_close(out["proj"].norm(dim=-1), torch.ones(3))
    assert out["exercise"].shape == (3, 9)  # all configured exercises
    assert out["score"].shape == (3,) and out["quality"].shape == (3, 3)
    assert out["items"].shape == (3, 3)  # CF1-CF3 at rep level


def test_video_level_heads_include_pf_items():
    m = CVDCModel(CFG, unit="video")
    assert m.heads.items == ["PF1", "PF2", "PF4", "CF1", "CF2", "CF3"]


def test_encoder_is_yan_et_al_size():
    enc = STGCN.from_config(CFG["model"], 2)
    assert len(enc.blocks) == 10
    n = sum(p.numel() for p in enc.parameters())
    assert 2.9e6 < n < 3.3e6  # ~3.1 M parameters, as the reference ST-GCN without its classifier


def test_views_share_weights():
    """Encoding the views of a sample in one batch equals encoding each view separately (same weights, no
    per-view parameters; eval mode so BatchNorm uses its running statistics)."""
    m = CVDCModel(CFG).eval()
    views = torch.randn(3, 2, 64, 17)  # F, L, R views of one rep
    together = m.encoder(views)
    apart = torch.cat([m.encoder(v[None]) for v in views])
    torch.testing.assert_close(together, apart, rtol=1e-4, atol=1e-5)


def test_exercise_head_does_not_see_the_exercise_embedding():
    m = CVDCModel(CFG).eval()
    x = torch.randn(2, 2, 64, 17)
    a = m(x, exercise=torch.tensor([0, 0]))
    b = m(x, exercise=torch.tensor([1, 1]))
    torch.testing.assert_close(a["exercise"], b["exercise"])
    assert not torch.allclose(a["score"], b["score"])  # the score head is conditioned on the exercise
    c = m(x)  # unknown exercise: the predicted one is used
    assert c["score"].shape == (2,)


def test_gradients_reach_every_encoder_parameter():
    m = CVDCModel(CFG).train()
    out = m(torch.randn(4, 2, 64, 17), exercise=torch.tensor([0, 1, 0, 1]), project=True)
    (out["proj"].sum() + out["score"].sum()).backward()
    missing = [n for n, p in m.encoder.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
