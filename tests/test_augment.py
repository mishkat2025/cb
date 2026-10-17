import torch

from cvdc.data.augment import augment, temporal_crop

AC = {"enabled": True, "rotate_deg": 10, "scale": 0.1, "shear": 0.1, "joint_dropout": 0.0, "crop_min": 0.8,
      "noise": 0.0, "share_temporal_crop": True}


def ramp(n=6, T=64, V=17):
    """x coordinate = time index, so a temporal crop shows up directly in the values."""
    x = torch.zeros(n, T, V, 2)
    x[..., 0] = torch.arange(T, dtype=torch.float32)[None, :, None]
    return x


def test_temporal_crop_full_range_is_identity():
    x = ramp()
    torch.testing.assert_close(temporal_crop(x, torch.ones(6), torch.zeros(6)), x)


def test_shared_crop_keeps_views_of_a_set_time_aligned():
    x = ramp()
    set_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    ac = {**AC, "rotate_deg": 0, "scale": 0, "shear": 0}
    out = augment(x, ac, set_ids=set_ids, generator=torch.Generator().manual_seed(0))
    t = out[:, :, 0, 0]
    torch.testing.assert_close(t[0], t[1])
    torch.testing.assert_close(t[0], t[2])
    torch.testing.assert_close(t[3], t[4])
    assert not torch.allclose(t[0], t[3])
    span = t[:, -1] - t[:, 0]
    assert (span >= 0.8 * 63 - 1e-3).all() and (span <= 63 + 1e-3).all()


def test_independent_crops_when_not_shared():
    x = ramp()
    ac = {**AC, "rotate_deg": 0, "scale": 0, "shear": 0, "share_temporal_crop": False}
    out = augment(x, ac, set_ids=torch.zeros(6, dtype=torch.long), generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(out[0, :, 0, 0], out[1, :, 0, 0])


def test_spatial_augmentation_is_per_item_and_keeps_visibility():
    x = torch.randn(4, 64, 17, 3)
    x[..., 2] = 0.7  # visibility channel (xyv input)
    out = augment(x, {**AC, "crop_min": 1.0}, coord_channels=2, generator=torch.Generator().manual_seed(1))
    assert out.shape == x.shape
    torch.testing.assert_close(out[..., 2], x[..., 2])
    ratio = out[..., :2].flatten(1).norm(dim=1) / x[..., :2].flatten(1).norm(dim=1)
    assert ((ratio > 0.8) & (ratio < 1.25)).all() and ratio.std() > 0  # small, different per item


def test_joint_dropout_moves_joints_to_the_root():
    x = torch.ones(200, 8, 17, 3)
    ac = {**AC, "crop_min": 1.0, "rotate_deg": 0, "scale": 0, "shear": 0, "joint_dropout": 0.1}
    out = augment(x, ac, generator=torch.Generator().manual_seed(2))
    dropped = (out == 0).all(-1).all(1)  # whole joint zero over time (coordinates and visibility)
    assert 0.07 < dropped.float().mean() < 0.13


def test_disabled_returns_input():
    x = torch.randn(2, 8, 17, 2)
    assert augment(x, {**AC, "enabled": False}) is x
