import torch

from cvdc.eval.consistency import vci_report, viewpoint_consistency
from cvdc.losses.supervised import SupervisedLoss, TargetScaler


def targets(n=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"score": 40 + 50 * torch.rand(n, generator=g),
            "items": torch.stack([torch.rand(n, generator=g), torch.ones(n)], 1),  # second item constant
            "quality": torch.randint(0, 3, (n,), generator=g), "exercise": torch.randint(0, 2, (n,), generator=g)}


def test_scaler_uses_training_items_only_and_inverts():
    tr, te = targets(seed=0), targets(seed=1)
    te["score"] = te["score"] + 30  # a shifted test distribution must not change the statistics
    sc = TargetScaler().fit(tr)
    z = sc.transform(tr)
    assert abs(z["score"].mean().item()) < 1e-5 and abs(z["score"].std(unbiased=False).item() - 1) < 1e-4
    torch.testing.assert_close(sc.stats["score_mean"], tr["score"].mean())
    back = sc.inverse({"score": sc.transform(te)["score"], "items": sc.transform(te)["items"]})
    torch.testing.assert_close(back["score"], te["score"], rtol=1e-5, atol=1e-4)
    assert sc.active.tolist() == [True, False]  # the constant item is skipped (T4)
    torch.testing.assert_close(back["items"][:, 1], torch.ones(100))  # inactive item -> training mean


def test_standardised_loss_weights_heads_comparably():
    y = targets()
    sc = TargetScaler().fit(y)
    loss = SupervisedLoss(["CF1", "CF3"], {"score": 1, "items": 1, "quality": 1, "exercise": 1}, sc)
    out = {"z": torch.zeros(100, 4), "score": torch.zeros(100), "items": torch.zeros(100, 2)}
    _, parts = loss(out, y)
    assert abs(parts["score"] - 1.0) < 1e-3  # predicting the mean costs 1 for any score scale
    assert abs(parts["items"] - 1.0) < 1e-3  # only the active item counts
    raw = SupervisedLoss(["CF1", "CF3"], {"score": 1, "items": 1, "quality": 1, "exercise": 1})
    _, p2 = raw({**out, "score": torch.full((100,), y["score"].mean().item() / 100)}, y)
    assert p2["score"] < 0.03  # the unscaled score / 100 loss is ~40x smaller at the mean


def test_centred_vci_ignores_a_common_offset():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(30, 16, generator=g)
    key = torch.arange(10).repeat_interleave(3)
    z = base.clone()
    z[1::3] = base[0::3] + 0.1 * torch.randn(10, 16, generator=g)  # view 2 close to view 1
    view, group = torch.arange(3).repeat(10), key.clone()
    shifted = z + 25.0  # e.g. non-negative pooled features sharing a large mean
    raw = viewpoint_consistency(shifted, key, view, group)
    cen = viewpoint_consistency(shifted, key, view, group, center=shifted.mean(0))
    ref = viewpoint_consistency(z, key, view, group, center=z.mean(0))
    assert raw["other"] > 0.95 and raw["vci"] < 0.05  # every cosine is high: the raw VCI is compressed
    assert abs(cen["vci"] - ref["vci"]) < 1e-5 and cen["vci"] > 0.2
    rep = vci_report(shifted, torch.nn.functional.normalize(z, dim=-1), key, view, group, shifted.mean(0))
    assert set(rep) >= {"vci_centred", "vci_raw", "vci_proj"}
