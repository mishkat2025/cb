"""Step (a2) score setup: video head pooling (T13), video batches and per-video weights (T6), item masks and the
item-activity rule (T14), the EMA / min-epoch early stopping (T8b), set-level baselines (E20)."""
import numpy as np
import pandas as pd
import pytest
import torch

from cvdc.config import load_config
from cvdc.eval.baselines import angle_range, item_activity, video_baselines
from cvdc.losses.supervised import SupervisedLoss, TargetScaler
from cvdc.models.heads import VideoHead, segment_softmax, segment_sum
from cvdc.models.model import CVDCModel
from cvdc.train.train_supervised import make_batches
from cvdc.train.trainer import EarlyStopping

CFG = load_config()


def test_segment_softmax_sums_to_one_per_video():
    s = torch.tensor([0.5, 2.0, -1.0, 3.0, 100.0, 101.0])
    seg = torch.tensor([0, 0, 0, 1, 2, 2])
    a = segment_softmax(s, seg, 3)
    torch.testing.assert_close(segment_sum(a, seg, 3), torch.ones(3))
    torch.testing.assert_close(a[:3], torch.softmax(s[:3], 0))
    assert a[3] == 1.0 and torch.isfinite(a).all()  # a one-rep video; large scores stay finite


def test_video_head_pools_per_video_and_ignores_rep_order():
    torch.manual_seed(0)
    h = VideoHead(8, 2, ["score", "PF4", "CF1"], ("mean", "attention"), 4, True, 3).eval()
    h.set_count_stats(torch.tensor([1.0, 10.0]))
    z = torch.randn(5, 8)
    seg = torch.tensor([0, 0, 0, 1, 1])
    o = h(z, seg, 2, torch.tensor([10.0, 1.0]), torch.tensor([0, 1]))
    assert o["score"].shape == (2,) and o["items"].shape == (2, 2)
    perm = torch.tensor([4, 2, 0, 3, 1])  # shuffle reps across the batch
    o2 = h(z[perm], seg[perm], 2, torch.tensor([10.0, 1.0]), torch.tensor([0, 1]))
    torch.testing.assert_close(o["score"], o2["score"])
    # a video's prediction depends only on its own reps
    o3 = h(z[:3], seg[:3], 1, torch.tensor([10.0]), torch.tensor([0]))
    torch.testing.assert_close(o["score"][:1], o3["score"])


def test_model_builds_the_video_head_only_at_rep_level():
    assert CVDCModel(CFG, "rep").video_head is not None
    assert CVDCModel(CFG, "video").video_head is None


class _Videos:
    """The PackData fields make_batches uses: 4 videos with 3, 1, 5 and 2 reps."""

    def __init__(self):
        sizes = [3, 1, 5, 2]
        self.video_of = torch.tensor(np.repeat(np.arange(4), sizes))
        self.v_nreps = torch.tensor(sizes)
        self.v_reps = [torch.nonzero(self.video_of == v).squeeze(1) for v in range(4)]

    def video_items(self, vids):
        parts = [self.v_reps[v] for v in vids.tolist()]
        return torch.cat(parts), torch.cat([torch.full((len(r),), j) for j, r in enumerate(parts)])


def test_video_batches_hold_whole_videos_once_per_epoch():
    d = _Videos()
    items = torch.arange(11)
    batches = make_batches(d, items, 4, torch.Generator().manual_seed(0), "video")
    seen = torch.cat([it for it, _, _ in batches])
    assert sorted(seen.tolist()) == list(range(11))  # every rep once
    for it, vids, seg in batches:
        assert sum(d.v_nreps[v].item() for v in vids) == len(it)
        assert (d.video_of[it] == vids[seg]).all()  # segment index points at the rep's own video
    assert all(len(it) >= 4 for it, _, _ in batches[:-1])


def test_per_video_weights_count_every_video_once():
    # video A: 4 reps with squared error 1; video B: 1 rep with squared error 9 (standardised units)
    loss = SupervisedLoss([], {"score": 1.0}, TargetScaler().fit({"score": torch.tensor([0.0, 2.0]), "items": None}))
    out = {"score": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])}
    y = {"score": torch.tensor([2.0, 2.0, 2.0, 2.0, 4.0]), "items": None, "quality": torch.full((5,), -1),
         "exercise": torch.full((5,), -1)}  # z = (y - 1) / 1 -> 1, 1, 1, 1, 3
    nreps = torch.tensor([4.0, 4, 4, 4, 1])
    _, p = loss(out, y, 1.0 / nreps)
    assert p["score"] == pytest.approx((1.0 + 9.0) / 2)  # mean over videos, not (4 * 1 + 9) / 5
    _, p = loss(out, y)
    assert p["score"] == pytest.approx((4 + 9) / 5)


def test_item_mask_removes_items_from_the_loss():
    sc = TargetScaler().fit({"score": torch.tensor([0.0, 1.0]), "items": torch.tensor([[0.0, 0.0], [1.0, 1.0]])})
    loss = SupervisedLoss(["CF1", "CF3"], {"score": 0.0, "items": 1.0}, sc)
    out = {"score": torch.zeros(2), "items": torch.zeros(2, 2)}
    y = {"score": torch.zeros(2), "items": torch.tensor([[1.0, 1.0], [1.0, 1.0]]), "quality": torch.full((2,), -1),
         "exercise": torch.full((2,), -1)}
    _, full = loss(out, y)
    _, masked = loss(out, {**y, "item_mask": torch.tensor([[True, False], [False, False]])})
    assert full["items"] == pytest.approx(1.0)  # z = (1 - 0.5) / 0.5 = 1
    assert masked["items"] == pytest.approx(1.0)  # only the one unmasked entry counts; row 2 has no item left


def test_item_activity_needs_enough_subjects_off_the_mode():
    rows = []
    for s in range(10):
        for _ in range(3):
            rows.append(dict(exercise="E01", group="expert", subject=f"P{s:02d}",
                             A=0.5 if s < 6 else 1.0,  # mode 0.5; off the mode in 4 subjects
                             B=0.0 if s < 2 else 1.0))  # off the mode in 2 subjects
    a = item_activity(pd.DataFrame(rows), ["A", "B"], min_subjects=4, min_sd=0.05).set_index("item")
    assert a.loc["A", "off_mode_subjects"] == 4 and a.loc["A", "varies"]
    assert a.loc["B", "off_mode_subjects"] == 2 and not a.loc["B", "varies"]


def test_early_stopping_min_epochs_and_ema():
    model = torch.nn.Linear(1, 1)
    es = EarlyStopping(patience=2, min_epochs=3, ema=0.5)
    vals = [0.1, 5.0, 5.0, 4.0, 4.0, 4.0]  # epoch 1 is lowest but too early to keep
    stopped = None
    for ep, v in enumerate(vals, 1):
        with torch.no_grad():
            model.weight.fill_(float(ep))
        if es.step(ep, v, model):
            stopped = ep
            break
    # smoothed: 0.1, 2.55, 3.775, 3.8875, 3.94 -> best eligible at epoch 3, two worse epochs -> stop at 5
    assert es.best_epoch == 3 and stopped == 5
    assert es.best_value == pytest.approx(3.775)
    es.restore(model)
    assert model.weight.item() == 3.0


def test_early_stopping_defaults_reproduce_the_step_a_rule():
    model = torch.nn.Linear(1, 1)
    a, b = EarlyStopping(patience=3), EarlyStopping(patience=3, min_epochs=1, ema=0.0)
    for ep, v in enumerate([5.0, 3.0, 4.0, 2.0, 2.5, 3.0, 2.2], 1):
        assert a.step(ep, v, model) == b.step(ep, v, model)
    assert a.best_epoch == b.best_epoch == 4


def test_circular_range_of_a_wrapped_angle():
    assert angle_range(np.array([170.0, -175.0, -160.0]), "horizontal_adduction") == pytest.approx(30.0)
    assert angle_range(np.array([18.0, 135.0, 60.0]), "horizontal_adduction") == pytest.approx(117.0)
    assert angle_range(np.array([10.0, 150.0]), "shoulder_elevation") == pytest.approx(140.0)


def test_within_group_baselines_are_fitted_on_the_training_fold():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(20):
        g = "expert" if s < 12 else "non-expert"
        for _ in range(4):
            n = rng.integers(5, 15)
            rows.append(dict(video_id=f"v{len(rows)}", subject=f"P{s:02d}", exercise="E01", group=g,
                             n_reps_set=n, rep_duration_s=rng.normal(2, 0.3), rom_deg=rng.normal(140, 10),
                             score=(50 if g == "expert" else 20) + 3 * n))
    v = pd.DataFrame(rows)
    subs = sorted(v.subject.unique())
    plans = [dict(fold=0, train=subs[::2], test=subs[1::2]), dict(fold=1, train=subs[1::2], test=subs[::2])]
    p = video_baselines(v, plans, ["score"]).merge(v, on="video_id")
    rc = p[p.method == "base_repcount"]
    np.testing.assert_allclose(rc.score_pred, rc.score, atol=1e-6)  # exact linear relation within each group
    gm = p[(p.method == "base_group") & (p.fold == 0)]
    tr = v[v.subject.isin(plans[0]["train"])]
    for g, d in gm.groupby("group"):
        assert np.allclose(d.score_pred, tr[tr.group == g].score.mean())
    assert p.groupby("method").video_id.nunique().eq(len(v)).all()
