import numpy as np
import pytest

from cvdc.config import load_config, resolve
from cvdc.models.lifter import H36M_LEFT, H36M_PARENTS, H36M_RIGHT, coco17_to_h36m, h36m_to_coco17, windows


def test_coco_to_h36m_mapping():
    X = np.random.default_rng(0).normal(size=(5, 17, 2))
    Y = coco17_to_h36m(X)
    np.testing.assert_allclose(Y[:, 0], (X[:, 11] + X[:, 12]) / 2)  # pelvis = mid-hip
    np.testing.assert_allclose(Y[:, 8], (X[:, 5] + X[:, 6]) / 2)  # thorax = mid-shoulder
    np.testing.assert_allclose(Y[:, 7], (Y[:, 0] + Y[:, 8]) / 2)  # spine
    np.testing.assert_allclose(Y[:, 10], (X[:, 1] + X[:, 2]) / 2)  # head = mid-eyes
    for h, c in {1: 12, 4: 11, 11: 5, 14: 6, 13: 9, 16: 10, 3: 16, 6: 15}.items():
        np.testing.assert_allclose(Y[:, h], X[:, c])
    back = h36m_to_coco17(Y)
    body = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    np.testing.assert_allclose(back[:, body], X[:, body])  # body joints survive the round trip


def test_h36m_skeleton_symmetry():
    assert len(H36M_PARENTS) == 17 and H36M_PARENTS[0] == -1
    for left, right in zip(H36M_LEFT, H36M_RIGHT):  # mirrored joints have mirrored parents
        pl, pr = H36M_PARENTS[left], H36M_PARENTS[right]
        assert pl == pr or (pl in H36M_LEFT and pr in H36M_RIGHT and H36M_LEFT.index(pl) == H36M_RIGHT.index(pr))


@pytest.mark.parametrize("n", [50, 243, 244, 1000])
def test_windows_cover_the_sequence(n):
    st = windows(n, 243, 81)
    covered = np.zeros(max(n, 243), int)
    for s in st:
        covered[s:s + 243] += 1
    assert covered[:n].min() >= 1
    assert max(st) + 243 >= n and min(st) == 0


def test_lifter_runs_if_the_weights_are_present():
    cfg = load_config()
    lc = cfg["lifter"]
    if not (resolve(lc["weights"]).exists() and resolve(lc["code_dir"]).exists()):
        pytest.skip("KTPFormer code / weights not downloaded")
    torch = pytest.importorskip("torch")
    from cvdc.models.lifter import KTPFormerLifter

    L = KTPFormerLifter(lc, "cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    xy = np.zeros((90, 17, 2))  # standing figure, arms moving: coco17, y down
    xy[:, 5], xy[:, 6] = [-0.3, -1.0], [0.3, -1.0]
    xy[:, 11], xy[:, 12] = [-0.2, 0.0], [0.2, 0.0]
    xy[:, 13], xy[:, 14], xy[:, 15], xy[:, 16] = [-0.2, 0.9], [0.2, 0.9], [-0.2, 1.8], [0.2, 1.8]
    xy[:, 0], xy[:, 1], xy[:, 2] = [0, -1.4], [-0.05, -1.45], [0.05, -1.45]
    ang = np.linspace(0, np.pi, 90)
    xy[:, 8] = xy[:, 6] + 0.5 * np.stack([np.sin(ang), np.cos(ang)], 1)
    xy[:, 10] = xy[:, 8] + 0.5 * np.stack([np.sin(ang), np.cos(ang)], 1)
    xy[:, 7], xy[:, 9] = [-0.3, -0.5], [-0.3, 0.0]
    out = L.lift_coco(xy + 0.002 * rng.normal(size=xy.shape), 30)
    w = out["world"]
    assert w.shape == (90, 17, 3) and np.isfinite(w).all()
    mid_hip = (w[:, 11] + w[:, 12]) / 2
    np.testing.assert_allclose(mid_hip, 0, atol=1e-5)
    torso = np.linalg.norm((w[:, 5] + w[:, 6]) / 2 - mid_hip, axis=-1)
    assert abs(np.median(torso) - 1) < 1e-5
    assert w[:, 16, 1].mean() > 1.0  # ankles below the hips (y down)
