"""Weight-space methods on synthetic data (CPU, a tiny network)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ringtrace import tabm, wspace  # noqa: E402
from ringtrace.classes import N_CLASSES  # noqa: E402


def tiny_spec(**kw) -> tabm.TabMSpec:
    base = dict(
        seeds=(0,), k=4, width=16, blocks=2, d_embed=4, n_freq=6, epochs=2, batch=64, device="cpu"
    )
    return tabm.TabMSpec(**{**base, **kw})


def toy(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 1.0, 1 + (X[:, 2] > 0).astype(np.int64), 0)
    return X, y.astype(np.int64)


def test_lerp_ends_and_extrapolation():
    a = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])}
    b = {"w": torch.tensor([3.0, 2.0]), "b": torch.tensor([1.0])}
    assert torch.equal(wspace.lerp(a, b, 0.0)["w"], a["w"])
    assert torch.equal(wspace.lerp(a, b, 1.0)["w"], b["w"])
    assert torch.allclose(wspace.lerp(a, b, 2.0)["w"], torch.tensor([5.0, 2.0]))
    assert wspace.distance(a, b) == pytest.approx(np.sqrt(5.0))
    assert torch.allclose(wspace.mean_state([a, b])["w"], torch.tensor([2.0, 2.0]))


def test_add_vectors_matches_single_lerp():
    a = {"w": torch.tensor([1.0, 2.0])}
    b = {"w": torch.tensor([3.0, 5.0])}
    c = {"w": torch.tensor([0.0, 2.0])}
    assert torch.allclose(wspace.add_vectors(a, [b], [1.5])["w"], wspace.lerp(a, b, 1.5)["w"])
    both = wspace.add_vectors(a, [b, c], [1.0, 1.0])["w"]
    assert torch.allclose(both, torch.tensor([2.0, 5.0]))
    assert wspace.cosine(a, b, b) == pytest.approx(1.0)


def test_barrier_is_zero_on_a_line_and_positive_on_a_bump():
    assert wspace.barrier([1.0, 1.5, 2.0, 2.5, 3.0]) == pytest.approx(0.0)
    assert wspace.barrier([1.0, 1.0, 1.4, 1.0, 1.0]) == pytest.approx(0.4)


def test_same_seed_gives_the_same_start_and_fine_tune_moves_the_weights():
    X, y = toy()
    spec = tiny_spec()
    Z = tabm.transform(tabm.fit_prep(X, 50), X)
    y_clean = y.copy()
    y_clean[:20] = 1
    recent = np.arange(len(y)) >= 300
    ends = wspace.endpoints(Z, y, y_clean, recent, spec, seed=3, log=lambda *_: None)
    again = wspace.state(tabm.train_one(Z, y, spec, 3, log=lambda *_: None))
    assert wspace.distance(ends["g"], again) == 0.0  # deterministic, so g and c share a start
    assert 0 < wspace.distance(ends["g"], ends["c"]) < wspace.norm(ends["g"])
    assert wspace.distance(ends["g"], ends["r"]) > 0
    arms = wspace.arm_weights(ends)
    assert wspace.distance(arms["ta1"], ends["c"]) < 1e-5
    assert wspace.distance(arms["ti1"], ends["r"]) < 1e-5
    p = wspace.proba(arms["ta1.5"], Z, spec)
    assert p.shape == (len(Z), N_CLASSES)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    losses = wspace.path_losses(ends["g"], ends["c"], Z, y, spec)
    assert len(losses) == len(wspace.PATH) and np.isfinite(losses).all()


def test_fine_tune_rejects_class_weights():
    X, y = toy(200)
    spec = tiny_spec(class_weight_power=0.5)
    Z = tabm.transform(tabm.fit_prep(X, 50), X)
    sd = wspace.state(tabm.build_model(Z.shape[1], spec))
    with pytest.raises(ValueError):
        wspace.fine_tune(sd, Z, y, spec, 0, log=lambda *_: None)
