"""TabM on a tiny synthetic problem (CPU): preprocessing, learning, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.classes import N_CLASSES

pytest.importorskip("torch")

from ringtrace import tabm  # noqa: E402
from ringtrace.scoring import macro_f1  # noqa: E402


def _problem(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, n)
    X[:, 0] = y + rng.normal(scale=0.2, size=n)  # the class is readable from column 0
    X[rng.random(n) < 0.1, 2] = np.nan
    return X, y


def _spec(**kw):
    base = dict(
        seeds=(0,),
        k=4,
        width=32,
        blocks=2,
        d_embed=4,
        n_freq=4,
        epochs=15,
        batch=256,
        lr=3e-3,
        device="cpu",
        quantiles=50,
    )
    base.update(kw)
    return tabm.TabMSpec(**base)


def test_prep_is_fitted_on_train_only_and_flags_gaps():
    X, _ = _problem(500, 0)
    prep = tabm.fit_prep(X, quantiles=50)
    assert list(prep.gap_cols) == [2]
    Z = tabm.transform(prep, X)
    assert Z.shape == (500, 6)
    assert np.isfinite(Z).all()
    assert set(np.unique(Z[:, -1])) == {0.0, 1.0}
    # monotone in each column, roughly standard normal
    order = np.argsort(X[:, 1])
    assert np.all(np.diff(Z[order, 1]) >= -1e-6)
    assert abs(Z[:, 1].mean()) < 0.2
    # out-of-range test values are clipped to the training range
    far = tabm.transform(prep, np.full((1, 5), 1e6, dtype=np.float32))
    assert far[0, 1] <= Z[:, 1].max() + 1e-6


def test_tabm_learns_and_is_deterministic():
    X, y = _problem(3000, 1)
    Xh, yh = _problem(1000, 2)
    period = np.full(len(y), 7)
    (p1,) = tabm.fit_predict(X, y, period, [Xh], _spec(), log=lambda *_: None)
    (p2,) = tabm.fit_predict(X, y, period, [Xh], _spec(), log=lambda *_: None)
    assert p1.shape == (1000, N_CLASSES)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)
    assert macro_f1(yh, p1.argmax(1)) > 0.8


def test_run_dispatches_on_model():
    from ringtrace import run

    spec, fn = run.model({"model": "tabm", "tabm": {"seeds": [3, 4], "epochs": 2}})
    assert fn is tabm.fit_predict and spec.seeds == (3, 4) and spec.epochs == 2
    spec, fn = run.model({})
    assert fn.__module__ == "ringtrace.cascade"
