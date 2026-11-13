"""Refit helpers and the screen-off cascade on synthetic data (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.cascade import BoosterSpec, CascadeSpec, fit_predict
from ringtrace.classes import N_CLASSES
from ringtrace.refit import argmax_changes, hedge, scaled_rounds

pytest.importorskip("xgboost")


def test_scaled_rounds():
    assert scaled_rounds(300, 300, [200, 200, 200]) == 450
    assert scaled_rounds(300, 100, [100, 100]) == 300


def test_hedge_is_row_normalised_mean():
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(N_CLASSES), 50)
    b = rng.dirichlet(np.ones(N_CLASSES), 50)
    h = hedge(a, b)
    np.testing.assert_allclose(h.sum(1), 1.0)
    np.testing.assert_allclose(h, 0.5 * (a + b), atol=1e-12)
    np.testing.assert_array_equal(hedge(a, a), a)


def test_argmax_changes_counts():
    a = np.eye(N_CLASSES)[[0, 1, 2, 3]]
    b = np.eye(N_CLASSES)[[0, 1, 5, 5]]
    r = argmax_changes(a, b)
    assert r["changed"] == 2 and r["rows"] == 4
    assert r["counts_b"][5] == 2 and sum(r["counts_a"]) == 4


def test_screen_off_fits_one_period_deterministically():
    rng = np.random.default_rng(1)
    n = 3000
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    fraud = rng.random(n) < 0.1
    y[fraud] = rng.integers(1, N_CLASSES, n)[fraud]
    X[fraud, 0] += 4.0
    X[fraud, 1] = y[fraud]
    period = np.full(n, 7)  # one month: no cross-fit possible
    spec = CascadeSpec(
        booster=BoosterSpec(rounds=20, depth=3, device="cpu"), seeds=(0,), screen=False
    )
    (p1,) = fit_predict(X, y, period, [X[:200]], spec, log=lambda *_: None)
    (p2,) = fit_predict(X, y, period, [X[:200]], spec, log=lambda *_: None)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    assert (p1.argmax(1) == y[:200]).mean() > 0.9
