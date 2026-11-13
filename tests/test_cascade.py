"""The cascade and decoder end to end on a tiny synthetic problem (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.cascade import BoosterSpec, CascadeSpec, fit_predict
from ringtrace.classes import N_CLASSES
from ringtrace.decode import decode_by_period
from ringtrace.scoring import macro_f1

pytest.importorskip("xgboost")


def _problem(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    fraud = rng.random(n) < 0.08
    kind = rng.integers(1, N_CLASSES, n)
    y[fraud] = kind[fraud]
    X[fraud, 0] += 4.0  # fraud is visible
    X[fraud, 1] = kind[fraud]  # and its type is readable
    period = rng.integers(7, 10, n)
    return X, y, period


def test_cascade_learns_and_is_deterministic():
    X, y, period = _problem(6000, 0)
    Xt, yt, _ = _problem(2000, 1)
    spec = CascadeSpec(booster=BoosterSpec(rounds=30, depth=4, device="cpu"), seeds=(0,),
                       screen_rounds=20)
    (p1,) = fit_predict(X, y, period, [Xt], spec, log=lambda *_: None)
    (p2,) = fit_predict(X, y, period, [Xt], spec, log=lambda *_: None)
    assert p1.shape == (2000, N_CLASSES)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)
    pred = decode_by_period(p1, np.full(2000, 10))
    assert macro_f1(yt, pred) > 0.8
