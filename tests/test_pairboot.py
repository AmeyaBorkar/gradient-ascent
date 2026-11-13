import numpy as np
import pytest

from ringtrace import pairboot as pb
from ringtrace.classes import N_CLASSES


def _dense(units, truth, a, b, strata, draws, seed):
    """Reference: the full weight matrix per stratum, no sharing."""
    _, g = np.unique(units, return_inverse=True)
    deltas, point = np.zeros(draws), 0.0
    ss = np.unique(strata)
    for s in ss:
        rows = strata == s
        ids = np.unique(g[rows])
        loc = np.searchsorted(ids, g[rows])
        w = pb.poisson_weights(seed, ids, draws).astype(np.float64)
        ca = pb._counts(loc, truth[rows], a[rows], len(ids)).toarray()
        cb = pb._counts(loc, truth[rows], b[rows], len(ids)).toarray()
        deltas += pb.macro(w @ cb) - pb.macro(w @ ca)
        point += pb.macro(cb.sum(0)[None])[0] - pb.macro(ca.sum(0)[None])[0]
    deltas /= len(ss)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point / len(ss), lo, hi


def test_weights_are_poisson_and_pure():
    w = pb.poisson_weights(3, np.arange(5000), 200)
    assert abs(w.mean() - 1) < 0.01 and abs(w.var() - 1) < 0.02
    assert np.array_equal(pb.poisson_weights(3, np.array([7, 9]), 200), w[:, [7, 9]])
    assert not np.array_equal(pb.poisson_weights(4, np.array([7]), 200), w[:, [7]])


def test_matches_dense_reference():
    rng = np.random.default_rng(0)
    n = 3000
    units = rng.integers(0, 900, n)
    strata = rng.integers(7, 10, n)
    truth = rng.integers(0, N_CLASSES, n)
    a = np.where(rng.random(n) < 0.6, truth, rng.integers(0, N_CLASSES, n))
    boot = pb.PairedBoot(units, truth, a, strata, draws=300, seed=5, chunk=128)
    for k in range(3):
        b = a.copy()
        flip = rng.random(n) < 0.05 * (k + 1)
        b[flip] = truth[flip]
        got = boot.compare(b)
        want = _dense(units, truth, a, b, strata, 300, 5)
        assert got["delta"] == pytest.approx(want[0], abs=1e-12)
        assert got["lo"] == pytest.approx(want[1], abs=1e-9)
        assert got["hi"] == pytest.approx(want[2], abs=1e-9)
        assert got["delta"] > 0


def test_identical_is_zero():
    rng = np.random.default_rng(1)
    t = rng.integers(0, N_CLASSES, 100)
    boot = pb.PairedBoot(np.arange(100), t, t, draws=50)
    assert boot.compare(t.copy()) == {"delta": 0.0, "lo": 0.0, "hi": 0.0}


def test_dict_of_targets_matches_single_targets():
    rng = np.random.default_rng(2)
    n = 500
    units = rng.integers(0, 200, n)
    t1, t2 = rng.integers(0, N_CLASSES, n), rng.integers(0, N_CLASSES, n)
    a = rng.integers(0, N_CLASSES, n)
    b = np.where(rng.random(n) < 0.1, t1, a)
    multi = pb.PairedBoot(units, {"x": t1, "y": t2}, a, draws=100).compare(b)
    assert multi["x"] == pb.PairedBoot(units, t1, a, draws=100).compare(b)
    assert multi["y"] == pb.PairedBoot(units, t2, a, draws=100).compare(b)


def test_keep_draws_matches_interval():
    rng = np.random.default_rng(3)
    n = 400
    t = rng.integers(0, N_CLASSES, n)
    a = rng.integers(0, N_CLASSES, n)
    b = np.where(rng.random(n) < 0.2, t, a)
    r = pb.PairedBoot(np.arange(n), t, a, draws=200).compare(b, keep_draws=True)
    assert r["lo"] == pytest.approx(np.percentile(r["draws"], 2.5))
