"""Entity pools and hashed id embeddings on tiny CPU problems."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.classes import N_CLASSES

pytest.importorskip("torch")

from ringtrace import nn_entity, tabm  # noqa: E402
from ringtrace.scoring import macro_f1  # noqa: E402


def _brute(X, entity, ts, n, cols):
    if cols is None:
        cols = np.arange(X.shape[1])
    cols = np.asarray(cols)
    out = np.empty((len(X), len(cols) + 1), dtype=np.float64)
    for i in range(len(X)):
        prev = np.flatnonzero((entity == entity[i]) & (ts < ts[i]))
        if len(prev) == 0:
            out[i, :-1] = np.nan
            out[i, -1] = 0.0
            continue
        take = prev[np.lexsort((prev, ts[prev]))][-n:]
        out[i, :-1] = np.asarray(X[take][:, cols], dtype=np.float64).mean(axis=0)
        out[i, -1] = float(len(take))
    return out


def _quiet(*_):
    return None


def _spec(**kw):
    base = dict(
        seeds=(0,),
        k=4,
        width=32,
        blocks=2,
        dropout=0.0,
        d_embed=4,
        n_freq=4,
        epochs=2,
        batch=64,
        lr=3e-3,
        weight_decay=0.0,
        device="cpu",
        quantiles=16,
        buckets=128,
        d_ent=4,
        ent_dropout=0.0,
    )
    base.update(kw)
    return nn_entity.EntitySpec(**base)


def test_past_pool_matches_brute_force_with_ties_and_no_leakage():
    # Interleaved entities, a same-timestamp pair, and a one-row entity.
    X = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [7.0]])
    entity = np.array([0, 0, 0, 1, 0, 1])
    ts = np.array([5, 5, 1, 9, 4, 2])
    got = nn_entity.past_pool(X, entity, ts, n=8)
    assert np.isnan(got[2, 0]) and got[2, 1] == 0  # earliest of entity 0
    assert got[4, 0] == 30 and got[4, 1] == 1  # only ts=1
    assert got[0, 0] == 40 and got[0, 1] == 2  # mean of 30 and 50; the tie is excluded
    assert got[1, 0] == 40 and got[1, 1] == 2
    assert np.isnan(got[5, 0]) and got[5, 1] == 0  # entity 1 has no earlier row
    assert got[3, 0] == 7 and got[3, 1] == 1  # entity 1 does not see entity 0
    got1 = nn_entity.past_pool(X, entity, ts, n=1)
    assert got1[0, 0] == 50 and got1[0, 1] == 1  # latest strictly earlier row
    assert got1[1, 0] == 50 and got1[1, 1] == 1

    rng = np.random.default_rng(0)
    n, d = 60, 4
    Xr = rng.normal(size=(n, d))
    Xr[7, 1] = np.nan
    Xr[8, 3] = np.nan
    entity_r = rng.integers(0, 8, n)
    ts_r = rng.integers(0, 12, n)
    entity_r[5] = entity_r[4]
    ts_r[5] = ts_r[4]
    for window in (1, 3, 8):
        for cols in (None, np.array([0, 2])):
            got = nn_entity.past_pool(Xr, entity_r, ts_r, n=window, cols=cols)
            exp = _brute(Xr, entity_r, ts_r, window, cols)
            np.testing.assert_allclose(got, exp, equal_nan=True)
            assert got.shape == exp.shape
            assert np.all((got[:, -1] >= 0) & (got[:, -1] <= window))
            empty = got[:, -1] == 0
            assert np.isnan(got[empty, :-1]).all()
            assert np.isfinite(got[:, -1]).all()

    # Bumping a row may change only strictly later rows of the same entity.
    base = nn_entity.past_pool(Xr, entity_r, ts_r, n=4)
    i = int(np.flatnonzero(base[:, -1] > 0)[0])
    bumped_x = Xr.copy()
    bumped_x[i, 0] += 1e6
    bumped = nn_entity.past_pool(bumped_x, entity_r, ts_r, n=4)
    untouched = ~((entity_r == entity_r[i]) & (ts_r > ts_r[i]))
    np.testing.assert_allclose(bumped[untouched], base[untouched], equal_nan=True)
    assert np.any(untouched) and untouched[i]


def test_fit_predict_entity_shapes_determinism_and_unseen_ids():
    rng = np.random.default_rng(1)
    n, d, e = 40, 4, 2
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[0, 1] = np.nan
    y = rng.integers(0, N_CLASSES, n)
    ids = rng.integers(0, 15, size=(n, e), dtype=np.int64)
    period = np.zeros(n, dtype=np.int64)
    X1 = rng.normal(size=(6, d)).astype(np.float32)
    ids1 = rng.integers(0, 15, size=(6, e), dtype=np.int64)
    X2 = rng.normal(size=(4, d)).astype(np.float32)
    ids2 = np.array(
        [
            [10**15, -(10**12)],
            [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
            [0, 1],
            [-7, 10**15],
        ],
        dtype=np.int64,
    )
    spec = _spec(seeds=(0, 1), epochs=1, batch=16, ent_dropout=0.4, buckets=64)
    kw = dict(spec=spec, log=_quiet)
    a = nn_entity.fit_predict_entity(X, ids, y, period, [X1, X2], [ids1, ids2], **kw)
    b = nn_entity.fit_predict_entity(X, ids, y, period, [X1, X2], [ids1, ids2], **kw)
    assert len(a) == 2
    assert a[0].shape == (6, N_CLASSES)
    assert a[1].shape == (4, N_CLASSES)
    for block in a:
        assert np.isfinite(block).all()
        np.testing.assert_allclose(block.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_recurring_entity_beats_numeric_only():
    rng = np.random.default_rng(0)
    reps_tr, reps_te, d = 36, 12, 3

    def draw(reps):
        ent = np.tile(np.arange(N_CLASSES, dtype=np.int64), reps)
        X = rng.normal(size=(len(ent), d)).astype(np.float32)
        perm = rng.permutation(len(ent))
        return X[perm], ent[perm, None], ent[perm]

    X, ids, y = draw(reps_tr)
    Xh, idsh, yh = draw(reps_te)
    period = np.zeros(len(y), dtype=np.int64)
    spec = _spec(
        epochs=40,
        batch=256,
        width=64,
        lr=1e-2,
        d_embed=8,
        n_freq=8,
        d_ent=8,
        buckets=4096,
        quantiles=32,
        ent_dropout=0.0,
        dropout=0.0,
    )
    hashed = nn_entity._hash_columns(ids, spec.buckets)
    assert len(np.unique(hashed)) == N_CLASSES
    ent = nn_entity.fit_predict_entity(X, ids, y, period, [Xh], [idsh], spec, log=_quiet)[0]
    num = tabm.fit_predict(X, y, period, [Xh], spec, log=_quiet)[0]
    ent_f1 = macro_f1(yh, ent.argmax(1))
    num_f1 = macro_f1(yh, num.argmax(1))
    assert np.isfinite(ent).all()
    np.testing.assert_allclose(ent.sum(1), 1.0, atol=1e-5)
    assert ent_f1 > num_f1 + 0.25, f"entity {ent_f1:.3f} numeric {num_f1:.3f}"
    assert ent_f1 > 0.45, f"entity {ent_f1:.3f} numeric {num_f1:.3f}"
