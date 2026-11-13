"""Synthetic checks for label-free neural input shifts and test-time averages."""

from __future__ import annotations

import tracemalloc

import numpy as np

from ringtrace import nn_shift


def _echo(A: np.ndarray) -> np.ndarray:
    """Row-wise stand-in for predict: finite values, with a second summary column."""
    filled = np.where(np.isfinite(A), A, 0.0)
    return np.column_stack((filled[:, 0], filled.sum(axis=1)))


def _freeze(X: np.ndarray) -> np.ndarray:
    return np.array(X, copy=True)


def _assert_same(X: np.ndarray, snap: np.ndarray) -> None:
    np.testing.assert_array_equal(X, snap)
    assert not np.shares_memory(X, snap)


def test_blank_tta_exact_outside_row_mask():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    X[rng.random(X.shape) < 0.15] = np.nan
    row_mask = rng.random(len(X)) < 0.5
    row_mask[0] = True
    row_mask[1] = False
    col_mask = np.array([False, True, False, True])
    snap = _freeze(X)
    first: dict[str, np.ndarray] = {}

    def predict_fn(A: np.ndarray) -> np.ndarray:
        # A fresh offset on every call, so a second evaluation cannot match the first.
        shift = np.random.default_rng().random()
        raw = np.where(np.isfinite(A), A, -3.0).sum(axis=1)
        pred = np.column_stack((raw + shift, raw - shift))
        if first.get("p") is None and A.shape == X.shape:
            first["p"] = pred.copy()
        return pred

    out = nn_shift.blank_tta(predict_fn, X, row_mask, col_mask, chunk=6)
    _assert_same(X, snap)
    np.testing.assert_array_equal(out[~row_mask], first["p"][~row_mask])
    # Masked rows were actually blanked: they are not the original call.
    assert not np.array_equal(out[row_mask], first["p"][row_mask])
    assert not np.shares_memory(out, X)


def test_blank_tta_chunk_matches_unchunked():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(37, 5))
    X[0, 1] = np.nan
    X[4, :] = np.nan
    row_mask = rng.random(len(X)) < 0.7
    col_mask = np.array([True, False, True, False, False])
    snap = _freeze(X)

    def predict_fn(A: np.ndarray) -> np.ndarray:
        return _echo(A)

    whole = nn_shift.blank_tta(predict_fn, X, row_mask, col_mask, chunk=None)
    by_one = nn_shift.blank_tta(predict_fn, X, row_mask, col_mask, chunk=1)
    by_seven = nn_shift.blank_tta(predict_fn, X, row_mask, col_mask, chunk=7)
    np.testing.assert_array_equal(whole, by_one)
    np.testing.assert_array_equal(whole, by_seven)
    _assert_same(X, snap)

    base = predict_fn(X)
    blanked = np.array(X, copy=True)
    blanked[np.ix_(np.flatnonzero(row_mask), np.flatnonzero(col_mask))] = np.nan
    expect = np.array(base, copy=True)
    expect[row_mask] = 0.5 * (base[row_mask] + predict_fn(blanked)[row_mask])
    np.testing.assert_array_equal(whole, expect)
    np.testing.assert_array_equal(whole[~row_mask], base[~row_mask])


def test_rank_by_group_monotone_unit_interval_and_nan():
    # One group with a tie, one group with a single finite value, inf left alone.
    X = np.array(
        [
            [10.0, 5.0, 1.0],
            [30.0, 5.0, np.inf],
            [30.0, np.nan, 1.0],
            [40.0, 9.0, 2.0],
            [4.0, 0.0, np.nan],
            [5.0, 0.0, -np.inf],
            [5.0, 1.0, 3.0],
            [np.nan, 8.0, 0.0],
        ],
        dtype=np.float64,
    )
    group = np.array([0, 0, 0, 0, 7, 7, 7, 7])
    col_mask = np.array([True, False, True])
    snap = _freeze(X)
    group_snap = group.copy()
    out = nn_shift.rank_by_group(X, col_mask, group)
    _assert_same(X, snap)
    np.testing.assert_array_equal(group, group_snap)
    assert not np.shares_memory(out, X)

    # Group 0, column 0: 10, 30, 30, 40 -> 0, 0.5, 0.5, 1.
    np.testing.assert_array_equal(out[:4, 0], np.array([0.0, 0.5, 0.5, 1.0]))
    # Group 7, column 0: 4, 5, 5, and a NaN that takes no place.
    # 0-based average positions 0, 1.5, 1.5 over m - 1 = 2.
    np.testing.assert_allclose(out[4:7, 0], np.array([0.0, 0.75, 0.75]))
    assert np.isnan(out[7, 0])
    # Unmasked column is bitwise unchanged, NaN included.
    np.testing.assert_array_equal(out[:, 1], X[:, 1])
    # Non-finite values are not ranked and do not move.
    assert out[1, 2] == np.inf and out[5, 2] == -np.inf and np.isnan(out[4, 2])
    # Group 0, column 2: finite values 1, 1, 2 (inf is not ranked). Tied minima sit at 0.25.
    np.testing.assert_allclose(out[[0, 2, 3], 2], np.array([0.25, 0.25, 1.0]))

    finite = np.isfinite(out[:, col_mask])
    assert finite.any()
    assert out[:, col_mask][finite].min() >= 0.0
    assert out[:, col_mask][finite].max() <= 1.0

    for g in np.unique(group):
        member = group == g
        for j in (0, 2):
            ok = member & np.isfinite(X[:, j])
            vals = X[ok, j]
            ranks = out[ok, j]
            order = np.argsort(vals, kind="mergesort")
            sv = vals[order]
            sr = ranks[order]
            assert np.all(np.diff(sr) >= -1e-15)
            same = np.diff(sv) == 0
            assert np.all(sr[1:][same] == sr[:-1][same])
            rose = np.diff(sv) > 0
            assert np.all(sr[1:][rose] > sr[:-1][rose])

    # Independent reference on random tied data, several groups, NaNs ignored.
    rng = np.random.default_rng(2)
    n, d = 180, 3
    raw = rng.integers(0, 6, size=(n, d)).astype(np.float64)
    raw[rng.random(raw.shape) < 0.1] = np.nan
    raw[0, 0] = np.inf
    groups = rng.integers(0, 5, size=n)
    got = nn_shift.rank_by_group(raw, np.array([True, True, False]), groups)
    np.testing.assert_array_equal(got[:, 2], raw[:, 2])
    for j in (0, 1):
        for g in np.unique(groups):
            idx = np.flatnonzero((groups == g) & np.isfinite(raw[:, j]))
            ref = _reference_ranks(raw[idx, j])
            np.testing.assert_allclose(got[idx, j], ref, atol=0, rtol=0)


def _reference_ranks(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.empty(0)
    if n == 1:
        return np.array([0.5])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_v = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        avg_pos = 0.5 * (i + (j - 1))
        ranks[order[i:j]] = avg_pos / (n - 1)
        i = j
    return ranks


def test_noise_tta_zero_sigma_and_seed():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(48, 4))
    X[1, 0] = np.nan
    X[2, 2] = np.inf
    X[3, 2] = -np.inf
    col_mask = np.array([True, False, True, False])
    snap = _freeze(X)

    def predict_fn(A: np.ndarray) -> np.ndarray:
        return np.array(A, dtype=np.float64, copy=True)

    base = predict_fn(X)
    zero = nn_shift.noise_tta(predict_fn, X, n_copies=5, sigma=0.0, seed=11, col_mask=col_mask)
    np.testing.assert_array_equal(zero, base)
    none = nn_shift.noise_tta(predict_fn, X, n_copies=0, sigma=0.3, seed=11, col_mask=col_mask)
    np.testing.assert_array_equal(none, base)

    kwargs = dict(n_copies=3, sigma=0.15, col_mask=col_mask)
    a = nn_shift.noise_tta(predict_fn, X, seed=9, chunk=10, **kwargs)
    b = nn_shift.noise_tta(predict_fn, X, seed=9, chunk=None, **kwargs)
    c = nn_shift.noise_tta(predict_fn, X, seed=9, chunk=4, **kwargs)
    other = nn_shift.noise_tta(predict_fn, X, seed=10, chunk=10, **kwargs)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    assert not np.array_equal(a, other, equal_nan=True)

    # One copy: untouched columns average back exactly, non-finite entries stay put.
    one = nn_shift.noise_tta(predict_fn, X, n_copies=1, sigma=0.25, seed=4, col_mask=col_mask)
    np.testing.assert_array_equal(one[:, 1], X[:, 1])
    np.testing.assert_array_equal(one[:, 3], X[:, 3])
    assert np.isnan(one[1, 0])
    assert one[2, 2] == np.inf and one[3, 2] == -np.inf
    finite0 = np.isfinite(X[:, 0])
    assert not np.array_equal(one[finite0, 0], X[finite0, 0])
    _assert_same(X, snap)

    # The column set, not the order it is written, selects the streams.
    as_bool = nn_shift.noise_tta(predict_fn, X, n_copies=2, sigma=0.2, seed=5, col_mask=col_mask)
    as_idx = nn_shift.noise_tta(
        predict_fn, X, n_copies=2, sigma=0.2, seed=5, col_mask=np.array([2, 0])
    )
    np.testing.assert_array_equal(as_bool, as_idx)


def test_scale_rows_multiplies_masked_cells_only():
    big = np.arange(60, dtype=np.float64).reshape(12, 5)
    big[2, 1] = np.nan
    X = big[::2]  # non-contiguous view; helpers must not write through it
    snap = _freeze(X)
    big_snap = _freeze(big)
    row_mask = np.array([True, False, True, False, True, False])
    col_mask = np.array([True, True, False, False, False])
    out = nn_shift.scale_rows(X, row_mask, col_mask, 2.0)
    _assert_same(X, snap)
    _assert_same(big, big_snap)
    assert not np.shares_memory(out, X)
    np.testing.assert_array_equal(out[~row_mask], X[~row_mask])
    np.testing.assert_array_equal(out[:, ~col_mask], X[:, ~col_mask])
    np.testing.assert_array_equal(out[row_mask][:, col_mask], X[row_mask][:, col_mask] * 2.0)
    assert np.isnan(out[1, 1])  # row 1 of the view is unmasked; NaN stays

    scaled = nn_shift.scale_rows(X, row_mask, col_mask, 1.166)
    cells = np.ix_(np.flatnonzero(row_mask), np.flatnonzero(col_mask))
    np.testing.assert_allclose(scaled[cells], X[cells] * 1.166)
    # Integer indices name the same cells as the boolean mask, once each.
    out_idx = nn_shift.scale_rows(X, np.array([4, 0, 2, 0]), np.array([1, 0, 0]), 2.0)
    np.testing.assert_array_equal(out_idx, out)


def test_inputs_unchanged_after_every_helper():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(24, 3))
    X[5, 1] = np.nan
    row_mask = rng.random(len(X)) < 0.5
    col_mask = np.array([False, True, True])
    group = rng.integers(0, 3, size=len(X))
    snaps = (_freeze(X), row_mask.copy(), col_mask.copy(), group.copy())

    def predict_fn(A: np.ndarray) -> np.ndarray:
        return _echo(A)

    nn_shift.blank_tta(predict_fn, X, row_mask, col_mask, chunk=5)
    nn_shift.noise_tta(predict_fn, X, 2, 0.1, 1, col_mask, chunk=8)
    nn_shift.rank_by_group(X, col_mask, group)
    nn_shift.scale_rows(X, row_mask, col_mask, 1.166)
    _assert_same(X, snaps[0])
    np.testing.assert_array_equal(row_mask, snaps[1])
    np.testing.assert_array_equal(col_mask, snaps[2])
    np.testing.assert_array_equal(group, snaps[3])


def test_blank_tta_respects_chunk_and_peak_memory():
    n, d = 200_000, 50
    chunk = 20_000
    X = np.ones((n, d), dtype=np.float64)
    row_mask = np.ones(n, dtype=bool)
    col_mask = np.zeros(d, dtype=bool)
    col_mask[0] = True
    seen: list[int] = []

    def predict_fn(A: np.ndarray) -> np.ndarray:
        seen.append(int(A.shape[0]))
        return np.zeros((A.shape[0], 2), dtype=np.float64)

    tracemalloc.start()
    try:
        out = nn_shift.blank_tta(predict_fn, X, row_mask, col_mask, chunk=chunk)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert out.shape == (n, 2)
    assert seen == [n] + [chunk] * (n // chunk)
    # The blanked block is one chunk. Peak must cover that copy and stay under
    # a second full matrix (X is 80 MB; a chunk of 20_000 rows is 8 MB).
    block_bytes = chunk * d * X.dtype.itemsize
    assert block_bytes <= peak < X.nbytes, f"peak traced memory {peak}, chunk {block_bytes}"
    np.testing.assert_array_equal(X, np.ones_like(X))
