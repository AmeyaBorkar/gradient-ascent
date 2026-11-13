"""Label-free input shifts and test-time averages for the neural member.

The model sees a raw float matrix (NaN = missing) and quantile-normalises it in
``transform``. Everything here runs on that raw matrix and is generic: row and
column roles are masks, so no column name is known.

``blank_tta`` and ``noise_tta`` average probabilities with equal weight. They do
not look inside ``predict_fn``. A function that already returns a mean over
seeds or ensemble members stays a mean of those means; this is not a logit
average and it does not reweight the inner average.

Both averages are row-wise. ``predict_fn`` is called on the whole matrix once,
then on row chunks of the modified matrix (default 200_000 rows). That matches
a single call on a fully modified matrix when ``predict_fn`` itself is row-wise,
which ``transform`` and ``predict`` are, and it keeps only one chunk of extra
feature memory instead of a second copy of X. A 1.5M x 300 float64 matrix is
about 3.6 GB, so a second full copy is the wrong default.

``sigma == 0``, ``n_copies == 0``, or an empty column mask skips the noisy
copies and returns that one ``predict_fn(X)`` call. Averaging identical
probabilities is not bitwise-identical once more than one copy is added.
"""

from __future__ import annotations

import operator

import numpy as np

_DEFAULT_CHUNK = 200_000


def _matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-d array, got ndim {X.ndim}")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError(f"X must be floating point (NaN marks missing), got {X.dtype}")
    return X


def _indices(mask: np.ndarray, n: int, what: str) -> np.ndarray:
    """Boolean mask or integer indices, as a sorted unique intp array."""
    m = np.asarray(mask)
    if m.dtype == np.bool_:
        if m.shape != (n,):
            raise ValueError(f"{what} boolean mask must have shape ({n},), got {m.shape}")
        return np.flatnonzero(m)
    if m.ndim != 1:
        raise ValueError(f"{what} must be 1-d, got shape {m.shape}")
    return np.unique(np.asarray(m, dtype=np.intp))


def _chunk(chunk: int | None, n: int) -> int:
    if chunk is None:
        return max(n, 1)
    step = operator.index(chunk)
    if step <= 0:
        raise ValueError("chunk must be a positive integer or None")
    return step


def _call(predict_fn, rows: np.ndarray) -> np.ndarray:
    pred = np.asarray(predict_fn(rows))
    if pred.shape[0] != rows.shape[0]:
        raise ValueError(
            f"predict_fn returned {pred.shape[0]} rows for {rows.shape[0]} inputs"
        )
    return pred


def blank_tta(
    predict_fn,
    X: np.ndarray,
    row_mask: np.ndarray,
    col_mask: np.ndarray,
    chunk: int | None = _DEFAULT_CHUNK,
) -> np.ndarray:
    """Mean of ``predict_fn(X)`` and the same rows with masked columns set to NaN.

    Rows outside ``row_mask`` are copied from the single ``predict_fn(X)`` call,
    not averaged. Masked rows are blanked and predicted in chunks of ``chunk``
    (``None`` sends every masked row in one call). ``X`` is not modified.
    """
    X = _matrix(X)
    out = np.array(_call(predict_fn, X), copy=True)
    rows = _indices(row_mask, X.shape[0], "row_mask")
    cols = _indices(col_mask, X.shape[1], "col_mask")
    if rows.size == 0 or cols.size == 0:
        return out
    step = _chunk(chunk, int(rows.shape[0]))
    for start in range(0, int(rows.shape[0]), step):
        idx = rows[start : start + step]
        block = X[idx]
        # Advanced indexing returns a copy; copy again only if that ever changes.
        if np.may_share_memory(block, X):
            block = np.array(block, copy=True)
        block[:, cols] = np.nan
        pred = _call(predict_fn, block)
        if pred.shape[1:] != out.shape[1:]:
            raise ValueError(
                f"predict_fn returned shape {pred.shape}, "
                f"expected {(idx.shape[0],) + out.shape[1:]}"
            )
        out[idx] = 0.5 * (out[idx] + pred)
    return out


def noise_tta(
    predict_fn,
    X: np.ndarray,
    n_copies: int,
    sigma: float,
    seed: int,
    col_mask: np.ndarray | None = None,
    chunk: int | None = _DEFAULT_CHUNK,
) -> np.ndarray:
    """Mean of ``predict_fn(X)`` and ``n_copies`` multiplicatively noised copies.

    Finite entries of the masked columns (every column, when ``col_mask`` is
    None) become ``x * (1 + N(0, sigma))`` with ``sigma`` the standard
    deviation. NaN and inf are left alone, and other columns are not touched.

    Each copy and each masked column has its own stream, spawned from
    ``SeedSequence(seed)`` and drawn in row order, one normal per row. Draws on
    non-finite rows are discarded. Because the streams are per column, the
    noise does not depend on ``chunk``. Only one row-chunk of X is copied at
    a time.
    """
    X = _matrix(X)
    copies = operator.index(n_copies)
    if copies < 0:
        raise ValueError("n_copies must be >= 0")
    base = _call(predict_fn, X)
    cols = (
        np.arange(X.shape[1], dtype=np.intp)
        if col_mask is None
        else _indices(col_mask, X.shape[1], "col_mask")
    )
    if copies == 0 or float(sigma) == 0.0 or cols.size == 0 or X.shape[0] == 0:
        return np.array(base, copy=True)

    acc = np.array(base, dtype=np.float64, copy=True)
    scale = float(sigma)
    step = _chunk(chunk, X.shape[0])
    sequences = np.random.SeedSequence(seed).spawn(copies)
    for copy_ss in sequences:
        rngs = [np.random.default_rng(s) for s in copy_ss.spawn(int(cols.shape[0]))]
        for start in range(0, X.shape[0], step):
            end = min(start + step, X.shape[0])
            block = X[start:end].copy()
            width = end - start
            for j, rng in zip(cols, rngs, strict=True):
                draw = rng.normal(0.0, scale, size=width)
                col = block[:, int(j)]
                finite = np.isfinite(col)
                col[finite] *= 1.0 + draw[finite]
            pred = _call(predict_fn, block)
            if pred.shape[1:] != acc.shape[1:]:
                raise ValueError(
                    f"predict_fn returned shape {pred.shape}, expected "
                    f"{(width,) + acc.shape[1:]}"
                )
            acc[start:end] += pred
    acc /= copies + 1
    if acc.dtype != base.dtype:
        return acc.astype(base.dtype, copy=False)
    return acc


def _unit_average_ranks(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Within-group average rank scaled to [0, 1].

    Ranks are 1-based and ties share the mean of the tied places (the usual
    average rank). The stored value is ``(rank - 1) / (m - 1)`` where ``m`` is
    the number of finite values in the group, so a unique minimum is 0 and a
    unique maximum is 1. A group with one value has no range and maps to 0.5.
    """
    n = int(values.shape[0])
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    order = np.lexsort((values, groups))
    if n == 1:
        out[order] = 0.5
        return out

    v = values[order]
    g = groups[order]
    new_group = np.empty(n, dtype=np.bool_)
    new_group[0] = True
    new_group[1:] = g[1:] != g[:-1]
    new_tie = np.empty(n, dtype=np.bool_)
    new_tie[0] = True
    new_tie[1:] = new_group[1:] | (v[1:] != v[:-1])

    index = np.arange(n, dtype=np.int64)
    group_start = np.maximum.accumulate(np.where(new_group, index, np.int64(0)))
    pos = index - group_start

    starts = np.flatnonzero(new_group)
    group_n = np.diff(np.append(starts, n))
    size = np.repeat(group_n, group_n)

    tie_starts = np.flatnonzero(new_tie)
    tie_ends = np.empty(tie_starts.shape[0], dtype=np.int64)
    tie_ends[:-1] = tie_starts[1:] - 1
    tie_ends[-1] = n - 1
    avg_pos = 0.5 * (pos[tie_starts] + pos[tie_ends]).astype(np.float64)
    m = size[tie_starts]
    scaled = np.full(m.shape[0], 0.5, dtype=np.float64)
    multi = m > 1
    scaled[multi] = avg_pos[multi] / (m[multi] - 1)
    tie_n = np.diff(np.append(tie_starts, n))
    out[order] = np.repeat(scaled, tie_n)
    return out


def rank_by_group(X: np.ndarray, col_mask: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Replace finite masked entries with their within-group rank in [0, 1].

    Each masked column is ranked on its own. NaN and inf stay as they are and
    do not take a place. Other columns are copied through. ``group`` is an
    integer id per row (for example the month). Label-free: no targets are read.
    """
    X = _matrix(X)
    out = np.array(X, copy=True)
    cols = _indices(col_mask, X.shape[1], "col_mask")
    if cols.size == 0 or X.shape[0] == 0:
        return out
    groups = np.asarray(group)
    if groups.ndim != 1 or groups.shape[0] != X.shape[0]:
        raise ValueError(f"group must have shape ({X.shape[0]},), got {groups.shape}")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError(f"group must be an integer array, got {groups.dtype}")
    for j in cols:
        finite = np.isfinite(out[:, j])
        if not finite.any():
            continue
        ranked = _unit_average_ranks(
            out[finite, j].astype(np.float64, copy=False), groups[finite]
        )
        out[finite, j] = ranked
    return out


def scale_rows(
    X: np.ndarray, row_mask: np.ndarray, col_mask: np.ndarray, factor: float
) -> np.ndarray:
    """Copy of ``X`` with masked columns multiplied by ``factor`` on masked rows.

    This is the legit-amount shift (about x1.166) applied to rows the caller
    marks. The multiply is column by column so the only full-sized array is the
    returned copy. NaN stays NaN.
    """
    X = _matrix(X)
    out = np.array(X, copy=True)
    rows = _indices(row_mask, X.shape[0], "row_mask")
    cols = _indices(col_mask, X.shape[1], "col_mask")
    for j in cols:
        out[rows, j] *= factor
    return out
