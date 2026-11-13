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
        raise ValueError(f"predict_fn returned {pred.shape[0]} rows for {rows.shape[0]} inputs")
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
                    f"predict_fn returned shape {pred.shape}, expected {(width,) + acc.shape[1:]}"
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
        ranked = _unit_average_ranks(out[finite, j].astype(np.float64, copy=False), groups[finite])
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


# ---------------------------------------------------------------------------------------------
# covariate-shift reweighting toward a target frame (features only)


def column_auc_gap(
    X_a: np.ndarray, X_b: np.ndarray, sample: int = 200_000, seed: int = 0
) -> np.ndarray:
    """``|AUC - 0.5|`` of each column alone at telling rows of ``X_b`` from rows of ``X_a``.

    At most ``sample`` rows of each side, drawn by a seeded generator. NaN ranks below every
    number, so a change in missingness counts as drift. Ties take the average rank.
    """
    from scipy.stats import rankdata

    X_a = _matrix(X_a)
    X_b = _matrix(X_b)
    if X_a.shape[1] != X_b.shape[1]:
        raise ValueError(f"column counts differ: {X_a.shape[1]} and {X_b.shape[1]}")
    rng = np.random.default_rng(seed)
    ia = np.sort(rng.choice(len(X_a), min(sample, len(X_a)), replace=False))
    ib = np.sort(rng.choice(len(X_b), min(sample, len(X_b)), replace=False))
    na, nb = ia.size, ib.size
    gap = np.empty(X_a.shape[1], dtype=np.float64)
    for j in range(X_a.shape[1]):
        v = np.concatenate([X_a[ia, j], X_b[ib, j]]).astype(np.float64)
        v[np.isnan(v)] = -np.inf
        r = rankdata(v)
        auc = (r[na:].sum() - nb * (nb + 1) / 2) / (na * nb)
        gap[j] = abs(auc - 0.5)
    return gap


def domain_weights(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    clip: tuple[float, float] = (0.2, 5.0),
    sample: int = 300_000,
    rounds: int = 200,
    seed: int = 0,
    threads: int = 8,
    log=print,
) -> tuple[np.ndarray, dict]:
    """Importance weights ``p_tgt(x) / p_src(x)`` for every source row, from a domain classifier.

    A LightGBM classifier tells source rows (0) from target rows (1) on the raw matrix; it reads
    no labels. Both sets are split in two halves by a seeded permutation; a model fitted on
    (at most ``sample`` rows of) each half scores the other half's source rows, so no source
    row is scored by a model that saw it. The odds are scaled by ``n_src / n_tgt``, clipped to
    ``clip`` and divided by their mean, so the weights average 1. ``info`` has the cross-fitted
    AUC on the source rows plus the scored target rows, and the effective sample share
    ``(sum w)^2 / (n sum w^2)``.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    X_src = _matrix(X_src)
    X_tgt = _matrix(X_tgt)
    if X_src.shape[1] != X_tgt.shape[1]:
        raise ValueError(f"column counts differ: {X_src.shape[1]} and {X_tgt.shape[1]}")
    n_s, n_t = X_src.shape[0], X_tgt.shape[0]
    if n_s < 4 or n_t < 4:
        raise ValueError("need at least 4 source and 4 target rows")
    rng = np.random.default_rng(seed)
    half_s = rng.permutation(n_s) % 2
    half_t = rng.permutation(n_t) % 2
    p_src = np.empty(n_s, dtype=np.float64)
    p_tgt = np.empty(n_t, dtype=np.float64)
    params = {
        "objective": "binary",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "num_threads": threads,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    for h in (0, 1):
        s_idx = np.flatnonzero(half_s == h)
        t_idx = np.flatnonzero(half_t == h)
        cap = max(sample // 2, 1)
        if s_idx.size > cap:
            s_idx = np.sort(rng.choice(s_idx, cap, replace=False))
        if t_idx.size > cap:
            t_idx = np.sort(rng.choice(t_idx, cap, replace=False))
        Xh = np.vstack([X_src[s_idx], X_tgt[t_idx]]).astype(np.float32)
        dh = np.r_[np.zeros(s_idx.size), np.ones(t_idx.size)]
        # balance the two domains inside the fit, then correct with n_src / n_tgt below
        wh = np.r_[np.full(s_idx.size, 0.5 / s_idx.size), np.full(t_idx.size, 0.5 / t_idx.size)]
        wh *= len(dh)
        booster = lgb.train(params, lgb.Dataset(Xh, dh, weight=wh), num_boost_round=rounds)
        del Xh
        other_s = np.flatnonzero(half_s != h)
        other_t = np.flatnonzero(half_t != h)
        for start in range(0, other_s.size, _DEFAULT_CHUNK):
            idx = other_s[start : start + _DEFAULT_CHUNK]
            p_src[idx] = booster.predict(X_src[idx].astype(np.float32))
        p_tgt[other_t] = booster.predict(X_tgt[other_t].astype(np.float32))
    eps = 1e-6
    p = np.clip(p_src, eps, 1 - eps)
    # The fit was balanced, so p / (1 - p) is already the density ratio p_tgt / p_src.
    w = np.clip(p / (1 - p), clip[0], clip[1])
    w = w / w.mean()
    auc = float(roc_auc_score(np.r_[np.zeros(n_s), np.ones(n_t)], np.r_[p_src, p_tgt]))
    ess = float(w.sum() ** 2 / (n_s * np.square(w).sum()))
    info = {
        "auc": auc,
        "ess_share": ess,
        "w_min": float(w.min()),
        "w_max": float(w.max()),
        "clipped_share": float(((p / (1 - p) <= clip[0]) | (p / (1 - p) >= clip[1])).mean()),
    }
    log(
        f"  shift weights: domain AUC {auc:.4f}, effective share {ess:.3f}, "
        f"w {info['w_min']:.3f}..{info['w_max']:.3f}, clipped {info['clipped_share']:.4f}"
    )
    return w, info


def fit_predict_weighted(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec,
    log=print,
) -> list[np.ndarray]:
    """TabM on one-hot labels with per-row weights; probabilities for every matrix in ``X_out``.

    Preprocessing is ``tabm.fit_prep`` on ``X_tr``; the update is
    ``nn_selftrain.train_one_soft`` (class weights from the row-weighted counts, a
    row-weighted cross-entropy). Seeds are averaged.
    """
    import torch

    from ringtrace import nn_selftrain, tabm

    prep = tabm.fit_prep(X_tr, spec.quantiles)
    Z = tabm.transform(prep, X_tr)
    Zo = [tabm.transform(prep, X) for X in X_out]
    T = nn_selftrain._one_hot(y_tr)
    outs = [np.zeros((len(X), T.shape[1]), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        model = nn_selftrain.train_one_soft(
            Z, T, np.asarray(w_tr, dtype=np.float64), spec, seed, log
        )
        for o, z in zip(outs, Zo, strict=True):
            o += tabm.predict(model, z, spec)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [o / len(spec.seeds) for o in outs]
