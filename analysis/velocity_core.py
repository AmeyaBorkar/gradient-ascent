"""Pure helpers for the velocity analysis (``analysis/velocity.py``): device sessions at any gap,
a burst-size test of "whole" against "thinned" events, per-100k rates, and an October F1 estimate
that rescales the structured and ordinary parts of every class by label-free ratios."""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu

K = 12


def sessions(
    key: np.ndarray, split: np.ndarray, ts: np.ndarray, row: np.ndarray, gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Session id and session size per row: consecutive rows of one key (inside one split) with
    gaps of at most ``gap`` seconds. Ties in time are broken by ``row``, so the result does not
    depend on input order."""
    n = len(key)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.lexsort((row, ts, key, split))
    k = np.asarray(key, dtype=np.int64)[order]
    s = np.asarray(split, dtype=np.int64)[order]
    t = np.asarray(ts, dtype=np.int64)[order]
    start = np.ones(n, dtype=bool)
    start[1:] = (k[1:] != k[:-1]) | (s[1:] != s[:-1]) | (t[1:] - t[:-1] > gap)
    first = np.flatnonzero(start)
    lengths = np.diff(np.r_[first, n])
    sid = np.empty(n, dtype=np.int64)
    size = np.empty(n, dtype=np.int64)
    sid[order] = np.repeat(first, lengths)
    size[order] = np.repeat(lengths, lengths)
    return sid, size


def per_100k(mask: np.ndarray, month: np.ndarray, months) -> list[float]:
    """Rows of ``mask`` per 100,000 rows of each month."""
    out = []
    for m in months:
        s = month == m
        out.append(float(mask[s].sum() / max(s.sum(), 1) * 1e5))
    return out


def size_test(test_sizes, train_sizes, keep: float = 0.5, draws: int = 20, seed: int = 0) -> dict:
    """Are the test bursts whole train-like bursts, or train bursts thinned to ``keep``?

    Returns the Mann-Whitney p-value of the test sizes against the whole and the thinned train
    sizes, and the chance that every test burst reaches the smallest observed test size under
    each reading."""
    test_sizes = np.asarray(test_sizes, dtype=np.int64)
    train_sizes = np.asarray(train_sizes, dtype=np.int64)
    rng = np.random.default_rng(seed)
    thinned = rng.binomial(np.repeat(train_sizes, draws), keep)
    lo = test_sizes.min()
    p_whole = float((train_sizes >= lo).mean())
    p_thin = float((thinned >= lo).mean())
    return {
        "test_sizes_q": np.quantile(test_sizes, [0, 0.5, 1]).tolist(),
        "train_sizes_q": np.quantile(train_sizes, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "thinned_sizes_q": np.quantile(thinned, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "mwu_p_whole": float(mannwhitneyu(test_sizes, train_sizes).pvalue),
        "mwu_p_thinned": float(mannwhitneyu(test_sizes, thinned).pvalue),
        "p_all_ge_min_whole": p_whole ** len(test_sizes),
        "p_all_ge_min_thinned": p_thin ** len(test_sizes),
    }


def rescaled_f1(
    lab: np.ndarray,
    gen: np.ndarray,
    calls_tr: np.ndarray,
    zord_tr: np.ndarray,
    seg_tr: np.ndarray,
    calls_te: np.ndarray,
    zord_te: np.ndarray,
    seg_te: np.ndarray,
) -> dict:
    """Expected per-class F1 on a test month from the train anatomy of every class.

    The true rows of a class split into structured rows (``gen`` agrees) and ordinary rows, and
    the calls into calls on zone-ordinary rows (``zord``: single-use fraud-pass devices with no
    template) and the rest. The structured part is rescaled by the test/train ratio of our calls
    outside the zone rows (a), the ordinary part by the ratio of zone-ordinary rows in the class's
    segment (b). No test label is read. With a = b = 1 this is the pooled train F1."""
    ntr, nte = len(lab), len(calls_te)
    out = {"f1_train": [], "f1_test": [], "a": [], "b": [], "ordinary_share": []}
    for c in range(K):
        t_s = np.sum((lab == c) & (gen == c)) / ntr
        t_o = np.sum((lab == c) & (gen != c)) / ntr
        tp_s = np.sum((lab == c) & (gen == c) & (calls_tr == c)) / ntr
        tp_o = np.sum((lab == c) & (gen != c) & (calls_tr == c)) / ntr
        k_s = np.sum((calls_tr == c) & ~zord_tr) / ntr
        k_z = np.sum((calls_tr == c) & zord_tr) / ntr
        if c == 0:
            a = b = 1.0
        else:
            a = (np.sum((calls_te == c) & ~zord_te) / nte) / max(k_s, 1e-12)
            z_tr = np.sum((seg_tr == c) & zord_tr) / ntr
            z_te = np.sum((seg_te == c) & zord_te) / nte
            b = z_te / z_tr if z_tr > 0 else 1.0
        den_tr = k_s + k_z + t_s + t_o
        den_te = (k_s + t_s) * a + (k_z + t_o) * b
        out["f1_train"].append(float(2 * (tp_s + tp_o) / den_tr) if den_tr else 0.0)
        out["f1_test"].append(float(2 * (tp_s * a + tp_o * b) / den_te) if den_te else 0.0)
        out["a"].append(float(a))
        out["b"].append(float(b))
        out["ordinary_share"].append(float(t_o / (t_s + t_o)) if t_s + t_o else 0.0)
    out["macro_train"] = float(np.mean(out["f1_train"]))
    out["macro_test"] = float(np.mean(out["f1_test"]))
    return out


def thin_keep(n: int, protect: np.ndarray, keep: float, seed: int) -> np.ndarray:
    """A random row sample of rate ``keep`` that always keeps the rows in ``protect``."""
    rng = np.random.default_rng(seed)
    return protect | (rng.random(n) < keep)
