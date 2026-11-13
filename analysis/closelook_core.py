"""Pure functions behind ``analysis/closelook.py`` (tested on synthetic data in
``tests/test_closelook_core.py``).

The A-clean proxy stands in for clean test labels: the cleaned train labels still leave the
unreported ordinary-looking fraud rows legit, and nothing tells those rows apart from legit
traffic, so each cleaned-legit row is scored as hidden class k with one common chance per month.
The month's hidden count of class k is ``HIDDEN_TOTAL`` shared out in proportion to that month's
ordinary-looking class-k labels (cleaned fraud, generator-only legit). Macro-F1 is then read off
the exact expected confusion matrix.
"""

from __future__ import annotations

import numpy as np

K = 12
HIDDEN_TOTAL = 3482  # train fraud rows that the cleaned labels leave legit (earlier measurement)


def hidden_counts(
    cleaned: np.ndarray, gen: np.ndarray, period: np.ndarray, total: float = HIDDEN_TOTAL
) -> dict[int, np.ndarray]:
    """Expected hidden fraud rows per month and class (index 0 stays 0)."""
    ordinary = (cleaned > 0) & (gen == 0)
    grand = max(int(ordinary.sum()), 1)
    out = {}
    for m in np.unique(period):
        sel = ordinary & (period == m)
        out[int(m)] = np.bincount(cleaned[sel], minlength=K).astype(float) * total / grand
    return out


def expected_confusion(truth: np.ndarray, pred: np.ndarray, hidden=None, k: int = K) -> np.ndarray:
    """Confusion (truth x call); with ``hidden``, the cleaned-legit rows carry ``hidden[c]`` rows
    of class c spread evenly over them."""
    cm = np.bincount(truth * k + pred, minlength=k * k).reshape(k, k).astype(float)
    if hidden is None:
        return cm
    legit = cm[0].copy()
    h = np.asarray(hidden, dtype=float) / max(legit.sum(), 1.0)
    h = h.copy()
    h[0] = 0.0
    cm[0] -= h.sum() * legit
    cm += h[:, None] * legit[None, :]
    return cm


def f1_of(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def macro(truth: np.ndarray, pred: np.ndarray, hidden=None) -> float:
    return float(f1_of(expected_confusion(truth, pred, hidden)).mean())


def class_table(truth: np.ndarray, pred: np.ndarray, hidden=None) -> dict:
    """Per class: true rows, calls, TP, FP, FN, precision, recall, F1, and the F1-optimal call
    count if the class's precision held at the current value for more or fewer calls
    (``2 TP / (N + P)`` with TP = prec x P is maximised at P = N, so this is a reference only)."""
    cm = expected_confusion(truth, pred, hidden)
    tp = np.diag(cm)
    n = cm.sum(axis=1)
    p = cm.sum(axis=0)
    f1 = f1_of(cm)
    out = {}
    for c in range(K):
        out[str(c)] = {
            "true": float(n[c]),
            "calls": float(p[c]),
            "tp": float(tp[c]),
            "fp": float(p[c] - tp[c]),
            "fn": float(n[c] - tp[c]),
            "precision": float(tp[c] / p[c]) if p[c] else 0.0,
            "recall": float(tp[c] / n[c]) if n[c] else 0.0,
            "f1": float(f1[c]),
        }
    out["macro"] = float(f1.mean())
    return out


def fixed_point_scaled(
    proba: np.ndarray,
    class_mult: np.ndarray | None = None,
    fraud_scale: float = 0.875,
    max_iter: int = 50,
    damping: float = 0.5,
) -> np.ndarray:
    """The fixed point of ``ringtrace.decode`` with one admission multiplier per class: a row
    joins class k when ``2 p_ik - m_k F_k`` (over ``N_k + P_k``) is its best gain. ``m_k < 1``
    lowers class k's cut-off (more calls), ``m_k > 1`` raises it; all ones is the plain fixed
    point."""
    p = np.asarray(proba, dtype=np.float64).copy()
    p[:, 1:] /= fraud_scale
    p /= p.sum(axis=1, keepdims=True)
    k = p.shape[1]
    mult = np.ones(k) if class_mult is None else np.asarray(class_mult, dtype=np.float64)
    expected = p.sum(axis=0)
    pred = p.argmax(axis=1)
    prev = None
    for _ in range(max_iter):
        counts = np.bincount(pred, minlength=k).astype(np.float64)
        tp = np.bincount(pred, weights=p[np.arange(len(p)), pred], minlength=k)
        denom = expected + counts
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
        inv = 1.0 / np.maximum(denom, 1e-12)
        if prev is not None:
            inv = damping * prev[0] + (1 - damping) * inv
            f1 = damping * prev[1] + (1 - damping) * f1
        prev = (inv, f1)
        gain = (2 * p - (mult * f1)[None, :]) * inv[None, :]
        new = gain.argmax(axis=1)
        if np.array_equal(new, pred):
            break
        pred = new
    return pred


def pair_split_best(
    n_a: float, n_b: float, tp_a: float, tp_b: float, p_a: float, p_b: float, pool: float, q: float
) -> dict:
    """Two look-alike classes A and B and a pool of ``pool`` rows that are A with chance ``q``
    and B otherwise (nothing ranks them). Outside the pool A has ``tp_a`` true positives from
    ``p_a`` calls and ``n_a`` true rows in total (pool rows included); likewise B. Giving a share
    s of the pool to A (the rest to B) gives F1_A + F1_B; return the best s on a grid and the
    sums at s = 0, 1 and the best."""

    def total(s: float) -> float:
        fa = 2 * (tp_a + s * pool * q) / max(n_a + p_a + s * pool, 1e-12)
        fb = 2 * (tp_b + (1 - s) * pool * (1 - q)) / max(n_b + p_b + (1 - s) * pool, 1e-12)
        return fa + fb

    grid = np.linspace(0.0, 1.0, 101)
    vals = np.array([total(s) for s in grid])
    i = int(vals.argmax())
    return {
        "best_share_a": float(grid[i]),
        "best": float(vals[i]),
        "all_a": total(1.0),
        "all_b": total(0.0),
    }
