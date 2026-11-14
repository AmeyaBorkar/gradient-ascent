"""Turn class probabilities into labels that maximise expected macro-F1 on one month.

The rule needs no labels. For a candidate labelling, the expected per-class quantities are

    N_k  = sum_i p_ik                      (expected true count)
    P_k  = #{i : pred_i = k}               (predicted count)
    TP_k = sum_{i : pred_i = k} p_ik       (expected true positives)
    F_k  = 2 TP_k / (N_k + P_k)

Moving row i into class k changes macro-F1 at first order by (2 p_ik - F_k) / (N_k + P_k), so
the labelling that is a fixed point of "give every row its best marginal class" is a stationary
point of expected macro-F1. We start from argmax and iterate that map until no row moves.

That iteration is damped and moves every row at once, so it can stop short of the labelling that
maximises the objective exactly. ``local_search`` starts from the fixed point and applies
single-row moves. Moving row i from class a to class b changes only those two classes,

    TP_a' = TP_a - p_ia,  P_a' = P_a - 1
    TP_b' = TP_b + p_ib,  P_b' = P_b + 1

with every N_k fixed, so the change in mean F1 is known from the two classes' current TP and P.
A round takes a batch of such moves whose classes do not overlap (their changes add), recomputes
the totals, and stops when no move improves by more than ``tol``. If a batch ever fails to raise
the objective, the rest of the search applies one move at a time. A row's destination tie goes to
the lower class code; a tie between rows goes to the lower row index.

Train labels under-report fraud by about 12 %, so a model fitted on them understates every fraud
probability; ``fraud_scale`` divides the fraud columns before decoding to undo that.
"""

from __future__ import annotations

import numpy as np


def _adjusted(proba: np.ndarray, fraud_scale: float) -> np.ndarray:
    """Fraud-scaled, row-normalised probabilities. Same adjustment as ``fixed_point``."""
    p = np.asarray(proba, dtype=np.float64).copy()
    p[:, 1:] /= fraud_scale
    p /= p.sum(axis=1, keepdims=True)
    return p


def _totals(p: np.ndarray, pred: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(pred, minlength=k).astype(np.float64)
    weights = p[np.arange(len(p)), pred]
    etp = np.bincount(pred, weights=weights, minlength=k).astype(np.float64)
    return etp, counts


def _class_f1(etp: np.ndarray, counts: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Per-class ``2 ETP / (P + T)``; a class with ``P + T = 0`` scores 0."""
    denom = counts + total
    out = np.zeros(len(denom), dtype=np.float64)
    np.divide(2.0 * etp, denom, out=out, where=denom > 0.0)
    return out


def expected_macro_f1(
    proba: np.ndarray, assignment: np.ndarray, fraud_scale: float = 0.875
) -> float:
    """Mean over classes of ``2 ETP / (P + T)`` for one assignment.

    ``T`` and the row probabilities use the same fraud-scale adjustment as ``fixed_point``. This
    is the ratio of expectations, not the expected value of the F1 ratio.
    """
    p = _adjusted(proba, fraud_scale)
    pred = np.asarray(assignment, dtype=np.int64)
    if pred.shape != (len(p),):
        raise ValueError(f"assignment shape {pred.shape} does not match {len(p)} rows")
    k = p.shape[1]
    etp, counts = _totals(p, pred, k)
    return float(_class_f1(etp, counts, p.sum(axis=0)).mean())


def fixed_point(
    proba: np.ndarray,
    fraud_scale: float = 0.875,
    max_iter: int = 50,
    damping: float = 0.5,
    hidden_factor: float = 1.0,
) -> np.ndarray:
    """``hidden_factor`` multiplies each fraud class's expected true count: the model is trained
    to give the ordinary-looking fraud rows a legit score, yet they still count as fraud in the
    labels, so the true class sizes are larger than the probabilities say (1.0 = trust them)."""
    p = np.asarray(proba, dtype=np.float64).copy()
    p[:, 1:] /= fraud_scale
    p /= p.sum(axis=1, keepdims=True)
    k = p.shape[1]
    expected = p.sum(axis=0)
    expected[1:] *= hidden_factor
    pred = p.argmax(axis=1)
    weight_prev = None
    for _ in range(max_iter):
        counts = np.bincount(pred, minlength=k).astype(np.float64)
        tp = np.bincount(pred, weights=p[np.arange(len(p)), pred], minlength=k)
        denom = expected + counts
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
        inv = 1.0 / np.maximum(denom, 1e-12)
        if weight_prev is not None:
            inv = damping * weight_prev[0] + (1 - damping) * inv
            f1 = damping * weight_prev[1] + (1 - damping) * f1
        weight_prev = (inv, f1)
        gain = (2 * p - f1[None, :]) * inv[None, :]
        new = gain.argmax(axis=1)
        if np.array_equal(new, pred):
            break
        pred = new
    return pred


def _best_moves(
    p: np.ndarray,
    pred: np.ndarray,
    etp: np.ndarray,
    counts: np.ndarray,
    total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact objective change of each row's best move, and that move's class.

    The change depends only on the source and destination totals. The current class is not a
    move; ties between destinations go to the lower class code (``argmax``).
    """
    n, k = p.shape
    f1 = _class_f1(etp, counts, total)
    src = pred
    rows = np.arange(n)
    p_src = p[rows, src]
    d_src = counts[src] - 1.0 + total[src]
    f1_src_new = np.zeros(n, dtype=np.float64)
    np.divide(2.0 * (etp[src] - p_src), d_src, out=f1_src_new, where=d_src > 0.0)
    d_dst = counts + 1.0 + total
    f1_dst_new = np.empty_like(p)
    np.divide(2.0 * (etp + p), d_dst, out=f1_dst_new)
    delta = (f1_src_new[:, None] - f1[src][:, None]) + (f1_dst_new - f1)
    delta[rows, src] = -np.inf
    dest = delta.argmax(axis=1)
    return delta[rows, dest] / k, dest


def _matching(
    pred: np.ndarray, delta: np.ndarray, dest: np.ndarray, tol: float, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """One class-disjoint batch of the improving moves, best first.

    Moves that share a class are not additive: each delta was computed from the current totals.
    Within a class pair only the best row can ever be chosen, so the scan is over one row per
    pair. Order is greatest improvement, then lower row index, then lower destination class.
    """
    candidates = np.flatnonzero(delta > tol)
    if len(candidates) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    pair = pred[candidates].astype(np.int64) * np.int64(k) + dest[candidates].astype(np.int64)
    by_pair = np.lexsort((candidates, -delta[candidates], pair))
    ordered = candidates[by_pair]
    pairs = pair[by_pair]
    first = np.empty(len(ordered), dtype=bool)
    first[0] = True
    first[1:] = pairs[1:] != pairs[:-1]
    winners = ordered[first]
    winners = winners[np.lexsort((dest[winners], winners, -delta[winners]))]
    used = np.zeros(k, dtype=bool)
    chosen: list[int] = []
    for raw in winners:
        i = int(raw)
        source = int(pred[i])
        target = int(dest[i])
        if used[source] or used[target]:
            continue
        used[source] = True
        used[target] = True
        chosen.append(i)
    if not chosen:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    picked = np.asarray(chosen, dtype=np.int64)
    return picked, dest[picked].astype(np.int64, copy=False)


def local_search(
    proba: np.ndarray,
    fraud_scale: float = 0.875,
    start: np.ndarray | None = None,
    max_rounds: int = 50,
    tol: float = 0.0,
) -> np.ndarray:
    """Labels maximising expected macro-F1 by exact single-row moves.

    Starts from ``fixed_point`` (or from ``start``) and accepts a move only when the recomputed
    objective rises. ``tol`` is the smallest improvement, in macro-F1, that a move must show.
    """
    p = _adjusted(proba, fraud_scale)
    n, k = p.shape
    total = p.sum(axis=0)
    if start is None:
        pred = np.asarray(fixed_point(proba, fraud_scale=fraud_scale), dtype=np.int64).copy()
    else:
        pred = np.asarray(start, dtype=np.int64).copy()
    if pred.shape != (n,):
        raise ValueError(f"start shape {pred.shape} does not match {n} rows")
    etp, counts = _totals(p, pred, k)
    obj = float(_class_f1(etp, counts, total).mean())
    single = False
    for _ in range(max_rounds):
        delta, dest = _best_moves(p, pred, etp, counts, total)
        chosen, new_class = _matching(pred, delta, dest, tol, k)
        if len(chosen) == 0:
            break
        if single:
            chosen = chosen[:1]
            new_class = new_class[:1]
        previous = pred[chosen].copy()
        old_etp, old_counts = etp, counts
        pred[chosen] = new_class
        etp, counts = _totals(p, pred, k)
        new_obj = float(_class_f1(etp, counts, total).mean())
        if new_obj <= obj:
            pred[chosen] = previous
            etp, counts = old_etp, old_counts
            if single:
                break
            single = True
            continue
        obj = new_obj
    return pred


def decode_block(proba: np.ndarray, **kw) -> np.ndarray:
    """Decode one block of rows. ``method`` selects the rule.

    Absent or ``fixed_point`` is :func:`fixed_point`, and every other keyword is passed through.
    ``local_search`` reads ``fraud_scale``, ``start``, ``max_rounds`` and ``tol``.
    """
    method = kw.get("method", "fixed_point")
    if method == "fixed_point":
        if "method" in kw:
            kw = {key: value for key, value in kw.items() if key != "method"}
        return fixed_point(proba, **kw)
    if method == "local_search":
        keep = {key: kw[key] for key in ("fraud_scale", "start", "max_rounds", "tol") if key in kw}
        return local_search(proba, **keep)
    raise ValueError(
        f"unknown decoder method {method!r}; known methods are 'fixed_point' and 'local_search'"
    )


def decode_by_period(proba: np.ndarray, period: np.ndarray, **kw) -> np.ndarray:
    """Solve the rule separately on each month, as each month is scored on its own.

    ``kw`` is the recipe's decoder settings. ``method`` absent or ``fixed_point`` is the damped
    fixed point; ``local_search`` is the exact local search. A ``start`` array, if passed, is
    handed to every month, so it has to be that month's length.
    """
    out = np.zeros(len(proba), dtype=np.int64)
    for value in np.unique(period):
        m = period == value
        out[m] = decode_block(proba[m], **kw)
    return out
