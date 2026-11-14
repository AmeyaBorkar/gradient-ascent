"""Turn class probabilities into labels that maximise expected macro-F1 on one month.

The rule needs no labels. For a candidate labelling, the expected per-class quantities are

    N_k  = sum_i p_ik                      (expected true count)
    P_k  = #{i : pred_i = k}               (predicted count)
    TP_k = sum_{i : pred_i = k} p_ik       (expected true positives)
    F_k  = 2 TP_k / (N_k + P_k)

Moving row i into class k changes macro-F1 at first order by (2 p_ik - F_k) / (N_k + P_k), so
the labelling that is a fixed point of "give every row its best marginal class" is a stationary
point of expected macro-F1. We start from argmax and iterate that map until no row moves. If the
map is still moving when the iteration budget runs out, the last labelling is just wherever the
path stopped; the one returned is the labelling on that path with the highest plug-in macro-F1.

Train labels under-report fraud by about 12 %, so a model fitted on them understates every fraud
probability; ``fraud_scale`` divides the fraud columns before decoding to undo that.
"""

from __future__ import annotations

import numpy as np


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
    sums = p.sum(axis=1, keepdims=True)
    # A row with no finite positive mass (zeros, NaN, a zero scale) would make every class
    # total NaN and the whole month would collapse to class 0. Call it legit and leave it
    # out of the totals, so one bad row cannot move any other call.
    good = np.isfinite(p).all(axis=1) & np.isfinite(sums).ravel() & (sums.ravel() > 0)
    out = np.zeros(len(p), dtype=np.int64)
    if not good.any():
        return out
    if not good.all():
        p = p[good]
        sums = sums[good]
    p /= sums
    k = p.shape[1]
    expected = p.sum(axis=0)
    expected[1:] *= hidden_factor
    pred = p.argmax(axis=1)
    best_pred = pred.copy()
    best_score = -1.0
    weight_prev = None
    settled = False
    for _ in range(max_iter):
        counts = np.bincount(pred, minlength=k).astype(np.float64)
        tp = np.bincount(pred, weights=p[np.arange(len(p)), pred], minlength=k)
        denom = expected + counts
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
        score = float(f1.mean())
        if score > best_score:
            best_score = score
            best_pred = pred.copy()
        inv = 1.0 / np.maximum(denom, 1e-12)
        if weight_prev is not None:
            inv = damping * weight_prev[0] + (1 - damping) * inv
            f1 = damping * weight_prev[1] + (1 - damping) * f1
        weight_prev = (inv, f1)
        gain = (2 * p - f1[None, :]) * inv[None, :]
        new = gain.argmax(axis=1)
        if np.array_equal(new, pred):
            settled = True
            break
        pred = new
    if settled:
        # A fixed point of the (damped) map is the labelling the iteration is defined to return,
        # even when an earlier step scored higher.
        out[good] = pred
        return out
    counts = np.bincount(pred, minlength=k).astype(np.float64)
    tp = np.bincount(pred, weights=p[np.arange(len(p)), pred], minlength=k)
    denom = expected + counts
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    if float(f1.mean()) > best_score:
        best_pred = pred
    out[good] = best_pred
    return out


def decode_by_period(proba: np.ndarray, period: np.ndarray, **kw) -> np.ndarray:
    """Solve the rule separately on each month, as each month is scored on its own."""
    out = np.zeros(len(proba), dtype=np.int64)
    for p in np.unique(period):
        m = period == p
        out[m] = fixed_point(proba[m], **kw)
    return out
