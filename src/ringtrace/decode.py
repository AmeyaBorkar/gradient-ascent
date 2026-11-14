"""Turn class probabilities into labels that maximise expected macro-F1 on one month.

The rule needs no labels. For a candidate labelling, the expected per-class quantities are

    N_k  = sum_i p_ik                      (expected true count)
    P_k  = #{i : pred_i = k}               (predicted count)
    TP_k = sum_{i : pred_i = k} p_ik       (expected true positives)
    F_k  = 2 TP_k / (N_k + P_k)

Moving row i into class k changes macro-F1 at first order by (2 p_ik - F_k) / (N_k + P_k), so
the labelling that is a fixed point of "give every row its best marginal class" is a stationary
point of expected macro-F1. We start from argmax and iterate that map until no row moves.

Train labels under-report fraud by about 12 %, so a model fitted on them understates every fraud
probability; ``fraud_scale`` divides the fraud columns before decoding to undo that.
"""

from __future__ import annotations

import numpy as np


def clean_posterior(
    proba: np.ndarray,
    unreported: float | list[float] = 0.121,
    legit_flip: float = 0.0007,
    type_flip: float = 0.0,
) -> np.ndarray:
    """Invert the train label noise, so that rows carry probabilities of the clean labels.

    The noise model, measured on the train months: a fraud row of class k is filed legit with
    chance ``u_k`` (``unreported``, one value or one per class with index 0 ignored), a legit row
    is filed as a fraud class with chance ``legit_flip`` (spread evenly over the 11 types), and a
    reported fraud row carries another fraud type, uniformly, with chance ``type_flip``. A model
    fitted on the given labels estimates

        q_k = (1 - u_k) t_k + legit_flip / 11 * t_0       (types before the flip)

    so the clean posterior is t_k = (q_k - legit_flip / 11 * t_0) / (1 - u_k), with t_0 = 1 - sum
    t_k solved jointly; the type flip is undone on the fraud types' split (T^-1 of the uniform
    flip). Negative values are clipped and rows renormalised."""
    p = np.asarray(proba, dtype=np.float64)
    k = p.shape[1]
    u = np.full(k, float(unreported)) if np.isscalar(unreported) else np.asarray(unreported, float)
    if u.shape != (k,):
        raise ValueError(f"unreported needs one value or {k}, got {u.shape}")
    r = float(legit_flip) / (k - 1)
    keep = 1.0 - u[1:]
    # q_k = keep_k t_k + r t_0 with t_0 = 1 - sum t_k gives
    # t_0 = (1 - sum q_k / keep_k) / (1 - r sum 1 / keep_k)
    t0 = (1.0 - (p[:, 1:] / keep).sum(axis=1)) / (1.0 - r * (1.0 / keep).sum())
    t0 = np.clip(t0, 0.0, 1.0)
    t = np.clip((p[:, 1:] - r * t0[:, None]) / keep, 0.0, None)
    if type_flip > 0:
        f = t.sum(axis=1, keepdims=True)
        split = t / np.maximum(f, 1e-12)
        e, m = float(type_flip), k - 1
        split = np.clip((split - e / (m - 1)) / (1.0 - e - e / (m - 1)), 0.0, None)
        split /= np.maximum(split.sum(axis=1, keepdims=True), 1e-12)
        t = split * f
    out = np.empty_like(p)
    out[:, 1:] = t
    out[:, 0] = np.clip(1.0 - t.sum(axis=1), 0.0, None)
    return out / out.sum(axis=1, keepdims=True)


def fixed_point(
    proba: np.ndarray,
    fraud_scale: float = 0.875,
    max_iter: int = 50,
    damping: float = 0.5,
    hidden_factor: float = 1.0,
    clean: dict | None = None,
) -> np.ndarray:
    """``hidden_factor`` multiplies each fraud class's expected true count: the model is trained
    to give the ordinary-looking fraud rows a legit score, yet they still count as fraud in the
    labels, so the true class sizes are larger than the probabilities say (1.0 = trust them).

    ``clean`` (the keyword arguments of :func:`clean_posterior`) first maps the probabilities to
    the clean labels by inverting the measured noise; use it with ``fraud_scale`` 1.0."""
    p = np.asarray(proba, dtype=np.float64).copy()
    if clean is not None:
        p = clean_posterior(p, **clean)
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


def decode_by_period(proba: np.ndarray, period: np.ndarray, **kw) -> np.ndarray:
    """Solve the rule separately on each month, as each month is scored on its own."""
    out = np.zeros(len(proba), dtype=np.int64)
    for p in np.unique(period):
        m = period == p
        out[m] = fixed_point(proba[m], **kw)
    return out
