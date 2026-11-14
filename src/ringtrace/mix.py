"""Class-mix weights for training, and a label-shift correction for a decoded month.

A target month can carry a different fraud mix from the months a model trains on. Two
optional, separate corrections:

* ``class_mix_weights`` reweights fraud rows so the weighted training mix moves toward
  that month. The weights go into the cascade's two stage fits.
* ``em_priors`` is the prior correction of Saerens, Latinne and Decaestecker (2002). The
  model learned probabilities under the training class shares; before decoding, a month's
  probabilities are adjusted to a prior estimated from that month alone.

Both stay off unless a recipe asks for them.
"""

from __future__ import annotations

import numpy as np

from ringtrace.classes import CODE, LEGIT, N_CLASSES


def ratios_from_names(d: dict[str, float]) -> dict[int, float]:
    """Map fraud class names to codes.

    Unknown names, ``legit`` and non-positive ratios are rejected. Legit rows stay at
    weight 1, and a non-positive weight would drop a class or reverse its gradient.
    """
    if not isinstance(d, dict):
        raise ValueError("class mix ratios must be a mapping of class name to a positive ratio")
    out: dict[int, float] = {}
    for name, ratio in d.items():
        if name == "legit":
            raise ValueError("class mix ratios do not take legit; legit rows stay at weight 1")
        if not isinstance(name, str) or name not in CODE:
            raise ValueError(f"unknown class {name!r} in class mix ratios")
        try:
            value = float(ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"class mix ratio for {name} must be a positive number, got {ratio!r}"
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"class mix ratio for {name} must be positive, got {ratio!r}")
        out[CODE[name]] = value
    return out


def _check_ratios(ratios: dict[int, float]) -> dict[int, float]:
    clean: dict[int, float] = {}
    for code, ratio in ratios.items():
        try:
            key = int(code)
            value = float(ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"class mix ratio {code!r}: {ratio!r} is not usable") from exc
        if key == LEGIT:
            raise ValueError("class mix weights keep legit rows at 1")
        if key not in range(1, N_CLASSES):
            raise ValueError(f"class code {key} is not a fraud class")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"class mix ratio for class {key} must be positive, got {ratio!r}")
        clean[key] = value
    return clean


def _event_majority(y: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Label of each row's event majority. Ties take the smaller code.

    A negative event id (``-1`` means the row is in no event) keeps that row's own label.
    Majority is computed from the rows passed in, which are the training rows.
    """
    events = np.asarray(events, dtype=np.int64)
    if len(events) != len(y):
        raise ValueError(f"events has length {len(events)}, labels have length {len(y)}")
    labels = np.array(y, dtype=np.int64, copy=True)
    inside = events >= 0
    if not inside.any():
        return labels
    _ids, inv = np.unique(events[inside], return_inverse=True)
    counts = np.bincount(inv * N_CLASSES + labels[inside], minlength=len(_ids) * N_CLASSES).reshape(
        len(_ids), N_CLASSES
    )
    labels[inside] = counts.argmax(axis=1)[inv]
    return labels


def class_mix_weights(
    y: np.ndarray, ratios: dict[int, float], events: np.ndarray | None = None
) -> np.ndarray:
    """Weight per training row that moves the fraud mix toward a target month.

    A row of fraud class ``k`` gets ``ratios.get(k, 1.0)``; a legit row gets 1.0. With
    ``events`` (one event id per row, negative when the row is in no event), every row of
    an event takes the weight of the event's majority label, and a tie goes to the smaller
    code.

    Tree boosters need no global renormalisation of these weights. They are instance
    weights already expressed as per-class ratios; rescaling every row would move them
    relative to ``min_child_weight`` and L2, which are in absolute weight units.
    """
    labels = np.asarray(y, dtype=np.int64)
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= N_CLASSES):
        raise ValueError(f"labels must be in 0..{N_CLASSES - 1}")
    if events is not None:
        labels = _event_majority(labels, events)
    ratios = _check_ratios(ratios)
    weights = np.ones(len(labels), dtype=np.float64)
    for code, ratio in ratios.items():
        weights[labels == code] = ratio
    return weights


def class_shares(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Share of each class in a training target. This is the source prior for label shift."""
    labels = np.asarray(y, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("no rows to estimate class shares")
    if int(labels.min()) < 0 or int(labels.max()) >= n_classes:
        raise ValueError(f"labels must be in 0..{n_classes - 1}")
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return counts / counts.sum()


def em_priors(
    post: np.ndarray,
    train_priors: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one block's class prior and adjust its probabilities for label shift.

    ``post`` rows are the model's probabilities, learned under ``train_priors`` (the
    training class shares). Starting from ``q = train_priors``, each step sets
    ``adjusted_ik`` proportional to ``post_ik * q_k / train_priors_k``, renormalises the
    rows, and replaces ``q`` by the column mean of ``adjusted``, until ``q`` moves by less
    than ``tol`` or ``max_iter`` steps have run. Returns ``(q, adjusted)``.

    A class with a zero training prior has no ratio. Its adjusted column stays zero and
    the other classes take the row's mass, so a zero prior cannot divide or put mass on a
    class the training rows never contained. A row whose mass sat only on such classes
    keeps its input row.
    """
    post = np.asarray(post, dtype=np.float64)
    train = np.asarray(train_priors, dtype=np.float64)
    if post.ndim != 2:
        raise ValueError(f"posteriors must be a 2-d array, got shape {post.shape}")
    if train.shape != (post.shape[1],):
        raise ValueError(
            f"train priors have length {train.shape[0]}, posteriors have {post.shape[1]} classes"
        )
    if train.size == 0 or not np.isfinite(train).all() or np.any(train < 0) or train.sum() <= 0:
        raise ValueError("train priors must be finite, non-negative and not all zero")
    if not np.isfinite(post).all():
        raise ValueError("posteriors must be finite")

    alive = train > 0
    q = train.copy()
    adjusted = post.copy()
    if len(post) == 0:
        return q, adjusted

    for _ in range(max_iter):
        ratio = np.zeros(train.shape, dtype=np.float64)
        np.divide(q, train, out=ratio, where=alive)
        weighted = post * ratio
        totals = weighted.sum(axis=1, keepdims=True)
        good = (totals.ravel() > 0) & np.isfinite(totals.ravel())
        adjusted = np.empty_like(weighted)
        adjusted[good] = weighted[good] / totals[good]
        if not good.all():
            adjusted[~good] = post[~good]
        finite = np.isfinite(adjusted).all(axis=1)
        if not finite.all():
            adjusted[~finite] = post[~finite]
        q_new = adjusted.mean(axis=0)
        shift = float(np.max(np.abs(q_new - q)))
        q = q_new
        if shift <= tol:
            break
    return q, adjusted
