"""Pure helpers for the gap scan: per-class F1 under an October-like truth.

``shifted_f1`` answers "what would this file score if October held ``r`` times the train rate of
ordinary-looking fraud rows, and if the file also caught a share ``q`` of those rows at precision
``prec``?" It works on the counts of a held-out month, so it needs no extra rows:

- ordinary rows of class k are the truth's class-k rows that the generator-only target calls legit;
- the extra ``(r - 1)`` ordinary rows per class are taken from legit rows the file calls legit
  (they look like legit rows, so the file calls them legit);
- the caught rows become true positives of their class, and their false companions
  (``q r O_k (1/prec - 1)``) come out of legit rows the file called legit.
"""

from __future__ import annotations

import numpy as np

K = 12


def counts(truth: np.ndarray, pred: np.ndarray, k: int = K) -> tuple[np.ndarray, ...]:
    """(true positives, predicted, true) per class."""
    idx = truth.astype(np.int64) * k + pred.astype(np.int64)
    cm = np.bincount(idx, minlength=k * k).reshape(k, k).astype(float)
    return np.diag(cm).copy(), cm.sum(axis=0), cm.sum(axis=1)


def ordinary_counts(truth: np.ndarray, gen: np.ndarray, k: int = K) -> np.ndarray:
    """Per class, the truth's fraud rows that the generator-only target leaves legit."""
    o = np.bincount(truth[(truth > 0) & (gen == 0)], minlength=k).astype(float)
    o[0] = 0.0
    return o


def shifted_f1(
    truth: np.ndarray,
    pred: np.ndarray,
    gen: np.ndarray,
    r: float = 1.0,
    q: float = 0.0,
    prec: float = 1.0,
) -> np.ndarray:
    """Per-class F1 with ordinary rows scaled by ``r`` and a share ``q`` caught at ``prec``."""
    if r < 0 or not 0 <= q <= 1 or not 0 < prec <= 1:
        raise ValueError(f"bad arguments r={r} q={q} prec={prec}")
    tp, p, n = counts(truth, pred)
    o = ordinary_counts(truth, gen)
    extra = (r - 1.0) * o
    n = n + extra
    n[0] -= extra.sum()
    tp[0] -= extra.sum()
    caught = q * r * o
    false = caught * (1.0 / prec - 1.0)
    tp = tp + caught
    p = p + caught + false
    p[0] -= (caught + false).sum()
    tp[0] -= false.sum()
    if (tp < -1e-9).any() or (p < -1e-9).any():
        raise ValueError("the shift asks for more legit rows than the month holds")
    den = p + n
    return np.divide(2.0 * tp, den, out=np.zeros(K), where=den > 0)


def catch_needed(
    truth: np.ndarray,
    pred: np.ndarray,
    gen: np.ndarray,
    target: float,
    r: float = 1.0,
    prec: float = 1.0,
    tol: float = 1e-4,
) -> float:
    """Smallest share of ordinary rows to catch at ``prec`` so that macro-F1 reaches ``target``.

    Returns ``nan`` when even catching all of them falls short.
    """
    if shifted_f1(truth, pred, gen, r, 1.0, prec).mean() < target:
        return float("nan")
    lo, hi = 0.0, 1.0
    if shifted_f1(truth, pred, gen, r, 0.0, prec).mean() >= target:
        return 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if shifted_f1(truth, pred, gen, r, mid, prec).mean() >= target:
            hi = mid
        else:
            lo = mid
    return hi


def implied_r(
    truth: np.ndarray, pred: np.ndarray, gen: np.ndarray, score: float, hi: float = 2.0
) -> float:
    """The ordinary-row scale ``r`` at which ``pred`` scores ``score``."""
    lo = 0.0
    if (
        not shifted_f1(truth, pred, gen, hi).mean()
        <= score
        <= shifted_f1(truth, pred, gen, lo).mean()
    ):
        raise ValueError(f"score {score} is outside the range reachable with r in [{lo}, {hi}]")
    for _ in range(50):
        mid = (lo + hi) / 2
        if shifted_f1(truth, pred, gen, mid).mean() > score:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def template_majority(
    typist: np.ndarray, tmpl: np.ndarray, pairs: dict[int, tuple[int, int]]
) -> np.ndarray:
    """A structure-only typist: inside each look-alike template it calls the pair's majority."""
    out = typist.copy()
    for t, (a, b) in pairs.items():
        m = (tmpl == t) & np.isin(typist, [a, b])
        if m.any():
            vals, cnt = np.unique(typist[m], return_counts=True)
            out[m] = vals[np.argmax(cnt)]
    return out
