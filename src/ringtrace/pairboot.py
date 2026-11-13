"""A paired Poisson bootstrap of macro-F1 change that is fast when many challengers share a base.

Every resampling unit (an event, or a row on its own) gets a Poisson(1) weight per draw, computed
from a counter-based hash of (seed, unit, draw) rather than drawn from a stream, so the weights of
any subset of units can be recomputed exactly. The base's resampled confusion counts are built once
(in chunks of units); a challenger's counts are the base's plus the weighted difference over the
units where any call differs, which is usually a few hundred. The change is the mean over strata
(months) of the per-stratum macro-F1 differences, as in ``ringtrace.compare.bootstrap``; the
interval is 2.5 / 97.5 % by default.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from ringtrace.classes import N_CLASSES

K = N_CLASSES
_M1, _M2, _M3 = (
    np.uint64(0x9E3779B97F4A7C15),
    np.uint64(0xBF58476D1CE4E5B9),
    np.uint64(0x94D049BB133111EB),
)
# inverse CDF of Poisson(1): weight k when cdf[k-1] <= u < cdf[k]
_CDF = np.cumsum([math.exp(-1) / math.factorial(k) for k in range(20)])


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser on uint64 (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        x = x + _M1
        x = (x ^ (x >> np.uint64(30))) * _M2
        x = (x ^ (x >> np.uint64(27))) * _M3
        return x ^ (x >> np.uint64(31))


def poisson_weights(seed: int, units: np.ndarray, draws: int) -> np.ndarray:
    """(draws x len(units)) Poisson(1) weights, a pure function of (seed, unit, draw)."""
    u64 = np.asarray(units, dtype=np.uint64)
    d = np.arange(draws, dtype=np.uint64)[:, None]
    with np.errstate(over="ignore"):
        key = _mix64(np.uint64(seed) * _M3 + u64[None, :]) ^ (d * _M1)
    u = (_mix64(key) >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    return np.searchsorted(_CDF, u, side="right").astype(np.float32)


def _counts(g: np.ndarray, truth: np.ndarray, pred: np.ndarray, ng: int) -> sp.csr_matrix:
    hit = truth == pred
    rows = np.concatenate([g[hit], g, g])
    cols = np.concatenate([truth[hit], K + pred, 2 * K + truth])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(ng, 3 * K))


def macro(c: np.ndarray) -> np.ndarray:
    tp, pr, tr = c[:, :K], c[:, K : 2 * K], c[:, 2 * K :]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(pr + tr > 0, 2 * tp / (pr + tr), 0.0)
    return f1.mean(axis=1)


class PairedBoot:
    """Resampled counts of one base, ready to compare any number of challengers."""

    def __init__(
        self,
        units: np.ndarray,
        truth: np.ndarray,
        base: np.ndarray,
        strata: np.ndarray | None = None,
        draws: int = 2000,
        seed: int = 0,
        chunk: int = 4096,
    ):
        self.draws, self.seed = draws, seed
        self.truth = np.asarray(truth)
        self.base = np.asarray(base)
        _, self.g = np.unique(units, return_inverse=True)
        strata = np.zeros(len(units), dtype=np.int64) if strata is None else np.asarray(strata)
        self.strata = []
        for s in np.unique(strata):
            rows = np.flatnonzero(strata == s)
            ids = np.unique(self.g[rows])  # global unit ids in this stratum
            local = np.searchsorted(ids, self.g[rows])
            ca = _counts(local, self.truth[rows], self.base[rows], len(ids))
            tot = np.zeros((draws, 3 * K))
            for i in range(0, len(ids), chunk):
                w = poisson_weights(seed, ids[i : i + chunk], draws)
                tot += np.asarray(ca[i : i + chunk].T.dot(w.T).T)
            point = np.asarray(ca.sum(axis=0)).ravel()
            self.strata.append({"rows": rows, "ids": ids, "local": local, "tot": tot, "pt": point})

    def compare(self, chal: np.ndarray, q=(2.5, 97.5)) -> dict:
        chal = np.asarray(chal)
        deltas = np.zeros(self.draws)
        point = 0.0
        for st in self.strata:
            rows = st["rows"]
            diff = self.base[rows] != chal[rows]
            if not diff.any():
                continue
            touched = np.unique(st["local"][diff])
            sel = np.isin(st["local"], touched)
            r = rows[sel]
            loc = np.searchsorted(touched, st["local"][sel])
            n = len(touched)
            dc = _counts(loc, self.truth[r], chal[r], n) - _counts(
                loc, self.truth[r], self.base[r], n
            )
            w = poisson_weights(self.seed, st["ids"][touched], self.draws)
            tot_b = st["tot"] + np.asarray(dc.T.dot(w.T).T)
            deltas += macro(tot_b) - macro(st["tot"])
            pb = st["pt"] + np.asarray(dc.sum(axis=0)).ravel()
            point += float(macro(pb[None])[0] - macro(st["pt"][None])[0])
        k = len(self.strata)
        deltas /= k
        lo, hi = np.percentile(deltas, list(q))
        return {"delta": point / k, "lo": float(lo), "hi": float(hi)}
