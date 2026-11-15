"""Border posteriors of a fraud phase: every class border as an unknown position on the number line.

A fraud phase is a run of minted numbers in which each class owns one consecutive zone. The
borders are not marked, and in a short phase (October's is about a sixth of train's) a border can
sit several phones away from where anchors alone would put it. This module reads the phase on the
*number line*, used and unused numbers alike, with three pieces of evidence fixed from the train
phase:

* how densely a class uses its numbers (``refined.GAP_MU``: an unused number is more likely in a
  sparse zone, and it is charged to the zone it lies in, not to the phone above it);
* which anchors its phones carry and how many rows they have (``phones.emission_table``,
  ``refined.ROW_PMF``);
* how long its zone should be (``LENGTH_SHARE`` of the phase, a wide log-normal prior).

``segment`` sums over every placement of the ten borders (a segmental forward-backward, exact) and
returns the posterior of each border's position and, for every number, of its class. Not a feature
family: it runs in milliseconds on a test-sized phase and is never hashed into the feature cache.
"""

from __future__ import annotations

import numpy as np

from ringtrace.evidence.phones import K, emission_table
from ringtrace.evidence.refined import GAP_MU, ROW_FLOOR, ROW_PMF

# Share of the train fraud phase's numbers each class's zone holds (zones from a monotone fit to
# the cleaned labels), fixed here so the model stays label-free when it runs.
LENGTH_SHARE = np.array(
    [0.1206, 0.0911, 0.0867, 0.0753, 0.3171, 0.0726, 0.1248, 0.0409, 0.0242, 0.0189, 0.0277]
)
SIGMA = 0.8  # log-sd of a zone's length around its share of the phase


def line_ll(nums: np.ndarray, counts: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """``ll[x, k]``: log-likelihood of number ``lo + x`` under class ``k``; a used number carries
    its phone's anchors and row count, an unused one only the class's density."""
    n = int(hi) - int(lo) + 1
    dens = 1.0 / (1.0 + GAP_MU)
    ll = np.tile(np.log1p(-dens), (n, 1))
    pmf = np.maximum(ROW_PMF, ROW_FLOOR)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    rows = np.asarray(counts, dtype=np.float64).sum(axis=1).round().astype(np.int64)
    rb = np.clip(rows, 1, pmf.shape[1]) - 1
    used = np.log(dens)[None, :] + counts @ emission_table().T + np.log(pmf[:, rb].T)
    ll[np.asarray(nums, dtype=np.int64) - int(lo)] = used
    return ll


def _lse(m: np.ndarray, axis: int) -> np.ndarray:
    top = m.max(axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.log(np.exp(m - top).sum(axis=axis)) + np.squeeze(top, axis=axis)


def length_logprior(n: int, share: np.ndarray | None, sigma: float | None) -> np.ndarray:
    """``lp[k, L]`` for zone lengths 0..n; a zone holds at least one number."""
    lp = np.zeros((K, n + 1))
    lp[:, 0] = -np.inf
    if share is not None and sigma is not None:
        length = np.arange(1, n + 1, dtype=np.float64)
        for k in range(K):
            lp[k, 1:] = -0.5 * ((np.log(length) - np.log(max(share[k] * n, 0.5))) / sigma) ** 2
    return lp


def segment(ll: np.ndarray, share: np.ndarray | None = None, sigma: float | None = None) -> dict:
    """Exact posteriors over the ten borders of one phase.

    Returns ``cut[k, c]`` (k = 1..K-1: the posterior that zone ``k`` starts at number ``c``) and
    ``post[x, k]`` (the posterior that number ``x`` belongs to class ``k``)."""
    n = ll.shape[0]
    cum = np.vstack([np.zeros((1, K)), np.cumsum(ll, axis=0)])
    idx = np.arange(n + 1)
    length = idx[None, :] - idx[:, None]  # length[c0, c1] = c1 - c0
    ok = length >= 1
    lp = length_logprior(n, share, sigma)
    fwd = np.full((K + 1, n + 1), -np.inf)
    fwd[0, 0] = 0.0
    bwd = np.full((K + 1, n + 1), -np.inf)
    bwd[K, n] = 0.0
    pen = [np.where(ok, lp[k][np.clip(length, 0, n)], -np.inf) for k in range(K)]
    for k in range(K):
        fwd[k + 1] = _lse((fwd[k] - cum[:, k])[:, None] + pen[k], axis=0) + cum[:, k]
    for k in range(K - 1, -1, -1):
        bwd[k] = _lse((bwd[k + 1] + cum[:, k])[None, :] + pen[k], axis=1) - cum[:, k]
    z = fwd[K, n]
    cut = np.zeros((K + 1, n + 1))
    cut[0, 0] = 1.0
    cut[K, n] = 1.0
    for k in range(1, K):
        w = fwd[k] + bwd[k] - z
        cut[k] = np.exp(w - w.max())
        cut[k] /= cut[k].sum()
    started = np.cumsum(cut, axis=1)[:, :n]  # started[k, x] = P(zone k starts at or before x)
    post = (started[:K] - started[1 : K + 1]).T
    return {"cut": cut[1:K], "post": np.clip(post, 0.0, 1.0), "log_z": float(z)}


def decode(
    nums: np.ndarray,
    counts: np.ndarray,
    lo: int,
    hi: int,
    share: np.ndarray | None = LENGTH_SHARE,
    sigma: float | None = SIGMA,
) -> dict:
    """Per-phone class posterior of one phase, and the borders' posteriors on its number line."""
    got = segment(line_ll(nums, counts, lo, hi), share, sigma)
    at = np.asarray(nums, dtype=np.int64) - int(lo)
    return {"post": got["post"][at], "cut": got["cut"], "line_post": got["post"]}
