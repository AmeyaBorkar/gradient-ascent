"""Pure helpers for ``analysis/farlook.py`` (no data access, tested on synthetic arrays)."""

from __future__ import annotations

import numpy as np


def f1(tp: float, called: float, true: float) -> float:
    return 2.0 * tp / (called + true) if called + true > 0 else 0.0


def split_curve(a: dict, b: dict, pool: dict, steps: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Macro-F1 change (12 classes) when a share q of a pool moves from class A to class B.

    ``a`` and ``b`` hold each class's month totals ``tp``, ``called`` and ``true`` with the whole
    pool called A. ``pool`` holds its size ``n`` and its rows that are truly A (``a``) and truly B
    (``b``). The moved rows are a random share, so the expected counts are linear in q.
    """
    qs = np.linspace(0.0, 1.0, steps)
    base = f1(a["tp"], a["called"], a["true"]) + f1(b["tp"], b["called"], b["true"])
    out = np.empty(steps)
    for i, q in enumerate(qs):
        fa = f1(a["tp"] - q * pool["a"], a["called"] - q * pool["n"], a["true"])
        fb = f1(b["tp"] + q * pool["b"], b["called"] + q * pool["n"], b["true"])
        out[i] = (fa + fb - base) / 12.0
    return qs, out


def oracle_typing_gain(a: dict, b: dict, pool: dict) -> float:
    """Macro-F1 change if every pool row that is truly B were called B and nothing else moved."""
    base = f1(a["tp"], a["called"], a["true"]) + f1(b["tp"], b["called"], b["true"])
    fa = f1(a["tp"], a["called"] - pool["b"], a["true"])
    fb = f1(b["tp"] + pool["b"], b["called"] + pool["b"], b["true"])
    return (fa + fb - base) / 12.0


def pair_concordance(entity: np.ndarray, label: np.ndarray, pair: tuple[int, int]) -> tuple:
    """Share of event pairs on one entity that carry the same label, and the number of pairs.

    One element per (entity, event); ``label`` is the event's class, one of ``pair``.
    """
    _, inv = np.unique(entity, return_inverse=True)
    n_a = np.bincount(inv, weights=(label == pair[0]).astype(np.float64))
    n_b = np.bincount(inv, weights=(label == pair[1]).astype(np.float64))
    same = (n_a * (n_a - 1) / 2 + n_b * (n_b - 1) / 2).sum()
    total = ((n_a + n_b) * (n_a + n_b - 1) / 2).sum()
    return (float(same / total) if total > 0 else float("nan")), int(total)


def shuffled_concordance(
    entity: np.ndarray, event: np.ndarray, label: np.ndarray, pair: tuple[int, int], draws: int
) -> tuple[float, float]:
    """Mean and sd of ``pair_concordance`` when the event labels are permuted among events."""
    events, first, inv = np.unique(event, return_index=True, return_inverse=True)
    labels = label[first]
    rng = np.random.default_rng(0)
    vals = [pair_concordance(entity, rng.permutation(labels)[inv], pair)[0] for _ in range(draws)]
    return float(np.mean(vals)), float(np.std(vals))


def gap_stats(ids: np.ndarray) -> dict:
    """Gap structure of a set of integer ids. A counter pushed through an affine map leaves at
    most three distinct gaps; a uniform draw leaves geometric gaps (cv close to 1)."""
    gaps = np.diff(np.sort(np.asarray(ids, dtype=np.int64)))
    return {
        "ids": int(len(ids)),
        "duplicates": int((gaps == 0).sum()),
        "distinct_gaps": int(len(np.unique(gaps))),
        "gap_cv": float(gaps.std() / gaps.mean()),
    }


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation without ties handling beyond ordinal ranks (inputs are near-unique)."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def keep_units(n: int, rate: float, seed: int) -> np.ndarray:
    """Seeded Bernoulli thinning mask over ``n`` units."""
    return np.random.default_rng(seed).random(n) < rate
