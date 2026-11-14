"""Pure functions behind ``analysis/cleandec.py``: the train label-noise process, a clean-label
proxy target, macro-F1 on it, and the public-score likelihood of a look-alike switch.

Tested on synthetic data in ``tests/test_cleandec_core.py``.
"""

from __future__ import annotations

import numpy as np

K = 12
HIDDEN_TOTAL = (
    3482  # train fraud-pass rows that the cleaned labels leave legit (earlier measurement)
)


# ------------------------------------------------------------------------------- noise process


def noise_rates(label: np.ndarray, truth: np.ndarray, group: np.ndarray) -> dict:
    """Over rows known to be fraud of class ``truth`` (> 0), per value of ``group``: rows, the
    share filed legit (unreported) and the share filed under another fraud type (wrong type),
    with the chi-square statistic of the unreported rate's homogeneity across groups."""
    label, truth, group = map(np.asarray, (label, truth, group))
    if not (truth > 0).all():
        raise ValueError("noise_rates needs a population of fraud rows (truth > 0)")
    keys, inv = np.unique(group, return_inverse=True)
    n = np.bincount(inv, minlength=len(keys)).astype(float)
    unrep = np.bincount(inv, weights=(label == 0).astype(float), minlength=len(keys))
    wrong = np.bincount(
        inv, weights=((label > 0) & (label != truth)).astype(float), minlength=len(keys)
    )
    p = unrep.sum() / n.sum()
    e = n * p
    chi2 = float((((unrep - e) ** 2) / np.maximum(e * (1 - p), 1e-12)).sum())
    return {
        "groups": {
            str(k): {
                "rows": int(n[i]),
                "unreported": unrep[i] / n[i],
                "wrong_type": wrong[i] / n[i],
            }
            for i, k in enumerate(keys)
        },
        "pooled_unreported": float(p),
        "pooled_wrong_type": float(wrong.sum() / n.sum()),
        "chi2": chi2,
        "df": int(len(keys) - 1),
    }


def wrong_type_matrix(label: np.ndarray, truth: np.ndarray, k: int = K) -> np.ndarray:
    """Counts of (truth class, filed fraud class) over fraud rows that were reported."""
    s = (np.asarray(truth) > 0) & (np.asarray(label) > 0)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(truth)[s], np.asarray(label)[s]), 1)
    return cm


# ------------------------------------------------------------------------------- proxy target


def counter_proxy(
    cleaned: np.ndarray,
    block: np.ndarray,
    seg_class: np.ndarray,
    conf: np.ndarray,
    min_conf: float = 0.8,
    bot_block: int = 1,
    fraud_block: int = 2,
    bot_class: int = 2,
) -> np.ndarray:
    """Cleaned labels with the label-free counter imposed: a confident fraud-block row takes its
    segment's class, a bot-pool row is velocity_bot. Every other row keeps its cleaned label."""
    out = np.asarray(cleaned).copy()
    fb = (block == fraud_block) & (seg_class >= 1) & (seg_class < K) & (conf >= min_conf)
    out[fb] = seg_class[fb]
    out[block == bot_block] = bot_class
    return out


def hidden_counts(
    target: np.ndarray, gen: np.ndarray, period: np.ndarray, total: float = HIDDEN_TOTAL
) -> dict[int, np.ndarray]:
    """Expected fraud rows per month and class that the target still files legit (the unreported
    ordinary-looking rows), shared out in proportion to the month's ordinary-looking labels."""
    ordinary = (target > 0) & (gen == 0)
    grand = max(int(ordinary.sum()), 1)
    return {
        int(m): np.bincount(target[ordinary & (period == m)], minlength=K).astype(float)
        * total
        / grand
        for m in np.unique(period)
    }


def expected_confusion(truth: np.ndarray, pred: np.ndarray, hidden=None, k: int = K) -> np.ndarray:
    """Confusion (truth x call); ``hidden[c]`` rows of class c are spread evenly over the rows the
    target files legit."""
    cm = np.bincount(truth * k + pred, minlength=k * k).reshape(k, k).astype(float)
    if hidden is None:
        return cm
    legit = cm[0].copy()
    h = np.asarray(hidden, dtype=float).copy() / max(legit.sum(), 1.0)
    h[0] = 0.0
    cm[0] -= h.sum() * legit
    cm += h[:, None] * legit[None, :]
    return cm


def macro(truth: np.ndarray, pred: np.ndarray, hidden=None) -> float:
    cm = expected_confusion(truth, pred, hidden)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0).mean())


# ------------------------------------------------------------------------------- public evidence


def switch_effect(n_moved, k_true_a, a: dict, b: dict) -> np.ndarray:
    """Macro-F1 change (12 classes) when ``n_moved`` calls go from class a to class b and
    ``k_true_a`` of them are truly a (the rest truly b). ``a`` and ``b`` hold the slice's true
    count ``T``, call count ``P`` and true positives ``TP`` before the move."""
    n, k = np.asarray(n_moved, float), np.asarray(k_true_a, float)
    fa0 = 2 * a["TP"] / (a["T"] + a["P"])
    fb0 = 2 * b["TP"] / (b["T"] + b["P"])
    fa = 2 * (a["TP"] - k) / (a["T"] + a["P"] - n)
    fb = 2 * (b["TP"] + n - k) / (b["T"] + b["P"] + n)
    return (fa - fa0 + fb - fb0) / K


def switch_likelihood(
    observed: float,
    moved: int,
    share_a,
    a: dict,
    b: dict,
    public: float = 0.3,
    draws: int = 200_000,
    tol: float = 5e-4,
    seed: int = 0,
) -> dict:
    """For each share of true-a rows among the moved rows: the mean and sd of the public change,
    the chance of a change at or below ``observed`` and the density (within ``tol``) of it."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in np.atleast_1d(share_a):
        n = rng.binomial(moved, public, draws)
        k = rng.binomial(n, float(c))
        e = switch_effect(n, k, a, b)
        out[f"{float(c):.2f}"] = {
            "mean": float(e.mean()),
            "sd": float(e.std()),
            "p_at_or_below": float((e <= observed).mean()),
            "density": float((np.abs(e - observed) < tol).mean()),
        }
    return out
