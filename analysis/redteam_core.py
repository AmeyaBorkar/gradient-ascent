"""Pure helpers for the red-team analyses (no file access; tested on synthetic data)."""

from __future__ import annotations

import itertools

import numpy as np
from core import macro_from_cm


def gap_noise(
    y: np.ndarray,
    ours: np.ndarray,
    others: list[np.ndarray],
    n_pub: int,
    draws: int,
    seed: int = 0,
    k: int = 12,
    thresholds: tuple[float, ...] = (0.004, 0.0084),
) -> list[dict]:
    """For each other file: the full-set gap to ``ours``, the sd of (slice gap - full gap) over
    ``draws`` random slices of ``n_pub`` rows, the sd of one file's slice-minus-full score, and the
    share of slices on which the slice gap beats the full gap by each threshold."""
    rng = np.random.default_rng(seed)
    n = len(y)
    preds = [ours, *others]
    codes = [np.asarray(y) * k + np.asarray(p) for p in preds]
    full = np.array([macro_from_cm(np.bincount(c, minlength=k * k).reshape(k, k)) for c in codes])
    pub = np.zeros((draws, len(preds)))
    for d in range(draws):
        idx = rng.choice(n, n_pub, replace=False)
        for j, c in enumerate(codes):
            pub[d, j] = macro_from_cm(np.bincount(c[idx], minlength=k * k).reshape(k, k))
    dev = pub - full
    out = []
    for j in range(1, len(preds)):
        e = dev[:, j] - dev[:, 0]
        out.append(
            {
                "full_gap": float(full[j] - full[0]),
                "sd_gap_error": float(e.std()),
                "sd_file": float(dev[:, j].std()),
                "corr_with_ours": float(np.corrcoef(dev[:, j], dev[:, 0])[0, 1]),
                "p_exceed": {str(t): float((e >= t).mean()) for t in thresholds},
            }
        )
    return out


def cell_scan(
    feats: dict[str, np.ndarray],
    y_clean: np.ndarray,
    called_legit: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    order: int = 2,
    min_train: int = 20,
    min_purity: float = 0.9,
    min_excess: float = 3.0,
) -> list[dict]:
    """Cells (crosses of ``order`` discrete features) whose train rows are at least
    ``min_purity`` fraud on the cleaned labels and whose October rows are called legit more often
    than the out-of-fold rate predicts by at least ``min_excess`` rows. October enters only as
    counts per cell."""
    names = list(feats)
    fraud = (y_clean > 0).astype(np.int64)
    leg = called_legit.astype(np.int64)
    out = []
    for combo in itertools.combinations(names, order):
        code = np.zeros(len(y_clean), dtype=np.int64)
        for nm in combo:
            v = feats[nm].astype(np.int64)
            code = code * (int(v.max()) + 1) + v
        _, inv = np.unique(code, return_inverse=True)
        m = int(inv.max()) + 1
        n_tr = np.bincount(inv[tr], minlength=m)
        f_tr = np.bincount(inv[tr], weights=fraud[tr], minlength=m)
        l_tr = np.bincount(inv[tr], weights=leg[tr], minlength=m)
        n_te = np.bincount(inv[te], minlength=m)
        l_te = np.bincount(inv[te], weights=leg[te], minlength=m)
        for c in np.flatnonzero((n_tr >= min_train) & (n_te > 0)):
            purity = f_tr[c] / n_tr[c]
            if purity < min_purity:
                continue
            exp = n_te[c] * l_tr[c] / n_tr[c]
            if l_te[c] - exp < min_excess:
                continue
            first = np.flatnonzero(inv == c)[0]
            out.append(
                {
                    "cell": {nm: int(feats[nm][first]) for nm in combo},
                    "train_rows": int(n_tr[c]),
                    "train_fraud_share": float(purity),
                    "train_called_legit": int(l_tr[c]),
                    "oct_rows": int(n_te[c]),
                    "oct_called_legit": int(l_te[c]),
                    "oct_expected_legit": float(exp),
                }
            )
    out.sort(key=lambda r: -(r["oct_called_legit"] - r["oct_expected_legit"]))
    return out


def per_100k(calls: np.ndarray, mask: np.ndarray, k: int = 12) -> np.ndarray:
    """Calls of each class per 100,000 rows of ``mask``."""
    n = max(int(mask.sum()), 1)
    return np.bincount(calls[mask], minlength=k) * 1e5 / n
