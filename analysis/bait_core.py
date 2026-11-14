"""Pure helpers for ``analysis/bait.py`` (no file access; tested on synthetic data).

A rule that moves legit calls into one class ``k`` gains on class ``k`` when the moved rows are
right more often than half of ``k``'s F1 (adding a row with chance ``p`` of being right raises
F1 = 2 TP / (T + P) exactly when ``p > F1 / 2``), and its effect on legit is negligible. So every
move is reported with its precision next to that break-even, per month and per cell.
"""

from __future__ import annotations

import numpy as np

from ringtrace.scoring import macro_f1, per_class_f1


def move_table(truth: np.ndarray, before: np.ndarray, after: np.ndarray, cls: int) -> dict:
    """Rows moved into ``cls``, how many are ``cls`` / any fraud in ``truth``, the precision and
    the break-even (half the base F1 of ``cls``), and the macro-F1 change."""
    truth, before, after = (np.asarray(a) for a in (truth, before, after))
    moved = (before != after) & (after == cls)
    n = int(moved.sum())
    tp = int((truth[moved] == cls).sum())
    f1 = float(per_class_f1(truth, before)[cls])
    return {
        "moved": n,
        "right": tp,
        "fraud": int((truth[moved] > 0).sum()),
        "precision": tp / n if n else None,
        "break_even": f1 / 2,
        "gain": macro_f1(truth, after) - macro_f1(truth, before),
    }


def by_month(
    targets: dict[str, np.ndarray],
    before: np.ndarray,
    after: np.ndarray,
    period: np.ndarray,
    cls: int,
    months=(7, 8, 9),
) -> dict:
    """``move_table`` per target and month (rows with a target below 0 are left out)."""
    out = {}
    for t, y in targets.items():
        rows = []
        for m in months:
            r = (period == m) & (y >= 0)
            rows.append(move_table(y[r], before[r], after[r], cls))
        out[t] = rows
    return out


def passes_months(table: dict, targets=("given", "cleaned"), tol: float = 1e-12) -> bool:
    """Non-negative on every month of ``targets`` and positive in at least one."""
    g = [row["gain"] for t in targets for row in table[t]]
    return all(v >= -tol for v in g) and any(v > tol for v in g)


def other_month_label_acts(acct: np.ndarray, period: np.ndarray, fraud: np.ndarray):
    """(acct, period) pairs of fraud-labelled rows, the label-history control's acts."""
    a = np.asarray(acct)[np.asarray(fraud)]
    p = np.asarray(period)[np.asarray(fraud)]
    key = np.unique(np.stack([a.astype(np.int64), p.astype(np.int64)], axis=1), axis=0)
    return key[:, 0], key[:, 1]
