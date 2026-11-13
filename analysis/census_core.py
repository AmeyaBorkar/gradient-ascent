"""Pure helpers for ``analysis/census.py`` (no file access; tested on synthetic data).

A census unit is a label-free boolean mask over the ledger (a template, an event-family shape, a
counter zone, a mark). For each unit and class, the train rows give the out-of-fold call rate and
the cleaned-label rate; scaled by the unit's October row count they give the October calls we would
expect if the model behaved as it did out of fold. The surprise of the actual October calls is
``(actual - expected) / sqrt(expected)`` (expected floored at 1 so empty cells stay finite).
"""

from __future__ import annotations

import numpy as np

FRAUD = -1  # pseudo-class: any fraud call


def _rates(v: np.ndarray, k: int) -> np.ndarray:
    n = len(v)
    h = np.bincount(v, minlength=k).astype(float)
    return h / n if n else h


def unit_table(
    units: dict[str, np.ndarray],
    tr: np.ndarray,
    te: np.ndarray,
    oof_calls: np.ndarray,
    oct_calls: np.ndarray,
    clean: np.ndarray,
    k: int = 12,
    min_train: int = 20,
) -> list[dict]:
    """One row per (unit, class) and per (unit, any-fraud): counts, rates, expected and actual
    October calls and the surprise ``z``. ``oof_calls`` and ``clean`` are read on ``tr`` rows,
    ``oct_calls`` on ``te`` rows (all arrays are ledger-length)."""
    out = []
    for name, m in units.items():
        mtr, mte = m & tr, m & te
        n_tr, n_te = int(mtr.sum()), int(mte.sum())
        if n_tr < min_train or n_te == 0:
            continue
        oof = _rates(oof_calls[mtr], k)
        lab = _rates(clean[mtr], k)
        act = np.bincount(oct_calls[mte], minlength=k).astype(float)
        cls = list(range(k)) + [FRAUD]
        for c in cls:
            if c == FRAUD:
                r_oof, r_lab, a = oof[1:].sum(), lab[1:].sum(), act[1:].sum()
                hit = (oof_calls[mtr] > 0) & (clean[mtr] > 0)
                called = int((oof_calls[mtr] > 0).sum())
            else:
                r_oof, r_lab, a = oof[c], lab[c], act[c]
                hit = (oof_calls[mtr] == c) & (clean[mtr] == c)
                called = int((oof_calls[mtr] == c).sum())
            e = r_oof * n_te
            out.append(
                {
                    "unit": name,
                    "cls": int(c),
                    "n_train": n_tr,
                    "n_oct": n_te,
                    "oof_rate": float(r_oof),
                    "clean_rate": float(r_lab),
                    "oof_precision": float(hit.sum() / called) if called else float("nan"),
                    "expected": float(e),
                    "expected_clean": float(r_lab * n_te),
                    "actual": int(a),
                    "z": float((a - e) / np.sqrt(max(e, 1.0))),
                }
            )
    return out


def top(rows: list[dict], n: int, sign: int, min_gap: float = 3.0) -> list[dict]:
    """The ``n`` rows with the most negative (sign -1) or positive (sign +1) surprise whose
    absolute gap ``|actual - expected|`` is at least ``min_gap`` rows."""
    keep = [r for r in rows if sign * (r["actual"] - r["expected"]) >= min_gap]
    return sorted(keep, key=lambda r: -sign * r["z"])[:n]


def move(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean and median of a feature on two row sets and the standardised mean shift."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {"train_mean": None, "oct_mean": None, "shift_sd": None}
    sd = a.std() or 1.0
    return {
        "train_mean": float(a.mean()),
        "oct_mean": float(b.mean()),
        "train_median": float(np.median(a)),
        "oct_median": float(np.median(b)),
        "shift_sd": float((b.mean() - a.mean()) / sd),
    }
