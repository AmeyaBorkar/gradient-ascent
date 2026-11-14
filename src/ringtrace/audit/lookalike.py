"""Can the sibling type inside a look-alike template be told apart? Event-level separability.

A look-alike template is one event shape written under two fraud types
(``ringtrace.evidence.anchors.TEMPLATE_PAIR``). The template rule gives every event of a template
one class, which is the best call when the partner type is drawn at random per event. This audit
turns each train event of a template into one row (size, time span, mean and spread of the row
features, share of rows on fraud-phase phones) and asks a cross-validated booster to tell the two
types apart. An AUC near 0.5 supports the rule; a clearly higher AUC means an event-level typing
model would beat it. Events with rows on fraud-phase phones are also scored apart, because the
counter segment already names their type.
"""

from __future__ import annotations

import numpy as np

from ringtrace.audit import stats


def event_table(
    event: np.ndarray, x: np.ndarray, ts: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One row per event id >= 0: size, span (seconds), then mean and standard deviation of every
    column of ``x``. Returns (event ids, matrix, column names)."""
    event = np.asarray(event, dtype=np.int64)
    keep = event >= 0
    ev, x, ts = event[keep], np.asarray(x, dtype=np.float64)[keep], np.asarray(ts)[keep]
    order = np.argsort(ev, kind="stable")
    ev, x, ts = ev[order], x[order], ts[order]
    ids, start, size = np.unique(ev, return_index=True, return_counts=True)
    xs = np.nan_to_num(x, nan=0.0)
    s1 = np.add.reduceat(xs, start, axis=0)
    s2 = np.add.reduceat(xs * xs, start, axis=0)
    mean = s1 / size[:, None]
    std = np.sqrt(np.maximum(s2 / size[:, None] - mean**2, 0.0))
    span = np.maximum.reduceat(ts, start) - np.minimum.reduceat(ts, start)
    table = np.column_stack([size, span, mean, std])
    cols = ["size", "span"] + [f"mean_{n}" for n in names] + [f"std_{n}" for n in names]
    return ids, table, cols


def event_labels(
    event: np.ndarray, y: np.ndarray, pair: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Per event id >= 0: 1 if the majority of its rows labelled with a pair class carry the
    second class, 0 for the first, -1 when no row carries either (ties go to the first)."""
    event = np.asarray(event, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    keep = event >= 0
    ids, inv = np.unique(event[keep], return_inverse=True)
    a = np.bincount(inv, weights=(y[keep] == pair[0]).astype(float), minlength=len(ids))
    b = np.bincount(inv, weights=(y[keep] == pair[1]).astype(float), minlength=len(ids))
    lab = np.where(a + b == 0, -1, (b > a).astype(np.int64))
    return ids, lab


def audit_template(
    event: np.ndarray,
    x: np.ndarray,
    ts: np.ndarray,
    names: list[str],
    y: np.ndarray,
    pair: tuple[int, int],
    fraud_phase: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    folds: int = 5,
    threads: int | None = None,
) -> dict:
    """Event-level separability of the two types of one template (train rows only)."""
    fp = np.asarray(fraud_phase, dtype=np.float64)[:, None]
    ids, table, cols = event_table(
        event, np.hstack([np.asarray(x, dtype=np.float64), fp]), ts, [*names, "fraud_phase"]
    )
    lab_ids, lab = event_labels(event, y, pair)
    assert np.array_equal(ids, lab_ids)
    ok = lab >= 0
    out: dict = {
        "events": int(ok.sum()),
        "second_type_share": float(lab[ok].mean()) if ok.any() else None,
    }
    if ok.sum() == 0:
        return out
    t, lab = table[ok], lab[ok]
    out["all"] = stats.cv_auc(t, lab, names=cols, seeds=seeds, folds=folds, threads=threads)
    off = t[:, cols.index("mean_fraud_phase")] == 0
    out["off_fraud_phase"] = stats.cv_auc(
        t[off], lab[off], names=cols, seeds=seeds, folds=folds, threads=threads
    )
    out["off_fraud_phase"]["second_type_share"] = float(lab[off].mean()) if off.any() else None
    return out
