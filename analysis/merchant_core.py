"""Pure helpers for the merchant-class October check (mule, refund, collusion): call rates per
split period, evidence bands with their call rates and train precision, the counter-zone check
of ring events, and the train-month effect of a rule. Everything returns aggregates only.

The month of a row is its split period (``period``): train rows dated 1 October are tails of
September events and belong to September, October rows dated November belong to October.
"""

from __future__ import annotations

import numpy as np

from ringtrace.scoring import macro_f1

PER_100K = 1e5


def call_rates(calls: np.ndarray, period: np.ndarray, k: int = 12) -> dict[int, list[float]]:
    """Calls of each class per 100k rows of each period: {period: [rate of class 0..k-1]}."""
    out = {}
    for p in np.unique(period):
        c = calls[period == p]
        out[int(p)] = (np.bincount(c, minlength=k)[:k] / max(len(c), 1) * PER_100K).tolist()
    return out


def band_report(
    band: np.ndarray,
    calls: np.ndarray,
    period: np.ndarray,
    cls: int,
    targets: dict[str, np.ndarray],
    train: np.ndarray,
) -> dict[str, dict]:
    """Per band: rows per 100k and calls of ``cls`` per 100k in each period, and the share of
    train calls of ``cls`` in the band that each target labels ``cls`` (train precision).
    ``targets`` hold full-length label arrays (any value on non-train rows); ``train`` marks the
    rows whose labels may be read."""
    rows = {int(p): int((period == p).sum()) for p in np.unique(period)}
    out = {}
    for b in np.unique(band):
        inb = band == b
        hit = inb & (calls == cls)
        rep = {
            "rows": {p: float((inb & (period == p)).sum() / n * PER_100K) for p, n in rows.items()},
            "calls": {
                p: float((hit & (period == p)).sum() / n * PER_100K) for p, n in rows.items()
            },
            "precision": {},
        }
        t = hit & train
        for name, y in targets.items():
            rep["precision"][name] = float((y[t] == cls).mean()) if t.any() else None
        out[str(b)] = rep
    return out


def event_sizes(event: np.ndarray) -> np.ndarray:
    """Rows of each row's event (0 for rows outside events, ``event < 0``)."""
    out = np.zeros(len(event), dtype=np.int64)
    inside = event >= 0
    if inside.any():
        _, inv, cnt = np.unique(event[inside], return_inverse=True, return_counts=True)
        out[inside] = cnt[inv]
    return out


def zone_votes(
    event: np.ndarray, fraud_block: np.ndarray, seg_class: np.ndarray, group: np.ndarray
) -> dict[str, dict[int, int]]:
    """For events holding at least one fraud-block row: how many such rows sit in each counter
    segment class, per event group (for example a size band or the event's train label)."""
    out: dict[str, dict[int, int]] = {}
    sel = (event >= 0) & fraud_block
    for g in np.unique(group[sel]):
        s = seg_class[sel & (group == g)]
        vals, cnt = np.unique(s, return_counts=True)
        out[str(g)] = {int(v): int(c) for v, c in zip(vals, cnt, strict=True)}
    return out


def rule_effect(
    calls: np.ndarray,
    new: np.ndarray,
    period: np.ndarray,
    targets: dict[str, np.ndarray],
    months: tuple[int, ...],
) -> dict[str, dict]:
    """Macro-F1 of ``new`` minus that of ``calls`` per month and target (train rows only:
    ``period`` and the targets are train-length), plus the rows changed per month."""
    out = {}
    for m in months:
        r = period == m
        rep = {"changed": int((new[r] != calls[r]).sum())}
        for name, y in targets.items():
            rep[name] = macro_f1(y[r], new[r]) - macro_f1(y[r], calls[r])
        out[str(m)] = rep
    return out


def passes(effect: dict[str, dict], must: tuple[str, ...] = ("given", "cleaned")) -> bool:
    """The upload bar: never negative on the ``must`` targets in any month, positive in one."""
    vals = [effect[m][t] for m in effect for t in must]
    return all(v >= -1e-12 for v in vals) and any(v > 1e-12 for v in vals)
