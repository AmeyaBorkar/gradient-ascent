"""Pure helpers for the event-completion survey (``analysis/events.py``).

* ``partial_events``: inside each event, how many rows are called fraud; counts of events called
  whole, not at all, or in part.
* ``nearest_anchor``: for every query row, the time gap to the nearest anchor row with the same
  key (before, after, or either side) and that anchor's class.
* ``link_cells``: link volume and precision of linked rows per window.
* ``f1_ceiling``: per-class F1 of calling exactly the generator-made rows, when the ordinary rows
  of a class count as misses (with the unreported share added back, as on clean labels).
* ``evidence_share``: share of rows carrying label-free evidence, per period.
"""

from __future__ import annotations

import numpy as np

TS_BITS = 32  # timestamps (seconds) fit in 32 bits; keys are shifted above them


def partial_events(event: np.ndarray, called: np.ndarray) -> dict[str, int]:
    """``event`` holds an event id per row (-1 for none); ``called`` is True for fraud calls."""
    event = np.asarray(event)
    called = np.asarray(called, dtype=bool)
    inside = event >= 0
    if not inside.any():
        return {"events": 0, "rows": 0, "whole": 0, "none": 0, "partial": 0, "legit_in_partial": 0}
    _, g = np.unique(event[inside], return_inverse=True)
    n = np.bincount(g)
    nf = np.bincount(g, weights=called[inside].astype(np.float64)).astype(np.int64)
    part = (nf > 0) & (nf < n)
    return {
        "events": int(len(n)),
        "rows": int(inside.sum()),
        "whole": int((nf == n).sum()),
        "none": int((nf == 0).sum()),
        "partial": int(part.sum()),
        "legit_in_partial": int((n - nf)[part].sum()),
    }


def _composite(key: np.ndarray, ts: np.ndarray) -> np.ndarray:
    key = np.asarray(key, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.int64)
    if (ts < 0).any() or (ts >= 1 << TS_BITS).any():
        raise ValueError("timestamps must be non-negative seconds below 2**32")
    if (key < 0).any() or (key >= 1 << (62 - TS_BITS)).any():
        raise ValueError("keys must be non-negative and below 2**30")
    return (key << TS_BITS) | ts


def nearest_anchor(
    q_key: np.ndarray,
    q_ts: np.ndarray,
    a_key: np.ndarray,
    a_ts: np.ndarray,
    a_cls: np.ndarray,
    direction: str = "both",
) -> tuple[np.ndarray, np.ndarray]:
    """Gap in seconds (``inf`` when no anchor shares the key) and class of the nearest anchor.

    ``direction`` is ``"backward"`` (anchor at or before the query), ``"forward"`` (at or after)
    or ``"both"``. Ties between sides go to the earlier anchor."""
    if direction not in ("backward", "forward", "both"):
        raise ValueError(f"unknown direction {direction!r}")
    q_key = np.asarray(q_key, dtype=np.int64)
    q_ts = np.asarray(q_ts, dtype=np.int64)
    nq = len(q_key)
    gap = np.full(nq, np.inf)
    cls = np.full(nq, -1, dtype=np.int64)
    if len(a_key) == 0 or nq == 0:
        return gap, cls
    ac = _composite(a_key, a_ts)
    order = np.argsort(ac, kind="stable")
    ac, ak = ac[order], np.asarray(a_key, dtype=np.int64)[order]
    at, acl = np.asarray(a_ts, dtype=np.int64)[order], np.asarray(a_cls)[order]
    qc = _composite(q_key, q_ts)
    if direction in ("backward", "both"):
        i = np.searchsorted(ac, qc, side="right") - 1
        ok = (i >= 0) & (ak[np.clip(i, 0, None)] == q_key)
        ii = np.clip(i, 0, None)
        g = np.where(ok, (q_ts - at[ii]).astype(np.float64), np.inf)
        gap, cls = g, np.where(ok, acl[ii], -1)
    if direction in ("forward", "both"):
        j = np.searchsorted(ac, qc, side="left")
        jj = np.clip(j, 0, len(ac) - 1)
        ok = (j < len(ac)) & (ak[jj] == q_key)
        g = np.where(ok, (at[jj] - q_ts).astype(np.float64), np.inf)
        c = np.where(ok, acl[jj], -1)
        take = g < gap
        gap = np.where(take, g, gap)
        cls = np.where(take, c, cls)
    return gap, cls


def complete_by_link(
    calls: np.ndarray,
    period: np.ndarray,
    query_ok: np.ndarray,
    q_col: np.ndarray,
    a_col: np.ndarray,
    ts: np.ndarray,
    direction: str,
    window: float,
    q_person: np.ndarray | None = None,
    a_person: np.ndarray | None = None,
) -> np.ndarray:
    """Completion rule: inside each period, a legit-called row allowed by ``query_ok`` whose
    nearest fraud-called anchor (``q_col`` of the query equal to ``a_col`` of the anchor) lies
    within ``window`` seconds takes the anchor's class. ``q_person`` / ``a_person`` restrict the
    queries / anchors (for person-receiver keys)."""
    out = np.asarray(calls).copy()
    for p in np.unique(period):
        s = period == p
        a = s & (calls > 0)
        q = s & (calls == 0) & query_ok
        if a_person is not None:
            a &= a_person
        if q_person is not None:
            q &= q_person
        gap, acls = nearest_anchor(q_col[q], ts[q], a_col[a], ts[a], calls[a], direction)
        hit = gap <= window
        out[np.flatnonzero(q)[hit]] = acls[hit]
    return out


def link_cells(
    gap: np.ndarray,
    anchor_cls: np.ndarray,
    windows: dict[str, float],
    label: np.ndarray | None = None,
) -> dict[str, dict]:
    """Per window: linked rows, and with ``label`` the fraud share and the share whose label equals
    the anchor's class (the precision of calling the linked row the anchor's class)."""
    out = {}
    for name, w in windows.items():
        hit = gap <= w
        cell = {"linked": int(hit.sum())}
        if label is not None:
            y = np.asarray(label)[hit]
            cell["fraud"] = float((y > 0).mean()) if hit.any() else float("nan")
            cell["same"] = float((y == anchor_cls[hit]).mean()) if hit.any() else float("nan")
        out[name] = cell
    return out


def f1_ceiling(
    gen: np.ndarray, cleaned: np.ndarray, unreported: float = 0.12, legit_f1: float = 0.988
) -> dict:
    """F1 per fraud class when the calls are exactly the generator-made rows (``gen``), every one
    right, and the ordinary rows (cleaned fraud, gen legit) are missed. Clean labels also count
    the unreported part of the ordinary rows, so their count is divided by ``1 - unreported``."""
    gen = np.asarray(gen)
    cleaned = np.asarray(cleaned)
    n_cls = int(max(gen.max(), cleaned.max())) + 1
    per = {}
    for c in range(1, n_cls):
        t = int((gen == c).sum())
        o = float(((cleaned == c) & (gen == 0)).sum()) / (1.0 - unreported)
        per[c] = {
            "generator_rows": t,
            "ordinary_rows": o,
            "f1": 2 * t / (2 * t + o) if t + o > 0 else 0.0,
        }
    macro = (legit_f1 + sum(v["f1"] for v in per.values())) / n_cls
    return {"per_class": per, "macro": macro}


def evidence_share(mask: np.ndarray, evidence: np.ndarray, period: np.ndarray) -> dict:
    """Rows of ``mask`` per period and the share of them that carry ``evidence``."""
    out = {}
    for p in np.unique(period[mask]):
        s = mask & (period == p)
        out[int(p)] = {"rows": int(s.sum()), "evidence_share": float(evidence[s].mean())}
    return out
