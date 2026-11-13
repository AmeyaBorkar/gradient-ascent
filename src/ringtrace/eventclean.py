"""Event-cleaned training labels: label-only, read from each row's own label-free events.

Two variants, both computed from the labels of the rows that share an evidence event:

* ``clean``: a legit-labelled row inside an event whose labelled rows are at least 50 % fraud
  takes the event's class (its most frequent fraud label, ties to the lower class index);
* ``l2``: ``clean``, plus a fraud-labelled row whose type holds at most 20 % of its event's fraud
  rows, while the event's majority type has at least 4 rows, takes the event's class.

A row placed in events by several modules uses the event with the highest fraud share; the module
order breaks ties. Only the rows in ``mask`` vote and change (in a month fold: the fit months), so
a held-out month never shapes another month's training labels.
"""

from __future__ import annotations

import numpy as np

from ringtrace.classes import N_CLASSES

MIN_FRAUD_SHARE = 0.5
MINORITY_SHARE = 0.2
MINORITY_MAJOR_ROWS = 4
MODES = ("clean", "l2")


def module_view(y: np.ndarray, ev: np.ndarray, voters: np.ndarray):
    """Per row: fraud share of its event, event class, count of the row's own type in the event,
    the event's fraud rows and its majority-type rows. Rows outside an event get share -1."""
    n = len(y)
    share = np.full(n, -1.0)
    cls = np.zeros(n, dtype=np.int64)
    own = np.zeros(n, dtype=np.int64)
    nfraud = np.zeros(n, dtype=np.int64)
    top = np.zeros(n, dtype=np.int64)
    inside = voters & (ev >= 0)
    if not inside.any():
        return share, cls, own, nfraud, top
    ids, inv = np.unique(ev[inside], return_inverse=True)
    counts = np.bincount(inv * N_CLASSES + y[inside], minlength=len(ids) * N_CLASSES).reshape(
        len(ids), N_CLASSES
    )
    total = counts.sum(1)
    fraud = total - counts[:, 0]
    ev_cls = 1 + counts[:, 1:].argmax(1)  # argmax takes the lowest index on ties
    share[inside] = (fraud / total)[inv]
    cls[inside] = ev_cls[inv]
    own[inside] = counts[inv, y[inside]]
    nfraud[inside] = fraud[inv]
    top[inside] = counts[inv, ev_cls[inv]]
    return share, cls, own, nfraud, top


def clean_labels(
    y: np.ndarray, events: list[np.ndarray], mode: str = "clean", mask: np.ndarray | None = None
) -> np.ndarray:
    """Cleaned copy of ``y``; ``events`` holds one event-id array per module (-1 = none), in the
    fixed module order. Rows outside ``mask`` neither vote nor change."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    y = np.asarray(y, dtype=np.int64)
    voters = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    best = np.full(len(y), -1.0)
    cls = np.zeros(len(y), dtype=np.int64)
    own = np.zeros(len(y), dtype=np.int64)
    nfraud = np.zeros(len(y), dtype=np.int64)
    top = np.zeros(len(y), dtype=np.int64)
    for ev in events:
        s, c, o, f, t = module_view(y, np.asarray(ev, dtype=np.int64), voters)
        better = s > best  # strict: the earlier module wins a tie
        best[better], cls[better], own[better] = s[better], c[better], o[better]
        nfraud[better], top[better] = f[better], t[better]
    out = y.copy()
    legit_in = voters & (y == 0) & (best >= MIN_FRAUD_SHARE)
    out[legit_in] = cls[legit_in]
    if mode == "l2":
        minority = (
            voters
            & (y > 0)
            & (best >= 0)
            & (y != cls)
            & (own <= MINORITY_SHARE * nfraud)
            & (top >= MINORITY_MAJOR_ROWS)
        )
        out[minority] = cls[minority]
    return out


def ledger_events() -> list[np.ndarray]:
    """Event ids of every event-assigning evidence module, in the fixed module order."""
    from ringtrace import store, teacher

    return [
        store.family(f)[teacher.EVENT_COLUMNS[f]].fill_null(-1).to_numpy().astype(np.int64)
        for f in teacher.EVENT_FAMILIES
    ]
