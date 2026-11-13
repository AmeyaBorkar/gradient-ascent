"""Row anchors: the class a label-free family shape or mark points to, and the look-alike templates.

``row_anchor`` is the ``template_pred`` of a rules-only detector. Every row takes the class of the
first family (in the order bursts, drains, flows, requests, merchant, repeats) that put it in an
event of a near-single-class shape; rows in no such event fall back to a few one-class marks.
Look-alike shapes take their majority class (drain 1 -> sim_swap, bait answered within 4 h ->
phishing_collect, ...). 0 means "no anchor".

``template_of`` says which of the six look-alike templates a row belongs to (1 chain hop,
2 fresh-sim 0.25 drain, 3 mule burst, 4 bait answered within 4 h, 5 just-under collect, 6 small
merchant round; 0 none) and ``template_event`` gives that family's event id. The template's pair
and the class the plain rule assigns are fixed here so that the counter and the rules read one
table.

Every function accepts a frame that already carries the family columns (synthetic tests) or the
ledger, in which case the families are read from the feature store.
"""

from __future__ import annotations

import numpy as np
import polars as pl

DEPENDS_ON = (
    "ringtrace.evidence.bursts",
    "ringtrace.evidence.drains",
    "ringtrace.evidence.flows",
    "ringtrace.evidence.requests",
    "ringtrace.evidence.merchant",
    "ringtrace.evidence.repeats",
    "ringtrace.evidence.marks",
)

# family prefix -> store name, in anchor priority order
FAMILY_OF = {
    "bot": "bursts",
    "drn": "drains",
    "mf": "flows",
    "col": "requests",
    "rng": "merchant",
    "rep": "repeats",
}

MULE, BOT, CHAIN, SIM, ATO, PHISH, QR, REFUND, CASHB, LOAN, COLLU = range(1, 12)

# shape -> class code (spec-events s2-s8 shape purities; look-alikes take the majority class)
SHAPE_CLASS: dict[str, dict[int, int]] = {
    "bot": {1: BOT, 2: BOT, 3: BOT, 4: BOT},
    "drn": {1: SIM, 2: SIM, 3: ATO, 4: ATO},
    "mf": {1: CHAIN, 2: MULE, 3: MULE, 4: CHAIN},
    # 5 = N1X (single unanswered 10-rupee bait, 93 % legit): no anchor
    "col": {1: PHISH, 2: PHISH, 3: PHISH, 4: PHISH, 6: PHISH, 7: PHISH, 8: CASHB, 9: CASHB},
    "rng": {1: QR, 2: REFUND, 3: COLLU, 4: QR, 5: COLLU},
    "rep": {1: REFUND, 2: LOAN, 3: LOAN},
}

# store name of every prefix a caller may ask ``family_columns`` for
STORE_NAME = {**FAMILY_OF, "mrk": "marks", "ctr": "phones", "cpr": "phones_prior"}

# one-class marks, used only on rows that no family shape anchors
MARK_CLASS = (
    ("mrk_p2p_category", QR),
    ("mrk_qr_p2p_nocat", PHISH),
    ("mrk_bal102", MULE),
    ("mrk_step025", SIM),
    ("mrk_step035", ATO),
)

# template number -> (family prefix, shapes); numbering of spec-counter 4b
TEMPLATES: dict[int, tuple[str, tuple[int, ...]]] = {
    1: ("mf", (1,)),
    2: ("drn", (1,)),
    3: ("mf", (2, 3)),
    4: ("col", (7,)),
    5: ("col", (1,)),
    6: ("rng", (1,)),
}
TEMPLATE_NAMES = {1: "mf1", 2: "drain1", 3: "mf2+mf3", 4: "col7", 5: "col1", 6: "rng1"}
# the two classes a template's events are written as, and the class the plain rule gives them
TEMPLATE_PAIR = {
    1: (CHAIN, MULE),
    2: (SIM, ATO),
    3: (MULE, BOT),
    4: (CASHB, PHISH),
    5: (PHISH, CASHB),
    6: (QR, REFUND),
}
TEMPLATE_CLASS = {1: CHAIN, 2: SIM, 3: MULE, 4: CASHB, 5: PHISH, 6: QR}


def family_columns(led: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """The requested family columns, from ``led`` when present, else from the feature store."""
    got = {c: led[c] for c in columns if c in led.columns}
    missing = [c for c in columns if c not in got]
    if missing:
        from ringtrace import store

        for name in dict.fromkeys(STORE_NAME[c.split("_", 1)[0]] for c in missing):
            fam = store.family(name)
            if fam.height != led.height:
                raise ValueError(f"family {name} has {fam.height} rows, frame has {led.height}")
            got.update({c: fam[c] for c in missing if c in fam.columns})
    still = [c for c in columns if c not in got]
    if still:
        raise KeyError(f"family columns not found: {still}")
    return pl.DataFrame([got[c] for c in columns])


def _shape(cols: pl.DataFrame, prefix: str) -> np.ndarray:
    return cols[f"{prefix}_shape"].fill_null(0).to_numpy().astype(np.int64)


def row_anchor(led: pl.DataFrame) -> np.ndarray:
    """Class code (1..11) implied by the row's first near-single-class shape or mark, else 0."""
    shape_cols = [f"{p}_shape" for p in SHAPE_CLASS]
    mark_cols = [c for c, _ in MARK_CLASS]
    cols = family_columns(led, shape_cols + mark_cols)
    anchor = np.zeros(led.height, dtype=np.int8)
    for prefix, table in SHAPE_CLASS.items():
        shape = _shape(cols, prefix)
        lut = np.zeros(max(max(table), int(shape.max(initial=0))) + 1, dtype=np.int8)
        for s, c in table.items():
            lut[s] = c
        free = anchor == 0
        anchor[free] = lut[shape[free]]
    for col, c in MARK_CLASS:
        hit = (anchor == 0) & (cols[col].fill_null(0).to_numpy() > 0)
        anchor[hit] = c
    return anchor


def templates(led: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(template_of, template_event)`` in one pass."""
    prefixes = list(dict.fromkeys(p for p, _ in TEMPLATES.values()))
    cols = family_columns(led, [f"{p}_{k}" for p in prefixes for k in ("shape", "event")])
    tmpl = np.zeros(led.height, dtype=np.int8)
    event = np.full(led.height, -1, dtype=np.int64)
    # family priority as in row_anchor: a row is claimed by the first family whose template holds it
    order = [p for p in FAMILY_OF if p in prefixes]
    for prefix in order:
        shape = _shape(cols, prefix)
        ev = cols[f"{prefix}_event"].fill_null(-1).to_numpy().astype(np.int64)
        for t, (p, shapes) in TEMPLATES.items():
            if p != prefix:
                continue
            hit = (tmpl == 0) & np.isin(shape, shapes) & (ev >= 0)
            tmpl[hit] = t
            event[hit] = ev[hit]
    return tmpl, event


def template_of(led: pl.DataFrame) -> np.ndarray:
    """Look-alike template number of each row (1..6, spec-counter 4b order), 0 for none."""
    return templates(led)[0]


def template_event(led: pl.DataFrame) -> np.ndarray:
    """Event id (smallest row) of the row's template event in its own family, -1 for none."""
    return templates(led)[1]
