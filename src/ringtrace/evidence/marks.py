"""Row marks: single-row generator certificates, and how many marked rows sit around each row.

A mark is a fact about one row that ordinary traffic almost never shows (a stale QR-check field,
a balance that is exactly 1.1 x the amount, an exact balance step to the sender's next row ...).
Marks carry no events. The neighbour counts say how many other marked rows the same sender,
receiver and phone have close in time; windows never reach across the train/test split.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.windows import Timeline

PREFIX = "mrk"

HOUR = 3600
DAY = 86400

ROUND5 = (1000, 2000, 2500, 5000, 10000)
CLASSIC12 = (1950, 1990, 1999, 4950, 4990, 4999, 9950, 9990, 9999, 24950, 24990, 24999)
BAIT = (1, 2, 5, 10)
STEPS = {"step025": 0.25, "step035": 0.35}
STEP_GAP_S = 3 * HOUR
CERTIFICATES = ("odd", "p2p_category", "qr_p2p_nocat", "bal110", "bal102", "over_balance",
                "step025", "step035")
# neighbour windows (seconds either side, inclusive)
SENDER_W = DAY
RECEIVER_PERSON_W = 3 * DAY
RECEIVER_MERCHANT_W = DAY
DEVICE_W = DAY


def _cents(values: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64) * 100).astype(np.int64)


def _balance_is(amount_c: np.ndarray, balance_c: np.ndarray, factor: float) -> np.ndarray:
    """balance_before equals round(factor x amount, 2) to within one cent."""
    return np.abs(balance_c - np.round(amount_c * factor)) <= 1


def row_marks(led: pl.DataFrame) -> dict[str, np.ndarray]:
    """Marks readable from the row alone (boolean arrays in ledger order)."""
    qr = (led["txn_type"] == "QR").to_numpy()
    pay = (led["txn_type"] == "PAY").to_numpy()
    check = led["qr_vpa_mismatch"].fill_nan(None).is_not_null().to_numpy()
    person = led["to_person"].to_numpy()
    category = led["merchant_category"].is_not_null().to_numpy()
    amount_c = _cents(led["amount"].to_numpy())
    balance_c = _cents(led["balance_before"].to_numpy())

    m: dict[str, np.ndarray] = {}
    # Original rule (#29): the check is filled iff the row is QR. Corrected (#37): an empty
    # check on a QR row only counts when the receiver is a merchant (October blanks QR-to-person).
    m["odd"] = (qr & ~check) | (~qr & check)
    m["odd_v2"] = (qr & ~person & ~check) | (~qr & check)
    m["p2p_category"] = person & category
    m["qr_p2p_nocat"] = qr & person & ~category
    m["round5"] = ~person & np.isin(amount_c, np.array(ROUND5) * 100)
    m["classic12"] = np.isin(amount_c, np.array(CLASSIC12) * 100)
    m["bait"] = person & pay & np.isin(amount_c, np.array(BAIT) * 100)
    m["bal110"] = _balance_is(amount_c, balance_c, 1.1)
    m["bal102"] = _balance_is(amount_c, balance_c, 1.02)
    m["over_balance"] = amount_c > balance_c
    m["spill"] = (led["month"] > led["period"]).to_numpy()
    return m


def step_marks(led: pl.DataFrame) -> dict[str, np.ndarray]:
    """Both rows of an exact balance step to the sender's very next row within 3 h.

    Exact means equal to the cent: round(next balance, 2) == round(c x balance, 2). (The drain
    family links with a 0.0075 tolerance and to the next-but-one row as well.)"""
    n = led.height
    split = led["is_test"].to_numpy().astype(np.int64)
    sender = led["sender"].to_numpy().astype(np.int64)
    ts = led["ts"].to_numpy().astype(np.int64)
    row = led["row"].to_numpy().astype(np.int64)
    bal = led["balance_before"].to_numpy().astype(np.float64)
    order = np.lexsort((row, ts, sender, split))
    grp = (sender * 2 + split)[order]
    t = ts[order]
    b = bal[order]
    b_c = np.round(b * 100)
    nxt = (grp[1:] == grp[:-1]) & (t[1:] - t[:-1] <= STEP_GAP_S)
    out: dict[str, np.ndarray] = {}
    for name, c in STEPS.items():
        hit = nxt & (b_c[1:] == np.round(c * b[:-1] * 100))
        mark = np.zeros(n, dtype=bool)
        mark[:-1] |= hit
        mark[1:] |= hit
        res = np.empty(n, dtype=bool)
        res[order] = mark
        out[name] = res
    return out


def _neighbours(key: np.ndarray, ts: np.ndarray, width: np.ndarray | int,
                flags: list[np.ndarray]) -> list[np.ndarray]:
    """Other rows of the same key within [t - w, t + w]: all rows, then each flag's rows."""
    tl = Timeline(key, ts)
    if np.isscalar(width):
        lo, hi = tl.bounds(int(width), int(width))
    else:
        w = np.asarray(width)[tl.order]
        lo_a, hi_a = tl.bounds(int(w.max()), int(w.max()))
        lo_b, hi_b = tl.bounds(int(w.min()), int(w.min()))
        wide = w == w.max()
        lo, hi = np.where(wide, lo_a, lo_b), np.where(wide, hi_a, hi_b)
    res = [hi - lo - 1]
    for f in flags:
        fs = f[tl.order].astype(np.int64)
        cs = np.concatenate([[0], np.cumsum(fs)])
        res.append(cs[hi] - cs[lo] - fs)
    return [r[tl.rank].astype(np.int32) for r in res]


def build(led: pl.DataFrame) -> pl.DataFrame:
    marks = row_marks(led)
    marks.update(step_marks(led))
    count = np.sum([marks[c] for c in CERTIFICATES], axis=0)
    marks["count"] = count
    marks["any"] = count > 0

    split = led["is_test"].to_numpy().astype(np.int64)
    ts = led["ts"].to_numpy().astype(np.int64)
    person = led["to_person"].to_numpy()
    flags = [marks["odd"], marks["any"]]
    cols: dict[str, pl.Series] = {}
    for name, values in marks.items():
        cols[f"{PREFIX}_{name}"] = pl.Series(np.asarray(values, dtype=np.int8))
    rcv_w = np.where(person, RECEIVER_PERSON_W, RECEIVER_MERCHANT_W)
    for ent, col, width in (("snd", "sender", SENDER_W), ("rcv", "receiver", rcv_w),
                            ("dev", "device", DEVICE_W)):
        key = led[col].to_numpy().astype(np.int64) * 2 + split
        n_all, n_odd, n_any = _neighbours(key, ts, width, flags)
        cols[f"{PREFIX}_{ent}_n"] = pl.Series(n_all)
        cols[f"{PREFIX}_{ent}_odd_n"] = pl.Series(n_odd)
        cols[f"{PREFIX}_{ent}_any_n"] = pl.Series(n_any)
    return pl.DataFrame(cols)
