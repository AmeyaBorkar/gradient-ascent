"""The just-under first-time PAY rule: multiples of ten under a limit, to an operator receiver."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace import baitroles, rules
from ringtrace.classes import CODE

PHISH = CODE["phishing_collect"]

# (period, ts, sender, receiver, to_person, txn_type, device, amount, new_payee, call)
ROWS = [
    # 0  account 900 receives a QR to a person in October: its only operator act
    (10, 0, 1, 900, True, "QR", 1, 500.0, 1, 7),
    # 1  account 901 receives a QR to a person in August
    (8, 0, 2, 901, True, "QR", 2, 500.0, 1, 7),
    # 2  July Rs 1,990 to 900, first time, called legit: phishing (October's act counts)
    (7, 0, 10, 900, True, "PAY", 10, 1990.0, 1, 0),
    # 3  July Rs 1,999 to 900: not a multiple of ten (first_pay's amount)
    (7, 0, 11, 900, True, "PAY", 11, 1999.0, 1, 0),
    # 4  July Rs 1,990 to 900, not a new payee
    (7, 0, 12, 900, True, "PAY", 12, 1990.0, 0, 0),
    # 5  July Rs 1,990 collect to 900
    (7, 0, 13, 900, True, "COLLECT", 13, 1990.0, 1, 0),
    # 6  July Rs 1,990 to 902, who holds no role
    (7, 0, 14, 902, True, "PAY", 14, 1990.0, 1, 0),
    # 7  July Rs 1,990 to 900, called fraud already
    (7, 0, 15, 900, True, "PAY", 15, 1990.0, 1, 3),
    # 8  October Rs 1,950 to 900: its act is in October itself
    (10, 0, 16, 900, True, "PAY", 16, 1950.0, 1, 0),
    # 9  October Rs 4,950 to 901 (an August act): phishing
    (10, 0, 17, 901, True, "PAY", 17, 4950.0, 1, 0),
    # 10 August Rs 9,990 to 901: its act is in August itself
    (8, 0, 18, 901, True, "PAY", 18, 9990.0, 1, 0),
    # 11 September Rs 24,990 to 901: phishing
    (9, 0, 19, 901, True, "PAY", 19, 24990.0, 1, 0),
]
NAMES = (
    "period",
    "ts",
    "sender",
    "receiver",
    "to_person",
    "txn_type",
    "device",
    "amount",
    "new_payee",
)


def ledger() -> tuple[pl.DataFrame, np.ndarray]:
    cols = {k: [r[i] for r in ROWS] for i, k in enumerate(NAMES)}
    led = pl.DataFrame(cols).with_columns(
        pl.col("period").cast(pl.Int8),
        pl.lit(0, dtype=pl.Int8).alias("col_role"),
        pl.lit(0, dtype=pl.Int8).alias("col_shape"),
        pl.lit(0, dtype=pl.Int8).alias("rng_shape"),
        (pl.col("period") == 10).alias("is_test"),
    )
    return led, np.array([r[-1] for r in ROWS], dtype=np.int64)


def changed(**params) -> list[int]:
    led, pred = ledger()
    out = rules.apply([{"name": "just_under_pay", **params}], pred, None, led, log=lambda *_: None)
    assert (out[out != pred] == PHISH).all()
    return np.flatnonzero(out != pred).tolist()


def test_just_under_pay_uses_all_four_months_by_default():
    assert changed() == [2, 9, 11]


def test_just_under_pay_train_month_view_misses_october_operators():
    assert changed(months=[7, 8, 9]) == [9, 11]


def test_just_under_pay_own_month_and_amounts():
    assert changed(own_month=True) == [2, 8, 9, 10, 11]
    assert changed(amounts=[1999]) == [3]


def test_just_under_amounts_are_multiples_of_ten():
    assert all(a % 10 == 0 for a in baitroles.JUST_UNDER)
    assert "just_under_pay" in rules.RULES
