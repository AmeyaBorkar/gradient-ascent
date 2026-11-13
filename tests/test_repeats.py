"""Synthetic checks for the repeat-offender family (rep)."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ringtrace.evidence import repeats

M = 10_000_000  # merchant receiver offset
D0 = datetime(2025, 8, 4)


def make(rows: list[dict]) -> pl.DataFrame:
    """A tiny ledger with exactly the columns the family reads (no label, y, app_version or
    ip_risk_score). ``d`` is the day offset from D0 and ``hh`` the hour of day (fractional)."""
    base = {"is_test": False, "sender": 1, "receiver": M + 1, "to_person": False,
            "txn_type": "REFUND", "merchant_category": "ecommerce", "amount": 2345.67,
            "balance_before": 900.0, "new_payee": 0, "d": 0, "hh": 13.0}
    full = [{**base, **r} for r in rows]
    stamps = [D0 + timedelta(days=r["d"], hours=r["hh"]) for r in full]
    df = pl.DataFrame({
        "row": pl.Series(range(len(full)), dtype=pl.Int32),
        "timestamp": stamps,
        "is_test": [r["is_test"] for r in full],
        "sender": pl.Series([r["sender"] for r in full], dtype=pl.Int64),
        "receiver": pl.Series([r["receiver"] for r in full], dtype=pl.Int64),
        "to_person": [r["to_person"] for r in full],
        "txn_type": [r["txn_type"] for r in full],
        "merchant_category": pl.Series([r["merchant_category"] for r in full], dtype=pl.Utf8),
        "amount": pl.Series([r["amount"] for r in full], dtype=pl.Float64),
        "balance_before": pl.Series([r["balance_before"] for r in full], dtype=pl.Float64),
        "new_payee": pl.Series([r["new_payee"] for r in full], dtype=pl.Int8),
    })
    return df.with_columns(pl.col("timestamp").dt.epoch("s").alias("ts"))


def col(out: pl.DataFrame, name: str) -> list:
    return out[f"rep_{name}"].to_list()


def lend(**kw) -> dict:
    return {"txn_type": "PAY", "merchant_category": "lending", "amount": 5000.0, "hh": 21.5,
            "new_payee": 1, **kw}


def p2p(**kw) -> dict:
    return {"txn_type": "PAY", "to_person": True, "merchant_category": None, "amount": 2345.0,
            "hh": 22.0, "new_payee": 1, "receiver": 77, **kw}


def test_refund_event():
    rows = [
        {"d": 6, "receiver": M + 2},                                     # row 0, last in time
        {"d": 0},                                                        # row 1
        {"d": 3, "merchant_category": "electronics", "receiver": M + 3},  # row 2
        {"d": 1, "sender": 2, "merchant_category": "grocery"},          # wrong category
        {"d": 1, "sender": 3, "txn_type": "PAY"},                       # not a refund
    ]
    out = repeats.build(make(rows))
    assert col(out, "shape") == [1, 1, 1, 0, 0]
    assert col(out, "kind") == [1, 1, 1, 0, 0]
    assert col(out, "event") == [0, 0, 0, None, None]
    assert col(out, "size") == [3, 3, 3, None, None]
    assert col(out, "receivers")[:3] == [3, 3, 3]
    assert col(out, "days")[:3] == [3, 3, 3]
    assert col(out, "rank")[:3] == [3, 1, 2]
    assert col(out, "span_s")[0] == 6 * 86400
    assert set(out.columns) == {f"rep_{c}" for c in ("event", "shape", "size", "span_s",
                                "receivers", "days", "rank", "kind", "chain_n")}
    assert all(dt.is_numeric() for dt in out.dtypes)


def test_refund_near_misses():
    two = [{"sender": 1, "d": 0}, {"sender": 1, "d": 2}]                     # only 2 rows
    gapped = [{"sender": 2, "d": 0}, {"sender": 2, "d": 1}, {"sender": 2, "d": 9}]  # > 7 days
    split = [{"sender": 3, "d": 0}, {"sender": 3, "d": 1}, {"sender": 3, "d": 2, "is_test": True}]
    out = repeats.build(make(two + gapped + split))
    assert col(out, "shape") == [0] * 8
    assert col(out, "kind") == [1] * 8
    assert col(out, "chain_n") == [2, 2, 2, 2, 1, 2, 2, 1]


def test_refund_gap_exactly_seven_days_chains():
    rows = [{"d": 0}, {"d": 7}, {"d": 14}]
    out = repeats.build(make(rows))
    assert col(out, "shape") == [1, 1, 1]


def test_lending_event_and_near_misses():
    hit = [lend(sender=1, d=i, receiver=M + 10 + i) for i in range(4)]
    early = [lend(sender=2, d=0), lend(sender=2, d=1), lend(sender=2, d=2, hh=19.99)]
    paise = [lend(sender=3, d=0), lend(sender=3, d=1), lend(sender=3, d=2, amount=5000.5)]
    other = [lend(sender=4, d=i, merchant_category="ecommerce") for i in range(3)]
    mixed = [lend(sender=5, d=0), lend(sender=5, d=1), {"sender": 5, "d": 2}]  # 2 + refund
    out = repeats.build(make(hit + early + paise + other + mixed))
    assert col(out, "shape") == [2] * 4 + [0] * 12
    assert col(out, "receivers")[:4] == [4] * 4
    assert col(out, "kind")[4:] == [2, 2, 0, 2, 2, 0, 0, 0, 0, 2, 2, 1]
    assert col(out, "chain_n")[13:] == [2, 2, 1]  # L2: a lending chain of two


def test_lending_to_person():
    hit = [p2p(sender=1, d=i) for i in range(3)]                        # 3 evenings
    evening = [p2p(sender=2, d=0, hh=20.0 + 0.5 * i) for i in range(3)]  # one evening only
    tens = [p2p(sender=3, d=i, amount=2340.0) for i in range(3)]        # multiple of 10
    law = [p2p(sender=4, d=i, amount=2340.0, balance_before=2574.0) for i in range(3)]  # 1.1x
    known = [p2p(sender=5, d=i, new_payee=0) for i in range(3)]          # not a new payee
    big = [p2p(sender=6, d=i, amount=15001.0) for i in range(3)]         # above 15,000
    out = repeats.build(make(hit + evening + tens + law + known + big))
    shapes = col(out, "shape")
    assert shapes == [3] * 3 + [0] * 3 + [0] * 3 + [3] * 3 + [0] * 6
    assert col(out, "kind") == [3] * 6 + [0] * 3 + [3] * 3 + [0] * 6
    assert col(out, "chain_n")[3:6] == [3, 3, 3]
    assert col(out, "days")[:3] == [3, 3, 3]
    assert col(out, "event")[9:12] == [9, 9, 9]


def test_order_independent():
    rows = [{"sender": i % 3, "d": i % 9, "receiver": M + i % 4} for i in range(20)]
    rows += [lend(sender=9, d=i) for i in range(5)] + [p2p(sender=8, d=i) for i in range(4)]
    led = make(rows)
    a = repeats.build(led)
    shuffled = led.sample(fraction=1.0, shuffle=True, seed=11)
    b = repeats.build(shuffled)
    b = b.with_columns(shuffled["row"].alias("row")).sort("row").drop("row")
    assert a.equals(b)
