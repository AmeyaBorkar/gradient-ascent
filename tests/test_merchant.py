"""Synthetic checks for the merchant-ring family (rng)."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ringtrace.evidence import merchant

M = 10_000_000  # merchant receiver offset
T0 = datetime(2025, 7, 10, 12, 0, 0)
H = 3600


def make(rows: list[dict]) -> pl.DataFrame:
    """A tiny ledger with exactly the columns the family reads (no label, y, app_version or
    ip_risk_score, so reading them would fail). ``h`` is hours after T0."""
    base = {
        "is_test": False,
        "sender": 1,
        "receiver": M + 1,
        "to_person": False,
        "txn_type": "PAY",
        "merchant_category": "grocery",
        "amount": 1000.0,
        "h": 0.0,
    }
    full = [{**base, **r} for r in rows]
    stamps = [T0 + timedelta(hours=r["h"]) for r in full]
    df = pl.DataFrame(
        {
            "row": pl.Series(range(len(full)), dtype=pl.Int32),
            "timestamp": stamps,
            "is_test": [r["is_test"] for r in full],
            "sender": pl.Series([r["sender"] for r in full], dtype=pl.Int64),
            "receiver": pl.Series([r["receiver"] for r in full], dtype=pl.Int64),
            "to_person": [r["to_person"] for r in full],
            "txn_type": [r["txn_type"] for r in full],
            "merchant_category": pl.Series([r["merchant_category"] for r in full], dtype=pl.Utf8),
            "amount": pl.Series([r["amount"] for r in full], dtype=pl.Float64),
        }
    )
    return df.with_columns(pl.col("timestamp").dt.epoch("s").alias("ts"))


def col(out: pl.DataFrame, name: str) -> list:
    return out[f"rng_{name}"].to_list()


def test_ring_sizes_name_the_shape():
    rows = []
    for k, n in enumerate((1, 3, 4, 8, 9, 12)):
        rows += [
            {
                "receiver": M + 10 + k,
                "sender": i % 2,
                "h": 2.0 * i,
                "txn_type": "QR" if i % 2 else "PAY",
            }
            for i in range(n)
        ]
    out = merchant.build(make(rows))
    sizes = col(out, "size")
    shapes = col(out, "shape")
    expect = {1: 1, 3: 1, 4: 2, 8: 2, 9: 3, 12: 3}
    for s, sh in zip(sizes, shapes, strict=True):
        assert sh == expect[s]
    assert sorted(set(sizes)) == [1, 3, 4, 8, 9, 12]
    assert set(out.columns) == {
        f"rng_{c}" for c in ("event", "shape", "size", "span_s", "senders", "pair_n", "rank")
    }
    assert all(dt.is_numeric() for dt in out.dtypes)
    assert out["rng_event"].dtype == pl.Int64 and out["rng_shape"].dtype == pl.Int8


def test_ring_chaining_24h_event_id_and_measures():
    rows = [
        {"h": 30.0, "sender": 7},  # row 0: latest in time, smallest row
        {"h": 0.0, "sender": 5},  # row 1
        {"h": 6.0, "sender": 7, "amount": 2500.0, "txn_type": "QR"},  # row 2
        {"h": 54.0, "sender": 5},  # row 3: exactly 24 h after row 0 -> chained
        {"h": 78.5, "sender": 5},  # row 4: 24.5 h gap -> new event
    ]
    out = merchant.build(make(rows))
    assert col(out, "event") == [0, 0, 0, 0, 4]
    assert col(out, "size") == [4, 4, 4, 4, 1]
    assert col(out, "shape") == [2, 2, 2, 2, 1]
    assert col(out, "span_s") == [54 * H] * 4 + [0]
    assert col(out, "senders") == [2, 2, 2, 2, 1]
    assert col(out, "pair_n") == [2, 2, 2, 2, 1]
    assert col(out, "rank") == [3, 1, 2, 4, 1]


def test_ring_near_misses():
    rows = [
        {"amount": 1500.0},  # other multiple of 500
        {"amount": 999.99},  # not a ring amount
        {"amount": 1000.0, "txn_type": "COLLECT"},  # wrong type
        {"amount": 1000.0, "txn_type": "REFUND"},
        {
            "amount": 1000.0,
            "to_person": True,
            "receiver": 42,
            "merchant_category": None,
        },  # person, not merchant
    ]
    out = merchant.build(make([{**r, "h": float(i)} for i, r in enumerate(rows)]))
    assert col(out, "shape") == [0] * 5
    assert col(out, "event") == [None] * 5


def test_ring_never_crosses_split():
    rows = [{"h": 0.0}, {"h": 1.0}, {"h": 2.0, "is_test": True}, {"h": 3.0, "is_test": True}]
    out = merchant.build(make(rows))
    assert col(out, "event") == [0, 0, 2, 2]
    assert col(out, "size") == [2, 2, 2, 2]


def test_qr_to_person_with_category():
    rows = [
        {"to_person": True, "receiver": 50, "txn_type": "QR", "amount": 432.17},
        {"to_person": True, "receiver": 50, "txn_type": "QR", "amount": 1000.0, "h": 0.1},
        {"to_person": True, "receiver": 50, "txn_type": "PAY", "h": 0.2},  # PAY: no
        {
            "to_person": True,
            "receiver": 50,
            "txn_type": "QR",
            "merchant_category": None,
            "h": 0.3,
        },  # no category: no
    ]
    out = merchant.build(make(rows))
    assert col(out, "shape") == [4, 4, 0, 0]
    assert col(out, "event") == [0, 1, None, None]
    assert col(out, "size") == [1, 1, None, None]


def test_whole_amount_collusion():
    hit = [
        {"receiver": M + 1, "sender": i % 3, "amount": 1200.0 + 100 * i, "h": 10.0 * i}
        for i in range(6)
    ]  # 6 rows, 3 senders: 6 >= 2 x 3
    spread = [
        {"receiver": M + 2, "sender": i, "amount": 1300.0, "h": 10.0 * i} for i in range(6)
    ]  # 6 rows, 6 senders
    few = [
        {"receiver": M + 3, "sender": 1, "amount": 1300.0, "h": 10.0 * i} for i in range(4)
    ]  # 4 rows only
    gap = [
        {"receiver": M + 4, "sender": 1, "amount": 1300.0, "h": 25.0 * i} for i in range(6)
    ]  # gaps > 24 h
    small = [
        {"receiver": M + 5, "sender": 1, "amount": 899.0, "h": 1.0 * i} for i in range(6)
    ]  # below 900
    paise = [
        {"receiver": M + 6, "sender": 1, "amount": 1300.5, "h": 1.0 * i} for i in range(6)
    ]  # not whole rupees
    led = make(hit + spread + few + gap + small + paise)
    out = merchant.build(led)
    shapes = col(out, "shape")
    assert shapes[:6] == [5] * 6
    assert shapes[6:] == [0] * (len(shapes) - 6)
    assert col(out, "event")[:6] == [0] * 6
    assert col(out, "senders")[:6] == [3] * 6
    assert col(out, "pair_n")[:6] == [2] * 6


def test_order_independent():
    rows = [
        {
            "receiver": M + 1 + i % 3,
            "sender": i % 4,
            "h": 7.0 * i,
            "amount": (1000.0, 1300.0, 5000.0)[i % 3],
        }
        for i in range(40)
    ]
    rows += [{"to_person": True, "receiver": 9, "txn_type": "QR", "h": 1.0}]
    led = make(rows)
    a = merchant.build(led)
    shuffled = led.sample(fraction=1.0, shuffle=True, seed=3)
    b = merchant.build(shuffled)
    b = b.with_columns(shuffled["row"].alias("row")).sort("row").drop("row")
    assert a.equals(b)
