"""Marks fire on planted rows and stay quiet on near-misses (tiny synthetic ledgers)."""

from __future__ import annotations

import polars as pl
import pytest

from ringtrace.evidence import marks

T0 = 1_720_000_000
HOUR = 3600
DAY = 86400
MERCHANT = 10_000_000

DEFAULTS = {
    "txn_type": "PAY",
    "qr_vpa_mismatch": None,
    "to_person": True,
    "merchant_category": None,
    "amount": 123.45,
    "balance_before": 50_000.0,
    "month": 7,
    "period": 7,
    "is_test": False,
    "sender": 1,
    "receiver": 2,
    "device": 1,
    "ts": 0,
}
SCHEMA = {
    "txn_type": pl.Utf8,
    "qr_vpa_mismatch": pl.Float64,
    "to_person": pl.Boolean,
    "merchant_category": pl.Utf8,
    "amount": pl.Float64,
    "balance_before": pl.Float64,
    "month": pl.Int8,
    "period": pl.Int8,
    "is_test": pl.Boolean,
    "sender": pl.Int64,
    "receiver": pl.Int64,
    "device": pl.Int64,
    "ts": pl.Int64,
}


def ledger(rows: list[dict]) -> pl.DataFrame:
    recs = [{**DEFAULTS, **r} for r in rows]
    for r in recs:  # a merchant receiver unless told otherwise
        if not r["to_person"] and r["receiver"] < MERCHANT:
            r["receiver"] += MERCHANT
    df = pl.DataFrame(recs, schema=SCHEMA)
    return df.with_row_index("row").with_columns(
        pl.col("row").cast(pl.Int32), (pl.col("ts") + T0).alias("ts")
    )


def marks_of(rows: list[dict]) -> pl.DataFrame:
    led = ledger(rows)
    out = marks.build(led)
    assert out.height == led.height
    assert all(c.startswith("mrk_") for c in out.columns)
    assert all(dt.is_numeric() for dt in out.dtypes)
    return out


def test_odd_original_and_corrected():
    out = marks_of(
        [
            {"txn_type": "QR", "to_person": False},  # QR merchant, no check
            {"txn_type": "QR", "to_person": True},  # QR person, no check
            {"txn_type": "PAY", "qr_vpa_mismatch": 0.0},  # non-QR with a value
            {"txn_type": "QR", "to_person": False, "qr_vpa_mismatch": 0.0},  # normal QR
            {"txn_type": "PAY"},  # normal PAY
            {"txn_type": "COLLECT", "qr_vpa_mismatch": float("nan")},  # NaN = empty
        ]
    )
    assert out["mrk_odd"].to_list() == [1, 1, 1, 0, 0, 0]
    assert out["mrk_odd_v2"].to_list() == [1, 0, 1, 0, 0, 0]


def test_qr_person_category_marks():
    out = marks_of(
        [
            {
                "txn_type": "QR",
                "to_person": True,
                "merchant_category": "grocery",
                "qr_vpa_mismatch": 0.0,
            },
            {"txn_type": "QR", "to_person": True, "qr_vpa_mismatch": 0.0},
            {
                "txn_type": "QR",
                "to_person": False,
                "merchant_category": "grocery",
                "qr_vpa_mismatch": 0.0,
            },
            {"txn_type": "PAY", "to_person": True},
        ]
    )
    assert out["mrk_p2p_category"].to_list() == [1, 0, 0, 0]
    assert out["mrk_qr_p2p_nocat"].to_list() == [0, 1, 0, 0]
    assert out["mrk_any"].to_list() == [1, 1, 0, 0]


def test_amount_marks():
    out = marks_of(
        [
            {"to_person": False, "amount": 2500.0},
            {"to_person": True, "amount": 2500.0},
            {"to_person": False, "amount": 2500.5},
            {"amount": 4999.0},
            {"amount": 4998.0},
            {"txn_type": "PAY", "amount": 5.0},
            {"txn_type": "PAY", "amount": 3.0},
            {"txn_type": "PAY", "to_person": False, "amount": 5.0},
            {"txn_type": "COLLECT", "amount": 5.0},
        ]
    )
    assert out["mrk_round5"].to_list() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert out["mrk_classic12"].to_list() == [0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert out["mrk_bait"].to_list() == [0, 0, 0, 0, 0, 1, 0, 0, 0]
    # flags, not certificates
    assert out["mrk_any"].sum() == 0


def test_balance_marks():
    out = marks_of(
        [
            {"amount": 1000.0, "balance_before": 1100.0},
            {"amount": 1000.0, "balance_before": 1100.01},  # one cent: still a mark
            {"amount": 1000.0, "balance_before": 1100.02},  # two cents: not
            {"amount": 1234.57, "balance_before": 1358.03},  # round(1.1 x 1234.57, 2) = 1358.03
            {"amount": 1000.0, "balance_before": 1020.0},
            {"amount": 500.0, "balance_before": 499.99},
            {"amount": 500.0, "balance_before": 500.0},
        ]
    )
    assert out["mrk_bal110"].to_list() == [1, 1, 0, 1, 0, 0, 0]
    assert out["mrk_bal102"].to_list() == [0, 0, 0, 0, 1, 0, 0]
    assert out["mrk_over_balance"].to_list() == [0, 0, 0, 0, 0, 1, 0]


def test_spill():
    out = marks_of(
        [
            {"month": 9, "period": 9},
            {"month": 10, "period": 9},
            {"month": 10, "period": 10, "is_test": True},
            {"month": 11, "period": 10, "is_test": True},
        ]
    )
    assert out["mrk_spill"].to_list() == [0, 1, 0, 1]


@pytest.mark.parametrize("c, col", [(0.25, "mrk_step025"), (0.35, "mrk_step035")])
def test_balance_step_to_next_row(c, col):
    b = 8000.0
    out = marks_of(
        [
            # planted: sender 10 steps exactly within 3 h
            {"sender": 10, "ts": 0, "balance_before": b},
            {"sender": 10, "ts": 1800, "balance_before": round(c * b, 2)},
            # one cent off
            {"sender": 11, "ts": 0, "balance_before": b},
            {"sender": 11, "ts": 1800, "balance_before": round(c * b, 2) + 0.01},
            # too slow
            {"sender": 12, "ts": 0, "balance_before": b},
            {"sender": 12, "ts": 3 * HOUR + 1, "balance_before": round(c * b, 2)},
            # next-but-one only (an unrelated row in between)
            {"sender": 13, "ts": 0, "balance_before": b},
            {"sender": 13, "ts": 600, "balance_before": 7777.0},
            {"sender": 13, "ts": 1200, "balance_before": round(c * b, 2)},
            # across the split
            {"sender": 14, "ts": 0, "balance_before": b},
            {
                "sender": 14,
                "ts": 600,
                "balance_before": round(c * b, 2),
                "is_test": True,
                "month": 10,
                "period": 10,
            },
            # different senders
            {"sender": 15, "ts": 0, "balance_before": b},
            {"sender": 16, "ts": 600, "balance_before": round(c * b, 2)},
        ]
    )
    assert out[col].to_list() == [1, 1] + [0] * 11
    other = "mrk_step035" if col == "mrk_step025" else "mrk_step025"
    assert out[other].sum() == 0
    assert out["mrk_count"].to_list()[:2] == [1, 1]


def test_neighbour_counts():
    odd = {"txn_type": "PAY", "qr_vpa_mismatch": 1.0}
    out = marks_of(
        [
            {"sender": 1, "device": 1, "receiver": 7, "ts": 0, **odd},
            {"sender": 1, "device": 1, "receiver": 8, "ts": DAY},  # window edge: inside
            {"sender": 1, "device": 1, "receiver": 9, "ts": DAY + 1},  # 24 h + 1 s from row 0
            {
                "sender": 1,
                "device": 1,
                "receiver": 9,
                "ts": 600,
                "is_test": True,
                "month": 10,
                "period": 10,
                **odd,
            },  # other split
            {"sender": 2, "device": 2, "receiver": 7, "ts": 3 * DAY, **odd},  # person: 72 h
            {"sender": 3, "device": 3, "receiver": 7, "ts": 3 * DAY + 1},
            {"sender": 4, "device": 4, "receiver": 50, "to_person": False, "ts": 0, **odd},
            {"sender": 5, "device": 5, "receiver": 50, "to_person": False, "ts": DAY + 1},
        ]
    )
    assert out["mrk_snd_n"].to_list()[:4] == [1, 2, 1, 0]
    assert out["mrk_snd_odd_n"].to_list()[:4] == [0, 1, 0, 0]
    assert out["mrk_dev_odd_n"].to_list()[:4] == [0, 1, 0, 0]
    assert out["mrk_snd_any_n"].to_list()[:4] == [0, 1, 0, 0]
    # person receiver 7: rows at 0 and 3 d are 72 h apart (inside), 3 d + 1 s is not from 0
    assert out["mrk_rcv_n"].to_list()[0] == 1
    assert out["mrk_rcv_odd_n"].to_list()[0] == 1
    assert out["mrk_rcv_n"].to_list()[4:6] == [2, 1]
    assert out["mrk_rcv_odd_n"].to_list()[4:6] == [1, 1]
    # merchant receiver: 24 h window
    assert out["mrk_rcv_n"].to_list()[6:8] == [0, 0]


def test_order_independent():
    rows = [
        {"sender": 10, "ts": 0, "balance_before": 8000.0, "qr_vpa_mismatch": 1.0},
        {"sender": 10, "ts": 1800, "balance_before": 2000.0},
        {"sender": 10, "ts": 1800, "balance_before": 9000.0},
        {"sender": 11, "ts": 50, "device": 10, "amount": 1999.0},
    ]
    led = ledger(rows)
    ref = pl.concat([led.select("row"), marks.build(led)], how="horizontal")
    shuffled = led.sample(fraction=1.0, shuffle=True, seed=3)
    got = pl.concat([shuffled.select("row"), marks.build(shuffled)], how="horizontal")
    assert got.sort("row").equals(ref)
