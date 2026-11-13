"""Synthetic checks of the collect family: just-under requests, baits, answers and near-misses."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import requests

H = 3600
MERCHANT = 10_000_000


def ledger(rows: list[dict]) -> pl.DataFrame:
    """A tiny ledger with every column the family reads; ``row`` follows list order."""
    full = []
    for k, r in enumerate(rows):
        x = {"is_test": False, "txn_type": "PAY", "new_payee": 1, "balance_before": 1e6, "ts": 0}
        x.update(r)
        x["row"] = k
        x["to_person"] = x["receiver"] < MERCHANT
        full.append(x)
    return pl.DataFrame(full).select(
        pl.col("row").cast(pl.Int32),
        pl.col("is_test").cast(pl.Boolean),
        pl.col("sender").cast(pl.Int64),
        pl.col("receiver").cast(pl.Int64),
        pl.col("to_person").cast(pl.Boolean),
        pl.col("txn_type").cast(pl.Utf8),
        pl.col("ts").cast(pl.Int64),
        pl.col("amount").cast(pl.Float64),
        pl.col("new_payee").cast(pl.Int8),
        pl.col("balance_before").cast(pl.Float64),
    )


def col(df: pl.DataFrame, name: str) -> list:
    return df[f"col_{name}"].to_list()


def test_request_kinds_and_near_misses():
    rows = [
        {"sender": 1, "receiver": 101, "txn_type": "COLLECT", "amount": 4990},  # 0 U
        {
            "sender": 1,
            "receiver": 102,
            "txn_type": "COLLECT",
            "amount": 24950,
            "ts": 40 * 86400,
        },  # 1 U
        {"sender": 2, "receiver": 103, "txn_type": "COLLECT", "amount": 4980},  # 2 no
        {
            "sender": 3,
            "receiver": 104,
            "txn_type": "COLLECT",
            "amount": 1999,
            "new_payee": 0,
        },  # 3 no
        {"sender": 4, "receiver": MERCHANT + 5, "txn_type": "COLLECT", "amount": 1999},  # 4 no
        {"sender": 5, "receiver": 105, "txn_type": "PAY", "amount": 9999},  # 5 UO
        {"sender": 6, "receiver": 106, "txn_type": "QR", "amount": 9990},  # 6 no
        {"sender": 7, "receiver": 107, "txn_type": "COLLECT", "amount": 1237},  # 7 W
        {"sender": 8, "receiver": 108, "txn_type": "COLLECT", "amount": 1230},  # 8 no
        {
            "sender": 9,
            "receiver": 109,
            "txn_type": "QR",
            "amount": 1000,
            "balance_before": 1100,
        },  # 9 W
        {"sender": 10, "receiver": 110, "txn_type": "PAY", "amount": 1237},  # 10 no
        {"sender": 11, "receiver": 111, "txn_type": "COLLECT", "amount": 457},  # 11 no
        {
            "sender": 1,
            "receiver": 112,
            "txn_type": "COLLECT",
            "amount": 9999,
            "is_test": True,
        },  # 12 U
    ]
    out = requests.build(ledger(rows))
    assert col(out, "shape") == [1, 1, 0, 0, 0, 2, 0, 3, 0, 3, 0, 0, 1]
    assert col(out, "role")[:2] == [1, 1]
    assert col(out, "event")[:2] == [0, 0]  # one victim, one event in the split
    assert col(out, "size")[:2] == [2, 2]
    assert col(out, "span_s")[0] == 40 * 86400
    assert col(out, "event")[12] == 12  # never joins the train event
    assert col(out, "limit")[0] == 5000 and col(out, "offset")[0] == 10
    assert col(out, "limit")[1] == 25000 and col(out, "offset")[1] == 50
    assert col(out, "offset")[2] is None
    assert col(out, "limit")[4] is None  # merchant row
    assert col(out, "limit")[7] == 2000


def test_bait_sessions_and_answers():
    X, V = 500, 600
    rows = [
        {"sender": X, "receiver": V, "amount": 2, "ts": 0},  # 0 A1S
        {"sender": V, "receiver": X, "txn_type": "COLLECT", "amount": 4999, "ts": H},  # 1 answer
        {"sender": X + 1, "receiver": V + 1, "amount": 5, "ts": 0},  # 2 N1
        {"sender": X + 2, "receiver": V + 2, "amount": 10, "ts": 0},  # 3 N1X
        {"sender": X + 3, "receiver": V + 3, "amount": 1, "ts": 0},  # 4 N2
        {"sender": X + 3, "receiver": V + 3, "amount": 10, "ts": 600},  # 5 N2
        {"sender": X + 4, "receiver": V + 4, "amount": 1, "ts": 0},  # 6 A2
        {"sender": X + 4, "receiver": V + 4, "amount": 1, "ts": 300},  # 7 A2
        {"sender": V + 4, "receiver": X + 4, "amount": 3000, "ts": 30 * H},  # 8 answer
        {"sender": X + 5, "receiver": V + 5, "amount": 1, "ts": 0},  # 9 A1L
        {"sender": V + 5, "receiver": X + 5, "amount": 3000, "ts": 4 * H + 1},  # 10 answer
        {"sender": X + 6, "receiver": V + 6, "amount": 5, "ts": 0},  # 11 N1: late
        {"sender": V + 6, "receiver": X + 6, "amount": 3000, "ts": 72 * H + 1},  # 12 no
        {"sender": X + 7, "receiver": V + 7, "amount": 1, "ts": 0},  # 13 N1
        {"sender": X + 7, "receiver": V + 7, "amount": 1, "ts": 601},  # 14 N1
        {"sender": X + 8, "receiver": V + 8, "amount": 3, "ts": 0},  # 15 no bait
        {"sender": X + 9, "receiver": V + 9, "amount": 1, "ts": 0},  # 16 N1
        {"sender": V + 9, "receiver": X + 9, "amount": 3000, "ts": H, "is_test": True},  # 17 no
    ]
    out = requests.build(ledger(rows))
    assert col(out, "shape") == [7, 7, 4, 5, 6, 6, 9, 9, 9, 8, 8, 4, 0, 4, 4, 0, 4, 0]
    assert col(out, "role")[:2] == [2, 3]  # the 4,999 answer is not a request
    assert col(out, "event")[:2] == [0, 0]
    assert col(out, "size")[:2] == [2, 2]
    assert col(out, "delay_s")[:2] == [H, H]
    assert col(out, "short")[:2] == [1, 1]
    assert col(out, "answer_amount")[0] == 4999
    assert col(out, "short")[9] == 0
    assert col(out, "n_bait")[6:9] == [2, 2, 2]
    assert col(out, "event")[6:9] == [6, 6, 6]
    assert col(out, "span_s")[8] == 30 * H
    assert col(out, "bait_max")[4] == 10
    assert col(out, "event")[13:15] == [13, 14]  # 601 s gap: two sessions
    assert col(out, "short")[2] is None
    assert col(out, "rev_tiny_n72")[1] == 1 and col(out, "rev_tiny_dt")[1] == H
    assert col(out, "rev_tiny_n72")[12] == 0  # bait was 72 h + 1 s earlier
    assert col(out, "rev_tiny_n72")[17] == 0  # other split


def test_ring_counts_and_order_independence():
    X = 700
    rows = [
        {"sender": X, "receiver": 801, "amount": 1, "ts": 0},  # 0 N1
        {"sender": X, "receiver": 802, "amount": 2, "ts": 0},  # 1 N1
        {"sender": X, "receiver": 803, "amount": 10, "ts": 0},  # 2 N1X
        {"sender": 900, "receiver": X, "txn_type": "COLLECT", "amount": 1950},  # 3 U
        {"sender": 901, "receiver": X, "txn_type": "PAY", "amount": 50},  # 4 plain
        {"sender": 902, "receiver": 903, "txn_type": "COLLECT", "amount": 9950},  # 5 U
    ]
    led = ledger(rows)
    out = requests.build(led)
    assert col(out, "shape") == [4, 4, 5, 1, 0, 1]
    # X is on the fraud side of rows 0, 1 (baits) and 3 (request receiver); N1X is left out
    assert col(out, "s_ring_n")[:3] == [2, 2, 3]
    assert col(out, "r_ring_n")[3:5] == [2, 3]
    assert col(out, "r_ring_n")[5] == 0
    perm = np.random.default_rng(1).permutation(led.height)
    assert requests.build(led[perm]).equals(out[perm])
    assert all(c.startswith("col_") for c in out.columns)
    assert out["col_event"].dtype == pl.Int64 and out["col_shape"].dtype == pl.Int8
