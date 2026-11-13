"""Repeat offenders (refund_abuse / loan_app_extortion), label-free.

Candidate kinds (rep_kind, 0 = none):

1. REFUND to an ecommerce or electronics merchant;
2. PAY to a lending merchant in whole rupees, between 20:00 and 23:59;
3. P2P PAY between 20:00 and 23:59 to a new payee, whole rupees in [1,000, 15,000], and either
   not a multiple of 10 or exactly 1/1.1 of the balance (the lending law sent to a person).

Candidates are chained per sender and kind inside one split; a gap of more than 7 days starts a
new chain (rep_chain_n = chain length, on every candidate). A chain of >= 3 rows is an event whose
shape is its kind; kind 3 also needs rows on >= 2 calendar days (one-evening P2P runs are
takeovers). Columns: rep_event (smallest row), rep_shape, rep_size, rep_span_s, rep_receivers
(distinct receivers), rep_days (distinct calendar days), rep_rank (1-based position by time),
rep_kind, rep_chain_n.
"""

from __future__ import annotations

import polars as pl

PREFIX = "rep"

DAY = 86_400
CHAIN_GAP = 7 * DAY
MIN_ROWS = 3
P2P_MIN_DAYS = 2
EVENING = (20, 21, 22, 23)
REFUND_CATEGORIES = ("ecommerce", "electronics")
LENDING = "lending"
P2P_MIN, P2P_MAX = 1000, 15000
RATIO_110 = 1 / 1.1
RATIO_TOL = 2e-4

KIND_REFUND = 1
KIND_LENDING = 2
KIND_LENDING_P2P = 3

_EVENT_COLS = ("event", "shape", "size", "span_s", "receivers", "days", "rank")


def _kind(led: pl.DataFrame) -> pl.Expr:
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    whole = cents % 100 == 0
    rupees = cents // 100
    evening = pl.col("timestamp").dt.hour().is_in(EVENING)
    to_person = pl.col("to_person")
    ttype = pl.col("txn_type")
    cat = pl.col("merchant_category")
    ratio_110 = ((pl.col("amount") / pl.col("balance_before")) - RATIO_110).abs() < RATIO_TOL
    refund = (ttype == "REFUND") & ~to_person & cat.is_in(REFUND_CATEGORIES)
    lending = (ttype == "PAY") & ~to_person & (cat == LENDING) & whole & evening
    lending_p2p = (
        (ttype == "PAY")
        & to_person
        & evening
        & (pl.col("new_payee") == 1)
        & whole
        & (pl.col("amount") >= P2P_MIN)
        & (pl.col("amount") <= P2P_MAX)
        & ((rupees % 10 != 0) | ratio_110.fill_null(False))
    )
    return (
        pl.when(refund.fill_null(False))
        .then(KIND_REFUND)
        .when(lending.fill_null(False))
        .then(KIND_LENDING)
        .when(lending_p2p.fill_null(False))
        .then(KIND_LENDING_P2P)
        .otherwise(0)
        .cast(pl.Int8)
    )


def build(led: pl.DataFrame) -> pl.DataFrame:
    base = led.select(
        "row",
        "is_test",
        "ts",
        "timestamp",
        "sender",
        "receiver",
        "to_person",
        "txn_type",
        "merchant_category",
        "amount",
        "balance_before",
        "new_payee",
    )
    base = base.with_columns(_kind(base).alias("kind"))
    cand = base.filter(pl.col("kind") > 0).sort("is_test", "sender", "kind", "ts", "row")
    new = (
        (pl.col("sender") != pl.col("sender").shift(1))
        | (pl.col("kind") != pl.col("kind").shift(1))
        | (pl.col("is_test") != pl.col("is_test").shift(1))
        | ((pl.col("ts") - pl.col("ts").shift(1)) > CHAIN_GAP)
    ).fill_null(True)
    cand = cand.with_columns(new.cum_sum().alias("_chain"))
    cand = cand.with_columns(
        pl.col("row").min().over("_chain").cast(pl.Int64).alias("event"),
        pl.len().over("_chain").cast(pl.Int32).alias("chain_n"),
        (pl.col("ts").max() - pl.col("ts").min()).over("_chain").alias("span_s"),
        pl.col("receiver").n_unique().over("_chain").alias("receivers"),
        pl.col("timestamp").dt.date().n_unique().over("_chain").alias("days"),
        (pl.int_range(pl.len()).over("_chain") + 1).alias("rank"),
    )
    is_event = (pl.col("chain_n") >= MIN_ROWS) & (
        (pl.col("kind") != KIND_LENDING_P2P) | (pl.col("days") >= P2P_MIN_DAYS)
    )
    cand = cand.with_columns(
        pl.when(is_event).then(pl.col("kind")).otherwise(0).cast(pl.Int8).alias("shape"),
        pl.when(is_event).then(pl.col("chain_n")).alias("size"),
        *(
            pl.when(is_event).then(pl.col(c)).alias(c)
            for c in ("event", "span_s", "receivers", "days", "rank")
        ),
    ).select(
        "row",
        "kind",
        pl.col("chain_n").cast(pl.Int32),
        "event",
        "shape",
        pl.col("size").cast(pl.Int32),
        *(pl.col(c).cast(pl.Float32) for c in ("span_s", "receivers", "days", "rank")),
    )
    out = led.select("row").join(cand, on="row", how="left", maintain_order="left")
    out = out.with_columns(pl.col("shape").fill_null(0), pl.col("kind").fill_null(0))
    return out.select(pl.col(c).alias(f"{PREFIX}_{c}") for c in (*_EVENT_COLS, "kind", "chain_n"))
