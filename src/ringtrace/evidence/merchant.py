"""Merchant rings (qr_swap / refund_abuse / merchant_collusion), label-free.

* Ring rows: PAY or QR to a merchant for exactly 1,000 / 2,000 / 2,500 / 5,000 / 10,000. They are
  chained per merchant in time (new event after a gap of more than 24 h, never across the split)
  and the event's size names the shape: 1 = 1-3 rows, 2 = 4-8 rows, 3 = 9+ rows.
* Shape 4: a QR payment to a person that carries a merchant category (one-row event).
* Shape 5: the whole-amount form of collusion: PAY or QR to a merchant in whole rupees >= 900 (not
  a ring amount), chained per merchant with gaps <= 24 h; an event when it has >= 5 rows and at
  least twice as many rows as distinct senders (the same few payers come back).

Columns: rng_event (smallest row of the event), rng_shape, rng_size, rng_span_s, rng_senders
(distinct senders in the event), rng_pair_n (rows of this row's sender in the event), rng_rank
(1-based position in the event by time). The QR check column is never read.
"""

from __future__ import annotations

import polars as pl

PREFIX = "rng"

DAY = 86_400
RING_CENTS = (100_000, 200_000, 250_000, 500_000, 1_000_000)
WHOLE_MIN = 900
WHOLE_MIN_ROWS = 5
WHOLE_ROWS_PER_SENDER = 2

SHAPE_SMALL = 1  # 1-3 ring rows   -> qr_swap
SHAPE_MID = 2  # 4-8 ring rows     -> refund_abuse
SHAPE_BIG = 3  # 9+ ring rows      -> merchant_collusion
SHAPE_QR_PERSON = 4  # QR to a person with a category -> qr_swap
SHAPE_WHOLE = 5  # whole-amount collusion form

_COLS = ("event", "shape", "size", "span_s", "senders", "pair_n", "rank")


def _chain(cand: pl.DataFrame, key: str, gap: int) -> pl.DataFrame:
    """Chain candidate rows per (split, key) in time order; a gap of more than ``gap`` seconds
    starts a new chain. Adds the per-chain summaries (sorting by row breaks timestamp ties, so
    the result does not depend on input order)."""
    c = cand.sort("is_test", key, "ts", "row")
    new = (
        (pl.col(key) != pl.col(key).shift(1))
        | (pl.col("is_test") != pl.col("is_test").shift(1))
        | ((pl.col("ts") - pl.col("ts").shift(1)) > gap)
    ).fill_null(True)
    c = c.with_columns(new.cum_sum().alias("_chain"))
    return c.with_columns(
        pl.col("row").min().over("_chain").cast(pl.Int64).alias("event"),
        pl.len().over("_chain").cast(pl.Int32).alias("size"),
        (pl.col("ts").max() - pl.col("ts").min()).over("_chain").alias("span_s"),
        pl.col("sender").n_unique().over("_chain").alias("senders"),
        pl.len().over("_chain", "sender").alias("pair_n"),
        (pl.int_range(pl.len()).over("_chain") + 1).alias("rank"),
    )


def build(led: pl.DataFrame) -> pl.DataFrame:
    base = led.select("row", "is_test", "ts", "sender", "receiver", "to_person", "txn_type",
                      "merchant_category", "amount")
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    to_merchant = ~pl.col("to_person")
    pay_qr = pl.col("txn_type").is_in(["PAY", "QR"])
    ring_amount = cents.is_in(RING_CENTS)
    base = base.with_columns(
        (to_merchant & pay_qr & ring_amount).alias("_ring"),
        (pl.col("to_person") & (pl.col("txn_type") == "QR")
         & pl.col("merchant_category").is_not_null()).alias("_qr_person"),
        (to_merchant & pay_qr & ~ring_amount & (cents % 100 == 0)
         & (pl.col("amount") >= WHOLE_MIN)).alias("_whole"),
    )

    ring = _chain(base.filter("_ring"), "receiver", DAY).with_columns(
        pl.when(pl.col("size") <= 3).then(SHAPE_SMALL)
        .when(pl.col("size") <= 8).then(SHAPE_MID)
        .otherwise(SHAPE_BIG).alias("shape")
    )
    qr_person = base.filter("_qr_person").with_columns(
        pl.col("row").cast(pl.Int64).alias("event"),
        pl.lit(SHAPE_QR_PERSON).alias("shape"),
        pl.lit(1, dtype=pl.Int32).alias("size"),
        pl.lit(0).alias("span_s"),
        pl.lit(1).alias("senders"),
        pl.lit(1).alias("pair_n"),
        pl.lit(1).alias("rank"),
    )
    whole = _chain(base.filter("_whole"), "receiver", DAY).filter(
        (pl.col("size") >= WHOLE_MIN_ROWS)
        & (pl.col("size") >= WHOLE_ROWS_PER_SENDER * pl.col("senders"))
    ).with_columns(pl.lit(SHAPE_WHOLE).alias("shape"))

    parts = [p.select("row", *_COLS) for p in (ring, qr_person, whole)]
    parts = [p.with_columns(
        pl.col("event").cast(pl.Int64), pl.col("shape").cast(pl.Int8),
        pl.col("size").cast(pl.Int32),
        *(pl.col(c).cast(pl.Float32) for c in ("span_s", "senders", "pair_n", "rank")),
    ) for p in parts]
    events = pl.concat(parts, how="vertical")
    out = led.select("row").join(events, on="row", how="left", maintain_order="left")
    out = out.with_columns(pl.col("shape").fill_null(0))
    return out.select(pl.col(c).alias(f"{PREFIX}_{c}") for c in _COLS)
