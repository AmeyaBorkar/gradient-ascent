"""Read the organisers' CSV files once, type them, and stack train and test into one frame.

The stacked frame ("ledger") is the only table the rest of the pipeline reads. Train rows come
first in file order, then test rows; ``row`` is that position and never changes. Identifiers are
turned into integers so that joins and sorts stay cheap:

* accounts ``AC0123456`` -> 123456
* merchants ``MR012345`` -> 12345 + MERCHANT_OFFSET (so one ``receiver`` key covers both kinds)
* devices ``DV0123456`` -> 123456 (numbers above 420,000 are phones that no account owns)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from ringtrace import paths
from ringtrace.classes import CLASSES

MERCHANT_OFFSET = 10_000_000
HOME_DEVICES = 420_000
LAST_TRAIN_MONTH = 9

TXN_SCHEMA = {
    "txn_id": pl.Utf8,
    "timestamp": pl.Utf8,
    "sender_id": pl.Utf8,
    "receiver_id": pl.Utf8,
    "receiver_type": pl.Utf8,
    "device_id": pl.Utf8,
    "amount": pl.Float64,
    "txn_type": pl.Utf8,
    "merchant_category": pl.Utf8,
    "device_city": pl.Utf8,
    "sim_age_hours": pl.Float64,
    "device_age_days": pl.Float64,
    "sender_account_age_days": pl.Int32,
    "new_payee": pl.Int8,
    "pin_retries": pl.Int16,
    "login_to_txn_sec": pl.Float64,
    "balance_before": pl.Float64,
    "app": pl.Utf8,
    "app_version": pl.Float64,
    "ip_risk_score": pl.Float64,
    "qr_vpa_mismatch": pl.Float64,
    "label": pl.Utf8,
}


def file_digest(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()[:n]


def _id_number(col: str) -> pl.Expr:
    return pl.col(col).str.slice(2).cast(pl.Int64)


def _parse_transactions(path: Path) -> pl.DataFrame:
    schema = {k: v for k, v in TXN_SCHEMA.items()}
    head = pl.read_csv(path, n_rows=1)
    if "label" not in head.columns:
        schema.pop("label")
    df = pl.read_csv(path, schema_overrides=schema)
    return df.with_columns(
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("timestamp")
    )


def _cached(name: str, src: Path, build) -> pl.DataFrame:
    key = file_digest(src)
    target = paths.cache_dir() / "tables" / f"{name}-{key}.parquet"
    if target.exists():
        return pl.read_parquet(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    df = build(src)
    tmp = target.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)
    return df


def raw(name: str) -> pl.DataFrame:
    """One organiser file as a typed frame, cached by content hash."""
    src = paths.data_dir() / f"{name}.csv"
    if name in ("train", "test"):
        return _cached(name, src, _parse_transactions)
    return _cached(name, src, lambda p: pl.read_csv(p))


def stack(
    train: pl.DataFrame, test: pl.DataFrame, accounts: pl.DataFrame, merchants: pl.DataFrame
) -> pl.DataFrame:
    """Build the ledger: train then test, integer keys, account and merchant attributes joined."""
    test = test.with_columns(pl.lit(None, dtype=pl.Utf8).alias("label"))
    both = pl.concat(
        [
            train.with_columns(pl.lit(False).alias("is_test")),
            test.select(train.columns).with_columns(pl.lit(True).alias("is_test")),
        ],
        how="vertical",
    ).with_row_index("row")

    label_code = pl.col("label").replace_strict(
        {c: i for i, c in enumerate(CLASSES)}, default=-1, return_dtype=pl.Int8
    )
    month = pl.col("timestamp").dt.month().cast(pl.Int8)
    ledger = both.with_columns(
        pl.col("row").cast(pl.Int32),
        (pl.col("timestamp").dt.epoch("s")).alias("ts"),
        month.alias("month"),
        # Train rows dated October are the tails of September events: they belong to September.
        pl.when(pl.col("is_test"))
        .then(pl.lit(10, dtype=pl.Int8))
        .otherwise(pl.min_horizontal(month, pl.lit(LAST_TRAIN_MONTH, dtype=pl.Int8)))
        .alias("period"),
        label_code.alias("y"),
        _id_number("sender_id").alias("sender"),
        pl.when(pl.col("receiver_type") == "MERCHANT")
        .then(_id_number("receiver_id") + MERCHANT_OFFSET)
        .otherwise(_id_number("receiver_id"))
        .alias("receiver"),
        _id_number("device_id").alias("device"),
        (pl.col("receiver_type") == "P2P").alias("to_person"),
    )

    acc = accounts.select(
        _id_number("account_id").alias("acct"),
        pl.col("home_city"),
        pl.col("bank_code"),
        pl.col("kyc_level"),
        pl.col("opened_on").str.to_date("%Y-%m-%d").alias("opened_on"),
        pl.col("os"),
    )
    ledger = ledger.join(
        acc.rename({c: f"s_{c}" for c in acc.columns if c != "acct"}),
        left_on="sender",
        right_on="acct",
        how="left",
    )
    ledger = ledger.join(
        acc.select(
            "acct",
            pl.col("opened_on").alias("r_opened_on"),
            pl.col("home_city").alias("r_home_city"),
            pl.col("kyc_level").alias("r_kyc"),
        ),
        left_on="receiver",
        right_on="acct",
        how="left",
    )
    mer = merchants.select(
        (_id_number("merchant_id") + MERCHANT_OFFSET).alias("mkey"),
        pl.col("category").alias("m_category"),
        pl.col("city").alias("m_city"),
        pl.col("onboarded_on").str.to_date("%Y-%m-%d").alias("m_onboarded_on"),
    )
    ledger = ledger.join(mer, left_on="receiver", right_on="mkey", how="left")
    return ledger.sort("row")


def ledger() -> pl.DataFrame:
    """The stacked frame, cached by the hashes of the four input files."""
    names = ("train", "test", "accounts", "merchants")
    key = "-".join(file_digest(paths.data_dir() / f"{n}.csv", 8) for n in names)
    target = paths.cache_dir() / "tables" / f"ledger-{key}.parquet"
    if target.exists():
        return pl.read_parquet(target)
    df = stack(*(raw(n) for n in names))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)
    return df
