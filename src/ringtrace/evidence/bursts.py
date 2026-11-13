"""Bot bursts: velocity_bot sessions, many payments in minutes from one phone.

A device session is a run of consecutive rows on one device with gaps of at most 5 minutes;
a session of 10 or more rows is a burst (no ordinary session comes close). Sessions stay inside
one side of the train/test split. The rule never reads the device number; ``bot_range`` (the
reserved bot phones) and ``bot_mburst`` (a device-free merchant-side session) are separate
row columns.

Shapes: 1 = farm (more than 6 accounts, nearly one row each: the sender pool repeats an account
now and then); 2 = at most 6 accounts with scripted logins (most rows log in within 6 s); 3 = at
most 6 accounts with human logins; 4 = anything else (many accounts that repeat).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.tables import MERCHANT_OFFSET

PREFIX = "bot"

SESSION_GAP_S = 300
MIN_ROWS = 10
FEW_ACCOUNTS = 6
FARM_ACCOUNT_SHARE = 0.9  # accounts / rows of a farm (observed >= 0.944)
SCRIPTED_LOGIN_S = 6  # generator logins are round(U(1, 6)); human q05 is 9
BOT_DEVICES = (420_001, 422_999)
# R3: merchant-side session of >= 4 rows with at most 60 other rows that calendar month
MERCHANT_MIN_ROWS = 4
MERCHANT_MONTH_OTHERS = 60


def sessions(
    key: np.ndarray, split: np.ndarray, ts: np.ndarray, row: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Session id (sorted position of its first row), size and position inside it, per row."""
    n = len(key)
    order = np.lexsort((row, ts, key, split))
    grp = (np.asarray(key, dtype=np.int64) * 2 + split)[order]
    t = ts[order]
    start = np.ones(n, dtype=bool)
    start[1:] = (grp[1:] != grp[:-1]) | (t[1:] - t[:-1] > SESSION_GAP_S)
    first = np.flatnonzero(start)
    sid_sorted = np.repeat(first, np.diff(np.r_[first, n]))
    size_sorted = np.repeat(np.diff(np.r_[first, n]), np.diff(np.r_[first, n]))
    sid = np.empty(n, dtype=np.int64)
    size = np.empty(n, dtype=np.int64)
    pos = np.empty(n, dtype=np.int64)
    sid[order] = sid_sorted
    size[order] = size_sorted
    pos[order] = np.arange(n) - sid_sorted
    return sid, size, pos


def masked(n: int, at: np.ndarray, values: np.ndarray, dtype) -> pl.Series:
    """A length-n series holding ``values`` at positions ``at`` and null elsewhere."""
    values = np.asarray(values)
    full = np.zeros(n, dtype=values.dtype if len(values) else np.int64)
    full[at] = values
    mask = np.zeros(n, dtype=bool)
    mask[at] = True
    frame = pl.DataFrame({"v": full, "m": mask})
    return frame.select(pl.when(pl.col("m")).then(pl.col("v").cast(dtype))).to_series()


def merchant_bursts(led: pl.DataFrame) -> np.ndarray:
    """R3: rows of a short merchant-side session at a merchant that is otherwise quiet."""
    split = led["is_test"].to_numpy().astype(np.int64)
    ts = led["ts"].to_numpy().astype(np.int64)
    row = led["row"].to_numpy().astype(np.int64)
    merchant = ~led["to_person"].to_numpy()
    receiver = led["receiver"].to_numpy().astype(np.int64)
    key = np.where(merchant, receiver, -1 - np.arange(len(receiver)))  # persons: singletons
    sid, size, _ = sessions(key, split, ts, row)
    frame = pl.DataFrame(
        {
            "r": receiver,
            "s": split,
            "m": led["month"].to_numpy(),
            "sid": sid,
            "size": size,
            "ts": ts,
            "row": row,
        }
    )
    month_rows = frame.group_by("r", "s", "m").agg(pl.len().alias("month_n"))
    first = (
        frame.sort("ts", "row")
        .group_by("sid")
        .agg(pl.col("r", "s", "m").first())
        .join(month_rows, on=["r", "s", "m"], how="left")
    )
    month_n = (
        frame.select("sid")
        .join(first.select("sid", "month_n"), on="sid", how="left", maintain_order="left")[
            "month_n"
        ]
        .to_numpy()
    )
    return merchant & (size >= MERCHANT_MIN_ROWS) & (month_n - size <= MERCHANT_MONTH_OTHERS)


def build(led: pl.DataFrame) -> pl.DataFrame:
    n = led.height
    split = led["is_test"].to_numpy().astype(np.int64)
    ts = led["ts"].to_numpy().astype(np.int64)
    row = led["row"].to_numpy().astype(np.int64)
    device = led["device"].to_numpy().astype(np.int64)
    sender = led["sender"].to_numpy().astype(np.int64)
    receiver = led["receiver"].to_numpy().astype(np.int64)
    sid, size, pos = sessions(device, split, ts, row)

    at = np.flatnonzero(size >= MIN_ROWS)
    rows = pl.DataFrame(
        {
            "i": at,
            "sid": sid[at],
            "pos": pos[at],
            "row": row[at],
            "ts": ts[at],
            "sender": sender[at],
            "receiver": receiver[at],
            "login": led["login_to_txn_sec"].to_numpy()[at],
        }
    )
    events = (
        rows.group_by("sid")
        .agg(
            pl.col("row").min().alias("event"),
            pl.len().alias("size"),
            (pl.col("ts").max() - pl.col("ts").min()).alias("span"),
            pl.col("sender").n_unique().alias("accounts"),
            pl.col("receiver")
            .filter(pl.col("receiver") >= MERCHANT_OFFSET)
            .n_unique()
            .alias("merchants"),
            (pl.col("login") <= SCRIPTED_LOGIN_S).mean().alias("scripted"),
        )
        .with_columns(
            pl.when(
                (pl.col("accounts") > FEW_ACCOUNTS)
                & (pl.col("accounts") >= FARM_ACCOUNT_SHARE * pl.col("size"))
            )
            .then(1)
            .when((pl.col("accounts") <= FEW_ACCOUNTS) & (pl.col("scripted") > 0.5))
            .then(2)
            .when(pl.col("accounts") <= FEW_ACCOUNTS)
            .then(3)
            .otherwise(4)
            .cast(pl.Int8)
            .alias("shape")
        )
    )
    rows = rows.join(events, on="sid", how="left", maintain_order="left")
    at = rows["i"].to_numpy()
    in_burst = np.zeros(n, dtype=bool)
    in_burst[at] = True

    # bursts each account sent in, then per row: the sender's other bursts, the receiver's bursts
    sent = rows.select("sender", "event").unique()
    per_acct = sent.group_by("sender").agg(pl.len().alias("k"))
    base = pl.DataFrame({"sender": sender, "receiver": receiver})
    snd_k = (
        base.join(per_acct, on="sender", how="left", maintain_order="left")["k"]
        .fill_null(0)
        .to_numpy()
        .astype(np.int64)
    )
    rcv_k = (
        base.join(
            per_acct.rename({"sender": "receiver"}),
            on="receiver",
            how="left",
            maintain_order="left",
        )["k"]
        .fill_null(0)
        .to_numpy()
        .astype(np.int64)
    )
    own = np.full(n, -1, dtype=np.int64)
    own[at] = rows["event"].to_numpy()
    rcv_own = (
        pl.DataFrame({"receiver": receiver, "event": own})
        .join(
            sent.rename({"sender": "receiver"}).with_columns(pl.lit(1).alias("hit")),
            on=["receiver", "event"],
            how="left",
            maintain_order="left",
        )["hit"]
        .fill_null(0)
        .to_numpy()
    )

    shape = np.zeros(n, dtype=np.int8)
    shape[at] = rows["shape"].to_numpy()
    on_burst = {
        "event": pl.Int64,
        "size": pl.Int16,
        "span": pl.Float32,
        "pos": pl.Int16,
        "accounts": pl.Int16,
        "merchants": pl.Int16,
    }
    cols = [pl.Series(f"{PREFIX}_shape", shape, dtype=pl.Int8)]
    for name, dtype in on_burst.items():
        out = "span_s" if name == "span" else name
        cols.append(masked(n, at, rows[name].to_numpy(), dtype).alias(f"{PREFIX}_{out}"))
    cols.insert(0, cols.pop(1))  # event first
    in_range = (device >= BOT_DEVICES[0]) & (device <= BOT_DEVICES[1])
    cols += [
        pl.Series(f"{PREFIX}_range", in_range, dtype=pl.Int8),
        pl.Series(f"{PREFIX}_mburst", merchant_bursts(led), dtype=pl.Int8),
        pl.Series(f"{PREFIX}_snd_bursts", snd_k - in_burst, dtype=pl.Int32),
        pl.Series(f"{PREFIX}_rcv_bursts", rcv_k - rcv_own, dtype=pl.Int32),
    ]
    return pl.DataFrame(cols)
