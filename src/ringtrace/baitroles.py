"""Unanswered small baits recovered from the sender's operator roles, and a first-time PAY rule.

The fraud side draws its operator accounts (bait senders, collect and QR receivers, bot accounts,
payers of round merchant amounts) from one fixed pool for all four months, so an account that
acts as an operator in one month is an operator in the others. A lone unanswered Rs 10 P2P PAY
(requests shape N1X) is legit on about 93 % of train rows, but the phishing routine writes its
unanswered baits with Rs 10 as well as Rs 1, 2 and 5; the Rs 10 ones are found by their sender.

``operator_roles`` lists the label-free operator acts of every account per month:

* ``qr_receiver``: receives a QR payment addressed to a person;
* ``collect_receiver``: receives a just-under-limit COLLECT (requests shape U);
* ``bait_sender``: sends a bait in a recognised bait session (every shape except the lone
  unanswered Rs 10, N1X);
* ``answer_receiver``: receives a bait session's answer (the payback);
* ``bot_sender``: sends from a phone of the bot pool (device numbers 420,001 to 422,999);
* ``round_payer``: pays a merchant a ring amount (1,000 / 2,000 / 2,500 / 5,000 / 10,000) inside
  a merchant ring (merchant shapes 1 to 3).

Every one of these acts is fraud on more than 97 % of train rows on the cleaned labels.

``bait_roles`` (a rule): a small unanswered P2P PAY bait called legit becomes phishing_collect when
its sender holds an operator role in a train month other than the bait's own month (an October
bait: any train month). By default the bait is the lone unanswered Rs 10 (shape N1X); with
``band`` it is any P2P PAY of Rs 1 to 10 with no P2P payment back within 72 h.

``first_pay`` (a rule): a P2P PAY to a new payee of at least ``min_amount`` in whole rupees, not
a multiple of ten, called legit, becomes phishing_collect; with ``operator_receiver`` only when
the receiver holds an operator role in another month (the victim pays the operator).

``own_month`` lets an act in the row's own month count as well (with ``months`` = July to October
every month then sees the same four months).

Both rules read only label-free columns and the row's own call.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.classes import CODE
from ringtrace.evidence import requests as rq

TRAIN_MONTHS = (7, 8, 9)
ROLES = (
    "qr_receiver",
    "collect_receiver",
    "bait_sender",
    "answer_receiver",
    "bot_sender",
    "round_payer",
)
BOT_DEVICES = (420_001, 422_999)
RING_SHAPES = (1, 2, 3)
RECOGNISED_BAITS = (rq.N1, rq.N2, rq.A1S, rq.A1L, rq.A2)
UNANSWERED = (rq.N1, rq.N1X, rq.N2)
BAND = (1.0, 10.0)
ANSWER_WINDOW = rq.ANSWER_WINDOW
LEDGER_COLS = (
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
FAMILY_COLS = ("col_role", "col_shape", "rng_shape")


def columns(led: pl.DataFrame) -> pl.DataFrame:
    """The ledger and family columns the rules read, one row per ledger row."""
    from ringtrace.evidence import anchors

    fam = anchors.family_columns(led, list(FAMILY_COLS))
    base = led.select(*LEDGER_COLS)
    return pl.concat([base, fam], how="horizontal")


def operator_roles(cols: pl.DataFrame) -> pl.DataFrame:
    """Unique (acct, period, role) operator acts; no label is read."""
    c = cols.with_columns(
        pl.col("to_person").fill_null(False),
        pl.col("txn_type").fill_null(""),
        pl.col("col_role").fill_null(0),
        pl.col("col_shape").fill_null(0),
        pl.col("rng_shape").fill_null(0),
        pl.col("device").fill_null(-1),
    )
    snd, rcv = pl.col("sender"), pl.col("receiver")
    acts = (
        ("qr_receiver", (pl.col("txn_type") == "QR") & pl.col("to_person"), rcv),
        (
            "collect_receiver",
            (pl.col("col_role") == rq.REQUEST) & (pl.col("col_shape") == rq.U),
            rcv,
        ),
        (
            "bait_sender",
            (pl.col("col_role") == rq.BAIT) & pl.col("col_shape").is_in(RECOGNISED_BAITS),
            snd,
        ),
        ("answer_receiver", pl.col("col_role") == rq.ANSWER, rcv),
        ("bot_sender", pl.col("device").is_between(*BOT_DEVICES), snd),
        ("round_payer", pl.col("rng_shape").is_in(RING_SHAPES), snd),
    )
    parts = [
        c.filter(cond).select(
            acct.cast(pl.Int64).alias("acct"),
            pl.col("period").cast(pl.Int8),
            pl.lit(name).alias("role"),
        )
        for name, cond, acct in acts
    ]
    return pl.concat(parts).unique().sort("acct", "period", "role")


def other_month(
    acct: np.ndarray,
    period: np.ndarray,
    acts: pl.DataFrame,
    months=TRAIN_MONTHS,
    roles=ROLES,
    own_month: bool = False,
) -> np.ndarray:
    """Per query row: does ``acct`` hold one of ``roles`` in a month of ``months`` other than the
    row's own ``period`` (``own_month``: the row's own month counts too)."""
    months = tuple(int(m) for m in months)
    bit = {m: 1 << i for i, m in enumerate(months)}
    a = acts.filter(pl.col("role").is_in(list(roles)) & pl.col("period").is_in(list(months)))
    mask = (
        a.select("acct", pl.col("period").replace_strict(bit, return_dtype=pl.Int64).alias("b"))
        .unique()
        .group_by("acct")
        .agg(pl.col("b").sum().alias("mask"))
    )
    q = pl.DataFrame(
        {"acct": np.asarray(acct, dtype=np.int64), "period": np.asarray(period, dtype=np.int64)}
    ).with_row_index("q")
    j = q.join(mask, on="acct", how="left").sort("q")
    own = np.array([bit.get(int(p), 0) for p in j["period"].to_numpy()], dtype=np.int64)
    if own_month:
        own[:] = 0
    m = j["mask"].fill_null(0).to_numpy().astype(np.int64)
    return (m & ~own) != 0


def _reverse_within(cols: pl.DataFrame, rows: np.ndarray, window: int) -> np.ndarray:
    """For each query row: a P2P row back from its receiver to its sender within ``window`` s."""
    q = cols[rows].select("sender", "receiver", "ts").with_row_index("q")
    back = cols.filter(pl.col("to_person").fill_null(False)).select(
        pl.col("receiver").alias("sender"),
        pl.col("sender").alias("receiver"),
        pl.col("ts").alias("tb"),
    )
    dt = pl.col("tb") - pl.col("ts")
    hit = q.join(back, on=["sender", "receiver"]).filter((dt > 0) & (dt <= window))
    out = np.zeros(len(rows), dtype=bool)
    out[hit["q"].unique().to_numpy()] = True
    return out


def small_unanswered(cols: pl.DataFrame, band: bool = False) -> np.ndarray:
    """Mask of unanswered small P2P PAY baits: the lone unanswered Rs 10 (shape N1X), or with
    ``band`` every P2P PAY of Rs 1 to 10 with no P2P payment back within 72 h (exact bait amounts
    by their session's shape, other amounts by a direct look for the payment back)."""
    shape = cols["col_shape"].fill_null(0).to_numpy()
    role = cols["col_role"].fill_null(0).to_numpy()
    if not band:
        return (role == rq.BAIT) & (shape == rq.N1X)
    p2p_pay = cols["to_person"].fill_null(False).to_numpy() & (cols["txn_type"] == "PAY").to_numpy()
    amt = cols["amount"].to_numpy()
    small = p2p_pay & (amt >= BAND[0]) & (amt <= BAND[1])
    bait = small & (role == rq.BAIT)
    out = bait & np.isin(shape, UNANSWERED)
    other = np.flatnonzero(small & ~bait & (role == 0))
    if len(other):
        out[other] = ~_reverse_within(cols, other, ANSWER_WINDOW)
    return out


def bait_roles(
    pred,
    proba,
    led,
    log=print,
    band: bool = False,
    roles=ROLES,
    months=TRAIN_MONTHS,
    own_month: bool = False,
    target: str = "phishing_collect",
) -> np.ndarray:
    cols = columns(led)
    acts = operator_roles(cols)
    cand = np.flatnonzero(small_unanswered(cols, band) & (np.asarray(pred) == 0))
    out = np.asarray(pred).copy()
    if len(cand):
        snd = cols["sender"].to_numpy()[cand]
        per = cols["period"].to_numpy()[cand]
        hit = cand[other_month(snd, per, acts, months, roles, own_month)]
        out[hit] = CODE[target]
    is_test = led["is_test"].to_numpy()
    ch = out != np.asarray(pred)
    log(
        f"  bait_roles: candidates {len(cand)}, changed train {int((ch & ~is_test).sum())} / "
        f"test {int((ch & is_test).sum())}"
    )
    return out


def first_pay_mask(cols: pl.DataFrame, min_amount: float = 1000.0) -> np.ndarray:
    """P2P PAY to a new payee, at least ``min_amount``, whole rupees, not a multiple of ten."""
    cents = np.round(cols["amount"].to_numpy() * 100).astype(np.int64)
    rupees = cents // 100
    return (
        cols["to_person"].fill_null(False).to_numpy()
        & (cols["txn_type"] == "PAY").to_numpy()
        & (cols["new_payee"].fill_null(0).to_numpy() == 1)
        & (cents >= round(min_amount * 100))
        & (cents % 100 == 0)
        & (rupees % 10 != 0)
    )


def first_pay(
    pred,
    proba,
    led,
    log=print,
    min_amount: float = 1000.0,
    operator_receiver: bool = False,
    roles=ROLES,
    months=TRAIN_MONTHS,
    own_month: bool = False,
    target: str = "phishing_collect",
) -> np.ndarray:
    """``operator_receiver``: only rows whose receiver holds an operator role in another month."""
    if operator_receiver:
        cols = columns(led)
    else:
        cols = led.select("to_person", "txn_type", "new_payee", "amount")
    hit = first_pay_mask(cols, min_amount) & (np.asarray(pred) == 0)
    if operator_receiver and hit.any():
        idx = np.flatnonzero(hit)
        rcv = cols["receiver"].to_numpy()[idx]
        per = cols["period"].to_numpy()[idx]
        hit[idx] = other_month(rcv, per, operator_roles(cols), months, roles, own_month)
    out = np.asarray(pred).copy()
    out[hit] = CODE[target]
    te = led["is_test"].to_numpy()
    log(f"  first_pay: changed train {int((hit & ~te).sum())} / test {int((hit & te).sum())}")
    return out
