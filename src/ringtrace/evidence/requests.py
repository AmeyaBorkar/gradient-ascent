"""Collect-request scams and tiny-payment baits (spec s6): phishing_collect / fake_cashback.

Two event kinds, disjoint by construction:

* requests (one event per victim = sender inside a split; the row shape is the row's own kind):
  1 U  - P2P COLLECT from a new payee for a "just under" amount L - d (L in 2,000 / 5,000 /
         10,000 / 25,000, d in 1 / 10 / 50);
  2 UO - P2P PAY/QR/AUTOPAY/REFUND from a new payee for 1,999 / 4,999 / 9,999 / 24,999;
  3 W  - P2P COLLECT or QR from a new payee, >= 500, whole rupees not a multiple of 10 or exactly
         1/1.1 of the balance, outside the twelve just-under amounts.
* bait sessions (same sender and receiver, P2P PAY of exactly 1/2/5/10, gaps <= 10 min) with their
  answer, the first P2P row back from the bait's receiver within 72 h after the last bait:
  4 N1 one bait of 1/2/5 unanswered, 5 N1X one unanswered 10, 6 N2 two+ baits unanswered,
  7 A1S one bait answered <= 4 h, 8 A1L one bait answered in 4-72 h, 9 A2 two+ baits answered.

A bait answer is never a request. Bait event id = smallest bait row. Every P2P row also gets the
nearest limit above its amount and the offset below it, the reverse tiny payments of the 72 h
before, and how many scam rows each of its two accounts has on the fraud side outside its own
event. Everything stays inside one split; labels and the two columns that die in October are
never read.
"""

from __future__ import annotations

import numpy as np
import polars as pl

PREFIX = "col"

LIMITS = (2000, 5000, 10000, 25000)
OFFSETS = (1, 10, 50)
JUST_UNDER = tuple(lim - d for lim in LIMITS for d in OFFSETS)
UO_AMOUNTS = tuple(lim - 1 for lim in LIMITS)
BAIT_AMOUNTS = (1, 2, 5, 10)
SESSION_GAP = 600
ANSWER_WINDOW = 72 * 3600
SHORT_ANSWER = 4 * 3600
W_MIN = 500.0
BAL_RATIO = 1 / 1.1
BAL_TOL = 2e-4

U, UO, W, N1, N1X, N2, A1S, A1L, A2 = range(1, 10)
REQUEST, BAIT, ANSWER = 1, 2, 3

INT_COLS = {"event": pl.Int64, "shape": pl.Int8, "size": pl.Int32, "role": pl.Int8,
            "n_bait": pl.Int8, "short": pl.Int8, "rev_tiny_n72": pl.Int32,
            "s_ring_n": pl.Int32, "r_ring_n": pl.Int32}


def _base(led: pl.DataFrame) -> pl.DataFrame:
    return led.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("i"),
        pl.col("row").cast(pl.Int64),
        pl.col("is_test").cast(pl.Int8).alias("split"),
        pl.col("sender").cast(pl.Int64).alias("snd"),
        pl.col("receiver").cast(pl.Int64).alias("rcv"),
        pl.col("to_person").fill_null(False).alias("person"),
        pl.col("txn_type").fill_null(""),
        pl.col("ts").cast(pl.Int64),
        pl.col("amount").cast(pl.Float64).alias("amt"),
        (pl.col("amount") * 100).round().cast(pl.Int64).alias("cents"),
        (pl.col("new_payee").fill_null(0) == 1).alias("new"),
        pl.col("balance_before").cast(pl.Float64).alias("bal"),
    )


def _cents(values) -> list[int]:
    return [100 * v for v in values]


def _sessions(b: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Bait rows numbered into sessions, and one line per session with its answer (if any)."""
    baits = b.filter(pl.col("bait")).sort("split", "snd", "rcv", "ts", "row")
    gap = pl.col("ts").diff().over("split", "snd", "rcv")
    baits = baits.with_columns(
        (gap.is_null() | (gap > SESSION_GAP)).cum_sum().cast(pl.Int64).alias("sess"))
    sess = baits.group_by("sess").agg(
        pl.col("split").first(), pl.col("snd").first(), pl.col("rcv").first(),
        pl.col("row").min().alias("event"), pl.len().alias("n_bait"),
        pl.col("amt").max().alias("bait_max"),
        pl.col("ts").min().alias("t_first"), pl.col("ts").max().alias("t_last"),
    )
    back = b.filter(pl.col("person") & ~pl.col("bait")).select(
        "split", pl.col("snd").alias("rcv"), pl.col("rcv").alias("snd"),
        pl.col("i").alias("ans_i"), pl.col("row").alias("ans_row"), pl.col("ts").alias("ans_t"),
        pl.col("amt").alias("ans_amt"))
    delay = pl.col("ans_t") - pl.col("t_last")
    first_back = (
        sess.join(back, on=["split", "snd", "rcv"])
        .filter((delay > 0) & (delay <= ANSWER_WINDOW))
        .sort("sess", "ans_t", "ans_row").unique("sess", keep="first", maintain_order=True)
        # one answer serves one session: the latest session before it keeps it
        .sort("ans_i", "t_last", "event", descending=[False, True, True])
        .unique("ans_i", keep="first", maintain_order=True)
        .select("sess", "ans_i", "ans_t", "ans_amt")
    )
    sess = sess.join(first_back, on="sess", how="left").with_columns(
        (pl.col("ans_t") - pl.col("t_last")).alias("delay"))
    answered = pl.col("ans_i").is_not_null()
    one = pl.col("n_bait") == 1
    sess = sess.with_columns(
        pl.when(one & ~answered & (pl.col("bait_max") == 10)).then(N1X)
        .when(one & ~answered).then(N1)
        .when(~answered).then(N2)
        .when(one & (pl.col("delay") <= SHORT_ANSWER)).then(A1S)
        .when(one).then(A1L)
        .otherwise(A2).cast(pl.Int8).alias("shape"),
        pl.when(answered).then((pl.col("delay") <= SHORT_ANSWER).cast(pl.Float64))
        .alias("short"),
        pl.coalesce("ans_t", "t_last").sub(pl.col("t_first")).alias("span"),
        (pl.col("n_bait") + answered.cast(pl.Int64)).alias("size"),
    )
    return baits.select("i", "sess"), sess


def build(led: pl.DataFrame) -> pl.DataFrame:
    b = _base(led)
    n = b.height
    names = ["event", "shape", "size", "span_s", "role", "n_bait", "bait_max", "delay_s",
             "answer_amount", "short", "limit", "offset", "rev_tiny_n72", "rev_tiny_dt",
             "s_ring_n", "r_ring_n"]
    out = {k: np.full(n, np.nan) for k in names}

    p2p, typ, cents = pl.col("person"), pl.col("txn_type"), pl.col("cents")
    b = b.with_columns((p2p & (typ == "PAY") & cents.is_in(_cents(BAIT_AMOUNTS))).alias("bait"))

    # --- bait sessions and answers --------------------------------------------------------------
    bait_rows, sess = _sessions(b)
    members = pl.concat([
        bait_rows.with_columns(pl.lit(BAIT, dtype=pl.Int8).alias("role")),
        sess.filter(pl.col("ans_i").is_not_null()).select(
            pl.col("ans_i").alias("i"), "sess", pl.lit(ANSWER, dtype=pl.Int8).alias("role")),
    ]).join(sess.drop("snd", "rcv", "split"), on="sess", how="left")
    mi = members["i"].to_numpy()
    for col, src in (("event", "event"), ("shape", "shape"), ("size", "size"),
                     ("span_s", "span"), ("role", "role"), ("n_bait", "n_bait"),
                     ("bait_max", "bait_max"), ("delay_s", "delay"),
                     ("answer_amount", "ans_amt"), ("short", "short")):
        out[col][mi] = members[src].cast(pl.Float64).fill_null(np.nan).to_numpy()
    is_answer = np.zeros(n, dtype=bool)
    is_answer[mi[members["role"].to_numpy() == ANSWER]] = True

    # --- request rows, one event per victim inside a split -------------------------------------
    whole = cents % 100 == 0
    off_ratio = ((pl.col("amt") / pl.col("bal")) - BAL_RATIO).abs() < BAL_TOL
    b = b.with_columns(pl.Series("answer", is_answer))
    new_p2p = p2p & pl.col("new") & ~pl.col("answer")
    kind = (
        pl.when(new_p2p & (typ == "COLLECT") & cents.is_in(_cents(JUST_UNDER))).then(U)
        .when(new_p2p & typ.is_in(["PAY", "QR", "AUTOPAY", "REFUND"])
              & cents.is_in(_cents(UO_AMOUNTS))).then(UO)
        .when(new_p2p & typ.is_in(["COLLECT", "QR"]) & (pl.col("amt") >= W_MIN)
              & ~cents.is_in(_cents(JUST_UNDER))
              & ((whole & ((cents // 100) % 10 != 0)) | off_ratio.fill_null(False))).then(W)
        .otherwise(0).cast(pl.Int8)
    )
    req = b.with_columns(kind.alias("kind")).filter(pl.col("kind") > 0).with_columns(
        pl.col("row").min().over("split", "snd").alias("event"),
        pl.len().over("split", "snd").alias("size"),
        (pl.col("ts").max() - pl.col("ts").min()).over("split", "snd").alias("span"),
    )
    ri = req["i"].to_numpy()
    out["event"][ri] = req["event"].to_numpy()
    out["shape"][ri] = req["kind"].to_numpy()
    out["size"][ri] = req["size"].to_numpy()
    out["span_s"][ri] = req["span"].to_numpy()
    out["role"][ri] = REQUEST

    # --- limit and offset of every P2P row ------------------------------------------------------
    amt = b["amt"].to_numpy()
    rcents = b["cents"].to_numpy()
    person = b["person"].to_numpy()
    limit = np.full(n, np.nan)
    for lim in sorted(LIMITS, reverse=True):
        limit[amt < lim] = lim
    limit[~person] = np.nan
    gap = np.round(limit * 100) - rcents
    hit = np.isin(gap, _cents(OFFSETS))
    out["limit"] = limit
    out["offset"] = np.where(hit, gap / 100, np.nan)

    # --- reverse tiny payments in the 72 h before -----------------------------------------------
    tiny = b.filter(pl.col("bait")).select(
        "split", pl.col("rcv").alias("snd"), pl.col("snd").alias("rcv"), pl.col("ts").alias("tt"))
    dt = pl.col("ts") - pl.col("tt")
    rev = (b.filter(p2p).select("i", "split", "snd", "rcv", "ts")
           .join(tiny, on=["split", "snd", "rcv"]).filter((dt > 0) & (dt <= ANSWER_WINDOW))
           .group_by("i").agg(pl.len().alias("n"), dt.min().alias("dt")))
    out["rev_tiny_n72"][person] = 0
    out["rev_tiny_n72"][rev["i"].to_numpy()] = rev["n"].to_numpy()
    out["rev_tiny_dt"][rev["i"].to_numpy()] = rev["dt"].to_numpy()

    # --- fraud-side scam rows of both accounts, own event excluded ------------------------------
    # receivers of requests and answers, senders of baits (single unanswered 10s are mostly
    # legit and are left out)
    shape = out["shape"]
    role = out["role"]
    ev = b.select("i", "split", "snd", "rcv").with_columns(
        pl.Series("event", out["event"], nan_to_null=True).cast(pl.Int64),
        pl.Series("shape", np.nan_to_num(shape)).cast(pl.Int8),
        pl.Series("role", np.nan_to_num(role)).cast(pl.Int8))
    scam = ev.filter((pl.col("shape") > 0) & (pl.col("shape") != N1X)).select(
        "split", "event",
        pl.when(pl.col("role") == BAIT).then(pl.col("snd")).otherwise(pl.col("rcv")).alias("acct"))
    total = scam.group_by("split", "acct").agg(pl.len().alias("n_all"))
    own = scam.group_by("split", "acct", "event").agg(pl.len().alias("n_own"))
    for side, col in (("snd", "s_ring_n"), ("rcv", "r_ring_n")):
        m = (ev.join(total, left_on=["split", side], right_on=["split", "acct"], how="left",
                     maintain_order="left")
             .join(own, left_on=["split", side, "event"], right_on=["split", "acct", "event"],
                   how="left", maintain_order="left"))
        out[col][m["i"].to_numpy()] = (m["n_all"].fill_null(0) - m["n_own"].fill_null(0)).to_numpy()

    shape = np.nan_to_num(out.pop("shape")).astype(np.int8)
    cols = []
    for k in names:
        if k == "shape":
            cols.append(pl.Series(f"{PREFIX}_shape", shape, dtype=pl.Int8))
            continue
        s = pl.Series(f"{PREFIX}_{k}", out[k], nan_to_null=True)
        cols.append(s.cast(INT_COLS.get(k, pl.Float32)))
    return pl.DataFrame(cols)
