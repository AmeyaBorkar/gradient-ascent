"""Money moving through mule and layering accounts (spec s5): hop chains, mule bursts, fan-ins.

Three event kinds, one event per row, in priority order chain > burst > fan-in:

* chain (shape 1): a P2P row of at least 5,000 whose receiver sends 96.49-99.51% of it on within
  6 h (any type); linked rows form connected components.
* burst (shape 2 inflow, 3 onward): one receiver's run of small whole P2P payments (multiple of 10,
  400-5,000, gaps <= 1 day) from >= 8 distinct senders, plus the receiver's own whole-rupee P2P
  payments of >= 500 from the first inflow to 2.5 h after the last.
* fan-in (shape 4): the same kind of run, new payees only, outside chains and bursts, from 3-7
  distinct senders. The sender count is kept: 4-7 reads layering, 8+ is a mule burst.

Soft, event-free columns describe every row's surroundings (the receiver's candidate run, the best
6 h pass-through partner, the sender's cash-out window, the 1/1.02 balance sweep, ring roles of the
two accounts). Everything stays inside one split (``is_test``); labels and the two columns that
die in October are never read.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ringtrace.tables import MERCHANT_OFFSET

PREFIX = "mf"

HOP_MIN_AMOUNT = 5000.0
HOP_WINDOW = 21_600
HOP_RATIO = (0.9649, 0.9951)
SLOW_HOP = 1_560
RUN_GAP = 86_400
INFLOW_RANGE = (400.0, 5000.0)
BURST_SENDERS = 8
ONWARD_MIN = 500.0
ONWARD_TAIL = 9_000
FAN_SENDERS = (3, 7)
PASS_TARGET = 0.98
SWEEP_RATIO = 1 / 1.02
SWEEP_TOL = 2e-4

CHAIN, INFLOW, ONWARD, FAN = 1, 2, 3, 4

INT_COLS = {"event": pl.Int64, "shape": pl.Int8, "size": pl.Int32, "pos": pl.Int32}


def _base(led: pl.DataFrame) -> pl.DataFrame:
    return led.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("i"),
        pl.col("row").cast(pl.Int64),
        pl.col("is_test").cast(pl.Int8).alias("split"),
        pl.col("sender").cast(pl.Int64).alias("snd"),
        pl.col("receiver").cast(pl.Int64).alias("rcv"),
        pl.col("to_person").fill_null(False).alias("person"),
        (pl.col("txn_type") == "PAY").fill_null(False).alias("pay"),
        pl.col("ts").cast(pl.Int64),
        pl.col("amount").cast(pl.Float64).alias("amt"),
        (pl.col("amount") * 100).round().cast(pl.Int64).alias("cents"),
        pl.col("new_payee").fill_null(0).cast(pl.Float64).alias("new"),
        pl.col("balance_before").cast(pl.Float64).alias("bal"),
    )


def _hops(b: pl.DataFrame) -> pl.DataFrame:
    """Every P2P row ``a`` paired with each row ``b`` its receiver sent within the next 6 h."""
    src = b.filter(pl.col("person")).select(
        pl.col("i").alias("a"), pl.col("row").alias("row_a"), "split",
        pl.col("rcv").alias("acct"), pl.col("ts").alias("ta"), pl.col("amt").alias("xa"))
    dst = b.select(
        pl.col("i").alias("b"), pl.col("row").alias("row_b"), "split",
        pl.col("snd").alias("acct"), pl.col("ts").alias("tb"), pl.col("amt").alias("xb"))
    dt = pl.col("tb") - pl.col("ta")
    return (
        src.lazy().join(dst.lazy(), on=["split", "acct"])
        .filter((dt > 0) & (dt <= HOP_WINDOW))
        .with_columns(dt.alias("dt"), (pl.col("xb") / pl.col("xa")).alias("ratio"))
        .with_columns((pl.col("ratio") - PASS_TARGET).abs().alias("miss"))
        .collect()
    )


def _best(pairs: pl.DataFrame, by: str, other_row: str) -> pl.DataFrame:
    """Per ``by`` the partner with ratio closest to 0.98, then shortest delay, then lowest row."""
    return pairs.sort(by, "miss", "dt", other_row).unique(by, keep="first", maintain_order=True)


def _runs(c: pl.DataFrame) -> pl.DataFrame:
    """Number candidate runs: per (split, receiver) in time order, new run after a > 1 day gap."""
    c = c.sort("split", "rcv", "ts", "row")
    gap = pl.col("ts").diff().over("split", "rcv")
    return c.with_columns((gap.is_null() | (gap > RUN_GAP)).cum_sum().cast(pl.Int64).alias("run"))


def _run_stats(c: pl.DataFrame) -> pl.DataFrame:
    return c.group_by("run").agg(
        pl.col("split").first(), pl.col("rcv").first(),
        pl.col("snd").n_unique().alias("senders"), pl.len().alias("rows"),
        pl.col("ts").min().alias("first"), pl.col("ts").max().alias("last"),
        pl.col("new").mean().alias("new_share"), pl.col("amt").sum().alias("total"),
    ).sort("run")


def _put(out: dict[str, np.ndarray], name: str, idx, values) -> None:
    out[name][np.asarray(idx, dtype=np.int64)] = np.asarray(values, dtype=np.float64)


def build(led: pl.DataFrame) -> pl.DataFrame:
    b = _base(led)
    n = b.height
    names = ["event", "shape", "size", "span_s", "pos", "pos_frac", "since_first_s",
             "ch_delay_in", "ch_ratio_in", "ch_delay_out", "ch_ratio_out", "ch_slow",
             "ch_first_amount", "bu_senders", "bu_inflows", "bu_sum", "bu_new_share",
             "bu_onward_n", "bu_onward_rank", "bu_onward_share", "bu_onward_delay",
             "fan_senders", "run_senders", "run_rows", "run_span_s", "run_new_share",
             "pt_fwd_ratio", "pt_fwd_delay", "pt_back_ratio", "pt_back_delay",
             "cashout_senders", "bal102", "s_ring_n", "r_ring_n"]
    out = {k: np.full(n, np.nan) for k in names}
    shape = np.zeros(n, dtype=np.int8)
    key = np.full(n, -1, dtype=np.int64)

    # --- 6 h pass-through partners (soft) and chain links --------------------------------------
    hops = _hops(b)
    fwd = _best(hops, "a", "row_b")
    _put(out, "pt_fwd_ratio", fwd["a"], fwd["ratio"])
    _put(out, "pt_fwd_delay", fwd["a"], fwd["dt"])
    back = _best(hops, "b", "row_a")
    _put(out, "pt_back_ratio", back["b"], back["ratio"])
    _put(out, "pt_back_delay", back["b"], back["dt"])

    lo, hi = HOP_RATIO
    links = hops.filter((pl.col("xa") >= HOP_MIN_AMOUNT) & (pl.col("ratio") >= lo)
                        & (pl.col("ratio") <= hi))
    la, lb = links["a"].to_numpy(), links["b"].to_numpy()
    graph = coo_matrix((np.ones(len(la), dtype=np.int8), (la, lb)), shape=(n, n))
    _, comp = connected_components(graph, directed=True, connection="weak")
    in_chain = np.zeros(n, dtype=bool)
    in_chain[la] = True
    in_chain[lb] = True
    shape[in_chain] = CHAIN
    key[in_chain] = comp[in_chain]

    lin = _best(links, "b", "row_a")
    _put(out, "ch_delay_in", lin["b"], lin["dt"])
    _put(out, "ch_ratio_in", lin["b"], lin["ratio"])
    lout = _best(links, "a", "row_b")
    _put(out, "ch_delay_out", lout["a"], lout["dt"])
    _put(out, "ch_ratio_out", lout["a"], lout["ratio"])
    slow = pl.DataFrame({"c": comp[la], "dt": links["dt"].to_numpy()}).group_by("c").agg(
        (pl.col("dt").max() > SLOW_HOP).cast(pl.Float64).alias("slow"))
    chain_rows = b.filter(pl.Series(in_chain)).with_columns(pl.Series("c", comp[in_chain]))
    chain_rows = chain_rows.join(slow, on="c", how="left").sort("c", "ts", "row").with_columns(
        pl.col("amt").first().over("c").alias("first_amt"),
        pl.col("snd").first().over("c").alias("first_snd"))
    _put(out, "ch_slow", chain_rows["i"], chain_rows["slow"].fill_null(0.0))
    _put(out, "ch_first_amount", chain_rows["i"], chain_rows["first_amt"])

    # --- small whole P2P payments: the receiver's candidate runs ---------------------------------
    is_cand = (pl.col("person") & pl.col("pay") & (pl.col("cents") % 1000 == 0)
               & (pl.col("amt") >= INFLOW_RANGE[0]) & (pl.col("amt") <= INFLOW_RANGE[1]))
    soft = _runs(b.filter(is_cand))
    soft_stats = _run_stats(soft)
    soft = soft.join(soft_stats.select("run", "senders", "rows", "first", "last", "new_share"),
                     on="run", how="left")
    _put(out, "run_senders", soft["i"], soft["senders"])
    _put(out, "run_rows", soft["i"], soft["rows"])
    _put(out, "run_span_s", soft["i"], soft["last"] - soft["first"])
    _put(out, "run_new_share", soft["i"], soft["new_share"])

    # sender's cash-out window: an own candidate run with first < t <= last + 2.5 h
    left = b.select("i", "split", "snd", "ts").sort("ts")
    right = soft_stats.select("split", pl.col("rcv").alias("snd"), "first", "last",
                              "senders").sort("first")
    cash = left.join_asof(right, left_on="ts", right_on="first", by=["split", "snd"],
                          strategy="backward", allow_exact_matches=False,
                          check_sortedness=False)
    ok = (cash["ts"] <= cash["last"] + ONWARD_TAIL).fill_null(False)
    _put(out, "cashout_senders", cash["i"],
         np.where(ok.to_numpy(), cash["senders"].fill_null(0).to_numpy(), 0.0))

    # --- mule bursts and their onward transfers ------------------------------------------------
    b = b.with_columns(pl.Series("taken", shape != 0))
    cand = _runs(b.filter(is_cand & ~pl.col("taken")))
    stats = _run_stats(cand)
    bursts = stats.filter(pl.col("senders") >= BURST_SENDERS)
    inflow = cand.join(bursts.select("run"), on="run")
    ii = inflow["i"].to_numpy()
    shape[ii] = INFLOW
    key[ii] = n + inflow["run"].to_numpy()

    b = b.with_columns(pl.Series("taken", shape != 0))
    onward = (
        b.filter(pl.col("person") & pl.col("pay") & (pl.col("cents") % 100 == 0)
                 & (pl.col("amt") >= ONWARD_MIN) & ~pl.col("taken"))
        .join(bursts.select("run", "split", pl.col("rcv").alias("snd"), "last", "total",
                            pl.col("first").alias("bfirst")), on=["split", "snd"])
        .filter((pl.col("ts") > pl.col("bfirst")) & (pl.col("ts") <= pl.col("last") + ONWARD_TAIL))
        .sort("i", "run").unique("i", keep="first", maintain_order=True)
        .sort("run", "ts", "row")
        .with_columns(pl.int_range(1, pl.len() + 1).over("run").alias("rank"))
    )
    oi = onward["i"].to_numpy()
    shape[oi] = ONWARD
    key[oi] = n + onward["run"].to_numpy()
    _put(out, "bu_onward_rank", oi, onward["rank"])
    _put(out, "bu_onward_share", oi, onward["amt"] / onward["total"])
    _put(out, "bu_onward_delay", oi, onward["ts"] - onward["last"])
    n_onward = onward.group_by("run").agg(pl.len().alias("onward_n"))
    bstat = bursts.join(n_onward, on="run", how="left").with_columns(
        pl.col("onward_n").fill_null(0))
    members = pl.concat([inflow.select("i", "run"), onward.select("i", "run")]).join(
        bstat, on="run", how="left")
    _put(out, "bu_senders", members["i"], members["senders"])
    _put(out, "bu_inflows", members["i"], members["rows"])
    _put(out, "bu_sum", members["i"], members["total"])
    _put(out, "bu_new_share", members["i"], members["new_share"])
    _put(out, "bu_onward_n", members["i"], members["onward_n"])

    # --- fan-ins: new-payee runs with 3-7 senders outside chains and bursts --------------------
    b = b.with_columns(pl.Series("taken", shape != 0))
    fan = _runs(b.filter(is_cand & (pl.col("new") == 1) & ~pl.col("taken")))
    fstats = _run_stats(fan).filter(pl.col("senders").is_between(*FAN_SENDERS))
    fan = fan.join(fstats.select("run", "senders"), on="run")
    fi = fan["i"].to_numpy()
    shape[fi] = FAN
    key[fi] = 2 * n + fan["run"].to_numpy()
    _put(out, "fan_senders", fi, fan["senders"])

    # --- event frame ----------------------------------------------------------------------------
    has = key >= 0
    ev = b.filter(pl.Series(has)).select("i", "row", "ts", "split", "snd", "rcv").with_columns(
        pl.Series("key", key[has]))
    ev = ev.sort("key", "ts", "row").with_columns(
        pl.col("row").min().over("key").alias("event"),
        pl.len().over("key").alias("size"),
        (pl.col("ts").max().over("key") - pl.col("ts").min().over("key")).alias("span"),
        pl.int_range(pl.len()).over("key").alias("pos"),
        (pl.col("ts") - pl.col("ts").min().over("key")).alias("since"),
    )
    ei = ev["i"].to_numpy()
    size = ev["size"].to_numpy().astype(np.float64)
    pos = ev["pos"].to_numpy().astype(np.float64)
    _put(out, "event", ei, ev["event"])
    _put(out, "size", ei, size)
    _put(out, "span_s", ei, ev["span"])
    _put(out, "pos", ei, pos)
    _put(out, "pos_frac", ei, pos / np.maximum(size - 1, 1))
    _put(out, "since_first_s", ei, ev["since"])

    # --- 1/1.02 sweep and ring roles of both accounts (own event excluded) ----------------------
    amt = b["amt"].to_numpy()
    bal = b["bal"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["bal102"] = (np.abs(amt / bal - SWEEP_RATIO) < SWEEP_TOL).astype(np.float64)

    # receivers of every event row (chain hops, mule, onward receivers, fan-in collector) and the
    # first-hop sender of every chain
    first_hop = chain_rows.select("split", pl.col("first_snd").alias("acct"),
                                  pl.col("c").cast(pl.Int64).alias("key"))
    roles = pl.concat([
        ev.select("split", pl.col("rcv").alias("acct"), "key"),
        first_hop,
    ]).filter(pl.col("acct") < MERCHANT_OFFSET).unique()
    per_acct = roles.group_by("split", "acct").agg(pl.len().alias("n_roles"))
    rows = b.select("i", "split", "snd", "rcv").with_columns(pl.Series("key", key))
    for side, col in (("snd", "s_ring_n"), ("rcv", "r_ring_n")):
        m = (rows.join(per_acct, left_on=["split", side], right_on=["split", "acct"], how="left",
                       maintain_order="left")
             .join(roles.with_columns(pl.lit(1).alias("own")), left_on=["split", side, "key"],
                   right_on=["split", "acct", "key"], how="left", maintain_order="left"))
        out[col][m["i"].to_numpy()] = (m["n_roles"].fill_null(0) - m["own"].fill_null(0)).to_numpy()

    cols = []
    for k in names:
        s = pl.Series(f"{PREFIX}_{k}", out[k], nan_to_null=True)
        if k == "shape":
            s = pl.Series(f"{PREFIX}_{k}", shape, dtype=pl.Int8)
        cols.append(s.cast(INT_COLS.get(k, pl.Float32)))
    return pl.DataFrame(cols)
