"""Drain chains: sim_swap / account_takeover sweeps that leave an exact balance trail.

Both generators empty an account in 2-4 quick payments, and each payment leaves the next row's
balance at c x the previous one (c = 0.25 for the sim-swap mechanism, 0.35 for takeover). A
sender's row links to the sender's next row, or the next-but-one if the next does not link,
when the balance stepped by one of the two constants; linked rows form a chain, and a chain is
an event when it lasts at most 3 h. Chains stay inside one side of the train/test split.

Shapes: 1 = 0.25 chain on a fresh SIM throughout; 2 = other 0.25; 3 = 0.35 chain on one extra
phone; 4 = other 0.35.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ringtrace.tables import HOME_DEVICES

PREFIX = "drn"

HOUR = 3600
CONSTANTS = ((25, 0.25), (35, 0.35))
TOL = 0.0075  # rupees; generator residual <= 0.005 (0.25) / 0.0065 (0.35)
MAX_SPAN_S = 3 * HOUR
SIM_FRESH_H = 30
PHONE_FIRST = 423_000  # extra phones minted by the counter (below: reserved bot devices)
PHONE_ROWS = (2, 4)


def step_code(b_from: np.ndarray, b_to: np.ndarray) -> np.ndarray:
    """25 or 35 when b_to = c x b_from (to TOL), else 0."""
    out = np.zeros(len(b_from), dtype=np.int8)
    for code, c in CONSTANTS[::-1]:
        out = np.where(np.abs(b_to - c * b_from) <= TOL, np.int8(code), out)
    return out


def _sender_order(led: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    split = led["is_test"].to_numpy().astype(np.int64)
    sender = led["sender"].to_numpy().astype(np.int64)
    ts = led["ts"].to_numpy().astype(np.int64)
    row = led["row"].to_numpy().astype(np.int64)
    order = np.lexsort((row, ts, sender, split))
    return order, (sender * 2 + split)[order]


def _links(grp: np.ndarray, bal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                      np.ndarray]:
    """Edges in sorted positions (src, dst, code) and the step to the immediate next row."""
    n = len(grp)
    step1 = np.zeros(n, dtype=np.int8)
    if n > 1:
        step1[:-1] = np.where(grp[1:] == grp[:-1], step_code(bal[:-1], bal[1:]), 0)
    step2 = np.zeros(n, dtype=np.int8)
    if n > 2:
        step2[:-2] = np.where(grp[2:] == grp[:-2], step_code(bal[:-2], bal[2:]), 0)
    s1 = np.flatnonzero(step1)
    s2 = np.flatnonzero((step2 > 0) & (step1 == 0))
    src = np.concatenate([s1, s2])
    dst = np.concatenate([s1 + 1, s2 + 2])
    code = np.concatenate([step1[s1], step2[s2]])
    return src, dst, code, step1


def masked(n: int, at: np.ndarray, values: np.ndarray, dtype) -> pl.Series:
    """A length-n series holding ``values`` at positions ``at`` and null elsewhere."""
    values = np.asarray(values)
    full = np.zeros(n, dtype=values.dtype if len(values) else np.int64)
    full[at] = values
    mask = np.zeros(n, dtype=bool)
    mask[at] = True
    frame = pl.DataFrame({"v": full, "m": mask})
    return frame.select(pl.when(pl.col("m")).then(pl.col("v").cast(dtype))).to_series()


def build(led: pl.DataFrame) -> pl.DataFrame:
    n = led.height
    order, grp = _sender_order(led)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    bal = led["balance_before"].to_numpy().astype(np.float64)[order]
    src, dst, code, step1 = _links(grp, bal)

    step_next = step1
    step_prev = np.zeros(n, dtype=np.int8)
    step_prev[1:] = step1[:-1]

    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    _, comp = connected_components(graph, directed=True, connection="weak")

    nodes = np.unique(np.concatenate([src, dst]))
    take = order[nodes]
    members = pl.DataFrame({
        "pos": nodes,
        "comp": comp[nodes],
        "row": led["row"].to_numpy().astype(np.int64)[take],
        "ts": led["ts"].to_numpy().astype(np.int64)[take],
        "sim": led["sim_age_hours"].to_numpy()[take],
        "device": led["device"].to_numpy().astype(np.int64)[take],
    })
    # a chain's constant is the one of its earliest link
    edge_c = (pl.DataFrame({"comp": comp[src], "src": src, "c": code})
              .sort("src").group_by("comp").agg(pl.col("c").first()))
    chains = (members.group_by("comp").agg(
        pl.col("row").min().alias("event"),
        pl.len().alias("size"),
        (pl.col("ts").max() - pl.col("ts").min()).alias("span"),
        (pl.col("sim") < SIM_FRESH_H).all().alias("sim_fresh"),
        ((pl.col("device").n_unique() == 1) & (pl.col("device").min() >= HOME_DEVICES))
        .alias("one_phone"),
    ).join(edge_c, on="comp").filter(pl.col("span") <= MAX_SPAN_S).with_columns(
        pl.when(pl.col("c") == 25)
        .then(pl.when(pl.col("sim_fresh")).then(1).otherwise(2))
        .otherwise(pl.when(pl.col("one_phone")).then(3).otherwise(4))
        .cast(pl.Int8).alias("shape")))
    rows = (members.join(chains, on="comp", how="inner")
            .with_columns((pl.col("pos").rank("ordinal").over("comp") - 1).alias("rank_in")))
    at = order[rows["pos"].to_numpy()]  # input positions of chain rows
    in_chain = np.zeros(n, dtype=bool)
    in_chain[at] = True

    def on_chain(values, dtype) -> pl.Series:
        return masked(n, at, values, dtype)

    # chains paying this row's receiver (whole ledger), the row's own chain excluded
    receiver = led["receiver"].to_numpy().astype(np.int64)
    pays = (pl.DataFrame({"receiver": receiver[at], "event": rows["event"].to_numpy()})
            .unique().group_by("receiver").agg(pl.len().alias("k")))
    k = (pl.DataFrame({"receiver": receiver})
         .join(pays, on="receiver", how="left", maintain_order="left")["k"]
         .fill_null(0).to_numpy())
    rcv_chains = k.astype(np.int64) - in_chain

    device = led["device"].to_numpy().astype(np.int64)
    dev_rows = pl.DataFrame({"d": device}).select(pl.len().over("d")).to_series().to_numpy()
    phone = (device >= PHONE_FIRST) & (dev_rows >= PHONE_ROWS[0]) & (dev_rows <= PHONE_ROWS[1])

    shape = np.zeros(n, dtype=np.int8)
    shape[at] = rows["shape"].to_numpy()
    return pl.DataFrame([
        on_chain(rows["event"].to_numpy(), pl.Int64).alias(f"{PREFIX}_event"),
        pl.Series(f"{PREFIX}_shape", shape, dtype=pl.Int8),
        on_chain(rows["size"].to_numpy(), pl.Int16).alias(f"{PREFIX}_size"),
        on_chain(rows["span"].to_numpy(), pl.Float32).alias(f"{PREFIX}_span_s"),
        on_chain(rows["c"].to_numpy(), pl.Int8).alias(f"{PREFIX}_c"),
        on_chain(rows["rank_in"].to_numpy(), pl.Int16).alias(f"{PREFIX}_pos"),
        pl.Series(f"{PREFIX}_step_prev", step_prev[rank], dtype=pl.Int8),
        pl.Series(f"{PREFIX}_step_next", step_next[rank], dtype=pl.Int8),
        pl.Series(f"{PREFIX}_rcv_chains", rcv_chains, dtype=pl.Int32),
        pl.Series(f"{PREFIX}_phone", phone, dtype=pl.Int8),
    ])
