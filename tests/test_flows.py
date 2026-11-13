"""Synthetic checks of the moneyflow family: planted chains, bursts and fan-ins, and near-misses."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import flows

H = 3600
MERCHANT = 10_000_000


def ledger(rows: list[dict]) -> pl.DataFrame:
    """A tiny ledger with every column the family reads; ``row`` follows list order."""
    full = []
    for k, r in enumerate(rows):
        x = {"is_test": False, "txn_type": "PAY", "new_payee": 1, "balance_before": 1e6}
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
    return df[f"mf_{name}"].to_list()


def chain(
    start: int, accts: list[int], t0: int, first: float, ratio: float = 0.98, delay: int = 600, **kw
) -> list[dict]:
    out, amt = [], first
    for k in range(len(accts) - 1):
        out.append(
            {
                "sender": accts[k],
                "receiver": accts[k + 1],
                "ts": t0 + k * delay,
                "amount": round(amt),
                **kw,
            }
        )
        amt *= ratio
    return out


def test_chain_links_and_near_misses():
    rows = chain(0, [1, 2, 3, 4], 0, 50_000)  # rows 0-2: a chain
    rows += chain(0, [11, 12, 13], 0, 50_000, ratio=0.996)  # 3-4: ratio above the band
    rows += chain(0, [21, 22, 23], 0, 50_000, delay=21_601)  # 5-6: hop slower than 6 h
    rows += chain(0, [31, 32, 33], 0, 4_900)  # 7-8: first amount < 5,000
    rows += chain(0, [41, 42, 43], 0, 50_000, delay=2_000)  # 9-10: slow chain
    rows += [
        {"sender": 51, "receiver": 52, "ts": 0, "amount": 50_000},
        {"sender": 52, "receiver": 53, "ts": 600, "amount": 49_000, "is_test": True},
    ]
    out = flows.build(ledger(rows))
    assert col(out, "shape") == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert col(out, "event")[:3] == [0, 0, 0]
    assert col(out, "size")[:3] == [3, 3, 3]
    assert col(out, "pos")[:3] == [0, 1, 2]
    assert col(out, "span_s")[:3] == [1200, 1200, 1200]
    assert col(out, "ch_ratio_in")[1] == np.float32(0.98)
    assert col(out, "ch_delay_out")[0] == 600
    assert col(out, "ch_first_amount")[2] == 50_000
    assert col(out, "ch_slow")[:3] == [0, 0, 0]
    assert col(out, "ch_slow")[9:11] == [1, 1]
    # the near-miss hops still show up as soft pass-through partners
    assert col(out, "pt_fwd_ratio")[3] == np.float32(0.996)
    assert col(out, "pt_fwd_delay")[5] is None


def inflows(
    receiver: int, senders: list[int], t0: int, gap: int = 2 * H, amount: float = 1000, **kw
) -> list[dict]:
    return [
        {"sender": s, "receiver": receiver, "ts": t0 + k * gap, "amount": amount, **kw}
        for k, s in enumerate(senders)
    ]


def test_burst_with_onward_transfers():
    rows = inflows(100, list(range(1, 9)), 0)  # rows 0-7: 8 senders
    last = 7 * 2 * H
    rows += [
        {"sender": 100, "receiver": 200, "ts": last + 3000, "amount": 4000},  # 8 onward
        {"sender": 100, "receiver": 201, "ts": last + 9000, "amount": 1500},  # 9 onward
        {"sender": 100, "receiver": 202, "ts": last + 9001, "amount": 1500},  # 10 too late
        {"sender": 100, "receiver": 203, "ts": last + 100, "amount": 1500.5},
    ]  # 11 paise
    out = flows.build(ledger(rows))
    assert col(out, "shape") == [2] * 8 + [3, 3, 0, 0]
    assert set(col(out, "event")[:10]) == {0}
    assert col(out, "size")[0] == 10
    assert col(out, "bu_senders")[0] == 8
    assert col(out, "bu_inflows")[9] == 8
    assert col(out, "bu_sum")[0] == 8000
    assert col(out, "bu_onward_n")[0] == 2
    assert col(out, "bu_onward_rank")[8:10] == [1, 2]
    assert col(out, "bu_onward_delay")[8:10] == [3000, 9000]
    assert col(out, "bu_onward_share")[8] == np.float32(0.5)
    assert col(out, "cashout_senders")[8] == 8
    assert col(out, "r_ring_n")[0] == 0  # own event excluded


def test_fan_in_and_burst_near_misses():
    rows = inflows(100, list(range(1, 8)), 0)  # 0-6: 7 new senders -> fan-in
    rows += inflows(200, [11, 12], 0)  # 7-8: 2 senders -> nothing
    rows += inflows(300, [21, 22, 23], 0, new_payee=0)  # 9-11: not new payees
    rows += inflows(400, [31, 32, 33, 34], 0, amount=1005)  # 12-15: not multiple of 10
    rows += inflows(500, list(range(41, 45)), 0)  # 16-19: 4 senders ...
    rows += inflows(500, list(range(45, 49)), 7 * 2 * H + 86_401)  # 20-23: ... and 4 after 1 day
    rows += inflows(600, [51, 52, 53], 0, amount=5010)  # 24-26: above 5,000
    out = flows.build(ledger(rows))
    shape = col(out, "shape")
    assert shape[:7] == [4] * 7
    assert col(out, "fan_senders")[0] == 7
    assert shape[7:16] == [0] * 9
    assert shape[16:24] == [4] * 8
    assert col(out, "event")[16:24] == [16] * 4 + [20] * 4
    assert shape[24:] == [0] * 3
    assert col(out, "run_senders")[7] == 2
    assert col(out, "run_new_share")[9] == 0


def test_bal102_split_and_order_independence():
    rows = chain(0, [1, 2, 3, 4], 0, 50_000)
    rows += inflows(100, list(range(10, 18)), 0)
    rows += [{"sender": 100, "receiver": 300, "ts": 50_000, "amount": 1000, "balance_before": 1020}]
    rows += inflows(700, [31, 32, 33], 0)
    rows += inflows(700, [34, 35], 3600, is_test=True)  # same receiver, other split
    led = ledger(rows)
    out = flows.build(led)
    assert col(out, "bal102")[11] == 1
    assert col(out, "shape")[11] == 3
    assert col(out, "shape")[12:] == [4, 4, 4, 0, 0]
    perm = np.random.default_rng(0).permutation(led.height)
    shuffled = flows.build(led[perm])
    assert shuffled.equals(out[perm])
    assert all(c.startswith("mf_") for c in out.columns)
    assert out.width == 34
    assert out["mf_event"].dtype == pl.Int64 and out["mf_shape"].dtype == pl.Int8
