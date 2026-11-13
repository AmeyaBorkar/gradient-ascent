"""Drain chains are found on planted balance trails and not on near-misses."""

from __future__ import annotations

import polars as pl

from ringtrace.evidence import drains

T0 = 1_720_000_000
HOUR = 3600

DEFAULTS = {
    "is_test": False, "sender": 1, "receiver": 900, "device": 1, "ts": 0,
    "balance_before": 10_000.0, "sim_age_hours": 5000.0,
}
SCHEMA = {
    "is_test": pl.Boolean, "sender": pl.Int64, "receiver": pl.Int64, "device": pl.Int64,
    "ts": pl.Int64, "balance_before": pl.Float64, "sim_age_hours": pl.Float64,
}


def ledger(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame([{**DEFAULTS, **r} for r in rows], schema=SCHEMA)
    return df.with_row_index("row").with_columns(
        pl.col("row").cast(pl.Int32), (pl.col("ts") + T0).alias("ts"))


def drains_of(rows: list[dict]) -> pl.DataFrame:
    led = ledger(rows)
    out = drains.build(led)
    assert out.height == led.height
    assert all(c.startswith("drn_") for c in out.columns)
    assert all(dt.is_numeric() for dt in out.dtypes)
    return out


def chain(sender: int, c: float, n: int, t0: int = 0, dt: int = 600, b0: float = 10_000.0,
          **extra) -> list[dict]:
    rows, b = [], b0
    for k in range(n):
        rows.append({"sender": sender, "ts": t0 + k * dt, "balance_before": b,
                     "receiver": 1000 + sender * 10 + k, **extra})
        b = round(c * b, 2)
    return rows


def test_sim_chain_shape_1_and_2():
    rows = chain(5, 0.25, 3, sim_age_hours=4.0)
    rows += chain(6, 0.25, 2, sim_age_hours=4.0)
    rows[-1]["sim_age_hours"] = 31.0  # one stale SIM row -> shape 2
    out = drains_of(rows)
    assert out["drn_event"].to_list() == [0, 0, 0, 3, 3]
    assert out["drn_shape"].to_list() == [1, 1, 1, 2, 2]
    assert out["drn_size"].to_list() == [3, 3, 3, 2, 2]
    assert out["drn_span_s"].to_list() == [1200.0, 1200.0, 1200.0, 600.0, 600.0]
    assert out["drn_c"].to_list() == [25] * 5
    assert out["drn_pos"].to_list() == [0, 1, 2, 0, 1]
    assert out["drn_step_prev"].to_list() == [0, 25, 25, 0, 25]
    assert out["drn_step_next"].to_list() == [25, 25, 0, 25, 0]


def test_takeover_chain_shape_3_and_4():
    rows = chain(7, 0.35, 3, device=423_500)          # one extra phone
    rows += chain(8, 0.35, 2, device=8)               # own phone
    rows += chain(9, 0.35, 2, device=423_600)
    rows[-1]["device"] = 423_601                      # two phones -> other
    out = drains_of(rows)
    assert out["drn_shape"].to_list() == [3, 3, 3, 4, 4, 4, 4]
    assert out["drn_c"].to_list() == [35] * 7


def test_next_but_one_link():
    rows = [
        {"sender": 3, "ts": 0, "balance_before": 10_000.0},
        {"sender": 3, "ts": 300, "balance_before": 7777.0},   # unrelated row in between
        {"sender": 3, "ts": 900, "balance_before": 2500.0},
    ]
    out = drains_of(rows)
    assert out["drn_event"].to_list() == [0, None, 0]
    assert out["drn_size"].to_list() == [2, None, 2]
    assert out["drn_shape"].to_list() == [2, 0, 2]
    assert out["drn_step_next"].to_list() == [0, 0, 0]


def test_near_misses():
    rows = []
    rows += [{"sender": 20, "ts": 0, "balance_before": 10_000.0},
             {"sender": 20, "ts": 600, "balance_before": 2500.01}]      # residual 0.01
    rows += [{"sender": 21, "ts": 0, "balance_before": 10_000.0},
             {"sender": 21, "ts": 3 * HOUR + 1, "balance_before": 2500.0}]  # too slow
    rows += [{"sender": 22, "ts": 0, "balance_before": 10_000.0},
             {"sender": 22, "ts": 600, "balance_before": 3000.0}]       # c = 0.30
    rows += [{"sender": 23, "ts": 0, "balance_before": 10_000.0},
             {"sender": 23, "ts": 600, "balance_before": 2500.0, "is_test": True}]  # split
    rows += [{"sender": 24, "ts": 0, "balance_before": 10_000.0},
             {"sender": 25, "ts": 600, "balance_before": 2500.0}]       # other sender
    rows += [{"sender": 26, "ts": 0, "balance_before": 10_000.0},
             {"sender": 26, "ts": 100, "balance_before": 9000.0},
             {"sender": 26, "ts": 200, "balance_before": 8000.0},
             {"sender": 26, "ts": 300, "balance_before": 2500.0}]       # three rows later
    out = drains_of(rows)
    assert out["drn_event"].null_count() == len(rows)
    assert out["drn_shape"].sum() == 0
    # the slow pair is still a step between consecutive rows
    assert out["drn_step_next"].to_list()[2] == 25


def test_tolerance_inside():
    rows = [{"sender": 30, "ts": 0, "balance_before": 10_000.0},
            {"sender": 30, "ts": 600, "balance_before": 2500.007}]
    out = drains_of(rows)
    assert out["drn_event"].to_list() == [0, 0]


def test_receiver_chains_and_phone():
    rows = chain(40, 0.25, 2, sim_age_hours=3.0)
    rows += chain(41, 0.25, 2, sim_age_hours=3.0)
    for r in rows:
        r["receiver"] = 77
    rows.append({"sender": 42, "ts": 50_000, "receiver": 77, "balance_before": 123.0})
    rows += [{"sender": 50 + k, "ts": 90_000 + k, "device": 423_100, "balance_before": 1.0 + k}
             for k in range(3)]
    rows += [{"sender": 60 + k, "ts": 90_000 + k, "device": 423_200, "balance_before": 1.0 + k}
             for k in range(5)]
    rows += [{"sender": 70 + k, "ts": 90_000 + k, "device": 421_000, "balance_before": 1.0 + k}
             for k in range(2)]
    out = drains_of(rows)
    assert out["drn_rcv_chains"].to_list()[:5] == [1, 1, 1, 1, 2]
    assert out["drn_phone"].to_list()[5:] == [1] * 3 + [0] * 5 + [0] * 2


def test_order_independent():
    rows = chain(5, 0.25, 3, sim_age_hours=4.0) + chain(7, 0.35, 2, device=423_500)
    rows.append({"sender": 5, "ts": 300, "balance_before": 1.0})
    led = ledger(rows)
    ref = pl.concat([led.select("row"), drains.build(led)], how="horizontal")
    shuffled = led.sample(fraction=1.0, shuffle=True, seed=11)
    got = pl.concat([shuffled.select("row"), drains.build(shuffled)], how="horizontal")
    assert got.sort("row").equals(ref)
