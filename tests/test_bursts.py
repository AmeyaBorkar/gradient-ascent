"""Bot bursts are found on planted device sessions and not on near-misses."""

from __future__ import annotations

import polars as pl

from ringtrace.evidence import bursts

T0 = 1_720_000_000
MERCHANT = 10_000_000

DEFAULTS = {
    "is_test": False, "sender": 1, "receiver": MERCHANT + 1, "to_person": False, "device": 1,
    "ts": 0, "login_to_txn_sec": 40.0, "month": 7,
}
SCHEMA = {
    "is_test": pl.Boolean, "sender": pl.Int64, "receiver": pl.Int64, "to_person": pl.Boolean,
    "device": pl.Int64, "ts": pl.Int64, "login_to_txn_sec": pl.Float64, "month": pl.Int8,
}


def ledger(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame([{**DEFAULTS, **r} for r in rows], schema=SCHEMA)
    return df.with_row_index("row").with_columns(
        pl.col("row").cast(pl.Int32), (pl.col("ts") + T0).alias("ts"))


def bursts_of(rows: list[dict]) -> pl.DataFrame:
    led = ledger(rows)
    out = bursts.build(led)
    assert out.height == led.height
    assert all(c.startswith("bot_") for c in out.columns)
    assert all(dt.is_numeric() for dt in out.dtypes)
    return out


def session(device: int, n: int, senders, t0: int = 0, gap: int = 20, merchants=(1, 2),
            **extra) -> list[dict]:
    return [{"device": device, "ts": t0 + k * gap, "sender": senders[k % len(senders)],
             "receiver": MERCHANT + merchants[k % len(merchants)], **extra} for k in range(n)]


def test_farm_and_small_bursts():
    rows = session(420_500, 25, list(range(100, 125)))                       # farm
    rows += session(420_600, 20, [200, 201, 202], login_to_txn_sec=3.0)      # scripted
    rows += session(5, 20, [300, 301, 302], login_to_txn_sec=46.0)           # human
    out = bursts_of(rows)
    assert out["bot_event"].to_list() == [0] * 25 + [25] * 20 + [45] * 20
    assert out["bot_shape"].to_list() == [1] * 25 + [2] * 20 + [3] * 20
    assert out["bot_size"].to_list() == [25] * 25 + [20] * 40
    assert out["bot_span_s"].to_list()[:1] == [480.0]
    assert out["bot_pos"].to_list()[:3] == [0, 1, 2]
    assert out["bot_accounts"].to_list()[:1] + out["bot_accounts"].to_list()[25:26] == [25, 3]
    assert out["bot_merchants"].to_list()[0] == 2
    assert out["bot_range"].to_list() == [1] * 45 + [0] * 20


def test_farm_with_a_repeated_account():
    senders = list(range(100, 119)) + [100]                                  # 20 rows, 19 accts
    out = bursts_of(session(421_000, 20, senders))
    assert set(out["bot_shape"].to_list()) == {1}


def test_near_misses():
    rows = session(9, 9, [1, 2, 3])                                         # 9 rows only
    rows += session(10, 9, [1, 2, 3], gap=20) + session(10, 11, [1, 2, 3], t0=160 + 301)
    rows += session(11, 6, [4], gap=20) + session(11, 6, [4], t0=100, is_test=True)
    out = bursts_of(rows)
    ev = out["bot_event"].to_list()
    assert ev[:9] == [None] * 9
    assert ev[9:18] == [None] * 9 and ev[18:29] == [18] * 11                 # split at 301 s
    assert ev[29:] == [None] * 12                                            # split by is_test
    assert out["bot_shape"].to_list()[:18] == [0] * 18


def test_sender_and_receiver_counts():
    rows = session(420_100, 12, [500, 501, 502, 503])
    rows += session(420_200, 12, [500, 601, 602, 603], t0=10_000)
    rows.append({"sender": 500, "ts": 50_000, "device": 500})               # plain row
    rows.append({"sender": 9, "ts": 60_000, "device": 9, "receiver": 500, "to_person": True})
    rows.append({"sender": 9, "ts": 70_000, "device": 9, "receiver": 999, "to_person": True})
    out = bursts_of(rows)
    snd = out["bot_snd_bursts"].to_list()
    assert snd[0] == 1 and snd[1] == 0 and snd[12] == 1 and snd[24] == 2
    assert out["bot_rcv_bursts"].to_list()[25:27] == [2, 0]


def test_merchant_side_session():
    rows = [{"sender": 700 + k, "device": 700 + k, "ts": k * 100, "receiver": MERCHANT + 50}
            for k in range(4)]                                               # 4-row session
    rows += [{"sender": 800 + k, "device": 800 + k, "ts": k * 100, "receiver": MERCHANT + 60}
             for k in range(3)]                                              # only 3 rows
    rows += [{"sender": 900 + k, "device": 900 + k, "ts": k * 100, "receiver": MERCHANT + 70}
             for k in range(4)]
    rows += [{"sender": 1000 + k, "device": 1000 + k, "ts": 100_000 + k * 1000,
              "receiver": MERCHANT + 70} for k in range(61)]                 # busy merchant
    rows += [{"sender": 2000 + k, "device": 2000 + k, "ts": k * 100, "receiver": 3000,
              "to_person": True} for k in range(4)]                          # a person
    out = bursts_of(rows)
    mb = out["bot_mburst"].to_list()
    assert mb[:4] == [1] * 4
    assert mb[4:7] == [0] * 3
    assert mb[7:11] == [0] * 4
    assert mb[-4:] == [0] * 4


def test_order_independent():
    rows = session(420_500, 12, list(range(100, 112))) + session(7, 3, [1])
    rows.append({"sender": 100, "ts": 5, "device": 420_500, "receiver": MERCHANT + 1})
    led = ledger(rows)
    ref = pl.concat([led.select("row"), bursts.build(led)], how="horizontal")
    shuffled = led.sample(fraction=1.0, shuffle=True, seed=5)
    got = pl.concat([shuffled.select("row"), bursts.build(shuffled)], how="horizontal")
    assert got.sort("row").equals(ref)
