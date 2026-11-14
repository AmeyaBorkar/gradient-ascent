"""Synthetic checks for the October drift scan: rates, chi-square, pile-up, suppression, bands."""

from __future__ import annotations

import math
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from drift_core import (  # noqa: E402
    HOP_DELAY_BAND_S,
    TEST,
    bait_amount,
    bait_answer,
    bait_gap,
    bot_gap,
    bot_rows,
    burst_senders,
    chi_square_scaled,
    collect_offset,
    drain_span,
    drain_step,
    evening,
    fan_senders,
    hop_delay,
    hop_ratio_high,
    hop_ratio_low,
    month_cell,
    month_comparison,
    per_100k,
    pileup_flags,
    repeat_gap,
    ring_amount,
    shown_count,
    size_table,
)

EPOCH = datetime(2026, 7, 1)
MERCHANT = 10_000_000
H = 3600
DAY = 86_400


def frame(rows: list[dict]) -> pl.DataFrame:
    """A tiny ledger. Only ``ts`` sets the clock; everything else has a quiet default."""

    def col(name, default, dtype):
        return pl.Series(name, [r.get(name, default) for r in rows], dtype=dtype)

    ts = [int(r.get("ts", 0)) for r in rows]
    return pl.DataFrame(
        [
            col("is_test", False, pl.Boolean),
            col("sender", 1, pl.Int64),
            col("receiver", 2, pl.Int64),
            col("device", 3, pl.Int64),
            col("to_person", True, pl.Boolean),
            col("amount", 123.45, pl.Float64),
            col("txn_type", "PAY", pl.Utf8),
            col("merchant_category", None, pl.Utf8),
            col("balance_before", 1_000_000.0, pl.Float64),
            col("new_payee", 0, pl.Int64),
            pl.Series("ts", ts, dtype=pl.Int64),
            pl.Series("timestamp", [EPOCH + timedelta(seconds=t) for t in ts]),
        ]
    )


def hits(mask: pl.Series) -> list[int]:
    return [i for i, flag in enumerate(mask.to_list()) if flag]


def cell(comp: dict, month: int) -> dict:
    return next(c for c in comp["months"] if c["month"] == month)


def test_rate_scales_by_month_rows():
    rows = {7: 200_000, 8: 200_000, 9: 200_000, 10: 100_000}
    comp = month_comparison({7: 200, 8: 200, 9: 200, 10: 200}, rows)
    assert cell(comp, 7)["rate"] == pytest.approx(100)
    assert cell(comp, TEST)["rate"] == pytest.approx(200)
    assert comp["ratio"] == pytest.approx(2)
    assert comp["z"] == pytest.approx(10)  # mu = 100, (200 - 100) / 10

    # unequal month lengths: the baseline is the mean of the three rates, not the pooled rate
    uneven_rows = {7: 100, 8: 100, 9: 300, 10: 100}
    uneven = month_comparison({7: 10, 8: 10, 9: 0, 10: 10}, uneven_rows)
    assert uneven["ratio"] == pytest.approx(1.5)
    assert uneven["z"] == pytest.approx(math.sqrt(15) / 3)
    assert per_100k(10, 100) == pytest.approx(10_000)


def test_zero_baseline_does_not_divide_by_zero():
    rows = {7: 100, 8: 100, 9: 100, 10: 80}
    both = month_comparison({7: 0, 8: 0, 9: 0, 10: 0}, rows)
    assert both["ratio"] is None
    assert both["z"] is None
    assert both["no_baseline"] is False
    assert both["test_rate"] == 0.0
    assert cell(both, TEST)["n"] == 0

    only_test = month_comparison({7: 0, 8: 0, 9: 0, 10: 8}, rows)
    assert only_test["no_baseline"] is True
    assert only_test["ratio"] is None
    assert only_test["z"] is None
    assert only_test["test_rate"] == pytest.approx(per_100k(8, 80))


def test_small_test_groups_are_hidden():
    assert shown_count(0, test=True) == 0
    assert shown_count(4, test=True) == "<5"
    assert shown_count(5, test=True) == 5
    assert shown_count(2, test=False) == 2
    hidden = month_cell(3, 100_000, test=True)
    assert hidden == {"n": "<5", "rate": None}
    assert month_cell(3, 100_000, test=False)["n"] == 3

    rows = {7: 1000, 8: 1000, 9: 1000, 10: 1000}
    comp = month_comparison({7: 50, 8: 50, 9: 50, 10: 3}, rows)
    assert cell(comp, TEST)["n"] == "<5"
    assert cell(comp, TEST)["rate"] is None
    assert comp["test_rate"] is None
    assert comp["ratio"] is None
    assert cell(comp, 7)["n"] == 50
    assert comp["z"] == pytest.approx((3 - 50) / math.sqrt(50))


def test_chi_square_uses_row_scaled_expectations():
    train = np.array([100.0, 100.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        matched = chi_square_scaled(train, np.array([50.0, 50.0, 0.0]), 1000, 500)
        # same shape at twice the rate: not renormalised to the test total, so chi2 is not 0
        shifted = chi_square_scaled(train, np.array([100.0, 100.0, 4.0]), 1000, 500)
    assert matched["expected"].tolist() == pytest.approx([50, 50, 0])
    assert matched["chi2"] == pytest.approx(0)
    assert matched["df"] == 2
    assert matched["p"] == pytest.approx(1)
    assert shifted["chi2"] == pytest.approx(100)
    assert shifted["df"] == 2
    assert bool(shifted["unexpected"][2]) is True
    assert math.isnan(shifted["z"][2])


def test_pileup_flags_only_an_excess_beside_the_cut():
    bins = np.array([3, 4, 5, 8, 9, 12])
    z = np.array([0.5, 3.0, 4.0, 2.5, -3.0, 5.0])
    expected = np.full(6, 10.0)
    observed = np.ones(6)
    flags = pileup_flags(bins, z, observed, expected, cuts=(3, 8))
    assert flags.tolist() == [False, True, False, True, False, False]

    novel = pileup_flags(
        np.array([8, 9, 12]),
        np.full(3, np.nan),
        np.array([0, 6, 6]),
        np.zeros(3),
        cuts=(8,),
    )
    assert novel.tolist() == [False, True, False]


def test_size_table_flags_a_pileup_next_to_the_cut():
    # 30 train events in bin 5 and in bin 8, 3,000 train rows; 500 test rows -> expected 5 each.
    values = [5] * 30 + [8] * 30 + [5] * 5 + [8] * 20 + [9] * 8 + [12] * 20
    period = (
        [7] * 10 + [8] * 10 + [9] * 10 + [7] * 10 + [8] * 10 + [9] * 10 + [10] * (5 + 20 + 8 + 20)
    )
    rows = {7: 1000, 8: 1000, 9: 1000, 10: 500}
    rep = size_table(
        np.array(values), np.array(period), np.zeros(len(values)), rows, cuts=(8,), names=("legit",)
    )
    by = {b["bin"]: b for b in rep["bins"]}
    assert rep["chi2"] == pytest.approx(45)  # (20 - 5)^2 / 5, bin 5 matches
    assert rep["df"] == 2
    assert by[8]["pileup"] is True
    assert by[8]["z"] == pytest.approx(15 / math.sqrt(5))
    assert by[5]["pileup"] is False
    assert by[12]["pileup"] is False  # a new bin away from the cut
    assert by[9]["pileup"] is True  # new mass on the high side of cut 8
    assert by[9]["novel"] is True


def _hop_pair(k, onward, dt, source=10_000.0, **dst):
    sender = 1000 + 2 * k
    receiver = sender + 1
    return [
        {"sender": sender, "receiver": receiver, "amount": source, "ts": 0, "to_person": True},
        {
            "sender": receiver,
            "receiver": 7,
            "amount": onward,
            "ts": dt,
            "to_person": dst.get("to_person", True),
            "is_test": dst.get("is_test", False),
        },
    ]


def test_hop_bands_select_each_side_and_not_the_law():
    rows: list[dict] = []
    want = {"low": [], "high": [], "delay": []}

    def add(kind, pair):
        start = len(rows)
        rows.extend(pair)
        if kind:
            want[kind].extend(range(start, len(rows)))

    add("low", _hop_pair(0, 9500, H))  # 0.95 within 6 h
    add("high", _hop_pair(1, 9990, H))  # 0.999 within 6 h
    add(None, _hop_pair(2, 9800, H))  # 0.98 within 6 h: the law
    add("delay", _hop_pair(3, 9800, 12 * H))  # in-band ratio, 12 h
    add("low", _hop_pair(4, 2850, 10 * H, source=3000))  # 2,000-5,000, first send after 6 h
    add(None, _hop_pair(5, 1425, H, source=1500))  # below the 2,000 floor
    add(None, _hop_pair(6, 8900, H))  # ratio 0.89, below the band
    add("low", _hop_pair(7, 9000, H))  # ratio 0.90, the floor of the band
    add("high", _hop_pair(8, 10_000, H))  # ratio 1
    add(None, _hop_pair(9, 10_100, H))  # ratio 1.01
    add(None, _hop_pair(10, 9800, 6 * H))  # exactly 6 h is still the law
    add("delay", _hop_pair(11, 9800, 6 * H + 1))
    add("delay", _hop_pair(12, 9800, HOP_DELAY_BAND_S))
    add(None, _hop_pair(13, 9800, HOP_DELAY_BAND_S + 1))
    add(None, _hop_pair(14, 9500, H, is_test=True))  # other split
    add("low", _hop_pair(15, 9500, H, to_person=False))  # onward row of any type
    # a low first send does not count when a later send inside 6 h is the law
    mule = [
        {"sender": 50, "receiver": 51, "amount": 10_000.0, "ts": 0},
        {"sender": 51, "receiver": 7, "amount": 9500.0, "ts": 100},
        {"sender": 51, "receiver": 7, "amount": 9800.0, "ts": 200},
    ]
    add(None, mule)
    # in-band but under 5,000 is not a delay near-miss (the amount floor still holds)
    add(None, _hop_pair(16, 2940, 12 * H, source=3000))

    led = frame(rows)
    assert len(rows) == led.height
    assert hits(hop_ratio_low(led)) == want["low"]
    assert hits(hop_ratio_high(led)) == want["high"]
    assert hits(hop_delay(led)) == want["delay"]


def _pay(receiver, n, amount=1000.0, gap=H, new_payee=1, txn="PAY", t0=0):
    return [
        {
            "sender": 10_000 + receiver * 40 + i,
            "receiver": receiver,
            "amount": amount,
            "ts": t0 + i * gap,
            "new_payee": new_payee,
            "txn_type": txn,
            "to_person": True,
        }
        for i in range(n)
    ]


def test_sender_count_bands():
    parts = {
        "burst": _pay(10, 7) + _pay(11, 6),
        "neither": _pay(12, 5) + _pay(13, 8) + _pay(14, 1, new_payee=1) + _pay(15, 3, new_payee=1),
        "old": _pay(16, 2, new_payee=0),
        "fan": _pay(17, 2, new_payee=1),
        "not_whole": _pay(18, 7, amount=1005),
        "split_runs": _pay(19, 2, new_payee=1) + _pay(19, 2, new_payee=1, t0=2 * DAY),
    }
    rows, expect_burst, expect_fan = [], [], []
    for name, part in parts.items():
        start = len(rows)
        rows.extend(part)
        ids = list(range(start, len(rows)))
        if name == "burst":
            expect_burst.extend(ids)
        if name == "fan":
            expect_fan.extend(ids)
        if name == "split_runs":
            expect_fan.extend(ids)  # a >1 day gap makes two runs of 2
    led = frame(rows)
    assert hits(burst_senders(led)) == expect_burst
    assert hits(fan_senders(led)) == expect_fan


def _session(device, n, gap, t0=0, is_test=False):
    return [
        {"device": device, "sender": device * 1000 + i, "ts": t0 + i * gap, "is_test": is_test}
        for i in range(n)
    ]


def test_session_gap_and_short_rows():
    short = _session(1, 9, 60) + _session(4, 8, 60) + _session(6, 9, 60, t0=0)
    short += _session(6, 9, 60, t0=8 * 60 + 1801)  # gap of 1801 s splits the relaxed session
    short += _session(8, 9, 300)  # a gap of exactly 5 min stays one session
    gap = _session(2, 6, 60) + _session(2, 6, 60, t0=5 * 60 + 600)
    gap += _session(7, 5, 60) + _session(7, 5, 60, t0=4 * 60 + 1800)  # exactly 30 min still joins
    quiet = _session(3, 10, 60) + _session(5, 7, 60)
    quiet += _session(9, 6, 60) + _session(9, 6, 60, t0=5 * 60 + 60, is_test=True)
    rows = short + gap + quiet
    led = frame(rows)
    assert hits(bot_rows(led)) == list(range(len(short)))
    assert hits(bot_gap(led)) == list(range(len(short), len(short) + len(gap)))


def test_drain_step_and_span_bands():
    def step(sender, balance, dt):
        return [
            {"sender": sender, "balance_before": 10_000.0, "ts": 0},
            {"sender": sender, "balance_before": balance, "ts": dt},
        ]

    rows, step_ids, span_ids = [], [], []

    def add(kind, pair):
        start = len(rows)
        rows.extend(pair)
        ids = list(range(start, len(rows)))
        if kind == "step":
            step_ids.extend(ids)
        if kind == "span":
            span_ids.extend(ids)

    add(None, step(1, 2500.0, 600))  # exact 0.25 inside 3 h: the law
    add("step", step(2, 2500.01, 600))  # residual 0.01, outside TOL, ratio still ~0.25
    add("step", step(3, 2600.0, 600))  # 0.26, within ±0.02, far outside TOL
    add(None, step(4, 2800.0, 600))  # 0.28, outside ±0.02 of both constants
    add(None, step(5, 3500.0, 600))  # exact 0.35: the law
    add("step", step(6, 3650.0, 600))  # 0.365, within ±0.02 of 0.35
    add("span", step(7, 2500.0, 3 * H + 1))
    add(None, step(8, 2500.0, 3 * H))  # exactly 3 h is still the law
    add("span", step(9, 2500.0, 6 * H))
    add(None, step(10, 2500.0, 6 * H + 1))
    add(
        None,
        [  # different senders
            {"sender": 11, "balance_before": 10_000.0, "ts": 0},
            {"sender": 12, "balance_before": 2500.0, "ts": 600},
        ],
    )
    add(
        None,
        [  # the other split
            {"sender": 13, "balance_before": 10_000.0, "ts": 0},
            {"sender": 13, "balance_before": 2600.0, "ts": 600, "is_test": True},
        ],
    )
    led = frame(rows)
    assert hits(drain_step(led)) == step_ids
    assert hits(drain_span(led)) == span_ids


def _collect(amount, **kw):
    row = {
        "txn_type": "COLLECT",
        "new_payee": 1,
        "to_person": True,
        "amount": amount,
        "sender": 20,
        "receiver": 21,
    }
    row.update(kw)
    return row


def test_collect_offset_band():
    yes = [1995, 4998, 4980, 4975, 4900, 9998, 24900]  # offsets 5, 2, 20, 25, 100 and two limits
    no = [
        _collect(4990),  # real offset 10
        _collect(4999),  # real offset 1
        _collect(4997),  # offset 3, not in the near set
        _collect(4995, txn_type="PAY"),
        _collect(4995, new_payee=0),
        _collect(4995, to_person=False, receiver=MERCHANT + 1),
    ]
    rows = [_collect(a) for a in yes] + no
    led = frame(rows)
    assert hits(collect_offset(led)) == list(range(len(yes)))


def test_bait_amount_band():
    yes = [{"amount": a} for a in (3, 20, 50)]
    no = [{"amount": a} for a in (1, 2, 5, 10, 4)]
    no += [{"amount": 3, "txn_type": "QR"}, {"amount": 3, "to_person": False, "receiver": MERCHANT}]
    led = frame(yes + no)
    assert hits(bait_amount(led)) == [0, 1, 2]


def test_bait_gap_band():
    rows = [
        {"sender": 1, "receiver": 2, "amount": 1, "ts": 0},
        {"sender": 1, "receiver": 2, "amount": 2, "ts": 100},  # gap 100 s: inside the law
        {"sender": 1, "receiver": 2, "amount": 5, "ts": 800},  # gap 700 s
        {"sender": 3, "receiver": 4, "amount": 10, "ts": 0},
        {"sender": 3, "receiver": 4, "amount": 1, "ts": 600},  # gap == 10 min: still the law
        {"sender": 3, "receiver": 4, "amount": 2, "ts": 600 + 1800},  # gap == 30 min
        {"sender": 5, "receiver": 6, "amount": 1, "ts": 0},
        {"sender": 5, "receiver": 6, "amount": 1, "ts": 1801},
        {"sender": 7, "receiver": 8, "amount": 1, "ts": 0, "is_test": True},
        {"sender": 7, "receiver": 8, "amount": 1, "ts": 900},  # other split, not paired with row 0
    ]
    led = frame(rows)
    assert hits(bait_gap(led)) == [1, 2, 4, 5]


def test_bait_answer_band():
    window = 72 * H
    rows = [
        {"sender": 1, "receiver": 2, "amount": 1, "ts": 0},
        {"sender": 2, "receiver": 1, "amount": 400, "ts": window + 1},
        {"sender": 3, "receiver": 4, "amount": 2, "ts": 0},
        {"sender": 4, "receiver": 3, "amount": 400, "ts": window},  # exactly 72 h: the law
        {"sender": 5, "receiver": 6, "amount": 5, "ts": 0},
        {"sender": 6, "receiver": 5, "amount": 400, "ts": window + DAY},  # 96 h
        {"sender": 7, "receiver": 8, "amount": 10, "ts": 0},
        {"sender": 8, "receiver": 7, "amount": 400, "ts": window + DAY + 1},
        {"sender": 9, "receiver": 10, "amount": 1, "ts": 0},
        {"sender": 10, "receiver": 9, "amount": 400, "ts": H},  # inside 72 h
        {"sender": 11, "receiver": 12, "amount": 1, "ts": 0},
        {"sender": 11, "receiver": 12, "amount": 1, "ts": 80 * H},
        {"sender": 12, "receiver": 11, "amount": 400, "ts": 80 * H + 10},  # a bait 10 s earlier
    ]
    led = frame(rows)
    assert hits(bait_answer(led)) == [1, 5]


def _refund(sender, ts, receiver=MERCHANT + 1):
    return {
        "sender": sender,
        "receiver": receiver,
        "ts": ts,
        "to_person": False,
        "txn_type": "REFUND",
        "merchant_category": "ecommerce",
        "amount": 400.0,
    }


def test_repeat_gap_band():
    rows = [
        _refund(1, 0),
        _refund(1, 10 * DAY),
        _refund(2, 0),
        _refund(2, 7 * DAY),  # exactly 7 days still chains
        _refund(3, 0),
        _refund(3, 7 * DAY + 1),
        _refund(4, 0),
        _refund(4, 14 * DAY),
        _refund(5, 0),
        _refund(5, 14 * DAY + 1),
        _refund(6, 0),
        _refund(7, 10 * DAY),  # different sender
        _refund(8, 0),
        {  # a lending row is a different kind, so it does not bridge
            "sender": 8,
            "receiver": MERCHANT + 2,
            "ts": 10 * DAY,
            "to_person": False,
            "txn_type": "PAY",
            "merchant_category": "lending",
            "amount": 5000.0,
        },
    ]
    # hour 21 so the lending row is a real evening candidate, not an accident of the clock
    rows[-1]["ts"] = 10 * DAY + 21 * H
    led = frame(rows)
    assert hits(repeat_gap(led)) == [0, 1, 4, 5, 6, 7]


def test_evening_band():
    def lend(ts, **kw):
        row = {
            "ts": ts,
            "txn_type": "PAY",
            "to_person": False,
            "receiver": MERCHANT + 3,
            "merchant_category": "lending",
            "amount": 5000.0,
        }
        row.update(kw)
        return row

    rows = [
        lend(18 * H),
        lend(19 * H + 1800),
        lend(20 * H),  # the law's evening
        lend(17 * H),
        lend(18 * H, amount=5000.5),
        lend(18 * H, merchant_category="grocery"),
        {
            "ts": 18 * H,
            "txn_type": "PAY",
            "to_person": True,
            "receiver": 70,
            "new_payee": 1,
            "amount": 2345.0,
        },
        {
            "ts": 18 * H,
            "txn_type": "PAY",
            "to_person": True,
            "receiver": 71,
            "new_payee": 1,
            "amount": 2340.0,  # multiple of 10, balance is not 1.1x
        },
        {
            "ts": 18 * H,
            "txn_type": "PAY",
            "to_person": True,
            "receiver": 72,
            "new_payee": 1,
            "amount": 2340.0,
            "balance_before": 2574.0,  # 1.1 x 2340
        },
        {
            "ts": 18 * H,
            "txn_type": "PAY",
            "to_person": True,
            "receiver": 73,
            "new_payee": 0,
            "amount": 2345.0,
        },
        {
            "ts": 18 * H,
            "txn_type": "PAY",
            "to_person": True,
            "receiver": 74,
            "new_payee": 1,
            "amount": 500.0,
        },
    ]
    led = frame(rows)
    assert hits(evening(led)) == [0, 1, 6, 8]


def test_ring_amount_band():
    def pay(amount, **kw):
        row = {
            "amount": amount,
            "to_person": False,
            "receiver": MERCHANT + 4,
            "txn_type": "PAY",
        }
        row.update(kw)
        return row

    rows = [
        pay(1500),
        pay(300),
        pay(100),
        pay(20_000),
        pay(4000, txn_type="QR"),
        pay(1000),
        pay(2500),
        pay(5000),
        pay(10_000),
        pay(250),
        pay(20_100),
        pay(1500, txn_type="COLLECT"),
        pay(1500, to_person=True, receiver=80),
    ]
    led = frame(rows)
    assert hits(ring_amount(led)) == [0, 1, 2, 3, 4]
