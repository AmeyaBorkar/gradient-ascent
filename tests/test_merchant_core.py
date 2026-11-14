import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from merchant_core import (  # noqa: E402
    band_report,
    call_rates,
    event_sizes,
    passes,
    rule_effect,
    zone_votes,
)


def test_call_rates_per_period():
    calls = np.array([0, 1, 1, 0, 2, 0, 0, 0])
    period = np.array([7, 7, 7, 7, 10, 10, 10, 10])
    r = call_rates(calls, period, k=3)
    assert r[7] == pytest.approx([50_000.0, 50_000.0, 0.0])
    assert r[10] == pytest.approx([75_000.0, 0.0, 25_000.0])


def test_band_report_rates_and_train_precision():
    band = np.array(["a", "a", "a", "b", "a", "b"])
    calls = np.array([1, 1, 0, 1, 1, 0])
    period = np.array([7, 7, 7, 7, 10, 10])
    train = period < 10
    y = np.array([1, 0, 0, 1, -1, -1])
    rep = band_report(band, calls, period, 1, {"given": y}, train)
    assert rep["a"]["rows"][7] == pytest.approx(75_000.0)
    assert rep["a"]["calls"][10] == pytest.approx(50_000.0)
    assert rep["a"]["precision"]["given"] == pytest.approx(0.5)  # October call never read
    assert rep["b"]["precision"]["given"] == pytest.approx(1.0)


def test_band_report_no_train_calls_gives_none():
    rep = band_report(
        np.array(["x", "x"]),
        np.array([0, 0]),
        np.array([7, 7]),
        1,
        {"given": np.array([1, 1])},
        np.array([True, True]),
    )
    assert rep["x"]["precision"]["given"] is None


def test_event_sizes_ignores_rows_outside_events():
    ev = np.array([-1, 5, 5, 9, 5, -1])
    assert event_sizes(ev).tolist() == [0, 3, 3, 1, 3, 0]


def test_zone_votes_counts_fraud_block_rows_per_group():
    ev = np.array([1, 1, 2, 2, 3, -1])
    fb = np.array([True, False, True, True, False, True])
    seg = np.array([8, -1, 11, 11, -1, 8])
    group = np.array(["refund", "refund", "collusion", "collusion", "refund", "refund"])
    v = zone_votes(ev, fb, seg, group)
    assert v == {"collusion": {11: 2}, "refund": {8: 1}}


def test_rule_effect_and_bar():
    rng = np.random.default_rng(0)
    n = 3000
    period = np.repeat([7, 8, 9], n // 3)
    y = rng.choice(12, n, p=[0.7] + [0.3 / 11] * 11)
    calls = np.where(rng.random(n) < 0.7, y, 0)
    better = calls.copy()
    fix = (calls != y) & (rng.random(n) < 0.5)
    better[fix] = y[fix]
    eff = rule_effect(calls, better, period, {"given": y, "cleaned": y}, (7, 8, 9))
    assert all(eff[m]["given"] > 0 for m in eff)
    assert sum(eff[m]["changed"] for m in eff) == int(fix.sum())
    assert passes(eff)
    worse = rule_effect(better, calls, period, {"given": y, "cleaned": y}, (7, 8, 9))
    assert not passes(worse)
    same = rule_effect(calls, calls, period, {"given": y, "cleaned": y}, (7, 8, 9))
    assert not passes(same)  # a tie everywhere is not an upload
