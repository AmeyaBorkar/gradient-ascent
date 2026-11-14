"""Composition and ceiling arithmetic, checked by hand."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.audit import ceiling as ceil


def test_rate_and_hidden_rows():
    assert ceil.new_device_rate(30, 970) == pytest.approx(0.03)
    assert ceil.hidden_rows(np.array([3.0, 0.0]), 0.03) == pytest.approx([97.0, 0.0])


def test_ceiling_matches_hand_computation():
    # one fraud class: 40 reachable, 60 hidden, 1000 rows
    res = ceil.ceiling(np.array([40.0]), np.array([60.0]), 1000)
    f1_fraud = 2 * 40 / (2 * 40 + 60)
    legit_true, legit_pred = 900, 960
    f1_legit = 2 * 900 / (legit_true + legit_pred)
    assert res["f1"].tolist() == pytest.approx([f1_legit, f1_fraud])
    assert res["macro"] == pytest.approx((f1_legit + f1_fraud) / 2)
    assert res["hidden_share"] == pytest.approx(0.6)


def test_expected_scores_perfect_calls_score_one():
    true = np.array([10.0, 20.0])
    res = ceil.expected_scores(true, true, true, 500)
    assert res["macro"] == pytest.approx(1.0)


def test_extra_rows_for_target():
    pred = np.array([40.0, 40.0])
    tp = np.array([38.0, 38.0])
    hidden = np.array([60.0, 60.0])
    true = tp + hidden
    now = ceil.expected_scores(pred, tp, true, 2000)["macro"]
    same = ceil.extra_rows_for(now, pred, tp, true, hidden, 2000)
    assert same["extra_rows"] == 0.0
    need = ceil.extra_rows_for(now + 0.05, pred, tp, true, hidden, 2000)
    assert need["feasible"] and 0 < need["share_of_hidden"] < 1
    x = np.array(need["per_class"])
    got = ceil.expected_scores(pred + x, tp + x, true, 2000)["macro"]
    assert got == pytest.approx(now + 0.05, abs=1e-5)
    assert not ceil.extra_rows_for(0.999, pred, tp, true, hidden, 2000)["feasible"]


def test_split_scores_random_order_is_neutral_and_front_loading_shifts_score():
    pred = np.array([400.0, 300.0])
    tp = 0.95 * pred
    true = tp + np.array([700.0, 500.0])
    n, n_pub = 100_000, 30_000
    full = ceil.expected_scores(pred, tp, true, n)["macro"]
    rnd = ceil.split_scores(pred, tp, true, n, n_pub, n_pub / n)
    assert rnd["public"] == pytest.approx(full, abs=2e-3)
    assert rnd["private"] == pytest.approx(full, abs=2e-3)
    front = ceil.split_scores(pred, tp, true, n, n_pub, 0.45)
    assert front["public"] > full + 0.02 and front["private"] < full - 0.02
