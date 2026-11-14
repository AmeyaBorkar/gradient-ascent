import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from pairs_core import (  # noqa: E402
    FEATURES,
    drain_event_features,
    event_label,
    event_scores,
    flip,
    matrix,
    month_gains,
    passes_bar,
)


def _rows():
    # event 7: two rows on the sender's own phone, new payees, short logins, 600 s apart
    # event 3: four rows, one old payee and one long login, a borrowed phone on one row
    return pl.DataFrame(
        {
            "drn_event": [7, 7, 3, 3, 3, 3],
            "ts": [1000, 1600, 0, 100, 200, 300],
            "hour": [2, 2, 14, 14, 14, 14],
            "sender": [5, 5, 9, 9, 9, 9],
            "device": [5, 5, 9, 9, 9, 44],
            "login_to_txn_sec": [10.0, 20.0, 10.0, 300.0, 5.0, 5.0],
            "new_payee": [1, 1, 1, 0, 1, 1],
            "pin_retries": [0, 1, 0, 0, 0, 0],
            "sim_age_hours": [3.0, 5.0, 1.0, 1.0, 1.0, 1.0],
            "balance_before": [800.0, 200.0, 4000.0, 1000.0, 250.0, 62.5],
            "age_err": [-1, 0, 0, 0, 0, 0],
        }
    )


def test_drain_event_features_flags_and_order():
    ev = drain_event_features(_rows())
    assert ev["drn_event"].to_list() == [3, 7]
    a, b = ev.row(0, named=True), ev.row(1, named=True)
    assert (a["size"], a["login_hi"], a["old_payee"], a["home_phone"]) == (4, 1.0, 1.0, 0.0)
    assert (b["size"], b["login_hi"], b["old_payee"], b["home_phone"]) == (2, 0.0, 0.0, 1.0)
    assert (a["size4"], a["size2"], b["size2"]) == (1.0, 0.0, 1.0)
    assert b["log_gap"] == pytest.approx(np.log1p(600.0))
    assert a["log_gap"] == pytest.approx(np.log1p(100.0))
    assert b["night"] == 1.0 and a["night"] == 0.0
    assert b["pin_share"] == 0.5 and b["age_minus1"] == 0.5
    assert matrix(ev).shape == (2, len(FEATURES))


def test_event_label_ties():
    assert event_label(np.array([3, 1, 2]), np.array([1, 2, 2])).tolist() == [0, 1, -1]


def test_event_scores_out_of_fold_and_deterministic():
    rng = np.random.default_rng(0)
    n = 900
    label = rng.integers(0, 2, n)
    X = np.column_stack([label + rng.normal(0, 0.7, n), rng.normal(0, 1, n)])
    period = rng.choice([7, 8, 9, 10], n)
    is_test = period == 10
    label[is_test] = -1
    label[:5] = -1  # ties never enter a fit
    s1 = event_scores(X, label, period, is_test)
    s2 = event_scores(X, label, period, is_test)
    assert np.array_equal(s1, s2)
    assert not np.isnan(s1).any()
    ok = (label >= 0) & ~is_test
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(label[ok], s1[ok]) > 0.8


def test_event_scores_held_out_month_never_fits_itself():
    # month 9 carries the opposite relation: an in-sample fit would score it well
    rng = np.random.default_rng(1)
    n = 600
    period = np.repeat([7, 8, 9], n // 3)
    label = rng.integers(0, 2, n)
    sign = np.where(period == 9, -1.0, 1.0)
    X = (sign * (2 * label - 1) + rng.normal(0, 0.3, n))[:, None]
    s = event_scores(X, label, period, np.zeros(n, bool))
    m9 = period == 9
    assert ((s[m9] > 0.5) == label[m9]).mean() < 0.2


def test_flip_moves_only_the_from_class():
    calls = np.array([4, 4, 0, 6, 4, 4])
    rows = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
    out = flip(calls, rows, np.array([True, True, False]), 4, 5)
    assert out.tolist() == [5, 5, 0, 6, 5, 4]
    assert calls.tolist() == [4, 4, 0, 6, 4, 4]
    assert flip(calls, rows, np.zeros(3, bool), 4, 5).tolist() == calls.tolist()


def test_month_gains_and_bar():
    y = np.array([0, 4, 5, 4, 0, 5, 4, 5])
    base = np.array([0, 4, 4, 4, 0, 4, 4, 4])
    new = np.array([0, 4, 5, 4, 0, 4, 4, 4])
    period = np.array([7, 7, 7, 8, 8, 8, 9, 9])
    g = month_gains(y, base, new, period)
    assert g[0] > 0 and g[1] == 0 and g[2] == 0
    assert passes_bar({"given": g, "cleaned": g})
    assert not passes_bar({"given": g, "cleaned": [0.0, -0.001, 0.0]})
    assert not passes_bar({"given": [0.0] * 3, "cleaned": [0.0] * 3})
