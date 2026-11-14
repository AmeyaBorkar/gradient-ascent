import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from bait_core import (  # noqa: E402
    by_month,
    move_table,
    other_month_label_acts,
    passes_months,
)

K = 6


def _month():
    # class 6 has 4 true rows; the base calls 2 of them and one legit row: F1 = 2*2/(4+3) = 4/7
    truth = np.array([6, 6, 6, 6, 0, 0, 0, 0, 0, 0])
    base = np.array([6, 6, 0, 0, 6, 0, 0, 0, 0, 0])
    return truth, base


def test_move_table_precision_and_break_even():
    truth, base = _month()
    new = base.copy()
    new[[2, 5]] = K  # one right, one wrong
    t = move_table(truth, base, new, K)
    assert t["moved"] == 2 and t["right"] == 1 and t["fraud"] == 1
    assert t["precision"] == pytest.approx(0.5)
    assert t["break_even"] == pytest.approx(2 / 7)
    assert t["gain"] > 0  # 0.5 is above the break-even


def test_below_break_even_loses():
    truth, base = _month()
    new = base.copy()
    new[[2, 5, 6, 7]] = K  # one right in four: 0.25 < 2/7
    t = move_table(truth, base, new, K)
    assert t["precision"] == pytest.approx(0.25)
    assert t["gain"] < 0


def test_no_move():
    truth, base = _month()
    t = move_table(truth, base, base, K)
    assert t["moved"] == 0 and t["precision"] is None and t["gain"] == 0


def test_by_month_and_passes():
    truth, base = _month()
    y = np.concatenate([truth, truth])
    a = np.concatenate([base, base])
    b = a.copy()
    b[2] = K  # a right move in month 7 only
    period = np.repeat([7, 8], len(truth))
    tab = by_month({"given": y, "cleaned": y}, a, b, period, K, months=(7, 8))
    assert tab["given"][0]["moved"] == 1 and tab["given"][1]["moved"] == 0
    assert passes_months(tab)
    b[10] = 0  # and the loss of a right call in month 8
    tab = by_month({"given": y, "cleaned": y}, a, b, period, K, months=(7, 8))
    assert not passes_months(tab)


def test_by_month_skips_negative_targets():
    truth, base = _month()
    y = truth.copy()
    y[5] = -1
    new = base.copy()
    new[5] = K
    tab = by_month({"given": y}, base, new, np.full(len(y), 7), K, months=(7,))
    assert tab["given"][0]["moved"] == 0


def test_label_acts_unique():
    acct, per = other_month_label_acts(
        np.array([3, 3, 4, 5]), np.array([7, 7, 8, 9]), np.array([True, True, True, False])
    )
    assert acct.tolist() == [3, 4] and per.tolist() == [7, 8]
