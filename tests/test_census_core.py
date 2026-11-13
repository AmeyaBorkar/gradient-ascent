import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from census_core import FRAUD, move, top, unit_table  # noqa: E402


def _frame():
    # 40 train rows, 10 October rows; unit "u" holds 20 train and 10 October rows
    tr = np.r_[np.ones(40, bool), np.zeros(10, bool)]
    te = ~tr
    u = np.r_[np.ones(20, bool), np.zeros(20, bool), np.ones(10, bool)]
    calls = np.zeros(50, np.int64)
    calls[:10] = 3  # half the unit's train rows called class 3
    calls[40:42] = 3  # only 2 of 10 October rows called 3
    clean = calls.copy()
    clean[10] = 3  # one more true 3 the model missed in train
    return {"u": u}, tr, te, calls, clean


def test_unit_table_expected_and_surprise():
    units, tr, te, calls, clean = _frame()
    rows = {(r["unit"], r["cls"]): r for r in unit_table(units, tr, te, calls, calls, clean, 4)}
    r = rows[("u", 3)]
    assert r["n_train"] == 20 and r["n_oct"] == 10
    assert r["oof_rate"] == 0.5 and r["clean_rate"] == 11 / 20
    assert r["expected"] == 5.0 and r["actual"] == 2
    assert np.isclose(r["z"], -3 / np.sqrt(5))
    assert r["oof_precision"] == 1.0
    f = rows[("u", FRAUD)]
    assert f["expected"] == 5.0 and f["actual"] == 2


def test_unit_table_skips_thin_units():
    units, tr, te, calls, clean = _frame()
    assert unit_table(units, tr, te, calls, calls, clean, 4, min_train=21) == []


def test_top_orders_and_filters():
    rows = [
        {"z": -3.0, "actual": 2, "expected": 5.0},
        {"z": -5.0, "actual": 0, "expected": 1.0},  # gap 1 < min_gap
        {"z": 4.0, "actual": 9, "expected": 1.0},
    ]
    assert [r["z"] for r in top(rows, 5, -1)] == [-3.0]
    assert [r["z"] for r in top(rows, 5, 1)] == [4.0]


def test_move_shift():
    m = move(np.array([0.0, 2.0, np.nan]), np.array([3.0, 3.0]))
    assert m["train_mean"] == 1.0 and m["oct_mean"] == 3.0 and m["shift_sd"] == 2.0
    assert move(np.array([]), np.array([1.0]))["shift_sd"] is None
