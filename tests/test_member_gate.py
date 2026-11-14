import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from member_gate import aclean_macro, round_bar  # noqa: E402

from ringtrace.classes import N_CLASSES  # noqa: E402
from ringtrace.scoring import macro_f1  # noqa: E402


def test_aclean_without_hidden_rows_is_macro_f1():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, N_CLASSES, 3000)
    pred = np.where(rng.random(3000) < 0.7, truth, rng.integers(0, N_CLASSES, 3000))
    assert aclean_macro(truth, pred, np.zeros(N_CLASSES)) == np.float64(macro_f1(truth, pred))


def test_hidden_rows_cost_the_legit_calls():
    truth = np.array([0] * 90 + [1] * 10)
    pred = truth.copy()
    hidden = np.zeros(N_CLASSES)
    hidden[1] = 9.0  # nine class-1 rows hide among the 90 legit rows, all called legit
    got = aclean_macro(truth, pred, hidden)
    assert got < aclean_macro(truth, pred, np.zeros(N_CLASSES))
    # class 1: TP 10, FN 9 -> F1 20 / 29; legit: TP 81 of 90 calls and 81 true -> F1 162 / 171
    assert abs(got - (20 / 29 + 162 / 171) / N_CLASSES) < 1e-12  # absent classes score 0


def test_round_bar():
    lomo = {"given": {"change": [0.0, 1e-5, 0.0]}, "cleaned": {"change": [0.0, 0.0, 0.0]}}
    octl = {
        "per_seed": {"given": {13: 1e-4, 14: -1e-5}, "cleaned": {13: 1e-4, 14: 1e-4}},
        "lower_bound": {"given": 1e-6, "cleaned": 1e-6},
    }
    assert round_bar(lomo, octl)["passes"]
    assert not round_bar(lomo, None)["passes"]
    octl["lower_bound"]["cleaned"] = -1e-6
    assert not round_bar(lomo, octl)["passes"]
    lomo["cleaned"]["change"][2] = -1e-6
    assert not round_bar(lomo, octl)["months_ok"]
    lomo = {"given": {"change": [0.0] * 3}, "cleaned": {"change": [0.0] * 3}}
    assert not round_bar(lomo, None)["months_ok"]
