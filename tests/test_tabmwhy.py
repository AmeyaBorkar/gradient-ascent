"""Gain anatomy helpers on synthetic calls."""

from __future__ import annotations

import numpy as np

from ringtrace import tabmwhy
from ringtrace.classes import N_CLASSES


def test_pockets_count_b_c_d():
    label = np.array([1, 2, 0, 3, 0])
    gen = np.array([1, 2, 0, 3, 0])
    pred = np.array([0, 3, 4, 3, 0])
    tmpl = np.array(["a", "b", "none", "a", "none"])
    p = tabmwhy.pockets(pred, label, gen, label, tmpl)
    assert (p["b"], p["c"], p["d"]) == (1, 1, 1)
    assert list(p["c_by"]) == ["b: velocity_bot -> layering_chain"]


def test_disagreements_credit_the_right_side():
    a = np.array([0, 0, 1, 2])
    b = np.array([1, 0, 0, 2])
    truth = np.array([1, 0, 1, 2])
    d = tabmwhy.disagreements(a, b, truth, np.array(["x", "x", "y", "x"]))
    assert d["n"] == 2
    assert d["pairs"]["legit -> mule_account"] == {"n": 1, "base_right": 0, "challenger_right": 1}
    assert d["templates"]["y"]["base_right"] == 1


def test_reliability_bins_and_mid_mass():
    p = np.zeros((4, N_CLASSES))
    p[:, 0] = [1.0, 0.5, 0.02, 0.0]
    p[:, 1] = 1 - p[:, 0]
    r = tabmwhy.reliability(p, np.array([0, 1, 1, 1]))
    assert sum(b["n"] for b in r["bins"]) == 4
    assert r["mid_rows"] == 1 and abs(r["mid_mass"] - 0.5) < 1e-12
    assert abs(r["fraud_mass"] - 2.48) < 1e-12
