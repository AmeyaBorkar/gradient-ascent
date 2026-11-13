import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from closelook_core import (  # noqa: E402
    class_table,
    expected_confusion,
    fixed_point_scaled,
    hidden_counts,
    macro,
    pair_split_best,
)

from ringtrace.decode import fixed_point  # noqa: E402
from ringtrace.scoring import confusion, macro_f1  # noqa: E402


def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(12, n, p=[0.7] + [0.3 / 11] * 11)
    pred = np.where(rng.random(n) < 0.7, y, rng.integers(0, 12, n))
    return y, pred


def test_no_hidden_is_plain_macro_f1():
    y, pred = _toy()
    assert np.array_equal(expected_confusion(y, pred), confusion(y, pred))
    assert np.isclose(macro(y, pred), macro_f1(y, pred))
    assert np.isclose(macro(y, pred, np.zeros(12)), macro_f1(y, pred))


def test_hidden_rows_keep_totals_and_lower_the_score():
    y, pred = _toy()
    h = np.r_[0.0, np.full(11, 5.0)]
    cm = expected_confusion(y, pred, h)
    assert np.isclose(cm.sum(), len(y))
    assert np.allclose(cm.sum(axis=0), np.bincount(pred, minlength=12))
    assert np.isclose(cm[1:].sum(axis=1).sum() - confusion(y, pred)[1:].sum(), h.sum())
    assert macro(y, pred, h) < macro(y, pred)


def test_hidden_counts_follow_ordinary_labels():
    cleaned = np.array([0, 1, 1, 2, 0, 1, 3, 3])
    gen = np.array([0, 0, 1, 0, 0, 0, 0, 3])
    period = np.array([7, 7, 7, 7, 8, 8, 8, 8])
    h = hidden_counts(cleaned, gen, period, total=8)
    # ordinary rows: (7, class 1), (7, class 2), (8, class 1), (8, class 3)
    assert h[7][1] == 2 and h[7][2] == 2 and h[8][1] == 2 and h[8][3] == 2
    assert h[7][0] == 0 and sum(h[7]) + sum(h[8]) == 8


def test_class_table_counts():
    y, pred = _toy(seed=1)
    t = class_table(y, pred)
    for c in range(12):
        r = t[str(c)]
        assert r["tp"] + r["fn"] == r["true"] and r["tp"] + r["fp"] == r["calls"]
    assert np.isclose(t["macro"], macro_f1(y, pred))


def _proba(n=4000, seed=2):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 12))
    logits[:, 0] += 4.0
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def test_unit_multipliers_reproduce_the_fixed_point():
    p = _proba()
    assert np.array_equal(fixed_point_scaled(p), fixed_point(p))
    assert np.array_equal(fixed_point_scaled(p, np.ones(12)), fixed_point(p))


def test_lower_multiplier_admits_more_rows():
    p = _proba(seed=3)
    base = (fixed_point_scaled(p) == 5).sum()
    m = np.ones(12)
    m[5] = 0.5
    more = (fixed_point_scaled(p, m) == 5).sum()
    m[5] = 2.0
    fewer = (fixed_point_scaled(p, m) == 5).sum()
    assert fewer <= base <= more


def test_pair_split_endpoints():
    r = pair_split_best(100, 100, 40, 40, 50, 50, pool=20, q=1.0)
    assert r["best_share_a"] == 1.0 and r["best"] >= r["all_b"]
    r = pair_split_best(100, 100, 40, 40, 50, 50, pool=20, q=0.0)
    assert r["best_share_a"] == 0.0


def test_decoder_class_mult_matches_the_analysis_decoder():
    from ringtrace.classes import CLASSES

    p = _proba(seed=4)
    assert np.array_equal(fixed_point(p, class_mult={}), fixed_point(p))
    m = np.ones(12)
    m[CLASSES.index("qr_swap")] = 0.5
    assert np.array_equal(fixed_point(p, class_mult={"qr_swap": 0.5}), fixed_point_scaled(p, m))
