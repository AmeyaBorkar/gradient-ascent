import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from farlook_core import (  # noqa: E402
    gap_stats,
    keep_units,
    oracle_typing_gain,
    pair_concordance,
    rank_correlation,
    shuffled_concordance,
    split_curve,
)

from ringtrace.scoring import per_class_f1  # noqa: E402


def _totals(y, pred, c):
    return {
        "tp": float(((y == c) & (pred == c)).sum()),
        "called": float((pred == c).sum()),
        "true": float((y == c).sum()),
    }


def test_split_curve_endpoints_match_a_real_relabel():
    rng = np.random.default_rng(0)
    n = 6000
    y = rng.choice(12, n, p=[0.5] + [0.5 / 11] * 11)
    pred = np.where(rng.random(n) < 0.7, y, rng.integers(0, 12, n))
    in_pool = (pred == 4) & (rng.random(n) < 0.5)
    pool = {
        "n": float(in_pool.sum()),
        "a": float((in_pool & (y == 4)).sum()),
        "b": float((in_pool & (y == 5)).sum()),
    }
    qs, delta = split_curve(_totals(y, pred, 4), _totals(y, pred, 5), pool)
    assert qs[0] == 0.0 and delta[0] == 0.0
    flipped = pred.copy()
    flipped[in_pool] = 5
    before, after = per_class_f1(y, pred), per_class_f1(y, flipped)
    want = (after[4] + after[5] - before[4] - before[5]) / 12
    assert abs(delta[-1] - want) < 1e-12


def test_split_curve_is_concave():
    a = {"tp": 500.0, "called": 700.0, "true": 1500.0}
    b = {"tp": 150.0, "called": 200.0, "true": 600.0}
    _, delta = split_curve(a, b, {"n": 300.0, "a": 140.0, "b": 150.0})
    assert (np.diff(delta, 2) <= 1e-12).all()


def test_oracle_typing_gain_is_positive_and_exact():
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.choice(12, n)
    pred = np.where(np.isin(y, (4, 5)), 4, y)
    in_pool = pred == 4
    pool = {"n": float(in_pool.sum()), "a": float((y == 4).sum()), "b": float((y == 5).sum())}
    gain = oracle_typing_gain(_totals(y, pred, 4), _totals(y, pred, 5), pool)
    before = per_class_f1(y, pred)
    assert abs(gain - (2.0 - before[4] - before[5]) / 12) < 1e-12
    assert gain > 0


def test_pair_concordance_loyal_and_mixed():
    entity = np.array([1, 1, 2, 2, 3, 3])
    loyal = np.array([4, 4, 5, 5, 4, 4])
    mixed = np.array([4, 5, 4, 5, 4, 5])
    assert pair_concordance(entity, loyal, (4, 5)) == (1.0, 3)
    assert pair_concordance(entity, mixed, (4, 5)) == (0.0, 3)


def test_shuffled_concordance_sits_at_the_independence_rate():
    rng = np.random.default_rng(2)
    event = np.repeat(np.arange(400), 2)
    entity = rng.integers(0, 60, len(event))
    label = np.repeat(rng.choice([4, 5], 400, p=[0.7, 0.3]), 2)
    mean, sd = shuffled_concordance(entity, event, label, (4, 5), draws=50)
    assert abs(mean - (0.7**2 + 0.3**2)) < 0.03
    assert sd < 0.05


def test_gap_stats_tells_a_counter_from_a_draw():
    counter = (np.arange(5000, dtype=np.int64) * 2654435761 + 12345) % 2**32
    assert gap_stats(counter)["distinct_gaps"] <= 3
    draw = np.random.default_rng(3).choice(2**32, 5000, replace=False)
    stats = gap_stats(draw)
    assert (
        stats["distinct_gaps"] > 1000 and 0.9 < stats["gap_cv"] < 1.1 and stats["duplicates"] == 0
    )


def test_rank_correlation_and_keep_units():
    x = np.arange(50.0)
    assert rank_correlation(x, x**3) > 0.999
    assert rank_correlation(x, -x) < -0.999
    keep = keep_units(6000, 1 / 6, seed=5)
    assert (keep == keep_units(6000, 1 / 6, seed=5)).all()
    assert 850 < keep.sum() < 1150
