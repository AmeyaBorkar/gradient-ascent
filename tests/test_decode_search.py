"""Exact local search over the expected macro-F1 decoder, and the recipe switch that selects it."""

from __future__ import annotations

import time
import types

import numpy as np
import polars as pl
import pytest

from ringtrace import final, octlike
from ringtrace.classes import N_CLASSES
from ringtrace.decode import (
    decode_block,
    decode_by_period,
    expected_macro_f1,
    fixed_point,
    local_search,
)


def _simplex(rng, n):
    proba = rng.random((n, N_CLASSES))
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def _problems():
    """Random blocks, one with a 3-row class and one class that has no probability mass."""
    problems = [_simplex(np.random.default_rng(seed), n) for seed, n in ((0, 1), (1, 8), (2, 40))]
    rng = np.random.default_rng(5)
    proba = rng.random((80, N_CLASSES))
    proba[:, 4] = 0.0
    proba[:3, 4] = rng.random(3) + 1.0
    proba[:, 11] = 0.0
    proba /= proba.sum(axis=1, keepdims=True)
    problems.append(proba)
    return problems


def _no_improving_move(proba, pred, scale, tol=1e-10):
    base = expected_macro_f1(proba, pred, scale)
    for i in range(len(proba)):
        for label in range(proba.shape[1]):
            if label == int(pred[i]):
                continue
            trial = pred.copy()
            trial[i] = label
            if expected_macro_f1(proba, trial, scale) > base + tol:
                return False
    return True


def _tied_pair():
    """Two identical rows. The fixed point moves them together and cannot split them."""
    proba = np.zeros((2, N_CLASSES))
    proba[:, 0] = 0.1
    proba[:, 1] = 0.45
    proba[:, 2] = 0.45
    return proba


def test_expected_macro_f1_is_the_ratio_of_expectations():
    proba = _tied_pair()
    both = expected_macro_f1(proba, [2, 2], fraud_scale=1.0)
    split = expected_macro_f1(proba, [1, 2], fraud_scale=1.0)
    assert both == pytest.approx(3 / 58)
    assert split == pytest.approx(3 / 38)
    rng = np.random.default_rng(1)
    raw = rng.random((25, N_CLASSES))
    raw /= raw.sum(axis=1, keepdims=True)
    pred = rng.integers(0, N_CLASSES, 25)
    total = raw.sum(axis=0)
    etp = np.zeros(N_CLASSES)
    counts = np.zeros(N_CLASSES)
    for i, label in enumerate(pred):
        etp[label] += raw[i, label]
        counts[label] += 1
    f1 = []
    for k in range(N_CLASSES):
        denom = counts[k] + total[k]
        f1.append(0.0 if denom == 0 else 2 * etp[k] / denom)
    assert expected_macro_f1(raw, pred, fraud_scale=1.0) == pytest.approx(float(np.mean(f1)))


def test_search_never_loses_to_the_fixed_point_and_is_deterministic():
    tiny = _problems()[-1]
    assert np.count_nonzero(tiny[:, 4]) == 3
    assert not tiny[:, 11].any()
    for proba in _problems():
        for scale in (1.0, 0.875):
            first = local_search(proba, fraud_scale=scale)
            second = local_search(proba, fraud_scale=scale)
            assert np.array_equal(first, second)
            base = fixed_point(proba, fraud_scale=scale)
            assert (
                expected_macro_f1(proba, first, scale)
                >= expected_macro_f1(proba, base, scale) - 1e-12
            )
            if len(proba) <= 40:
                assert _no_improving_move(proba, first, scale)


def test_search_splits_rows_the_fixed_point_keeps_together():
    proba = _tied_pair()
    base = fixed_point(proba, fraud_scale=1.0)
    assert base.tolist() == [2, 2]
    untouched = proba.copy()
    found = local_search(proba, fraud_scale=1.0)
    assert np.array_equal(proba, untouched)
    assert found.tolist() == [1, 2]
    assert expected_macro_f1(proba, found, 1.0) > expected_macro_f1(proba, base, 1.0)
    assert np.array_equal(found, local_search(proba, fraud_scale=1.0, start=base))
    assert np.array_equal(local_search(proba, fraud_scale=1.0, max_rounds=0), base)
    assert np.array_equal(local_search(proba, fraud_scale=1.0, tol=1.0), base)


def test_ties_break_by_lower_class_and_lower_row():
    proba = np.zeros((1, N_CLASSES))
    proba[0, 1] = 0.5
    proba[0, 2] = 0.5
    assert local_search(proba, fraud_scale=1.0, start=np.array([0])).tolist() == [1]

    rows = np.zeros((2, N_CLASSES))
    rows[:, 0] = 0.5
    rows[:, 1] = 0.5
    start = np.array([0, 0])
    found = local_search(rows, fraud_scale=1.0, start=start)
    assert start.tolist() == [0, 0]
    assert found.tolist() == [1, 0]
    assert expected_macro_f1(rows, found, 1.0) > expected_macro_f1(rows, [1, 1], 1.0)


def test_decode_by_period_keeps_the_fixed_point_unless_asked():
    proba = np.vstack([_tied_pair(), _tied_pair()])
    period = np.array([7, 7, 8, 8])
    kw = {"fraud_scale": 0.9, "damping": 0.2, "max_iter": 4, "hidden_factor": 1.4}
    got = decode_by_period(proba, period, **kw)
    for month in (7, 8):
        sel = period == month
        assert np.array_equal(got[sel], fixed_point(proba[sel], **kw))
    assert np.array_equal(got, decode_by_period(proba, period, method="fixed_point", **kw))

    pair = _tied_pair()
    plain = decode_by_period(pair, np.array([7, 7]), fraud_scale=1.0)
    assert np.array_equal(plain, fixed_point(pair, fraud_scale=1.0))
    searched = decode_by_period(pair, np.array([7, 7]), method="local_search", fraud_scale=1.0)
    assert searched.tolist() == [1, 2]
    with pytest.raises(ValueError, match="unknown decoder method"):
        decode_by_period(_tied_pair(), np.array([7, 7]), method="nope")


def _pair_bench(tmp_path):
    pair = _tied_pair()
    period = np.array([7, 7, 8, 8, 9, 9, 10, 10])
    led = pl.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(len(period))],
            "period": period,
            "is_test": period == 10,
        }
    )
    run = tmp_path / "pair"
    run.mkdir()
    np.save(run / "oof.npy", np.vstack([pair, pair, pair]))
    np.save(run / "test.npy", pair.copy())
    bench = final.Bench(
        led, {"given": np.zeros(6, dtype=np.int64)}, np.arange(6), tmp_path, log=lambda *_: None
    )
    return bench


def _recipe(decoder=None):
    recipe = {"name": "r", "members": [{"name": "pair", "run": "pair"}], "rules": []}
    if decoder is not None:
        recipe["decoder"] = decoder
    return final.check_recipe(recipe)


def _by_fixed_point(bench, recipe, scale):
    arrays = [bench.member(m) for m in recipe["members"]]
    mixed = final.mix(arrays, final.static_weights(recipe), [m["role"] for m in recipe["members"]])
    out = np.zeros(bench.led.height, dtype=np.int64)
    for month in (7, 8, 9):
        rows = bench.tr[bench.period[bench.tr] == month]
        out[rows] = fixed_point(mixed[rows], fraud_scale=scale)
    out[bench.te] = fixed_point(mixed[bench.te], fraud_scale=scale)
    return out, mixed


def test_recipe_without_method_matches_fixed_point_calls(tmp_path):
    bench = _pair_bench(tmp_path)
    recipe = _recipe()
    assert "method" not in recipe["decoder"]
    manual, mixed = _by_fixed_point(bench, recipe, 0.875)
    assert np.array_equal(bench.calls(recipe)["pred"], manual)
    explicit = _recipe({"method": "fixed_point", "fraud_scale": 0.875})
    bench._calls.clear()
    assert np.array_equal(bench.calls(explicit)["pred"], manual)
    rows = bench.tr[bench.period[bench.tr] == 7]
    assert np.array_equal(
        bench.decode_rows(rows, mixed[rows], recipe), fixed_point(mixed[rows], fraud_scale=0.875)
    )


def test_recipe_can_select_local_search(tmp_path):
    bench = _pair_bench(tmp_path)
    recipe = _recipe({"method": "local_search", "fraud_scale": 0.875})
    base = _recipe()
    found = bench.calls(recipe)["pred"]
    manual, mixed = _by_fixed_point(bench, base, 0.875)
    assert not np.array_equal(found, manual)
    for month in (7, 8, 9, 10):
        rows = bench.te if month == 10 else bench.tr[bench.period[bench.tr] == month]
        assert expected_macro_f1(mixed[rows], found[rows], 0.875) > expected_macro_f1(
            mixed[rows], manual[rows], 0.875
        )
    rows = bench.tr[bench.period[bench.tr] == 7]
    assert bench.decode_rows(rows, mixed[rows], recipe).tolist() == [1, 2]
    frame = types.SimpleNamespace(hold=np.array([True, True]), led=types.SimpleNamespace(height=2))
    assert octlike.decode(frame, _tied_pair(), recipe["decoder"], rules=[]).tolist() == [1, 2]
    assert np.array_equal(
        octlike.decode(frame, _tied_pair(), {"fraud_scale": 1.0}, rules=[]),
        fixed_point(_tied_pair(), fraud_scale=1.0),
    )
    with pytest.raises(ValueError, match="unknown decoder method"):
        bench.calls(_recipe({"method": "nope"}))


def test_search_settings_pass_through_decode_block():
    rows = np.zeros((2, N_CLASSES))
    rows[:, 0] = 0.5
    rows[:, 1] = 0.5
    start = np.array([0, 0])
    found = decode_block(
        rows, method="local_search", fraud_scale=1.0, start=start, max_rounds=5, tol=0.0
    )
    assert found.tolist() == [1, 0]
    assert start.tolist() == [0, 0]
    blocked = decode_block(_tied_pair(), method="local_search", fraud_scale=1.0, tol=1.0)
    assert np.array_equal(blocked, fixed_point(_tied_pair(), fraud_scale=1.0))


def test_local_search_handles_300k_rows_quickly():
    proba = _simplex(np.random.default_rng(0), 300_000)
    started = time.perf_counter()
    found = local_search(proba)
    assert time.perf_counter() - started < 30
    base = fixed_point(proba)
    assert expected_macro_f1(proba, found) >= expected_macro_f1(proba, base) - 1e-12
