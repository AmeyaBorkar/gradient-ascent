import json

import numpy as np
import polars as pl
import pytest

from ringtrace import final
from ringtrace.classes import N_CLASSES


def _probs(rng, y, sharp):
    logits = rng.normal(size=(len(y), N_CLASSES))
    logits[np.arange(len(y)), y] += sharp
    p = np.exp(logits)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


@pytest.fixture
def bench(tmp_path):
    rng = np.random.default_rng(0)
    n_tr, n_te = 600, 200
    period = np.r_[np.repeat([7, 8, 9], n_tr // 3), np.full(n_te, 10)]
    is_test = period == 10
    y = np.where(rng.random(n_tr) < 0.6, 0, rng.integers(1, N_CLASSES, n_tr))
    led = pl.DataFrame(
        {"txn_id": [f"t{i}" for i in range(n_tr + n_te)], "period": period, "is_test": is_test}
    )
    y_te = rng.integers(0, N_CLASSES, n_te)
    for name, sharp in (("good", 4.0), ("weak", 1.0)):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "oof.npy", _probs(rng, y, sharp))
        np.save(d / "test.npy", _probs(rng, y_te, sharp))
    d = tmp_path / "good-refit"
    d.mkdir()
    np.save(d / "test.npy", _probs(rng, y_te, 4.0))
    targets = {"given": y, "cleaned": y.copy()}
    groups = np.arange(n_tr)
    return final.Bench(led, targets, groups, tmp_path, log=lambda *_: None)


def _recipe(**kw):
    r = {
        "name": "r",
        "members": [{"name": "good", "run": "good"}, {"name": "weak", "run": "weak"}],
        "rules": [],
    }
    r.update(kw)
    return final.check_recipe(json.loads(json.dumps(r)))


def test_check_recipe_defaults_and_errors():
    r = _recipe()
    assert r["weights"] == {"mode": "equal"}
    assert r["members"][0]["variant"] == "fold"
    with pytest.raises(final.RecipeError):
        _recipe(weights={"mode": "fixed", "values": {"good": 1}})
    with pytest.raises(final.RecipeError):
        _recipe(weights={"mode": "magic"})
    with pytest.raises(final.RecipeError):
        _recipe(base_rate_rules=[{"name": "x"}])
    with pytest.raises(final.RecipeError):
        final.check_recipe({"members": [{"name": "a", "run": "a", "variant": "later"}]})


def test_recipe_key_ignores_name():
    assert final.recipe_key(_recipe(name="a")) == final.recipe_key(_recipe(name="b"))
    assert final.recipe_key(_recipe()) != final.recipe_key(
        _recipe(weights={"mode": "fixed", "values": {"good": 1, "weak": 0}})
    )


def test_static_weights_normalise():
    w = final.static_weights(_recipe(weights={"mode": "fixed", "values": {"good": 3, "weak": 1}}))
    assert np.allclose(w, [0.75, 0.25])


def test_greedy_picks_best_and_stops():
    scores = {(1, 0): 0.5, (0, 1): 0.4, (2, 0): 0.5, (1, 1): 0.45}
    c = final.greedy_counts(lambda k: scores.get(k, 0.0), 2)
    assert c.tolist() == [1, 0]


def test_capped_raises_above_cap():
    is_test = np.array([False, True, True])
    a, b = np.array([0, 0, 0]), np.array([1, 1, 1])
    assert final.capped("x", a, b, is_test, 2) == 2
    with pytest.raises(final.RecipeError):
        final.capped("x", a, b, is_test, 1)


def test_calls_deterministic_and_identical_recipes_tie(bench):
    r = _recipe()
    c1 = bench.calls(r)["pred"]
    bench._calls.clear()
    c2 = bench.calls(r)["pred"]
    assert np.array_equal(c1, c2)
    cmp = bench.compare(r, _recipe(name="same"), draws=50)
    assert cmp["rows_changed"] == {"train": 0, "october": 0}
    assert all(v == 0 for v in cmp["given"]["change"])


def test_nested_prefers_the_good_member(bench):
    r = _recipe(weights={"mode": "nested", "steps": 5})
    w = bench.weights(r)
    for k in (7, 8, 9, "october"):
        assert w[k][0] > w[k][1]
    good_only = _recipe(weights={"mode": "fixed", "values": {"good": 1, "weak": 0}})
    lo = bench.lomo(good_only)["given"]["mean"]
    assert bench.lomo(r)["given"]["mean"] == pytest.approx(lo, abs=0.02)


def test_variant_changes_october_only(bench):
    fold = _recipe(members=[{"name": "good", "run": "good"}])
    refit = _recipe(members=[{"name": "good", "run": "good", "variant": "refit"}])
    a, b = bench.calls(fold)["pred"], bench.calls(refit)["pred"]
    assert np.array_equal(a[bench.tr], b[bench.tr])
    assert (a[bench.te] != b[bench.te]).any()
    other = _recipe(
        members=[{"name": "w", "run": "weak", "variant": "refit", "variant_run": "good"}]
    )
    c = bench.calls(other)["pred"]
    assert np.array_equal(c[bench.te], b[bench.te])


def test_gate_verdict():
    lomo = {
        "given": {"change": [0.0, 0.001, 0.0], "lo": 0.0001},
        "cleaned": {"change": [0.0, 0.001, 0.0], "lo": 0.0001},
        "rows_changed": {},
    }
    octl = {
        "per_seed": {"given": {13: 0.1, 14: 0.1}, "cleaned": {13: 0.1, 14: 0.1}},
        "lower_bound": {"given": 0.01, "cleaned": 0.01},
    }
    assert final.gate_verdict(lomo, octl)["passes"]
    octl["complete"] = False
    assert not final.gate_verdict(lomo, octl)["passes"]
    octl["complete"] = True
    octl["per_seed"]["given"][14] = 0.0
    assert not final.gate_verdict(lomo, octl)["passes"]
    assert not final.gate_verdict(lomo, None)["passes"]


def test_pair_counts():
    got = final.pair_counts(np.array([0, 1, 1]), np.array([0, 2, 2]))
    assert got == {f"{final.CLASSES[1]} -> {final.CLASSES[2]}": 2}


def test_fraud_role_moves_only_the_legit_split():
    a = np.array([[0.6, 0.3, 0.1] + [0.0] * (N_CLASSES - 3)])
    b = np.array([[0.2, 0.1, 0.7] + [0.0] * (N_CLASSES - 3)])
    out = final.mix([a, b], [0.5, 0.5], ["full", "fraud"])
    assert out[0, 0] == pytest.approx(0.4)
    assert out[0, 1] / out[0, 2] == pytest.approx(3.0)
    assert out.sum() == pytest.approx(1.0)
    assert np.allclose(final.mix([a, b], [0.5, 0.5]), 0.5 * a + 0.5 * b)


def test_fraud_role_with_zero_full_weight_keeps_the_trees_types():
    a = np.array([[0.6, 0.3, 0.1] + [0.0] * (N_CLASSES - 3)])
    b = np.array([[0.2, 0.1, 0.7] + [0.0] * (N_CLASSES - 3)])
    out = final.mix([a, b], [0.0, 1.0], ["full", "fraud"])
    assert out[0, 0] == pytest.approx(0.2)
    assert out[0, 1] / out[0, 2] == pytest.approx(3.0)
