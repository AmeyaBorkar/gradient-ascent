"""Seed-by-seed fits: seed lists, the store, and byte equality with ``fit_predict`` (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace import member
from ringtrace.cascade import BoosterSpec, CascadeSpec, fit_predict, training_rows
from ringtrace.classes import N_CLASSES

pytest.importorskip("xgboost")


def _problem(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    fraud = rng.random(n) < 0.08
    kind = rng.integers(1, N_CLASSES, n)
    y[fraud] = kind[fraud]
    X[fraud, 0] += 4.0
    X[fraud, 1] = kind[fraud]
    return X, y, rng.integers(7, 9, n)


def _spec(**kw):
    return CascadeSpec(
        booster=BoosterSpec(rounds=20, depth=3, device="cpu"), screen_rounds=10, **kw
    )


def test_parse_seeds_and_outputs():
    assert member.parse_seeds("0-2,7") == (0, 1, 2, 7)
    assert member.parse_seeds("3") == (3,)
    for bad in ("", "1,1", "2-1x"):
        with pytest.raises(ValueError):
            member.parse_seeds(bad)
    assert member.parse_outputs(["a=0-2", "b=0-4"], groups=False) == {
        "a": [(0, 1, 2)],
        "b": [(0, 1, 2, 3, 4)],
    }
    assert member.parse_outputs(["a=0-2/3-5"], groups=True) == {"a": [(0, 1, 2), (3, 4, 5)]}
    with pytest.raises(ValueError):
        member.parse_outputs(["a=0-2/3-5"], groups=False)
    with pytest.raises(ValueError):
        member.parse_outputs(["a=0", "a=1"], groups=False)


def test_training_rows_follow_the_cuts():
    fraud = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    score = np.array([0.02, 0.05, 0.5, 0.1, 0.69, 0.71])
    keep, typ = training_rows(fraud, score, _spec())
    assert keep.tolist() == [True, True, True, True, True, False]
    assert typ.tolist() == [False, False, True, False, False, False]
    _, typ04 = training_rows(fraud, score, _spec(type_min_score=0.04))
    assert typ04.tolist() == [False, True, True, False, False, False]


def test_seed_subsets_equal_fit_predict_byte_for_byte(tmp_path):
    X, y, period = _problem(3000, 0)
    Xt, _, _ = _problem(500, 1)
    spec = _spec(seeds=(0, 1, 2, 3), type_min_score=0.04)
    st = member.SeedStore(tmp_path)
    log = lambda *_: None  # noqa: E731
    done = member.fit_seeds(st, ["a", "b"], X, y, period, [Xt, X[:100]], spec, (0, 1, 2, 3), log)
    assert done == (0, 1, 2, 3)
    for sub in ((0, 1, 2), (1, 3), (0, 1, 2, 3)):
        sub_spec = _spec(seeds=sub, type_min_score=0.04)
        pa, pb = fit_predict(X, y, period, [Xt, X[:100]], sub_spec, log=log)
        np.testing.assert_array_equal(st.mean("a", sub), pa)
        np.testing.assert_array_equal(st.mean("b", sub), pb)
    # stored seeds are not refitted; a missing one is
    assert member.fit_seeds(st, ["a", "b"], X, y, period, [Xt, X[:100]], spec, (0, 1), log) == ()
    st.path("b", 2).unlink()
    assert member.fit_seeds(st, ["a", "b"], X, y, period, [Xt, X[:100]], spec, (2,), log) == (2,)


def test_fit_seeds_refuses_mismatched_keys(tmp_path):
    X, y, period = _problem(500, 0)
    with pytest.raises(ValueError):
        member.fit_seeds(member.SeedStore(tmp_path), ["a"], X, y, period, [X, X], _spec(), (0,))
    with pytest.raises(TypeError):
        member.fit_seeds(member.SeedStore(tmp_path), ["a"], X, y, period, [X], object(), (0,))


def test_member_recipes_parse():
    from ringtrace.run import cascade_spec, load_recipe

    champ = load_recipe("recipes/champion.json")
    for name, seeds in (("member-x5", (0, 1, 2, 3, 4)), ("member-cut04", (0, 1, 2))):
        r = load_recipe(f"recipes/{name}.json")
        assert (r["families"], r["drop"], r["rules"]) == (
            champ["families"],
            champ["drop"],
            champ["rules"],
        )
        s, c = cascade_spec(r), cascade_spec(champ)
        assert s.seeds == seeds and s.booster == c.booster
        assert (s.type_min_score, s.drop_legit_min_score) == (0.04, 0.70)
        assert c.type_min_score == 0.10


def test_cpu_forced_fits_are_stored_apart(monkeypatch, tmp_path):
    monkeypatch.setenv("RINGTRACE_OUT", str(tmp_path))
    monkeypatch.delenv("RINGTRACE_DEVICE", raising=False)
    assert member.store_root("member-x5", "lomo") == tmp_path / "member" / "member-x5" / "lomo"
    monkeypatch.setenv("RINGTRACE_DEVICE", "cpu")
    assert member.store_root("member-x5", "s13") == tmp_path / "member" / "member-x5-cpu" / "s13"


def test_cli_refuses_unfitted_seeds_unless_shared(monkeypatch):
    called = {}
    monkeypatch.setattr(member, "cells", lambda *a: called.setdefault("cells", a))
    argv = ["cells", "r.json", "--frames", "13", "--seeds", "5-9", "--cells", "c"]
    with pytest.raises(SystemExit):
        member.main([*argv, "--as", "m=0-4/5-9"])
    member.main([*argv, "--as", "m=0-4/5-9", "--shared"])
    assert called["cells"][3] == {"m": [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)]}


def test_unfitted_names_the_seeds_missing_an_output(tmp_path):
    st = member.SeedStore(tmp_path)
    a = np.zeros((3, N_CLASSES))
    st.save("hold7", 0, a)
    st.save("oct7", 0, a)
    st.save("hold7", 1, a)  # seed 1 lacks its October output
    assert member.unfitted(st, ["hold7", "oct7"], (0, 1, 2)) == (1, 2)
    assert member.unfitted(st, ["hold7"], (0, 1)) == ()
    # nothing left to fit: fit_seeds returns before touching its (absent) inputs
    assert member.fit_seeds(st, ["hold7"], None, None, None, [None], _spec(), (0, 1)) == ()
