"""The cascade and decoder end to end on a tiny synthetic problem (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.cascade import BoosterSpec, CascadeSpec, fit_predict
from ringtrace.classes import N_CLASSES
from ringtrace.decode import decode_by_period
from ringtrace.scoring import macro_f1

pytest.importorskip("xgboost")


def _problem(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    fraud = rng.random(n) < 0.08
    kind = rng.integers(1, N_CLASSES, n)
    y[fraud] = kind[fraud]
    X[fraud, 0] += 4.0  # fraud is visible
    X[fraud, 1] = kind[fraud]  # and its type is readable
    period = rng.integers(7, 10, n)
    return X, y, period


def test_cascade_learns_and_is_deterministic():
    X, y, period = _problem(6000, 0)
    Xt, yt, _ = _problem(2000, 1)
    spec = CascadeSpec(
        booster=BoosterSpec(rounds=30, depth=4, device="cpu"), seeds=(0,), screen_rounds=20
    )
    (p1,) = fit_predict(X, y, period, [Xt], spec, log=lambda *_: None)
    (p2,) = fit_predict(X, y, period, [Xt], spec, log=lambda *_: None)
    assert p1.shape == (2000, N_CLASSES)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)
    pred = decode_by_period(p1, np.full(2000, 10))
    assert macro_f1(yt, pred) > 0.8


def test_lightgbm_cascade_learns_and_is_deterministic():
    pytest.importorskip("lightgbm")
    from ringtrace.cascade import LgbmSpec

    X, y, period = _problem(6000, 0)
    Xt, yt, _ = _problem(2000, 1)
    spec = CascadeSpec(
        booster=LgbmSpec(rounds=30, num_leaves=15),
        seeds=(0, 1),
        screen=BoosterSpec(rounds=30, depth=4, device="cpu"),
        screen_rounds=20,
    )
    (p1,) = fit_predict(X, y, period, [Xt], spec, log=lambda *_: None)
    (p2,) = fit_predict(X, y, period, [Xt], spec, log=lambda *_: None)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)
    assert macro_f1(yt, decode_by_period(p1, np.full(2000, 10))) > 0.8


def test_catboost_stages_learn_and_are_deterministic_on_cpu():
    pytest.importorskip("catboost")
    from ringtrace.cascade import CatBoostSpec

    X, y, period = _problem(6000, 0)
    Xt, yt, _ = _problem(2000, 1)
    spec = CascadeSpec(
        booster=BoosterSpec(rounds=30, depth=4, device="cpu"),
        seeds=(0, 1),
        screen_rounds=20,
        stages="catboost",
        catboost=CatBoostSpec(iterations=40, depth=4, task_type="CPU"),
    )
    (p1,) = fit_predict(X, y, period, [Xt], spec, log=lambda *_: None)
    (p2,) = fit_predict(X, y, period, [Xt], spec, log=lambda *_: None)
    assert p1.shape == (2000, N_CLASSES)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)
    assert macro_f1(yt, decode_by_period(p1, np.full(2000, 10))) > 0.8


def test_lgbm_recipe_keeps_the_champion_screen_and_features():
    from ringtrace.cascade import LgbmSpec
    from ringtrace.run import cascade_spec, load_recipe

    champ, lg = load_recipe("recipes/champion.json"), load_recipe("recipes/lgbm.json")
    assert (lg["families"], lg["drop"]) == (champ["families"], champ["drop"])
    c, s = cascade_spec(champ), cascade_spec(lg)
    assert isinstance(s.booster, LgbmSpec) and s.seeds == c.seeds == (0, 1, 2)
    assert s.screen == c.booster and c.screen is None
    assert (s.screen_rounds, s.type_min_score, s.drop_legit_min_score) == (
        c.screen_rounds,
        c.type_min_score,
        c.drop_legit_min_score,
    )
    p = s.booster.params(2)
    assert p["seed"] == p["bagging_seed"] == p["feature_fraction_seed"] == 2
    assert p["deterministic"] and p["force_row_wise"]


def test_catboost_recipe_parses(monkeypatch):
    from ringtrace.run import cascade_spec, load_recipe

    monkeypatch.delenv("RINGTRACE_DEVICE", raising=False)

    spec = cascade_spec(load_recipe("recipes/catboost.json"))
    assert spec.stages == "catboost"
    assert spec.seeds == (0, 1, 2)
    p = spec.catboost.params(1, "Logloss")
    assert (p["iterations"], p["depth"], p["learning_rate"], p["border_count"]) == (
        300,
        8,
        0.08,
        254,
    )
    assert p["task_type"] == "GPU" and p["random_seed"] == 1 and p["verbose"] is False


def test_device_switch_forces_cpu(monkeypatch):
    from ringtrace.cascade import CatBoostSpec, forced_device

    monkeypatch.delenv("RINGTRACE_DEVICE", raising=False)
    assert forced_device() is None
    assert BoosterSpec().params(0)["device"] == "cuda"  # default unchanged
    assert CatBoostSpec().params(0, "Logloss")["task_type"] == "GPU"
    monkeypatch.setenv("RINGTRACE_DEVICE", "cpu")
    assert BoosterSpec().params(0)["device"] == "cpu"
    cat = CatBoostSpec(gpu_ram_part=0.5).params(0, "Logloss")
    assert cat["task_type"] == "CPU"
    assert "devices" not in cat and "gpu_ram_part" not in cat
    monkeypatch.setenv("RINGTRACE_DEVICE", "gpu")
    with pytest.raises(ValueError, match="RINGTRACE_DEVICE"):
        BoosterSpec().params(0)


def test_device_switch_forces_cuda(monkeypatch):
    from ringtrace.cascade import CatBoostSpec, LgbmSpec, forced_device

    monkeypatch.setenv("RINGTRACE_DEVICE", "cuda")
    assert forced_device() == "cuda"
    assert BoosterSpec(device="cpu").params(0)["device"] == "cuda"
    cat = CatBoostSpec(task_type="CPU", gpu_ram_part=0.2).params(0, "Logloss")
    assert cat["task_type"] == "GPU" and cat["devices"] == "0" and cat["gpu_ram_part"] == 0.2
    assert LgbmSpec().params(0)["device_type"] == "cpu"  # LightGBM has no GPU build here


def test_device_switch_fits_a_cuda_spec_on_cpu(monkeypatch):
    monkeypatch.setenv("RINGTRACE_DEVICE", "cpu")
    X, y, period = _problem(3000, 0)
    spec = CascadeSpec(
        booster=BoosterSpec(rounds=10, depth=3, device="cuda"), seeds=(0,), screen_rounds=5
    )
    (p,) = fit_predict(X, y, period, [X[:100]], spec, log=lambda *_: None)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)


@pytest.mark.parametrize("n_class", [0, 4])
def test_cpu_training_loop_matches_xgb_train(n_class):
    import xgboost as xgb

    from ringtrace.cascade import train_cpu

    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = rng.integers(0, max(n_class, 2), 2000).astype(np.float32)
    extra = (
        {"objective": "multi:softprob", "num_class": n_class}
        if n_class
        else {"objective": "binary:logistic"}
    )
    params = BoosterSpec(rounds=15, depth=3, device="cpu").params(0, **extra)
    a = xgb.train(params, xgb.QuantileDMatrix(X, label=y, max_bin=256), 15)
    b = train_cpu(params, xgb.QuantileDMatrix(X, label=y, max_bin=256), 15)
    np.testing.assert_array_equal(a.inplace_predict(X), b.inplace_predict(X))
    assert a.save_raw("json") == b.save_raw("json")
