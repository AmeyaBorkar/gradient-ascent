"""TabM on a tiny synthetic problem (CPU): preprocessing, learning, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.classes import N_CLASSES

pytest.importorskip("torch")

from ringtrace import tabm  # noqa: E402
from ringtrace.scoring import macro_f1  # noqa: E402


def _problem(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, n)
    X[:, 0] = y + rng.normal(scale=0.2, size=n)  # the class is readable from column 0
    X[rng.random(n) < 0.1, 2] = np.nan
    return X, y


def _spec(**kw):
    base = dict(
        seeds=(0,),
        k=4,
        width=32,
        blocks=2,
        d_embed=4,
        n_freq=4,
        epochs=15,
        batch=256,
        lr=3e-3,
        device="cpu",
        quantiles=50,
    )
    base.update(kw)
    return tabm.TabMSpec(**base)


def test_prep_is_fitted_on_train_only_and_flags_gaps():
    X, _ = _problem(500, 0)
    prep = tabm.fit_prep(X, quantiles=50)
    assert list(prep.gap_cols) == [2]
    Z = tabm.transform(prep, X)
    assert Z.shape == (500, 6)
    assert np.isfinite(Z).all()
    assert set(np.unique(Z[:, -1])) == {0.0, 1.0}
    # monotone in each column, roughly standard normal
    order = np.argsort(X[:, 1])
    assert np.all(np.diff(Z[order, 1]) >= -1e-6)
    assert abs(Z[:, 1].mean()) < 0.2
    # out-of-range test values are clipped to the training range
    far = tabm.transform(prep, np.full((1, 5), 1e6, dtype=np.float32))
    assert far[0, 1] <= Z[:, 1].max() + 1e-6


def test_tabm_learns_and_is_deterministic():
    X, y = _problem(3000, 1)
    Xh, yh = _problem(1000, 2)
    period = np.full(len(y), 7)
    (p1,) = tabm.fit_predict(X, y, period, [Xh], _spec(), log=lambda *_: None)
    (p2,) = tabm.fit_predict(X, y, period, [Xh], _spec(), log=lambda *_: None)
    assert p1.shape == (1000, N_CLASSES)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)
    assert macro_f1(yh, p1.argmax(1)) > 0.8


def test_run_dispatches_on_model():
    from ringtrace import run

    spec, fn = run.model({"model": "tabm", "tabm": {"seeds": [3, 4], "epochs": 2}})
    assert fn is tabm.fit_predict and spec.seeds == (3, 4) and spec.epochs == 2
    spec, fn = run.model({})
    assert fn.__module__ == "ringtrace.cascade"


def test_check_proba_rejects_empty_or_nan_rows():
    from ringtrace.run import check_proba

    good = np.full((3, N_CLASSES), 1 / N_CLASSES)
    check_proba(good, "ok")
    for bad in (np.zeros((1, N_CLASSES)), np.full((1, N_CLASSES), np.nan)):
        with pytest.raises(ValueError):
            check_proba(np.vstack([good, bad]), "bad")


def test_decompose_splits_changes_and_credits_them():
    from ringtrace.decompose import categories, decompose

    a = np.array([0, 0, 1, 2, 3, 0])
    b = np.array([1, 0, 0, 3, 3, 0])
    y = np.array([1, 0, 1, 3, 3, 0])
    cats = categories(a, b)
    assert cats["legit->fraud"].tolist() == [True, False, False, False, False, False]
    assert cats["fraud->legit"].sum() == 1 and cats["type switch"].sum() == 1
    rows = {r["category"]: r for r in decompose(a, b, {"given": y}, np.full(6, 7), months=(7,))}
    assert rows["legit->fraud"]["prec_given"] == 1.0
    assert rows["fraud->legit"]["prec_given"] == 0.0
    assert rows["legit->fraud"]["d_given"][0] > 0 > rows["fraud->legit"]["d_given"][0]


def test_fraud_split_keeps_types_and_takes_the_fraud_mass():
    from ringtrace.blend import fraud_split_from

    types = np.zeros((2, N_CLASSES))
    types[0, [0, 1, 2]] = [0.6, 0.3, 0.1]
    types[1, 0] = 1.0  # no fraud mass: types spread evenly
    split = np.zeros((2, N_CLASSES))
    split[:, 0] = [0.2, 0.5]
    split[:, 3] = [0.8, 0.5]
    p = fraud_split_from(types, split)
    np.testing.assert_allclose(p.sum(1), 1.0)
    np.testing.assert_allclose(p[0, :3], [0.2, 0.6, 0.2])
    np.testing.assert_allclose(p[1, 1:], 0.5 / 11)


@pytest.mark.parametrize(
    "kw",
    [
        {"embedding": "ple", "n_bins": 32},
        {"class_weight_power": 0.5},
        {"recent_weight": 2.0},
    ],
)
def test_tabm_variants_learn_and_are_deterministic(kw):
    X, y = _problem(2000, 1)
    Xh, yh = _problem(500, 2)
    period = np.where(np.arange(len(y)) < 1000, 7, 8)
    (p1,) = tabm.fit_predict(X, y, period, [Xh], _spec(**kw), log=lambda *_: None)
    (p2,) = tabm.fit_predict(X, y, period, [Xh], _spec(**kw), log=lambda *_: None)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)
    assert macro_f1(yh, p1.argmax(1)) > 0.7


def test_ple_edges_are_strictly_increasing():
    Z = np.repeat(np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32), 50, axis=0)
    e = tabm.ple_edges(Z, 4)
    assert e.shape == (2, 5) and np.all(np.diff(e, axis=1) > 0)


def test_recency_weight_marks_the_latest_month():
    w = tabm.recency_weight(np.array([7, 8, 8, 7]), 2.0)
    assert w.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert tabm.recency_weight(np.array([7, 8]), 1.0) is None


def test_two_stage_multiplies_fraud_and_type():
    pytest.importorskip("xgboost")
    X, y = _problem(2000, 1)
    Xh, yh = _problem(500, 2)
    period = np.where(np.arange(len(y)) < 1000, 7, 8)
    from ringtrace import cascade

    spec = _spec(two_stage=True)
    old = cascade.BoosterSpec.device
    cascade.BoosterSpec.device = "cpu"
    try:
        (p,) = tabm.fit_predict(X, y, period, [Xh], spec, log=lambda *_: None)
    finally:
        cascade.BoosterSpec.device = old
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    assert macro_f1(yh, p.argmax(1)) > 0.6


def test_fit_target_modes():
    from ringtrace import run

    y = np.array([0, 1, 2, 0])
    fit = np.array([True, True, False, False])
    assert run.fit_target({}, y, fit) is y
    cl = np.array([1, 1, 1, 1])
    assert run.fit_target({"train_on": "cleaned"}, y, fit, cl).tolist() == [1, 1, 2, 0]
    with pytest.raises(ValueError):
        run.fit_target({"train_on": "nope"}, y, fit)


def test_sweep_configs_build_valid_recipes():
    from ringtrace import run, tabmsweep

    for name in tabmsweep.CONFIGS:
        r = tabmsweep.recipe_for(name)
        spec, _ = run.model(r)
        assert r["name"] == "tabm2-sw-" + name and spec.epochs > 0
    assert tabmsweep.recipe_for("l2")["train_on"] == "event_l2"
