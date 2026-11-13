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


def test_soft_targets_smooth_across_the_fraud_boundary_and_mix_the_teacher():
    y = np.array([0, 3])
    prior = np.full(N_CLASSES, 1.0)
    q = tabm.soft_targets(y, _spec(smooth_fraud=0.1), prior)
    np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-6)
    assert q[0, 0] == pytest.approx(0.9) and q[0, 1:].sum() == pytest.approx(0.1)
    assert q[1, 3] == pytest.approx(0.9) and q[1, 0] == pytest.approx(0.1)
    assert q[1, [1, 2, 4]].sum() == 0
    teacher = np.full((2, N_CLASSES), 1 / N_CLASSES)
    q2 = tabm.soft_targets(y, _spec(distill_alpha=0.5), prior, teacher)
    assert q2[0, 0] == pytest.approx(0.5 + 0.5 / N_CLASSES)
    with pytest.raises(ValueError):
        tabm.soft_targets(y, _spec(distill_alpha=0.5), prior)


@pytest.mark.parametrize(
    "kw",
    [
        {"loss": "focal"},
        {"loss": "logit_adjust"},
        {"smooth_fraud": 0.05},
        {"distill_alpha": 0.3},
        {"mixup_alpha": 0.2},
        {"noise_std": 0.05},
        {"ema_decay": 0.99},
        {"swa_last": 3},
        {"pretrain_epochs": 2},
        {"qr_blank_p": 0.5},
    ],
)
def test_every_technique_learns_and_is_deterministic(kw):
    X, y = _problem(3000, 1)
    Xh, yh = _problem(1000, 2)
    period = np.full(len(y), 7)
    rng = np.random.default_rng(5)
    teacher = np.full((len(y), N_CLASSES), 0.02)
    teacher[np.arange(len(y)), y] += 1 - teacher.sum(1)
    Xb = X.copy()
    Xb[:, 4] = np.nan
    ex = tabm.Extras(
        teacher=teacher,
        X_blank=Xb,
        blank_rows=np.flatnonzero(rng.random(len(y)) < 0.3),
        X_unlabelled=Xh,
    )
    spec = _spec(**kw)
    (p1,) = tabm.fit_predict(X, y, period, [Xh], spec, log=lambda *_: None, extras=ex)
    (p2,) = tabm.fit_predict(X, y, period, [Xh], spec, log=lambda *_: None, extras=ex)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)
    assert macro_f1(yh, p1.argmax(1)) > 0.7


def test_qr_blanking_needs_its_inputs():
    X, y = _problem(500, 1)
    with pytest.raises(ValueError):
        tabm.fit_predict(
            X, y, np.full(len(y), 7), [X], _spec(qr_blank_p=0.5, epochs=1), log=lambda *_: None
        )


def test_nnexp_variants_build_and_fraud_only_keeps_tree_types():
    from ringtrace import nnexp

    recipe = {"model": "tabm", "tabm": {"seeds": [0, 1, 2], "epochs": 10}}
    for name in nnexp.VARIANTS:
        nnexp.variant_spec(name, recipe)
    s = nnexp.variant_spec("ema+smooth", recipe)
    assert s.ema_decay > 0 and s.smooth_fraud > 0 and s.epochs == 10
    trees = np.zeros((1, N_CLASSES))
    trees[0, [0, 1, 2]] = [0.8, 0.15, 0.05]
    nn = np.zeros((1, N_CLASSES))
    nn[0, [0, 5]] = [0.4, 0.6]
    p = nnexp.fraud_only(nn, trees)
    assert p[0, 0] == pytest.approx(0.6)
    assert p[0, 1] / p[0, 2] == pytest.approx(3.0) and p[0, 5] == 0


def test_realmlp_prep_and_learning():
    from ringtrace import realmlp

    X, y = _problem(3000, 1)
    Xh, yh = _problem(1000, 2)
    prep = realmlp.fit_prep(X)
    Z = realmlp.transform(prep, X)
    assert Z.shape == (3000, 6) and np.isfinite(Z).all() and np.abs(Z[:, :5]).max() < 3.01
    spec = realmlp.RealMLPSpec(seeds=(0,), width=32, epochs=15, batch=256, lr=3e-3, device="cpu")
    (p1,) = realmlp.fit_predict(X, y, np.full(len(y), 7), [Xh], spec, log=lambda *_: None)
    (p2,) = realmlp.fit_predict(X, y, np.full(len(y), 7), [Xh], spec, log=lambda *_: None)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    assert macro_f1(yh, p1.argmax(1)) > 0.7
