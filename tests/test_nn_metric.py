"""Metric-loss TabM on synthetic CPU problems."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.classes import N_CLASSES

pytest.importorskip("torch")

import torch  # noqa: E402

from ringtrace import nn_metric, tabm  # noqa: E402
from ringtrace.scoring import macro_f1, per_class_f1  # noqa: E402


def _spec(**kw):
    base = dict(
        seeds=(0,),
        k=4,
        width=32,
        blocks=2,
        dropout=0.1,
        d_embed=4,
        n_freq=4,
        epochs=2,
        batch=64,
        lr=3e-3,
        device="cpu",
        quantiles=16,
    )
    base.update(kw)
    return tabm.TabMSpec(**base)


def _problem(n, seed, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, n)
    X[:, 0] = y + rng.normal(scale=0.3, size=n)
    X[rng.random(n) < 0.1, 2] = np.nan
    return X, y.astype(np.int64)


def _imbalanced(n, seed, signal=1.1):
    """About 95% class 0 and three rare classes, separated by a one-coordinate shift."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int64)
    n_rare = int(round(n * 0.05))
    idx = rng.choice(n, n_rare, replace=False)
    labels = np.resize(np.array([1, 2, 3], dtype=np.int64), n_rare)
    y[idx] = rng.permutation(labels)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    for c in (1, 2, 3):
        X[y == c, c - 1] += np.float32(signal)
    return X, y


def _quiet(*_):
    return None


def _rare_macro_f1(y, proba):
    return float(per_class_f1(y, proba.argmax(1))[1:4].mean())


def test_softf1_matches_the_batch_definition():
    p = torch.zeros(3, N_CLASSES)
    p[0, 0], p[0, 1] = 0.7, 0.3
    p[1, 0], p[1, 1] = 0.2, 0.8
    p[2, 0], p[2, 1] = 0.6, 0.4
    y = torch.tensor([0, 1, 0])
    eps = nn_metric._F1_EPS
    f0 = 2 * 1.3 / (2 * 1.3 + 0.2 + 0.7 + eps)
    f1 = 2 * 0.8 / (2 * 0.8 + 0.7 + 0.2 + eps)
    expected = 1.0 - (f0 + f1) / 2.0
    got = nn_metric._softf1_extra(p, y)
    torch.testing.assert_close(got, torch.tensor(expected, dtype=got.dtype))
    # a class with no rows in the batch is left out of the mean
    assert torch.isfinite(got)


def test_pairwise_matches_the_margin_definition_and_caps_pairs():
    # s = 1 - p(legit). Two fraud rows, three legit: m = min(8, 3) = 3, so every legit is used.
    p_legit = torch.tensor([0.2, 0.4, 0.9, 0.5, 0.7])
    p = torch.zeros(5, N_CLASSES)
    p[:, 0] = p_legit
    p[:, 1] = 1.0 - p_legit
    y = torch.tensor([1, 1, 0, 0, 0])
    s_fraud = 1.0 - p_legit[:2]
    s_legit = 1.0 - p_legit[2:]
    gaps = s_fraud[:, None] - s_legit[None, :]
    expected = torch.nn.functional.softplus(-gaps).mean()
    got = nn_metric._pairwise_extra(p, y)
    torch.testing.assert_close(got, expected)

    assert nn_metric._pair_budget(10, 100) == (10, 40)
    assert nn_metric._pair_budget(200, 10_000) == (200, 250)
    assert nn_metric._pair_budget(0, 10) == (0, 0)
    kept, m = nn_metric._pair_budget(60_000, 10)
    assert kept * m <= 50_000

    wide_y = torch.zeros(6000, dtype=torch.long)
    wide_y[:3000] = 1
    wide_p = torch.softmax(torch.randn(6000, N_CLASSES), dim=-1)
    wide = nn_metric._pairwise_extra(wide_p, wide_y)
    assert torch.isfinite(wide)
    assert nn_metric._pair_budget(3000, 3000)[0] * nn_metric._pair_budget(3000, 3000)[1] <= 50_000


def test_reinforce_is_the_score_function_and_matches_macro_f1():
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(24, N_CLASSES), dim=-1)
    y = torch.arange(24) % N_CLASSES
    baseline = 0.05
    torch.manual_seed(1)
    extra, new_baseline = nn_metric._reinforce_extra(p, y, baseline)
    torch.manual_seed(1)
    sampled = torch.multinomial(p, num_samples=1).squeeze(1)
    reward = macro_f1(y.numpy(), sampled.numpy())
    torch.testing.assert_close(
        nn_metric._macro_f1_torch(y, sampled), torch.tensor(reward), rtol=1e-6, atol=1e-6
    )
    logp = p.gather(1, sampled[:, None]).clamp_min(nn_metric._LOGP_EPS).log().mean()
    expected = -(reward - baseline) * logp
    torch.testing.assert_close(extra, expected)
    assert new_baseline == pytest.approx(
        nn_metric._BASELINE_EMA * baseline + (1.0 - nn_metric._BASELINE_EMA) * reward
    )
    assert torch.isfinite(extra)


def test_each_loss_is_finite_and_reaches_every_parameter():
    spec = _spec(dropout=0.0, k=3, width=16, blocks=2, d_embed=4, n_freq=3)
    rng = np.random.default_rng(0)
    n = 36
    Z = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.arange(n) % N_CLASSES
    for name in ("none", "softf1", "pairwise", "reinforce"):
        tabm._seed_all(0)
        model = tabm.build_model(Z.shape[1], spec)
        model.train()
        logits = model(torch.from_numpy(Z))
        yb = torch.from_numpy(y.astype(np.int64))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, N_CLASSES), yb.repeat_interleave(spec.k)
        )
        if name != "none":
            extra, _ = nn_metric._extra(logits, yb, name, 0.0)
            assert torch.isfinite(extra)
            loss = loss + 0.5 * extra
        assert torch.isfinite(loss)
        loss.backward()
        dead = [pname for pname, p in model.named_parameters() if p.grad is None]
        assert not dead, (name, dead)
        bad = [
            pname
            for pname, p in model.named_parameters()
            if not torch.isfinite(p.grad).all()
        ]
        assert not bad, (name, bad)


def test_none_matches_tabm_fit_predict():
    X, y = _problem(180, 0)
    hold_a, _ = _problem(40, 1, d=X.shape[1])
    hold_b, _ = _problem(25, 2, d=X.shape[1])
    period = np.zeros(len(y), dtype=np.int64)
    spec = _spec(seeds=(0, 1), epochs=2, batch=64, dropout=0.1)
    base = tabm.fit_predict(X, y, period, [hold_a, hold_b], spec, log=_quiet)
    got = nn_metric.fit_predict_metric(
        X, y, period, [hold_a, hold_b], spec, metric_loss="none", log=_quiet
    )
    assert len(got) == 2
    np.testing.assert_array_equal(got[0], base[0])
    np.testing.assert_array_equal(got[1], base[1])


def test_same_seed_repeats_for_every_loss():
    X, y = _problem(160, 3)
    hold, _ = _problem(30, 4, d=X.shape[1])
    period = np.zeros(len(y), dtype=np.int64)
    spec = _spec(seeds=(0,), epochs=2, batch=50, dropout=0.1)
    for name in ("none", "softf1", "pairwise", "reinforce"):
        a = nn_metric.fit_predict_metric(
            X, y, period, [hold], spec, metric_loss=name, lam=0.5, log=_quiet
        )[0]
        b = nn_metric.fit_predict_metric(
            X, y, period, [hold], spec, metric_loss=name, lam=0.5, log=_quiet
        )[0]
        assert a.shape == (len(hold), N_CLASSES)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a.sum(1), 1.0, atol=1e-5)
        np.testing.assert_array_equal(a, b)


def test_unknown_metric_loss_raises():
    spec = _spec(epochs=1, batch=8)
    Z = np.zeros((8, 3), dtype=np.float32)
    y = np.zeros(8, dtype=np.int64)
    with pytest.raises(ValueError):
        nn_metric.train_one_metric(Z, y, spec, 0, metric_loss="macro", log=_quiet)


def test_softf1_and_pairwise_hold_rare_class_f1():
    X, y = _imbalanced(5000, seed=10, signal=1.1)
    Xh, yh = _imbalanced(2500, seed=11, signal=1.1)
    period = np.zeros(len(y), dtype=np.int64)
    spec = _spec(
        seeds=(0,),
        k=4,
        width=48,
        blocks=2,
        dropout=0.0,
        d_embed=4,
        n_freq=4,
        epochs=12,
        batch=256,
        lr=3e-3,
        quantiles=32,
        device="cpu",
    )
    scores = {}
    for name in ("none", "softf1", "pairwise"):
        proba = nn_metric.fit_predict_metric(
            X, y, period, [Xh], spec, metric_loss=name, lam=0.5, log=_quiet
        )[0]
        scores[name] = _rare_macro_f1(yh, proba)
    # rare-class macro-F1 (classes 1, 2, 3) at argmax, held out
    assert scores["softf1"] >= scores["none"] - 0.02, scores
    assert scores["pairwise"] >= scores["none"] - 0.02, scores
