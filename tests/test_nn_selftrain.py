"""Soft self-training TabM on synthetic CPU problems."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.classes import N_CLASSES

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ringtrace import nn_selftrain, tabm  # noqa: E402


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


def _quiet(*_):
    return None


def _one_hot(y):
    t = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    t[np.arange(len(y)), y] = 1.0
    return t


def _probs(rows):
    """Stack rows of length 12 that already sum to 1."""
    p = np.asarray(rows, dtype=np.float64)
    assert p.shape[1] == N_CLASSES
    return p


def test_empty_pseudo_and_zero_weight_match_fit_predict():
    X, y = _problem(180, 0)
    hold_a, _ = _problem(40, 1, d=X.shape[1])
    hold_b, _ = _problem(25, 2, d=X.shape[1])
    period = np.zeros(len(y), dtype=np.int64)
    spec = _spec(seeds=(0, 1), epochs=2, batch=64, dropout=0.1, class_weight_power=0.5)
    base = tabm.fit_predict(X, y, period, [hold_a, hold_b], spec, log=_quiet)
    empty_x = np.zeros((0, X.shape[1]), dtype=np.float32)
    empty_p = np.zeros((0, N_CLASSES), dtype=np.float32)
    got = nn_selftrain.fit_predict_selftrain(
        X, y, empty_x, empty_p, [hold_a, hold_b], spec, w_pseudo=0.3, log=_quiet
    )
    assert len(got) == 2
    np.testing.assert_array_equal(got[0], base[0])
    np.testing.assert_array_equal(got[1], base[1])

    # A pseudo matrix that would move the quantile fit if it were included.
    rng = np.random.default_rng(4)
    junk = np.full((30, X.shape[1]), 1e6, dtype=np.float32)
    junk[0, 0] = np.nan
    raw = rng.random((30, N_CLASSES))
    raw /= raw.sum(axis=1, keepdims=True)
    zero = nn_selftrain.fit_predict_selftrain(
        X, y, junk, raw, [hold_a, hold_b], spec, w_pseudo=0.0, min_conf=0.0, log=_quiet
    )
    np.testing.assert_array_equal(zero[0], base[0])
    np.testing.assert_array_equal(zero[1], base[1])

    # Filtering every pseudo row is the same labelled fit.
    capped = nn_selftrain.fit_predict_selftrain(
        X,
        y,
        junk,
        raw,
        [hold_a, hold_b],
        spec,
        w_pseudo=1.0,
        min_conf=1.1,
        cap=np.zeros(N_CLASSES, dtype=np.int64),
        log=_quiet,
    )
    np.testing.assert_array_equal(capped[0], base[0])
    np.testing.assert_array_equal(capped[1], base[1])


def test_zero_weight_row_has_no_effect():
    spec = _spec(epochs=2, batch=16, dropout=0.1, k=3, width=16)
    rng = np.random.default_rng(0)
    n, d = 40, 5
    Z = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, n)
    T = _one_hot(y)
    w = np.ones(n, dtype=np.float64)
    extra_z = np.concatenate([Z, rng.normal(size=(3, d)).astype(np.float32)])
    extra_t = np.concatenate([T, rng.random((3, N_CLASSES)).astype(np.float32)])
    extra_t[n:] /= extra_t[n:].sum(axis=1, keepdims=True)
    extra_w = np.concatenate([w, np.zeros(3)])
    first = nn_selftrain.train_one_soft(Z, T, w, spec, seed=0, log=_quiet)
    second = nn_selftrain.train_one_soft(extra_z, extra_t, extra_w, spec, seed=0, log=_quiet)
    for pa, pb in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(pa, pb)

    # Inside the loss, a zero weight drops that row even if its logits change.
    torch.manual_seed(1)
    logits = torch.randn(4, spec.k, N_CLASSES)
    target = torch.softmax(torch.randn(4, N_CLASSES), dim=-1)
    row_w = torch.tensor([1.0, 0.5, 0.0, 2.0])
    cw = torch.ones(N_CLASSES)
    full = nn_selftrain._weighted_soft_ce(logits, target, row_w, cw)
    kept = nn_selftrain._weighted_soft_ce(
        logits[[0, 1, 3]], target[[0, 1, 3]], row_w[[0, 1, 3]], cw
    )
    torch.testing.assert_close(full, kept)
    changed = target.clone()
    changed[2] = torch.softmax(torch.randn(N_CLASSES), dim=-1)
    wild = logits.clone()
    wild[2] = 50.0
    again = nn_selftrain._weighted_soft_ce(wild, changed, row_w, cw)
    torch.testing.assert_close(again, full)
    wild.requires_grad_(True)
    nn_selftrain._weighted_soft_ce(wild, changed, row_w, cw).backward()
    assert torch.all(wild.grad[2] == 0)
    assert torch.any(wild.grad[0] != 0)

    with pytest.raises(ValueError):
        nn_selftrain.train_one_soft(Z, T, -w, spec, seed=0, log=_quiet)


def test_min_conf_and_cap_select_the_most_confident_rows():
    def row(cls, conf):
        p = np.full(N_CLASSES, (1.0 - conf) / (N_CLASSES - 1))
        p[cls] = conf
        return p

    # class 0: a below-threshold row must not take the only slot from a weaker but eligible row
    P = _probs(
        [
            row(0, 0.40),  # 0 below min_conf
            row(0, 0.70),  # 1 eligible, most confident class 0
            row(0, 0.65),  # 2 eligible class 0
            row(1, 0.90),  # 3 class 1
            row(1, 0.80),  # 4 class 1, tied with 5
            row(1, 0.80),  # 5 earlier tie wins if cap is 2? both 0.80, cap 2 keeps 4 and 5
            row(1, 0.50),  # 6 equal to min_conf, kept
            row(2, 0.49),  # 7 just below
        ]
    )
    snap = P.copy()
    got = nn_selftrain.select_pseudo(P, min_conf=0.50, cap={0: 1, 1: 2})
    np.testing.assert_array_equal(P, snap)
    # class 0 keeps index 1. class 1 cap 2 keeps 0.90 and the earlier 0.80.
    np.testing.assert_array_equal(got, [1, 3, 4])

    # The other 0.80 is kept when the cap has room, and the boundary confidence stays.
    wider = nn_selftrain.select_pseudo(P, min_conf=0.50, cap={0: 1, 1: 3})
    np.testing.assert_array_equal(wider, [1, 3, 4, 5])
    boundary = nn_selftrain.select_pseudo(P, min_conf=0.50, cap=None)
    np.testing.assert_array_equal(boundary, [1, 2, 3, 4, 5, 6])

    as_array = np.full(N_CLASSES, 10, dtype=np.int64)
    as_array[0] = 1
    as_array[1] = 2
    np.testing.assert_array_equal(
        nn_selftrain.select_pseudo(P, min_conf=0.50, cap=as_array), [1, 3, 4]
    )
    # A missing dict key does not cap that class. Class 2 is below min_conf here;
    # raise its confidence and leave it out of the dict.
    P2 = P.copy()
    P2[7] = row(2, 0.95)
    P2[6] = row(2, 0.60)
    kept = nn_selftrain.select_pseudo(P2, min_conf=0.50, cap={0: 1, 1: 1})
    assert 6 in kept and 7 in kept and 1 in kept
    # class 1 cap 1 keeps only the 0.90
    assert 3 in kept and 4 not in kept and 5 not in kept


def test_one_hot_soft_loss_matches_hard_cross_entropy():
    torch.manual_seed(0)
    b, k = 6, 3
    logits = torch.randn(b, k, N_CLASSES)
    y = torch.tensor([0, 1, 3, 0, 7, 11])
    counts = torch.bincount(y, minlength=N_CLASSES).double()
    power = 0.5
    cw = torch.where(counts > 0, (counts.sum() / counts.clamp_min(1)) ** power, 0.0)
    cw = cw / (cw * counts).sum() * counts.sum()
    cw = cw.float()
    target = F.one_hot(y, N_CLASSES).float()
    row_w = torch.ones(b)
    hard = F.cross_entropy(logits.reshape(-1, N_CLASSES), y.repeat_interleave(k), weight=cw)
    soft = nn_selftrain._weighted_soft_ce(logits, target, row_w, cw)
    torch.testing.assert_close(soft, hard, rtol=1e-6, atol=1e-6)

    # Non-uniform row weights: weighted mean of the class-weighted terms.
    row_w = torch.tensor([1.0, 0.0, 0.25, 3.0, 1.0, 0.5])
    nll = F.cross_entropy(logits.reshape(-1, N_CLASSES), y.repeat_interleave(k), reduction="none")
    cw_m = cw[y.repeat_interleave(k)]
    w_m = row_w.repeat_interleave(k)
    expected = (w_m * cw_m * nll).sum() / (w_m * cw_m).sum()
    got = nn_selftrain._weighted_soft_ce(logits, target, row_w, cw)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)

    logits1 = logits.detach().clone().requires_grad_(True)
    logits2 = logits.detach().clone().requires_grad_(True)
    F.cross_entropy(logits1.reshape(-1, N_CLASSES), y.repeat_interleave(k), weight=cw).backward()
    nn_selftrain._weighted_soft_ce(logits2, target, torch.ones(b), cw).backward()
    torch.testing.assert_close(logits1.grad, logits2.grad, rtol=1e-5, atol=1e-6)

    # Soft target: expectation of (class weight * NLL), not class weight times soft CE.
    logits_s = torch.zeros(1, 1, N_CLASSES)
    logits_s[0, 0, 1] = 2.0
    T = torch.zeros(1, N_CLASSES)
    T[0, 0] = 0.5
    T[0, 1] = 0.5
    cw_s = torch.ones(N_CLASSES)
    cw_s[1] = 10.0
    log_p = F.log_softmax(logits_s[0, 0], dim=-1)
    as_expectation = -(T[0] * cw_s * log_p).sum() / (T[0] * cw_s).sum()
    product = -(T[0] * log_p).sum()  # class weight would cancel if it only scaled the row
    loss = nn_selftrain._weighted_soft_ce(logits_s, T, torch.ones(1), cw_s)
    torch.testing.assert_close(loss, as_expectation)
    assert not torch.allclose(loss, product)

    # The counts behind those weights: one-hot unit rows match bincount; soft rows add mass.
    y_np = y.numpy()
    counts_np = nn_selftrain._expected_counts(_one_hot(y_np), np.ones(len(y_np)))
    np.testing.assert_array_equal(counts_np, np.bincount(y_np, minlength=N_CLASSES))
    soft_T = np.zeros((1, N_CLASSES), dtype=np.float64)
    soft_T[0, 1] = 1.0
    mixed = nn_selftrain._expected_counts(
        np.concatenate([_one_hot(y_np), soft_T]),
        np.concatenate([np.ones(len(y_np)), [0.3]]),
    )
    assert mixed[1] == pytest.approx(counts_np[1] + 0.3)
    cw_from_counts = nn_selftrain._class_weights(counts_np, power)
    np.testing.assert_allclose(cw_from_counts, cw.numpy(), rtol=1e-6, atol=1e-6)


def _shift_sample(n, seed, loc):
    """Same labelling rule, features recentred by ``loc`` (covariate shift)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    X += np.asarray(loc, dtype=np.float32)
    z = 1.4 * X[:, 0] + 0.9 * X[:, 1] - 0.7 * X[:, 2] + 0.4 * X[:, 0] * X[:, 1]
    y = np.zeros(n, dtype=np.int64)
    y[z > -0.4] = 1
    y[z > 0.6] = 2
    y[z > 1.5] = 3
    return X, y


def _teacher_probs(X, y_teacher_X, y_teacher, temperature=1.0):
    """Soft labels from class means on a labelled draw of the target month."""
    means = np.stack([X[y_teacher == c].mean(axis=0) for c in range(4)])
    # Distance in the shifted space. Unused classes stay at probability 0.
    dist = ((y_teacher_X[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)
    logits = -dist / temperature
    logits -= logits.max(axis=1, keepdims=True)
    raw = np.exp(logits)
    P = np.zeros((len(y_teacher_X), N_CLASSES), dtype=np.float32)
    P[:, :4] = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    return P


def test_selftrain_improves_accuracy_under_covariate_shift():
    loc = (1.1, -0.8, 0.6, 0.0, 0.0, 0.0)
    X, y = _shift_sample(1800, seed=10, loc=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    Xt, yt = _shift_sample(1600, seed=11, loc=loc)  # labelled target, teacher only
    Xp, yp = _shift_sample(1600, seed=12, loc=loc)
    Xh, yh = _shift_sample(1200, seed=13, loc=loc)
    P = _teacher_probs(Xt, Xp, yt, temperature=0.5)
    # The teacher should actually know the target month before it is asked to teach.
    teacher_acc = float((P.argmax(1) == yp).mean())
    assert teacher_acc > 0.7, teacher_acc
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
    period = np.zeros(len(y), dtype=np.int64)
    base = tabm.fit_predict(X, y, period, [Xh], spec, log=_quiet)[0]
    self_p = nn_selftrain.fit_predict_selftrain(
        X, y, Xp, P, [Xh], spec, w_pseudo=1.0, min_conf=0.0, log=_quiet
    )[0]
    acc_base = float((base.argmax(1) == yh).mean())
    acc_self = float((self_p.argmax(1) == yh).mean())
    assert acc_self > acc_base, (
        f"self {acc_self:.3f} vs base {acc_base:.3f} (teacher {teacher_acc:.3f})"
    )


def test_selftrain_is_deterministic():
    X, y = _problem(120, 5)
    rng = np.random.default_rng(6)
    Xp = rng.normal(size=(40, X.shape[1])).astype(np.float32)
    raw = rng.random((40, N_CLASSES)).astype(np.float32)
    raw /= raw.sum(axis=1, keepdims=True)
    hold, _ = _problem(20, 7, d=X.shape[1])
    spec = _spec(seeds=(0, 2), epochs=2, batch=32, dropout=0.1)
    a = nn_selftrain.fit_predict_selftrain(
        X, y, Xp, raw, [hold], spec, w_pseudo=0.3, min_conf=0.2, cap={0: 5, 1: 4}, log=_quiet
    )[0]
    b = nn_selftrain.fit_predict_selftrain(
        X, y, Xp, raw, [hold], spec, w_pseudo=0.3, min_conf=0.2, cap={0: 5, 1: 4}, log=_quiet
    )[0]
    assert a.shape == (len(hold), N_CLASSES)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(a, b)
