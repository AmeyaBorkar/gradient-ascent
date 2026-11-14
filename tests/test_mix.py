"""Class-mix weights and the label-shift prior correction, on synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.classes import CODE, N_CLASSES
from ringtrace.mix import class_mix_weights, class_shares, em_priors, ratios_from_names
from ringtrace.run import (
    apply_label_shift,
    class_mix_ratios,
    label_shift_on,
    reference_priors,
    sample_weight_for,
)


def test_reference_priors_use_the_other_months_and_fall_back_to_a_no_op():
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.5, 0.5]])
    per = np.array([7, 7, 8, 8])
    refs, whole = reference_priors(p, per, (7, 8))
    np.testing.assert_allclose(refs[7], [0.55, 0.45])  # month 8's mean
    np.testing.assert_allclose(refs[8], [0.85, 0.15])  # month 7's mean
    np.testing.assert_allclose(whole, p.mean(axis=0))
    one, _ = reference_priors(p[:2], per[:2], (7,))
    np.testing.assert_allclose(one[7], p[:2].mean(axis=0))
    adj, _ = apply_label_shift(p[:2], per[:2], one)
    np.testing.assert_allclose(adj, p[:2], atol=1e-6)


def test_stage_one_weights_only_its_positives(monkeypatch):
    import ringtrace.cascade as cascade

    seen = {}

    def fake_stages(X_bin, y_bin, X_type, y_type, spec, seed, w_bin=None, w_type=None):
        seen["y_bin"], seen["w_bin"], seen["w_type"] = y_bin, w_bin, w_type
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(cascade, "_fit_stages", fake_stages)
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [MULE] * 10)
    X = rng.normal(size=(len(y), 3)).astype(np.float32)
    spec = cascade.CascadeSpec(booster=cascade.booster_spec({"device": "cpu"}), use_screen=False)
    w = np.where(y == MULE, 3.0, 1.0)
    with pytest.raises(RuntimeError):
        cascade.fit_predict(
            X, y, np.full(len(y), 7), [X], spec, log=lambda *_: None, sample_weight=w
        )
    assert np.all(seen["w_bin"][seen["y_bin"] == 0] == 1.0)
    assert np.all(seen["w_bin"][seen["y_bin"] == 1] == 3.0)
    assert np.all(seen["w_type"] == 3.0)


MULE = CODE["mule_account"]
BOT = CODE["velocity_bot"]
CHAIN = CODE["layering_chain"]


def test_class_mix_weights_scale_with_counts_and_keep_legit_at_one():
    y = np.array([0, MULE, MULE, BOT, CHAIN, 4, 0, BOT])
    ratios = ratios_from_names({"mule_account": 2.0, "velocity_bot": 0.5, "layering_chain": 4.0})
    w = class_mix_weights(y, ratios)
    assert w.dtype == np.float64 and w.shape == (len(y),)
    expected = {0: 1.0, MULE: 2.0, BOT: 0.5, CHAIN: 4.0, 4: 1.0}
    for code, ratio in expected.items():
        rows = y == code
        np.testing.assert_allclose(w[rows], ratio)
        np.testing.assert_allclose(w[rows].sum(), ratio * rows.sum())
    assert np.all(w[y == 0] == 1.0)


def test_class_mix_weights_follow_the_event_majority():
    ratios = {MULE: 2.0, BOT: 0.5, CHAIN: 4.0}
    # event 10: legit, mule, mule, bot -> majority mule; event 7: chain; negatives stay themselves
    y = np.array([0, MULE, MULE, BOT, CHAIN, 0, BOT])
    events = np.array([10, 10, 10, 10, 7, -1, -1])
    w = class_mix_weights(y, ratios, events)
    np.testing.assert_array_equal(w[:4], np.full(4, 2.0))
    assert w[4] == 4.0
    assert w[5] == 1.0  # legit, no event
    assert w[6] == 0.5
    # a tie goes to the smaller code: mule and chain are level, mule wins
    tied = np.array([CHAIN, MULE, CHAIN, MULE])
    w_tie = class_mix_weights(tied, ratios, np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(w_tie, np.full(4, ratios[MULE]))
    # legit against mule is also a tie; the smaller code is legit, weight 1
    w_legit = class_mix_weights(np.array([0, MULE]), ratios, np.array([3, 3]))
    np.testing.assert_array_equal(w_legit, np.ones(2))


def test_ratios_from_names_rejects_legit_unknown_and_non_positive():
    with pytest.raises(ValueError, match="legit"):
        ratios_from_names({"legit": 2.0})
    with pytest.raises(ValueError, match="unknown"):
        ratios_from_names({"not_a_class": 2.0})
    with pytest.raises(ValueError, match="positive"):
        ratios_from_names({"mule_account": 0.0})
    with pytest.raises(ValueError, match="positive"):
        ratios_from_names({"mule_account": -1.5})
    with pytest.raises(ValueError, match="positive"):
        ratios_from_names({"velocity_bot": float("nan")})


def _bayes_posterior(train_prior, test_prior, n, seed, gap):
    """Training posterior of samples drawn from ``test_prior``, same class-conditionals."""
    rng = np.random.default_rng(seed)
    k = len(train_prior)
    y = rng.choice(k, size=n, p=test_prior)
    x = np.eye(k)[y] * gap + rng.normal(scale=1.0, size=(n, k))
    # ||x - gap e_j||^2 = ||x||^2 - 2 gap x_j + gap^2
    dist2 = (x * x).sum(1, keepdims=True) - 2.0 * gap * x + gap**2
    log_post = -0.5 * dist2 + np.log(train_prior)
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    return post


def test_em_priors_recovers_a_known_shift_and_stays_put_when_it_matches():
    train = np.array([0.40, 0.20, 0.15, 0.10, 0.10, 0.05])
    test = np.array([0.15, 0.35, 0.10, 0.20, 0.05, 0.15])
    post = _bayes_posterior(train, test, n=12_000, seed=0, gap=3.0)
    q, adjusted = em_priors(post, train)
    assert q.shape == train.shape and adjusted.shape == post.shape
    assert np.max(np.abs(q - test)) < 0.02
    np.testing.assert_allclose(adjusted.sum(1), 1.0, atol=1e-8)
    assert np.max(np.abs(q - train)) > 0.05

    # column means already equal the training prior: the correction is the identity
    same = np.array(
        [
            [0.50, 0.50, 0.0],
            [0.25, 0.75, 0.0],
        ]
    )
    prior = same.mean(axis=0)
    q_same, adj_same = em_priors(same, prior)
    np.testing.assert_array_equal(adj_same, same)
    np.testing.assert_allclose(q_same, prior)

    # first step starts at the training prior, so a flat ratio leaves the rows untouched
    flat = np.array([[0.9, 0.1], [0.9, 0.1]])
    q_one, adj_one = em_priors(flat, np.array([0.5, 0.5]), max_iter=1)
    np.testing.assert_array_equal(adj_one, flat)
    np.testing.assert_allclose(q_one, [0.9, 0.1])


def test_em_priors_zero_train_prior_stays_finite():
    post = np.array([[0.7, 0.3, 0.0], [0.2, 0.8, 0.0]])
    # the third class was absent in training and the model gives it nothing
    train = np.array([post[:, 0].mean(), post[:, 1].mean(), 0.0])
    q, adjusted = em_priors(post, train)
    assert np.isfinite(q).all() and np.isfinite(adjusted).all()
    np.testing.assert_allclose(adjusted.sum(1), 1.0)
    np.testing.assert_array_equal(adjusted, post)
    np.testing.assert_allclose(q[2], 0.0)


def test_recipe_helpers_parse_class_mix_and_label_shift():
    y = np.array([0, MULE, BOT, BOT])
    assert class_mix_ratios({}) is None
    assert class_mix_ratios({"class_mix": {}}) is None
    assert sample_weight_for({}, y) is None
    assert label_shift_on({}) is False
    assert label_shift_on({"label_shift_em": False}) is False
    assert label_shift_on({"label_shift_em": True}) is True

    recipe = {
        "class_mix": {"mule_account": 3.0, "velocity_bot": 0.5},
        "label_shift_em": True,
    }
    assert class_mix_ratios(recipe) == {MULE: 3.0, BOT: 0.5}
    np.testing.assert_array_equal(sample_weight_for(recipe, y), [1.0, 3.0, 0.5, 0.5])
    # event 4 ties legit with mule, so both rows take legit's weight
    events = np.array([4, 4, -1, -1])
    np.testing.assert_array_equal(sample_weight_for(recipe, y, events), [1.0, 1.0, 0.5, 0.5])
    assert class_mix_ratios({"model": "cascade", "class_mix": {"qr_swap": 1.5}}) == {
        CODE["qr_swap"]: 1.5
    }

    with pytest.raises(ValueError, match="cascade"):
        class_mix_ratios({"model": "tabm", "class_mix": {"mule_account": 2.0}})
    with pytest.raises(ValueError, match="legit"):
        class_mix_ratios({"class_mix": {"legit": 2.0}})
    with pytest.raises(ValueError, match="true or false"):
        label_shift_on({"label_shift_em": "yes"})

    prior_a = np.zeros(N_CLASSES)
    prior_b = np.zeros(N_CLASSES)
    prior_a[0], prior_a[MULE] = 0.9, 0.1
    prior_b[0], prior_b[BOT] = 0.5, 0.5
    post = np.vstack([prior_a, prior_a, prior_b, prior_b])
    period = np.array([7, 7, 8, 8])
    out, recorded = apply_label_shift(post, period, {7: prior_a, 8: prior_b})
    np.testing.assert_allclose(out, post)
    assert set(recorded) == {"7", "8"}
    np.testing.assert_allclose(recorded["7"]["estimated"], prior_a)
    np.testing.assert_allclose(recorded["7"]["train"], prior_a)
    np.testing.assert_allclose(recorded["8"]["estimated"], prior_b)
    # one shared training prior, checked against the training rows' own shares
    shares = class_shares(np.array([0, 0, 0, MULE]))
    block = np.tile(shares, (5, 1))
    out_b, rec_b = apply_label_shift(block, np.full(5, 10), shares)
    np.testing.assert_allclose(out_b, block)
    np.testing.assert_allclose(rec_b["10"]["estimated"], shares)
    np.testing.assert_allclose(rec_b["10"]["train"], shares)


def test_fit_predict_weights_match_ones_and_downweight_lowers_a_class():
    pytest.importorskip("xgboost")
    from ringtrace.cascade import BoosterSpec, CascadeSpec, fit_predict

    def problem(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 4)).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)
        draw = rng.random(n)
        y[draw < 0.15] = MULE
        y[(draw >= 0.15) & (draw < 0.30)] = BOT
        x[y > 0, 0] += 3.0
        x[y == MULE, 1] += 3.0
        x[y == BOT, 2] += 3.0
        period = np.resize(np.array([7, 8, 9]), n)
        return x, y, period

    x, y, period = problem(2400, 0)
    xt, _, _ = problem(800, 1)
    spec = CascadeSpec(
        booster=BoosterSpec(rounds=25, depth=3, device="cpu"),
        seeds=(0,),
        screen_rounds=12,
    )

    def quiet(*_):
        return None

    (base,) = fit_predict(x, y, period, [xt], spec, log=quiet)
    (ones,) = fit_predict(x, y, period, [xt], spec, log=quiet, sample_weight=np.ones(len(y)))
    np.testing.assert_array_equal(base, ones)
    np.testing.assert_allclose(base.sum(1), 1.0, atol=1e-5)
    assert base[:, MULE].mean() > 0.05

    down_w = class_mix_weights(y, {MULE: 1e-3})
    (down,) = fit_predict(x, y, period, [xt], spec, log=quiet, sample_weight=down_w)
    np.testing.assert_allclose(down.sum(1), 1.0, atol=1e-5)
    assert down[:, MULE].mean() < base[:, MULE].mean() * 0.5

    with pytest.raises(ValueError, match="length"):
        fit_predict(x, y, period, [xt], spec, log=quiet, sample_weight=np.ones(3))
