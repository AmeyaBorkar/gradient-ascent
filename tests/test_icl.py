"""The in-context member's pure pieces on synthetic data: threshold, feature ranking, context
subsampling, prior correction and the 12-class composition."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace import icl


def _proba(rng, n, k=12):
    a = rng.random((n, k))
    a[:, 0] += 5.0
    return a / a.sum(axis=1, keepdims=True)


def test_choose_tau_leaves_the_share():
    p = np.random.default_rng(0).random(100_000)
    tau = icl.choose_tau(p, 0.05)
    assert abs((p >= tau).mean() - 0.05) < 1e-3


def test_auc_score_extremes_and_missing():
    y = np.array([0, 0, 1, 1])
    assert icl.auc_score(np.array([0.0, 1.0, 2.0, 3.0]), y) == 1.0
    assert icl.auc_score(np.array([3.0, 2.0, 1.0, 0.0]), y) == 0.0
    assert icl.auc_score(np.ones(4), y) == 0.5
    # missing values rank lowest
    assert icl.auc_score(np.array([np.nan, np.nan, 1.0, 2.0]), y) == 1.0


def test_rank_features_picks_the_signal_and_skips_constants():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    X = rng.normal(size=(2000, 6))
    X[:, 4] += 2.0 * y
    X[:, 1] -= 1.0 * y
    X[:, 2] = 7.0
    assert icl.rank_features(X, y, 2).tolist() == [1, 4]
    assert 2 not in icl.rank_features(X, y, 5).tolist()
    assert len(icl.rank_features(X, y, 50)) == 6


def test_subsample_keeps_every_fraud_row_and_is_seeded():
    y = np.r_[np.ones(100, int), np.zeros(1000, int)]
    keep, q = icl.subsample_context(y, 0.5, seed=3)
    assert (y[keep] == 1).sum() == 100
    assert (y[keep] == 0).sum() == 50
    assert q == pytest.approx(0.05)
    again, _ = icl.subsample_context(y, 0.5, seed=3)
    assert np.array_equal(keep, again)
    allrows, q1 = icl.subsample_context(y, None, seed=3)
    assert len(allrows) == len(y) and q1 == 1.0


def test_prior_correct_inverts_the_shift():
    p_true = np.array([0.01, 0.2, 0.5, 0.9])
    q = 0.1
    odds_fit = p_true / (1 - p_true) / q
    p_fit = odds_fit / (1 + odds_fit)
    assert np.allclose(icl.prior_correct(p_fit, q), p_true)
    assert np.allclose(icl.prior_correct(p_true, 1.0), p_true)


def test_compose_member_matches_the_design():
    rng = np.random.default_rng(2)
    champ = _proba(rng, 50)
    cand = np.zeros(50, dtype=bool)
    cand[[3, 10, 20]] = True
    p_icl = np.array([0.0, 0.4, 1.0])
    out = icl.compose_member(champ, cand, p_icl)
    assert np.allclose(out[~cand], champ[~cand])
    assert np.allclose(out.sum(axis=1), 1.0) and np.isfinite(out).all() and (out >= 0).all()
    assert np.allclose(out[cand, 0], 1 - p_icl)
    share = champ[10, 1:] / champ[10, 1:].sum()
    assert np.allclose(out[10, 1:], 0.4 * share)


def test_compose_member_with_no_fraud_mass_spreads_evenly():
    champ = np.zeros((2, 12))
    champ[:, 0] = 1.0
    out = icl.compose_member(champ, np.array([True, False]), np.array([0.5]))
    assert np.allclose(out[0, 1:], 0.5 / 11)
    assert np.allclose(out[1], champ[1])


def test_blend_weights_sum_to_one():
    for w in (1 / 6, 1 / 3):
        ws = icl.blend_weights(5, w)
        assert len(ws) == 6 and ws[-1] == w and sum(ws) == pytest.approx(1.0)
    assert icl.blend_weights(5, 1 / 6)[0] == pytest.approx(1 / 6)
    assert icl.blend_weights(5, 1 / 3)[0] == pytest.approx(2 / 15)
