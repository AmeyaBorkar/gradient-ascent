"""The in-context member on synthetic data: threshold, feature ranking, context subsampling with
its prior correction, the 12-class composition and the month folds (with a stub model)."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace import icl
from ringtrace.classes import N_CLASSES


def _champ(rng, n, fraud_bias):
    logits = rng.normal(0, 1, (n, N_CLASSES))
    logits[:, 0] += 3.0 - fraud_bias
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def test_choose_tau_lets_the_share_in():
    rng = np.random.default_rng(0)
    p = rng.random(10_000)
    tau = icl.choose_tau(p, 0.05)
    assert (p >= tau).mean() == pytest.approx(0.05, abs=1e-3)


def test_auc_scores_rank_the_informative_column_first():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    X = rng.normal(size=(2000, 5))
    X[:, 3] += 2.0 * y
    X[:, 1] = 7.0  # constant
    X[rng.random(2000) < 0.1, 3] = np.nan
    X[0, 2] = np.inf
    s = icl.auc_scores(X, y)
    assert s.argmax() == 3
    assert s[1] == 0.0
    assert np.isfinite(s).all()
    assert icl.top_features(X, y, 2).tolist() == sorted([3, int(np.argsort(-s)[1])])


def test_subsample_context_keeps_fraud_and_reports_rate():
    y = np.r_[np.ones(100, int), np.zeros(9900, int)]
    idx, q = icl.subsample_context(y, 1000, seed=3)
    assert len(idx) == 1000 and y[idx].sum() == 100
    assert q == pytest.approx(900 / 9900)
    idx2, _ = icl.subsample_context(y, 1000, seed=3)
    assert np.array_equal(idx, idx2)
    all_idx, q1 = icl.subsample_context(y, 20_000, seed=3)
    assert len(all_idx) == len(y) and q1 == 1.0
    # fraud rows alone over the limit: all kept, the limit grows to leave legit 20 % of the room
    idx3, q3 = icl.subsample_context(y, 50, seed=3)
    assert y[idx3].sum() == 100 and len(idx3) == 125 and q3 == pytest.approx(25 / 9900)


def test_prior_correct_undoes_legit_subsampling():
    # true fraud rate 0.01; legit kept at q = 0.1 makes the context rate about 0.0917
    q, true = 0.1, 0.01
    seen = true / (true + q * (1 - true))
    assert icl.prior_correct(np.array([seen]), q)[0] == pytest.approx(true)
    p = np.array([0.0, 0.3, 1.0])
    assert np.array_equal(icl.prior_correct(p, 1.0), p)


def test_compose_rows_sum_to_one_and_keep_type_shares():
    rng = np.random.default_rng(2)
    champ = _champ(rng, 50, 2.0)
    pf = rng.random(50)
    out = icl.compose(pf, champ)
    assert out.shape == (50, N_CLASSES)
    assert np.allclose(out.sum(1), 1.0) and np.isfinite(out).all()
    assert np.allclose(out[:, 0], 1 - pf)
    share = champ[:, 1:] / champ[:, 1:].sum(1, keepdims=True)
    assert np.allclose(out[:, 1:] / out[:, 1:].sum(1, keepdims=True), share)
    zero = np.zeros((1, N_CLASSES))
    zero[0, 0] = 1.0
    assert np.allclose(icl.compose(np.array([0.5]), zero).sum(), 1.0)


class _Stub:
    """Logistic-free stand-in: fraud probability = the context fraud rate scaled by column 0."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        self.rate = float(np.mean(y))
        self.n_cols = X.shape[1]
        return self

    def predict_proba(self, X):
        assert X.shape[1] == self.n_cols
        p = np.clip(self.rate * (1 + np.nan_to_num(X[:, 0]) ** 2), 0, 1)
        return np.c_[1 - p, p]


def test_lomo_member_changes_only_candidates_and_never_peeks():
    rng = np.random.default_rng(4)
    n_tr, n_te = 3000, 600
    period = np.r_[np.repeat([7, 8, 9], n_tr // 3), np.full(n_te, 10)]
    is_test = period == 10
    y = np.r_[(rng.random(n_tr) < 0.05).astype(int), np.full(n_te, -1)]
    champ = _champ(rng, n_tr + n_te, 2.0 * (y > 0))
    X = rng.normal(size=(n_tr + n_te, 8)).astype(np.float32)
    X[:, 2] += 2.0 * (y > 0)

    out, rep = icl.lomo_member(X, (y > 0).astype(int), period, is_test, champ, _Stub, k=3,
                               max_context=150, seed=0, log=lambda *_: None)  # fmt: skip
    cand = icl.fraud_prob(champ) >= rep["tau"]
    assert np.allclose(out[~cand], champ[~cand])
    assert not np.allclose(out[cand], champ[cand])
    assert np.allclose(out.sum(1), 1.0) and np.isfinite(out).all()
    assert sum(rep["candidates"][str(m)] for m in (7, 8, 9)) >= 0.05 * n_tr
    assert set(rep["folds"]) == {"7", "8", "9", "october"}
    for f in rep["folds"].values():
        assert f["context_rows"] <= 150 and len(f["features"]) == 3
    # held-out month labels never reach their own fold: flipping them leaves that month unchanged
    y2 = y.copy()
    y2[period == 9] = 1 - y2[period == 9]
    out2, _ = icl.lomo_member(X, (y2 > 0).astype(int), period, is_test, champ, _Stub, k=3,
                              max_context=150, seed=0, log=lambda *_: None)  # fmt: skip
    assert np.allclose(out[period == 9], out2[period == 9])
