import numpy as np

from ringtrace import icl
from ringtrace.classes import N_CLASSES


def _champ(n, rng):
    p = rng.dirichlet(np.ones(N_CLASSES) * 0.3, size=n)
    return p


def test_choose_tau_share():
    p = np.linspace(0, 1, 1001)
    tau = icl.choose_tau(p, 0.05)
    assert abs((p >= tau).mean() - 0.05) < 0.002


def test_auc_score_matches_definition():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 200).astype(float)
    y = rng.integers(0, 2, 200)
    pos, neg = x[y == 1], x[y == 0]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (
        len(pos) * len(neg)
    )
    assert abs(icl.auc_score(x, y) - brute) < 1e-12


def test_rank_features_picks_signal_and_skips_constant():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    X = rng.normal(size=(500, 6)).astype(np.float32)
    X[:, 4] += 3 * y
    X[:, 2] = 7.0
    X[0, 1] = np.nan
    cols = icl.rank_features(X, y, 2)
    assert 4 in cols and 2 not in cols and len(cols) == 2


def test_subsample_keeps_fraud_and_prior_correction():
    y = np.r_[np.ones(100, int), np.zeros(10_000, int)]
    keep, rate = icl.subsample_context(y, 1100, seed=3)
    assert len(keep) == 1100 and y[keep].sum() == 100
    assert abs(rate - 0.1) < 1e-9
    keep2, _ = icl.subsample_context(y, 1100, seed=3)
    assert np.array_equal(keep, keep2)
    # a model fitted at 10x legit thinning reports odds 10x too high
    p_true = 0.2
    p_biased = (p_true / (1 - p_true) / rate) / (1 + p_true / (1 - p_true) / rate)
    assert abs(icl.prior_correct(np.array([p_biased]), rate)[0] - p_true) < 1e-12
    assert icl.prior_correct(np.array([0.3]), 1.0)[0] == 0.3


def test_compose_rows_sum_to_one_and_keep_type_shares():
    rng = np.random.default_rng(2)
    champ = _champ(50, rng)
    champ[3] = np.eye(N_CLASSES)[0]  # no fraud mass
    cand = np.zeros(50, bool)
    cand[:10] = True
    p = rng.uniform(size=10)
    out = icl.compose(champ, cand, p)
    assert np.allclose(out.sum(1), 1) and np.isfinite(out).all()
    assert np.array_equal(out[~cand], champ[~cand])
    assert np.allclose(out[cand, 0], 1 - p)
    r = 1
    ratio = out[r, 1:] / champ[r, 1:]
    assert np.allclose(ratio, ratio[0])


def test_run_member_folds_never_see_their_own_month():
    rng = np.random.default_rng(4)
    n = 4000
    period = np.repeat([7, 8, 9, 10], n // 4)
    is_test = period == 10
    y = np.where(is_test, 0, (rng.uniform(size=n) < 0.1).astype(int))
    champ = _champ(n, rng)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    X[:, 0] = period  # the predictor sees which months are in its context through column 0
    seen = {}

    def predictor(Xc, yc, Xq, seed):
        seen[seed] = (set(np.unique(Xc[:, 0])), set(np.unique(Xq[:, 0])))
        return np.full(len(Xq), 0.5)

    res = icl.run_member(X, y, period, is_test, champ, predictor, k=5, max_context=10**6, seed=0)
    for m in (7, 8, 9, 10):
        ctx, q = seen[m]
        assert q == {m} and m not in ctx and 10 not in ctx
    cand = res["cand"]
    assert abs(cand[~is_test].mean() - icl.SHARE) < 0.01
    assert np.allclose(res["proba"][cand, 0], 0.5)
    assert np.array_equal(res["proba"][~cand], champ[~cand])
