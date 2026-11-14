import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from cleandec_core import (  # noqa: E402
    counter_proxy,
    expected_confusion,
    hidden_counts,
    macro,
    noise_rates,
    switch_effect,
    switch_likelihood,
    wrong_type_matrix,
)

from ringtrace.decode import clean_posterior, fixed_point  # noqa: E402
from ringtrace.scoring import macro_f1  # noqa: E402


def _noisy_labels(rng, truth, u, flip):
    lab = truth.copy()
    lab[rng.random(len(truth)) < u] = 0
    f = (lab > 0) & (rng.random(len(truth)) < flip)
    lab[f] = (lab[f] - 1 + rng.integers(1, 11, f.sum())) % 11 + 1
    return lab


def test_noise_rates_recover_uniform_noise_and_flag_a_group_effect():
    rng = np.random.default_rng(0)
    n = 40_000
    truth = rng.integers(1, 12, n)
    group = rng.integers(0, 4, n)
    lab = _noisy_labels(rng, truth, 0.12, 0.03)
    r = noise_rates(lab, truth, group)
    assert abs(r["pooled_unreported"] - 0.12) < 0.01
    assert abs(r["pooled_wrong_type"] - 0.03 * 0.88) < 0.005
    assert r["df"] == 3 and r["chi2"] < 16  # homogeneous: chi2(3) is rarely above 16
    # a group that under-reports twice as often is flagged
    lab2 = lab.copy()
    g3 = (group == 3) & (lab2 > 0) & (rng.random(n) < 0.14)
    lab2[g3] = 0
    assert noise_rates(lab2, truth, group)["chi2"] > 50
    with pytest.raises(ValueError):
        noise_rates(lab, np.where(truth > 5, truth, 0), group)


def test_wrong_type_matrix_counts_reported_rows_only():
    truth = np.array([1, 1, 2, 2, 3])
    lab = np.array([1, 2, 0, 2, 1])
    cm = wrong_type_matrix(lab, truth)
    assert cm[1, 1] == 1 and cm[1, 2] == 1 and cm[2, 2] == 1 and cm[3, 1] == 1
    assert cm.sum() == 4


def test_counter_proxy_imposes_confident_segments_and_bot_pool():
    cleaned = np.array([0, 3, 0, 5, 0])
    block = np.array([2, 2, 2, 1, 0])
    seg = np.array([4, 4, 6, 0, 7])
    conf = np.array([0.9, 0.95, 0.5, -1.0, 0.99])
    out = counter_proxy(cleaned, block, seg, conf)
    assert out.tolist() == [4, 4, 0, 2, 0]


def test_expected_confusion_with_hidden_rows_keeps_totals():
    rng = np.random.default_rng(1)
    y = np.where(rng.random(5000) < 0.9, 0, rng.integers(1, 12, 5000))
    pred = np.where(rng.random(5000) < 0.8, y, 0)
    hid = np.zeros(12)
    hid[3] = 40.0
    cm0 = expected_confusion(y, pred)
    cm = expected_confusion(y, pred, hid)
    assert np.isclose(cm.sum(), cm0.sum())
    assert np.isclose(cm[3].sum() - cm0[3].sum(), 40.0)
    assert np.allclose(cm.sum(axis=0), cm0.sum(axis=0))  # calls never move
    assert np.isclose(macro(y, pred), macro_f1(y, pred))
    assert macro(y, pred, hid) < macro(y, pred)


def test_hidden_counts_share_the_total_by_ordinary_labels():
    target = np.array([1, 1, 2, 0, 3, 3])
    gen = np.array([0, 0, 0, 0, 3, 0])
    period = np.array([7, 7, 8, 8, 8, 8])
    h = hidden_counts(target, gen, period, total=8.0)
    assert np.isclose(h[7][1], 4.0) and np.isclose(h[8][2], 2.0) and np.isclose(h[8][3], 2.0)
    assert h[7][0] == 0 and np.isclose(sum(v.sum() for v in h.values()), 8.0)


def test_switch_effect_signs_and_likelihood_orders_the_shares():
    a = {"T": 100.0, "P": 50.0, "TP": 30.0}
    b = {"T": 300.0, "P": 100.0, "TP": 80.0}
    # moving rows that are truly a costs; moving rows that are truly b gains
    assert switch_effect(10, 10, a, b) < 0 < switch_effect(10, 0, a, b)
    lk = switch_likelihood(-0.01, 30, (0.0, 0.5, 1.0), a, b, public=1.0, draws=4000)
    means = [lk[k]["mean"] for k in ("0.00", "0.50", "1.00")]
    assert means[0] > means[1] > means[2]
    assert lk["1.00"]["p_at_or_below"] > lk["0.00"]["p_at_or_below"]


def test_clean_posterior_inverts_the_noise_process():
    rng = np.random.default_rng(2)
    n, k = 4000, 12
    t = rng.dirichlet(np.ones(k) * 0.3, n)
    u = np.r_[0.0, np.linspace(0.08, 0.16, k - 1)]
    r = 0.0007
    q = t.copy()
    q[:, 1:] = (1 - u[1:]) * t[:, 1:] + r / (k - 1) * t[:, [0]]
    q[:, 0] = 1 - q[:, 1:].sum(axis=1)
    back = clean_posterior(q, unreported=u.tolist(), legit_flip=r)
    assert np.allclose(back, t, atol=1e-9)
    # the uniform type flip is undone on the fraud split
    e = 0.03
    split = t[:, 1:] / t[:, 1:].sum(axis=1, keepdims=True)
    noisy = (1 - e) * split + e / (k - 2) * (1 - split)
    q2 = t.copy()
    q2[:, 1:] = noisy * t[:, 1:].sum(axis=1, keepdims=True)
    back2 = clean_posterior(q2, unreported=0.0, legit_flip=0.0, type_flip=e)
    ok = (split > e).all(axis=1)  # rows with no clipped class
    assert np.allclose(back2[ok], t[ok], atol=1e-9)
    assert np.allclose(back2.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        clean_posterior(q, unreported=[0.1, 0.1])


def test_fixed_point_with_clean_noise_off_equals_plain_decoder():
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.r_[50.0, np.ones(11) * 0.2], 3000)
    plain = fixed_point(p, fraud_scale=1.0)
    off = fixed_point(p, fraud_scale=1.0, clean={"unreported": 0.0, "legit_flip": 0.0})
    assert np.array_equal(plain, off)
    # inverting unreported fraud raises every fraud column: never fewer fraud calls
    up = fixed_point(p, fraud_scale=1.0, clean={"unreported": 0.12})
    assert (up > 0).sum() >= (plain > 0).sum()
