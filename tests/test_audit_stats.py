"""Audit statistics on synthetic data: a planted signal is found, pure noise is not."""

from __future__ import annotations

import numpy as np

from ringtrace.audit import stats

FAST = {"n_estimators": 60}


def test_cv_auc_finds_a_planted_signal_and_not_noise():
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.3).astype(int)
    noise = rng.normal(size=(n, 3))
    signal = np.column_stack([noise, y + rng.normal(scale=0.3, size=n)])
    hit = stats.cv_auc(signal, y, seeds=(0,), folds=3, params=FAST, names=["a", "b", "c", "s"])
    assert hit["auc"] > 0.95
    assert hit["top_features"][0][0] == "s"
    miss = stats.cv_auc(noise, y, seeds=(0,), folds=3, params=FAST)
    assert abs(miss["auc"] - 0.5) < 0.1


def test_cv_auc_too_few_positives_is_nan():
    y = np.array([1, 1] + [0] * 50)
    res = stats.cv_auc(np.zeros((52, 2)), y, folds=5)
    assert np.isnan(res["auc"]) and res["positives"] == 2


def test_univariate_rates_and_ranks():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(400), np.zeros(4000)].astype(bool)
    flag = np.r_[rng.random(400) < 0.5, rng.random(4000) < 0.2].astype(float)
    shifted = np.r_[rng.normal(1.0, 1, 400), rng.normal(0.0, 1, 4000)]
    flat = rng.normal(size=4400)
    res = stats.univariate({"flag": flag, "shifted": shifted, "flat": flat}, y)
    by = {r["feature"]: r for r in res}
    assert by["flag"]["kind"] == "rate" and by["flag"]["z"] > 5
    assert abs(by["flag"]["group1"] - 0.5) < 0.07 and abs(by["flag"]["group0"] - 0.2) < 0.03
    assert by["shifted"]["auc"] > 0.7 and by["flat"]["p"] > 0.001
    assert res[-1]["feature"] == "flat"  # sorted by p-value


def test_link_share_counts_other_members_within_window():
    key = np.array([1, 1, 1, 2, 2])
    ts = np.array([0, 50, 500, 0, 30])
    member = np.array([True, False, True, False, True])
    # row 0 (member): only itself within 100 s on key 1 -> False; row 1: member row 0 is 50 s away
    assert stats.link_share(key, ts, member, 100).tolist() == [False, True, False, True, False]
    assert stats.link_share(key, ts, member, 1000).tolist() == [True, True, True, True, False]


def test_adversarial_sees_a_small_b_only_group_in_the_tail():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3000, 3))
    b = rng.normal(size=(3000, 3))
    b[:150, 0] = 8.0  # 5 % of B sits where A never goes
    res = stats.adversarial(a, b, folds=3, seeds=(0,))
    assert res["auc"] < 0.62
    assert res["tail_share_b"]["0.01"] > 0.95
    same = stats.adversarial(a, rng.normal(size=(3000, 3)), folds=3, seeds=(0,))
    assert abs(same["auc"] - 0.5) < 0.05
