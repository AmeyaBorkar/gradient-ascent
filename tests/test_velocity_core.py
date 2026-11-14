import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from velocity_core import (  # noqa: E402
    mule_edge,
    per_100k,
    rescaled_f1,
    sessions,
    size_test,
    thin_keep,
)


def test_sessions_split_on_gap_key_and_split():
    key = np.array([1, 1, 1, 1, 2, 2, 1])
    split = np.array([0, 0, 0, 0, 0, 0, 1])
    ts = np.array([0, 100, 250, 1000, 0, 50, 120])
    row = np.arange(7)
    sid, size = sessions(key, split, ts, row, gap=200)
    assert size.tolist() == [3, 3, 3, 1, 2, 2, 1]
    assert sid[0] == sid[1] == sid[2] != sid[3]
    assert sid[4] == sid[5]
    assert sid[6] not in (sid[0], sid[3])


def test_sessions_do_not_depend_on_input_order():
    rng = np.random.default_rng(1)
    n = 500
    key = rng.integers(0, 20, n)
    split = rng.integers(0, 2, n)
    ts = rng.integers(0, 5000, n)
    row = np.arange(n)
    _, size = sessions(key, split, ts, row, gap=120)
    perm = rng.permutation(n)
    _, size_p = sessions(key[perm], split[perm], ts[perm], row[perm], gap=120)
    assert np.array_equal(size[perm], size_p)


def test_sessions_empty():
    sid, size = sessions(np.array([]), np.array([]), np.array([]), np.array([]), gap=60)
    assert len(sid) == 0 and len(size) == 0


def test_size_test_tells_whole_from_thinned():
    rng = np.random.default_rng(0)
    train = rng.integers(20, 61, 300)
    whole = rng.integers(20, 61, 12)
    thinned = rng.binomial(rng.integers(20, 61, 12), 0.5)
    w = size_test(whole, train)
    t = size_test(thinned, train)
    assert w["mwu_p_whole"] > 0.01 and w["mwu_p_thinned"] < 0.001
    assert t["mwu_p_thinned"] > 0.01 and t["mwu_p_whole"] < 0.001
    assert w["p_all_ge_min_whole"] > w["p_all_ge_min_thinned"]


def test_per_100k_and_thin_keep():
    month = np.array([7] * 1000 + [8] * 500)
    mask = np.zeros(1500, dtype=bool)
    mask[:10] = True
    mask[1000:1005] = True
    assert per_100k(mask, month, (7, 8)) == [1000.0, 1000.0]
    protect = np.zeros(10_000, dtype=bool)
    protect[:100] = True
    keep = thin_keep(10_000, protect, 0.5, seed=3)
    assert keep[:100].all()
    assert 0.45 < keep[100:].mean() < 0.55
    assert np.array_equal(keep, thin_keep(10_000, protect, 0.5, seed=3))


def _synthetic(n, rng):
    lab = rng.choice(12, n, p=[0.7] + [0.3 / 11] * 11)
    gen = np.where(rng.random(n) < 0.4, lab, 0)
    calls = np.where(gen > 0, gen, np.where(rng.random(n) < 0.02, rng.integers(1, 12, n), 0))
    zord = (gen == 0) & (lab > 0) & (rng.random(n) < 0.05)
    calls = np.where(zord, lab, calls)
    seg = np.where(zord, lab, 0)
    return lab, gen, calls, zord, seg


def test_rescaled_f1_on_a_copy_is_the_pooled_f1():
    rng = np.random.default_rng(0)
    lab, gen, calls, zord, seg = _synthetic(20_000, rng)
    r = rescaled_f1(lab, gen, calls, zord, seg, calls, zord, seg)
    direct = f1_score(lab, calls, labels=list(range(12)), average=None, zero_division=0)
    assert np.allclose(r["f1_train"], direct)
    assert np.allclose(r["f1_test"], direct)
    assert np.allclose(r["a"][1:], 1.0) and np.allclose(r["b"][1:], 1.0)


def test_rescaled_f1_tracks_a_month_with_half_the_structured_rows():
    rng = np.random.default_rng(2)
    lab, gen, calls, zord, seg = _synthetic(40_000, rng)
    # the test month drops half of class 2's structured rows and keeps the rest of the mix
    drop = (gen == 2) & (rng.random(len(lab)) < 0.5)
    keep = ~drop
    r = rescaled_f1(lab, gen, calls, zord, seg, calls[keep], zord[keep], seg[keep])
    # calls outside the dropped rows (false alarms) stay, so the ratio sits a little above 0.5
    assert 0.5 < r["a"][2] < 0.65
    actual = f1_score(lab[keep], calls[keep], labels=list(range(12)), average=None)
    # a is read from all non-zone calls, false alarms included, so the estimate is approximate
    assert abs(r["f1_test"][2] - actual[2]) < 0.05
    assert r["f1_test"][2] < r["f1_train"][2]


def test_mule_edge_moves_only_bot_calls_on_mule_zone_rows():
    calls = np.array([2, 2, 2, 1, 0, 2])
    zord = np.array([True, True, False, True, True, True])
    seg = np.array([1, 2, 1, 1, 1, 0])
    new, hit = mule_edge(calls, zord, seg)
    assert hit.tolist() == [True, False, False, False, False, False]
    assert new.tolist() == [1, 2, 2, 1, 0, 2]
    assert calls.tolist() == [2, 2, 2, 1, 0, 2]
