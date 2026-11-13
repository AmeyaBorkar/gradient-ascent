"""The counter's teaching prior: HMM terms, the learned prior and decoding a thinned phase."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import norm
from test_phones import GAP_MEAN, synthetic

from ringtrace.evidence import phones

K = phones.K


def _paths(n: int, k: int):
    """Every monotone path over n phones that starts in segment 0 and moves by 0 or +1."""
    for steps in itertools.product((0, 1), repeat=n - 1):
        path = np.concatenate([[0], np.cumsum(steps)])
        if path[-1] < k:
            yield path


def test_zero_terms_leave_the_hmm_unchanged():
    rng = np.random.default_rng(0)
    ll = rng.normal(size=(40, K))
    zeros = np.zeros((40, K)), np.zeros(K)
    assert np.array_equal(phones.forward_backward(ll), phones.forward_backward(ll, *zeros))
    assert np.array_equal(phones.viterbi(ll), phones.viterbi(ll, *zeros))


def test_forward_backward_and_viterbi_match_enumeration():
    rng = np.random.default_rng(1)
    n, k = 7, 4
    ll = rng.normal(size=(n, k))
    enter = rng.normal(size=(n, k))
    end = rng.normal(size=k)
    marg = np.zeros((n, k))
    best, best_w = None, -np.inf
    for path in _paths(n, k):
        w = ll[np.arange(n), path].sum() + end[path[-1]]
        moves = np.flatnonzero(np.diff(path) == 1) + 1
        w += enter[moves, path[moves]].sum()
        marg[np.arange(n), path] += np.exp(w)
        if w > best_w:
            best, best_w = path, w
    marg /= marg.sum(axis=1, keepdims=True)
    assert np.allclose(phones.forward_backward(ll, enter, end), marg)
    assert np.array_equal(phones.viterbi(ll, enter, end), best)


def test_interval_logmass():
    a = np.array([-0.3, -0.01, 0.0, 0.02, 0.2])
    b = a + np.array([0.05, 0.02, 0.001, 0.1, 0.03])
    got = phones.interval_logmass(a, b, 0.01, 0.05)
    want = np.log(norm.cdf(b, 0.01, 0.05) - norm.cdf(a, 0.01, 0.05))
    assert np.allclose(got, want)
    # far tails stay finite and ordered (no cancellation to log 0)
    far = phones.interval_logmass(np.array([2.0, 3.0]), np.array([2.001, 3.001]), 0.0, 0.05)
    assert np.isfinite(far).all() and far[0] > far[1]
    far_low = phones.interval_logmass(np.array([-3.001]), np.array([-3.0]), 0.0, 0.05)
    assert np.allclose(far_low, far[1:])


def _phase_from_truth(nums: np.ndarray, seg: np.ndarray) -> phones.Phase:
    n = len(nums)
    post = np.eye(K)[seg]
    return phones.Phase(nums, post, seg, np.ones(n), np.full(n, np.nan), None)


def test_learn_prior_reads_cuts_and_occupancy():
    # segment k: 20 phones, every (k % 3 + 1)-th number used
    nums, seg, n = [], [], 1000
    for k in range(K):
        for _ in range(20):
            nums.append(n)
            seg.append(k)
            n += k % 3 + 1
    nums, seg = np.array(nums), np.array(seg)
    lo, edge = 1000, int(nums[-1])
    prior = phones.learn_prior(_phase_from_truth(nums, seg), lo, edge, spread=0.02, gap_weight=0.5)
    length = edge - lo + 1
    for k in range(1, K):
        mid = (nums[seg < k].max() + nums[seg >= k].min()) / 2
        assert np.isclose(prior.cuts[k - 1], (mid - lo) / length)
    occ = 1.0 / (np.arange(K) % 3 + 1)
    rel = prior.occupancy / prior.occupancy.mean()
    assert np.allclose(rel, occ / occ.mean(), rtol=0.1)
    assert prior.spread == 0.02 and prior.gap_weight == 0.5
    # a teaching fit that never reached the last two segments leaves those cuts open
    short = phones.learn_prior(_phase_from_truth(nums, np.minimum(seg, K - 3)), lo, edge)
    assert np.isnan(short.cuts[-2:]).all() and np.isfinite(short.cuts[:-2]).all()
    assert np.isfinite(short.occupancy).all()


def _long_phase(rng, per_seg: int):
    """One fraud phase: phones, anchor counts and the true segment."""
    probs = np.exp(phones.emission_table())
    nums, counts, truth, n = [], [], [], 0
    for k in range(K):
        for _ in range(per_seg):
            n += rng.geometric(1 / (1 + GAP_MEAN[k])) - 1
            row = np.zeros(K + 1)
            row[rng.choice(K + 1, p=probs[k])] += 1
            nums.append(n)
            counts.append(row)
            truth.append(k)
            n += 1
    return np.array(nums), np.array(counts), np.array(truth), n


def _decode(nums, counts, edge, prior=None):
    used = np.zeros(edge + 2, dtype=bool)
    used[nums] = True
    blocks = phones.Blocks(f0=0, used=used, base=0)
    return phones.decode_phase(nums, counts, blocks, 0, edge, prior=prior, edge=edge)


def test_a_full_phase_teaches_a_thinned_one():
    rng = np.random.default_rng(3)
    nums, counts, truth, end = _long_phase(rng, per_seg=60)
    edge = end - 1
    base, taught = [], []
    for seed in range(12):
        keep = np.random.default_rng(100 + seed).random(len(nums)) < 1 / 6
        prior = phones.learn_prior(_decode(nums[~keep], counts[~keep], edge), 0, edge, 0.01, 1.0)
        plain = _decode(nums[keep], counts[keep], edge)
        with_prior = _decode(nums[keep], counts[keep], edge, prior)
        base.append((plain.seg == truth[keep]).mean())
        taught.append((with_prior.seg == truth[keep]).mean())
        assert np.allclose(with_prior.post.sum(axis=1), 1.0)
        assert (np.diff(with_prior.seg) >= 0).all()
    assert np.mean(taught) > np.mean(base) + 0.02, (np.mean(base), np.mean(taught))


def test_teach_changes_only_later_splits():
    arr, _ = synthetic(seed=5)
    dev, split, anchor = arr[:, 0], arr[:, 1], arr[:, 2]
    off = phones.counter(dev, split, anchor)
    on = phones.counter(dev, split, anchor, teach={"spread": 0.01, "gap_weight": 1.0})
    tr = split == 0
    for c in off:
        assert np.array_equal(off[c][tr], on[c][tr], equal_nan=True), c
    fb = (on["block"] == 2) & ~tr
    post = np.stack([on[f"p{k:02d}"][fb] for k in range(K)], axis=1)
    assert np.allclose(post.sum(axis=1), 1.0)
    truth = arr[:, 3]
    assert (on["seg"][fb] == truth[fb]).mean() >= 0.9
