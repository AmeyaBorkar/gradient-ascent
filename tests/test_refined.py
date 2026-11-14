"""The refined device counter: allocation geometry, on synthetic phases only."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import phones, refined


def test_allocation_evidence_separates_dense_sim_from_sparse_takeover():
    from ringtrace.evidence.refined import ATO, SIM, allocation_ll

    # 10 dense two-row phones (no gaps) then 10 sparse three-row phones (gaps of 3)
    nums = np.r_[np.arange(100, 110), 110 + 4 * np.arange(1, 11)]
    counts = np.zeros((20, phones.K + 1))
    counts[:10, 0] = 2
    counts[10:, 0] = 3
    ll = allocation_ll(nums, counts)
    assert np.isfinite(ll).all()
    assert (ll[1:10, SIM] > ll[1:10, ATO]).all()  # dense phones read as sim_swap
    assert (ll[10:, ATO] > ll[10:, SIM]).all()  # sparse phones read as account_takeover


def test_allocation_evidence_is_finite_for_unseen_row_counts():
    from ringtrace.evidence.refined import allocation_ll

    nums = np.array([5, 9, 10])
    counts = np.zeros((3, phones.K + 1))
    counts[:, 0] = [7, 1, 12]  # 5+ rows never seen in training
    ll = allocation_ll(nums, counts)
    assert np.isfinite(ll).all()


def test_build_matches_phones_columns_and_dtypes():
    rng = np.random.default_rng(0)
    n = 40
    device = rng.integers(1, 5000, n)
    split = rng.integers(0, 2, n)
    anchor = rng.integers(0, phones.K + 1, n)
    led = pl.DataFrame({"device": device, "is_test": split.astype(bool)})
    champ = phones.build(led, anchor=anchor)
    got = refined.build(led, anchor=anchor)
    assert got.columns == champ.columns
    assert got.schema == champ.schema
    assert got.height == champ.height == n


def _density_drop_phase():
    """Two phones of each segment, then eight dense and eight sparse phones of anchor class 5.

    Class 5 is what both sim_swap and account_takeover emit. A run of unused numbers sits between
    the dense phones and the sparse ones.
    """
    nums: list[int] = []
    anchor: list[int] = []
    n = 5000

    def add(count: int, cls: int, gap: int) -> None:
        nonlocal n
        for _ in range(count):
            nums.append(n)
            anchor.append(cls)
            n += 1 + gap

    for seg in range(3):
        add(2, seg + 1, 0)
    add(8, 5, 0)
    n += 8
    add(8, 5, 5)
    for seg in range(5, phones.K):
        add(2, seg + 1, 0)
    numbers = np.array(nums, dtype=np.int64)
    counts = np.zeros((len(numbers), phones.K + 1))
    counts[np.arange(len(numbers)), np.array(anchor, dtype=np.int64)] = 1.0
    lo, hi = int(numbers[0]), int(numbers[-1])
    used = np.zeros(hi - lo + 1, dtype=bool)
    used[numbers - lo] = True
    blocks = phones.Blocks(f0=lo, used=used, base=lo)
    return numbers, counts, blocks, lo, hi


def test_decoders_differ_on_a_dense_to_sparse_drop():
    nums, counts, blocks, lo, hi = _density_drop_phase()
    champ = phones.decode_phase(nums, counts, blocks, lo, hi)
    got = refined.decode_phase(nums, counts, blocks, lo, hi)
    same_seg = np.array_equal(champ.seg, got.seg)
    same_post = np.allclose(champ.post, got.post)
    assert not (same_seg and same_post)
