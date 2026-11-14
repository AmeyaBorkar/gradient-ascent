"""The counter-news rule, and the champion fraud-phase segmentation it reads."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace import counternews, rules
from ringtrace.evidence import phones, refined
from ringtrace.tables import HOME_DEVICES

# previous segment class, refined class, refined posterior, call, is_test
ROWS = [
    (4, 5, 0.70, 4, False),  # 0 threshold: 0.70 moves
    (4, 5, 0.699, 4, False),  # 1 0.699 stays
    (5, 6, 0.90, 0, False),  # 2 legit call stays
    (0, 1, 0.99, 0, False),  # 3 off the fraud phase, called legit
    (0, 1, 0.99, 1, True),  # 4 off the fraud phase, call is not the old class
    (4, 6, 0.99, 4, False),  # 5 two classes away
    (4, 2, 0.99, 4, False),  # 6 two classes the other way
    (4, 5, 0.99, 6, False),  # 7 call already differs from the previous segment
    (4, 5, 0.99, 5, False),  # 8 call is already the refined class
    (8, 7, 0.70, 8, True),  # 9 one class down, on the threshold, test split
    (3, 3, 0.99, 3, False),  # 10 refined class equals the previous class
]


def _patch(monkeypatch):
    old, new, post, call, is_test = (np.array([r[i] for r in ROWS]) for i in range(5))
    monkeypatch.setattr(
        counternews,
        "news",
        lambda led: {
            "old_class": old.astype(np.int64),
            "new_class": new.astype(np.int64),
            "new_post": post.astype(np.float64),
        },
    )
    led = pl.DataFrame({"is_test": is_test.astype(bool)})
    return led, call.astype(np.int64)


def test_counter_news_moves_only_an_adjacent_confident_old_call(monkeypatch):
    led, pred = _patch(monkeypatch)
    lines: list[str] = []
    out = counternews.counter_news(pred, None, led, log=lines.append)
    assert out.tolist() == [5, 4, 0, 0, 1, 4, 4, 6, 5, 7, 3]
    assert lines == ["  counter news: train 1 / test 1"]


def test_counter_news_respects_min_post(monkeypatch):
    led, pred = _patch(monkeypatch)
    loose = counternews.counter_news(pred, None, led, log=lambda *_: None, min_post=0.699)
    assert np.flatnonzero(loose != pred).tolist() == [0, 1, 9]
    strict = counternews.counter_news(pred, None, led, log=lambda *_: None, min_post=0.701)
    assert np.flatnonzero(strict != pred).tolist() == []


def test_counter_news_is_registered_and_deterministic(monkeypatch):
    led, pred = _patch(monkeypatch)
    a = rules.apply(["counter_news"], pred, None, led, log=lambda *_: None)
    b = rules.apply(
        [{"name": "counter_news", "min_post": 0.7}], pred, None, led, log=lambda *_: None
    )
    assert "counter_news" in rules.RULES
    assert np.array_equal(a, b)
    assert np.array_equal(a, counternews.counter_news(pred, None, led, log=lambda *_: None))
    held = pred.copy()
    counternews.counter_news(pred, None, led, log=lambda *_: None)
    assert np.array_equal(pred, held)


def _previous_segmentation(nums, counts, blocks, lo, hi) -> np.ndarray:
    """The segment path of ``decode_phase`` before allocation geometry (revision before d4f31a3).

    That function returned a phase (posteriors, side, cut) after this path. The path does not
    depend on them.
    """
    loge = phones.emission_table()
    ll = counts @ loge.T
    path = phones.viterbi(ll)
    zone = np.isin(path, (phones.SIM, phones.ATO))
    if zone.sum() >= 2 * phones.CUT_MIN_SIDE:
        idx = np.flatnonzero(zone)
        j = phones.change_point(nums[idx])
        if j is not None:
            peak = int(nums[idx[j - 1]]) + 1
            cut = phones.snap_cut(peak, blocks, lo, hi)
            ll = ll.copy()
            ll[nums < cut, phones.ATO] -= phones.SIDE_PENALTY
            ll[nums >= cut, phones.SIM] -= phones.SIDE_PENALTY
            path = phones.viterbi(ll)
    return path


def _density_drop_phase():
    """Two phones of each segment, except eight dense then eight sparse phones of anchor class 5.

    Class 5 is what both sim_swap and account_takeover emit. A run of unused numbers sits between
    the dense phones and the sparse ones, so the old change point has a cut to snap to. Anchors
    alone put the dense phones on account_takeover; the side penalty moves them to sim_swap.
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


# Dense phones (indices 6..13) land on sim_swap, sparse phones (14..21) on account_takeover.
_DROP_PATH = (
    [0, 0, 0, 1, 2, 2]
    + [phones.SIM] * 8
    + [phones.ATO] * 8
    + [5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
)


def test_legacy_decode_reproduces_the_previous_change_point_segmentation():
    nums, counts, blocks, lo, hi = _density_drop_phase()
    got = counternews.legacy_decode(nums, counts, blocks, lo, hi)
    assert got.tolist() == _DROP_PATH
    assert np.array_equal(got, _previous_segmentation(nums, counts, blocks, lo, hi))
    anchors_only = phones.viterbi(counts @ phones.emission_table().T)
    assert anchors_only[6:14].tolist() != [phones.SIM] * 8
    assert got[6:14].tolist() == [phones.SIM] * 8
    assert got[14:22].tolist() == [phones.ATO] * 8


def test_news_labels_only_the_fraud_phase(monkeypatch):
    rows: list[tuple[int, int]] = [(1, 0)] * 2
    rows += [(HOME_DEVICES + 10, 0)] * 4  # multi-row bot, below the first single-row phone
    n = HOME_DEVICES + 3000
    fraud_at: list[int] = []
    for _ in range(12):
        fraud_at.append(len(rows))
        rows.append((n, 0))
        n += 2
    n += 20
    rows += [(n + i, 0) for i in range(40)]  # dense legit block, after a sparse hole
    device = np.array([d for d, _ in rows], dtype=np.int64)
    split = np.array([s for _, s in rows], dtype=np.int64)
    anchor = np.zeros(len(rows), dtype=np.int64)
    anchor[fraud_at] = 5
    led = pl.DataFrame({"device": device, "is_test": split.astype(bool)})
    monkeypatch.setattr(counternews.anchors, "row_anchor", lambda led: anchor)

    blocks = phones.find_blocks(device, split)
    block = phones.block_codes(device, split, blocks)
    phase = np.flatnonzero(block == 2)
    assert phase.tolist() == fraud_at
    lo, _edge, hi = blocks.ranges[0]
    nums, pinv = np.unique(device[phase], return_inverse=True)
    counts = np.zeros((len(nums), phones.K + 1))
    np.add.at(counts, (pinv, anchor[phase]), 1.0)
    old = counternews.legacy_decode(nums, counts, blocks, lo, hi)
    post = refined.decode_phase(nums, counts, blocks, lo, hi).post
    choice = post.argmax(axis=1)

    got = counternews.news(led)
    off = block != 2
    assert np.array_equal(got["old_class"][off], np.zeros(off.sum(), dtype=np.int64))
    assert np.array_equal(got["new_class"][off], np.zeros(off.sum(), dtype=np.int64))
    assert np.array_equal(got["new_post"][off], np.zeros(off.sum()))
    assert np.array_equal(got["old_class"][phase], old[pinv] + 1)
    assert np.array_equal(got["new_class"][phase], choice[pinv] + 1)
    assert np.allclose(got["new_post"][phase], post[np.arange(len(choice)), choice][pinv])
    assert (got["old_class"][phase] >= 1).all()
    assert (got["new_class"][phase] >= 1).all()
