import numpy as np
import pytest

from ringtrace import repair as rp
from ringtrace.decode import fixed_point


def test_masks():
    assert rp.bit(1) == 1 and rp.bit(11) == 1 << 10 and rp.bit(0) == 0
    assert rp.span(2, 4) == 0b1110
    assert rp.span(1, 11) == rp.ALL
    assert list(rp.popcount(np.array([0, 1, 0b1011, rp.ALL]))) == [0, 1, 3, 11]
    assert list(rp.lowest_class(np.array([0, 0b100, 0b1100]))) == [0, 3, 3]


def test_shape_key_priority():
    tmpl = np.array([2, 0, 0, 0, 0])
    fam_a = np.array([1, 3, 0, 0, 0])
    fam_b = np.array([0, 4, 2, 0, 0])
    mark = np.array([1, 1, 1, 1, 0])
    key = rp.shape_key(tmpl, [fam_a, fam_b], [mark])
    assert list(key) == [102, 1003, 2002, 100_000, 0]


def test_shape_table_share_and_majority():
    key = np.array([5] * 100 + [6] * 10 + [7] * 3)
    lab = np.array([1] * 98 + [2] * 2 + [3] * 9 + [0] + [4] * 3)
    fit = np.ones(len(key), dtype=bool)
    t = rp.shape_table(key, lab, fit, min_share=0.05)
    assert t[5] == rp.bit(1)  # 2 % of class 2 is under the share
    assert t[6] == rp.bit(3)  # the legit row does not count
    assert t[7] == rp.bit(4)
    m = rp.shape_masks(np.array([5, 0, 99]), t)
    assert list(m) == [rp.bit(1), rp.ALL, rp.ALL]
    # unfit rows never vote
    t2 = rp.shape_table(key, lab, np.zeros(len(key), dtype=bool))
    assert t2 == {}


def test_group_and():
    g = np.array([3, 1, 3, 1, 7])
    m = np.array([0b110, 0b011, 0b011, 0b001, 0b100])
    assert list(rp.group_and(g, m)) == [0b010, 0b001, 0b010, 0b001, 0b100]


def test_longest_nondecreasing():
    c = np.array([1, 3, 2, 2, 5, 4, 4, 6])
    keep = rp.longest_nondecreasing(c)
    kept = c[keep]
    assert len(kept) == 6
    assert np.all(np.diff(kept) >= 0)
    assert np.array_equal(keep, rp.longest_nondecreasing(c))
    assert not rp.longest_nondecreasing(np.zeros(0, dtype=np.int64)).any()


def test_interval_masks():
    # phones in number order; 9 conflicts with the run 2, 2, 5 and is dropped
    a = np.array([0, 2, 0, 9, 2, 0, 5, 0])
    mask, kept = rp.interval_masks(a)
    assert list(kept) == [False, True, False, False, True, False, True, False]
    assert mask[0] == rp.span(1, 2)
    assert mask[1] == rp.bit(2)
    assert mask[2] == rp.bit(2)  # between two anchors of class 2
    assert mask[3] == rp.bit(2)  # the dropped anchor is read like an unanchored phone
    assert mask[5] == rp.span(2, 5)
    assert mask[7] == rp.span(5, 11)
    none, kept0 = rp.interval_masks(np.zeros(3, dtype=np.int64))
    assert not kept0.any() and np.all(none == rp.ALL)


def test_anchor_class_needs_one_class():
    phone = np.array([10, 10, 11, 11, 12])
    cls = np.array([3, 3, 3, 4, 0])
    assert list(rp.anchor_class(phone, cls)) == [3, 3, 0, 0, 0]


def test_phase_masks_intersects_shapes():
    phone = np.array([1, 2, 2, 3, 4])
    anchor = np.array([2, 0, 0, 0, 6])
    shape = np.array([rp.ALL, rp.span(1, 3), rp.span(3, 8), rp.ALL, rp.ALL])
    feas, st = rp.phase_masks(phone, anchor, shape)
    assert feas[0] == rp.bit(2)
    assert feas[1] == feas[2] == rp.bit(3)  # 2..6 within the device's shapes 3
    assert feas[3] == rp.span(2, 6)
    assert st["phones"] == 4 and st["anchored_phones"] == 2 and st["singleton_phones"] == 3


def test_retype_variants():
    calls = np.array([0, 5, 5, 7, 7, 2, 3])
    feas = np.array([rp.bit(4), rp.bit(4), rp.ALL, rp.span(3, 4), rp.ALL, 0, rp.bit(3)])
    group = np.array([0, 1, 1, 2, 2, 3, 4])
    value = np.zeros((7, 12))
    value[3, 4] = 1.0
    new, moved = rp.retype(calls, feas, group, value, "conservative")
    # legit stays, the event (rows 1-2) moves whole, two feasible classes wait, empty set waits
    assert list(new) == [0, 4, 4, 7, 7, 2, 3]
    assert list(moved) == [False, True, True, False, False, False, False]
    new_f, moved_f = rp.retype(calls, feas, group, value, "full")
    assert list(new_f) == [0, 4, 4, 4, 4, 2, 3]
    assert moved_f.sum() == 4
    with pytest.raises(ValueError):
        rp.retype(calls, feas, group, None, "full")
    with pytest.raises(ValueError):
        rp.retype(calls, feas, group, value, "bold")


def test_retype_same_event_different_class_moves_apart():
    calls = np.array([5, 6])
    feas = np.array([rp.bit(6), rp.bit(6)])
    new, moved = rp.retype(calls, feas, np.array([1, 1]))
    assert list(new) == [6, 6] and list(moved) == [True, False]


def test_decision_values_match_fixed_point():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(12) * 0.3, size=400)
    pred = fixed_point(p)
    v = rp.decision_values(p, pred)
    assert v.shape == (400, 12)
    # at the fixed point most rows already take their best marginal class
    assert (v.argmax(axis=1) == pred).mean() > 0.9


def test_break_even():
    truth = np.array([1, 1, 2, 0])
    pred = np.array([1, 0, 1, 0])
    # class 1: TP 1, true 2, called 2 -> F1 0.5, break-even 0.25
    assert rp.break_even(truth, pred, 1) == pytest.approx(0.25)
    assert rp.break_even(truth, pred, 9) == 0.0
