import numpy as np
import pytest

from ringtrace.eventclean import clean_labels


def test_legit_in_fraud_event_takes_majority_class():
    y = np.array([0, 3, 3, 2, 0, 0, 0])
    ev = np.array([5, 5, 5, 5, -1, 9, 9])
    out = clean_labels(y, [ev])
    assert out.tolist() == [3, 3, 3, 2, 0, 0, 0]


def test_half_fraud_is_enough_and_ties_go_to_lower_class():
    y = np.array([0, 0, 4, 2])
    out = clean_labels(y, [np.array([1, 1, 1, 1])])
    assert out.tolist() == [2, 2, 4, 2]


def test_below_half_keeps_label():
    y = np.array([0, 0, 0, 5, 5])
    out = clean_labels(y, [np.zeros(5, dtype=np.int64)])
    assert out.tolist() == y.tolist()


def test_multi_module_highest_share_then_module_order():
    y = np.array([0, 2, 0, 0, 7, 7, 8, 8])
    a = np.array([1, 1, 1, -1, -1, -1, -1, -1])  # row 0: share 1/3 -> no change from a
    b = np.array([2, -1, -1, 2, 2, 2, -1, -1])  # row 0: share 2/4, class 7
    out = clean_labels(y, [a, b])
    assert out[0] == 7
    # equal shares in two modules: the first module wins
    y2 = np.array([0, 3, 6])
    m1 = np.array([1, 1, -1])
    m2 = np.array([4, -1, 4])
    assert clean_labels(y2, [m1, m2])[0] == 3
    assert clean_labels(y2, [m2, m1])[0] == 6


def test_mask_rows_neither_vote_nor_change():
    y = np.array([0, 3, 3, 0])
    ev = np.array([1, 1, 1, 1])
    mask = np.array([True, True, False, False])
    out = clean_labels(y, [ev], mask=mask)
    assert out.tolist() == [3, 3, 3, 0]
    out = clean_labels(np.array([0, 0, 3, 3]), [ev], mask=np.array([True, True, False, True]))
    assert out.tolist() == [0, 0, 3, 3]  # 1 fraud of 3 voters


def test_l2_minority_threshold():
    # 4 of type 2 plus one of type 5: 5 is 1/5 = 20 % -> flips
    y = np.array([2, 2, 2, 2, 5, 0])
    ev = np.zeros(6, dtype=np.int64)
    assert clean_labels(y, [ev], "clean").tolist() == [2, 2, 2, 2, 5, 2]
    assert clean_labels(y, [ev], "l2").tolist() == [2, 2, 2, 2, 2, 2]
    # majority only 3 rows -> no flip
    y = np.array([2, 2, 2, 5])
    assert clean_labels(y, [np.zeros(4, dtype=np.int64)], "l2").tolist() == [2, 2, 2, 5]
    # minority above 20 % (2 of 6) -> no flip
    y = np.array([2, 2, 2, 2, 5, 5])
    assert clean_labels(y, [np.zeros(6, dtype=np.int64)], "l2").tolist() == y.tolist()


def test_unknown_mode():
    with pytest.raises(ValueError):
        clean_labels(np.zeros(2, dtype=np.int64), [], "x")
