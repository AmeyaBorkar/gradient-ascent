import numpy as np

from ringtrace import aclean
from ringtrace.decode import fixed_point
from ringtrace.scoring import confusion, macro_f1


def _toy(seed=0, n=4000, k=12):
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.9, 0, rng.integers(1, k, n))
    pred = np.where(rng.random(n) < 0.8, y, rng.integers(0, k, n))
    return y, pred


def test_zero_hidden_is_plain_confusion():
    y, pred = _toy()
    cm = aclean.expected_confusion(y, pred, np.zeros(12))
    assert np.allclose(cm, confusion(y, pred))
    assert np.isclose(aclean.aclean_f1(y, pred, np.zeros(12)), macro_f1(y, pred))


def test_hidden_rows_move_from_legit_truth_and_keep_totals():
    y, pred = _toy(1)
    hidden = np.zeros(12)
    hidden[3] = 40.0
    cm = aclean.expected_confusion(y, pred, hidden)
    base = confusion(y, pred).astype(float)
    assert np.isclose(cm.sum(), base.sum())
    assert np.allclose(cm.sum(axis=0), base.sum(axis=0))  # calls do not change
    assert np.isclose(cm[3].sum() - base[3].sum(), 40.0)
    assert np.isclose(base[0].sum() - cm[0].sum(), 40.0)
    assert aclean.aclean_f1(y, pred, hidden) < macro_f1(y, pred)


def test_hidden_counts_share_the_total_by_ordinary_labels():
    cleaned = np.array([0, 1, 1, 2, 2, 2, 0, 1])
    structured = np.array([0, 0, 1, 0, 0, 2, 0, 0])
    period = np.array([7, 7, 7, 7, 8, 8, 8, 8])
    h = aclean.hidden_counts(cleaned, structured, period, total=10)
    assert np.isclose(sum(v.sum() for v in h.values()), 10)
    # ordinary rows: month 7 -> class 1 once, class 2 once; month 8 -> class 2 once, class 1 once
    assert np.allclose(h[7][[1, 2]], [2.5, 2.5])
    assert h[7][0] == 0


def test_vector_hidden_factor_matches_scalar():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.r_[30.0, np.full(11, 0.3)], size=3000)
    hf = np.r_[1.0, np.full(11, 2.0)]
    assert np.array_equal(fixed_point(p, hidden_factor=2.0), fixed_point(p, hidden_factor=hf))


def test_call_margin_is_relative_to_half_f1():
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.r_[20.0, np.full(11, 0.5)], size=2000)
    pred = fixed_point(p)
    m = aclean.call_margin(p, pred, 2)
    assert m.shape == (2000,)
    assert np.all(m >= 0)
    # the fixed point admits rows above the cut unless a better class competes
    assert (pred[m > 1.2] != 0).mean() > 0.9


def test_sweep_grid_and_rare_classes():
    g = aclean.sweep_grid()
    assert {"hidden_factor": 1.0, "fraud_scale": 0.875, "rare_only": False} in g
    assert len(g) == 15
    p = np.tile(np.r_[0.5, np.arange(1, 12) / 132.0], (5, 1))
    assert list(aclean.rare_classes(p)) == [1, 2, 3, 4]
