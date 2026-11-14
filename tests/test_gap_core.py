import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from gap_core import (  # noqa: E402
    catch_needed,
    implied_r,
    ordinary_counts,
    shifted_f1,
    template_majority,
)

from ringtrace.scoring import per_class_f1  # noqa: E402


def _month(seed: int = 0, n: int = 20000):
    rng = np.random.default_rng(seed)
    truth = np.where(rng.random(n) < 0.05, rng.integers(1, 12, n), 0)
    gen = np.where(rng.random(n) < 0.4, truth, 0)  # 60 % of fraud rows are "ordinary"
    pred = np.where(gen > 0, gen, 0)
    flip = rng.random(n) < 0.002
    pred[flip] = rng.integers(0, 12, flip.sum())
    return truth, pred, gen


def test_no_shift_is_plain_f1():
    truth, pred, gen = _month()
    assert np.allclose(shifted_f1(truth, pred, gen), per_class_f1(truth, pred))


def test_catching_all_ordinary_rows_equals_relabelled_prediction():
    truth, pred, gen = _month(1)
    ordinary = (truth > 0) & (gen == 0)
    fixed = pred.copy()
    fixed[ordinary] = truth[ordinary]
    # ordinary rows are all called legit here, so catching them exactly reproduces the fixed file
    assert (pred[ordinary] == 0).mean() > 0.99
    keep = ~ordinary | (pred == 0)
    t, p, g = truth[keep], pred[keep], gen[keep]
    f = fixed[keep]
    assert np.allclose(shifted_f1(t, p, g, q=1.0, prec=1.0), per_class_f1(t, f))


def test_more_ordinary_rows_lower_the_score_and_catching_raises_it():
    truth, pred, gen = _month(2)
    base = shifted_f1(truth, pred, gen).mean()
    assert shifted_f1(truth, pred, gen, r=1.3).mean() < base
    assert shifted_f1(truth, pred, gen, q=0.2).mean() > base
    # low-precision catching of rare classes loses
    assert shifted_f1(truth, pred, gen, q=0.2, prec=0.02).mean() < base


def test_catch_needed_hits_target_and_handles_unreachable():
    truth, pred, gen = _month(3)
    base = shifted_f1(truth, pred, gen).mean()
    q = catch_needed(truth, pred, gen, base + 0.02)
    assert 0 < q < 1
    assert shifted_f1(truth, pred, gen, q=q).mean() >= base + 0.02 - 1e-9
    assert catch_needed(truth, pred, gen, base - 0.01) == 0.0
    assert np.isnan(catch_needed(truth, pred, gen, 1.01))


def test_bad_arguments_fail_loudly():
    truth, pred, gen = _month(4)
    with pytest.raises(ValueError):
        shifted_f1(truth, pred, gen, q=1.5)
    with pytest.raises(ValueError):
        shifted_f1(truth, pred, gen, prec=0.0)


def test_ordinary_counts_and_template_majority():
    truth = np.array([0, 1, 1, 2, 2, 2])
    gen = np.array([0, 1, 0, 0, 0, 2])
    assert ordinary_counts(truth, gen)[:3].tolist() == [0, 1, 2]
    typist = np.array([4, 4, 5, 4, 0, 5])
    tmpl = np.array([2, 2, 2, 2, 2, 0])
    out = template_majority(typist, tmpl, {2: (4, 5)})
    assert out.tolist() == [4, 4, 4, 4, 0, 5]


def test_implied_r_inverts_the_shift():
    truth, pred, gen = _month(5)
    score = shifted_f1(truth, pred, gen, r=1.2).mean()
    assert abs(implied_r(truth, pred, gen, score) - 1.2) < 1e-6
    with pytest.raises(ValueError):
        implied_r(truth, pred, gen, 0.999)
