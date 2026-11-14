"""Month resampling and prior shifts of the label-shift test, on synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from augment_em import (  # noqa: E402
    apply_rules,
    em_types,
    known_shift,
    macro_f1,
    ratio_vector,
    resample_rows,
    unit_classes,
)

from ringtrace.classes import CODE, N_CLASSES  # noqa: E402
from ringtrace.scoring import macro_f1 as reference_macro  # noqa: E402


def _proba(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(N_CLASSES), size=n)
    p[:, 0] += 3.0
    return p / p.sum(axis=1, keepdims=True)


def test_unit_classes_take_the_majority_and_the_smaller_code_on_ties():
    units = np.array([5, 5, 5, 9, 9, -1, -2])
    truth = np.array([3, 3, 0, 2, 4, 0, 7])
    inv, ucls = unit_classes(units, truth)
    assert ucls[inv].tolist() == [3, 3, 3, 2, 2, 0, 7]


def test_resample_rows_repeats_whole_units_at_the_target_ratio():
    # 2000 single-row units of class 1 at ratio 1.5, 2000 of class 2 at ratio 0.5, legit at 1.
    units = np.arange(6000)
    truth = np.repeat([1, 2, 0], 2000)
    inv, ucls = unit_classes(units, truth)
    ratio = np.ones(N_CLASSES)
    ratio[1], ratio[2] = 1.5, 0.5
    idx = resample_rows(inv, ucls, ratio, seed=3)
    counts = np.bincount(truth[idx], minlength=3)
    assert counts[0] == 2000
    assert counts[1] == pytest.approx(3000, rel=0.05)
    assert counts[2] == pytest.approx(1000, rel=0.08)
    assert np.array_equal(idx, resample_rows(inv, ucls, ratio, seed=3))

    # An event's rows always travel together.
    ev_units = np.array([0, 0, 0, 1, 1, 2])
    ev_truth = np.array([1, 1, 1, 0, 0, 1])
    inv, ucls = unit_classes(ev_units, ev_truth)
    r = np.ones(N_CLASSES)
    r[1] = 2.0
    idx = resample_rows(inv, ucls, r, seed=0)
    assert np.bincount(idx, minlength=6).tolist() == [2, 2, 2, 1, 1, 2]


def test_known_shift_moves_odds_and_keeps_rows_normalised():
    p = _proba(50)
    ratio = ratio_vector({"mule_account": 2.0, "loan_app_extortion": 0.5})
    full = known_shift(p, ratio, types_only=False)
    assert np.allclose(full.sum(axis=1), 1.0)
    odds = (full[:, CODE["mule_account"]] / full[:, 0]) / (p[:, CODE["mule_account"]] / p[:, 0])
    assert np.allclose(odds, 2.0)
    types = known_shift(p, ratio, types_only=True)
    assert np.allclose(types[:, 0], p[:, 0])
    assert np.allclose(types.sum(axis=1), 1.0)
    assert np.allclose(known_shift(p, np.ones(N_CLASSES), True), p)


def test_em_types_keeps_the_fraud_mass_and_is_a_no_op_at_its_own_mean():
    p = _proba(400, seed=1)
    ref = p.mean(axis=0)
    q, adj = em_types(p, ref)
    assert np.allclose(adj[:, 0], p[:, 0])
    assert np.allclose(adj.sum(axis=1), 1.0)
    # Referenced to its own fraud-weighted type mix, the month needs no correction.
    w = p[:, 1:].sum(axis=1)
    own = np.concatenate([[0.0], (p[:, 1:] / w[:, None] * (w / w.sum())[:, None]).sum(axis=0)])
    q2, adj2 = em_types(p, own + np.array([1.0] + [0.0] * (N_CLASSES - 1)))
    assert np.allclose(adj2, p, atol=1e-6)


def test_apply_rules_moves_template_pairs_and_blank_qr_rows():
    rt = {
        "tmpl": np.array([0, 1, 1, 0, 0]),
        "pair0": np.array([0, 7, 0, 0, 0, 0, 0]),
        "pair1": np.array([0, 8, 0, 0, 0, 0, 0]),
        "target": np.array([0, 8, 8, 0, 0]),
        "qr": np.array([False, False, False, True, True]),
    }
    p = _proba(5, seed=2)
    p[3, 1:] = 0.0
    p[3, 7] = 0.5
    p[3, 0] = 0.5
    pred = np.array([0, 7, 3, 0, 5])
    out = apply_rules(pred, p, rt, np.arange(5))
    assert out[:3].tolist() == [0, 8, 3]  # row 2 calls 3, not in the template's pair
    assert out[3] == 7  # blank QR called legit takes its fraud argmax
    assert out[4] == 5  # blank QR already fraud stays


def test_macro_f1_matches_the_package_scorer():
    rng = np.random.default_rng(4)
    truth = rng.integers(0, N_CLASSES, size=3000)
    pred = np.where(rng.random(3000) < 0.6, truth, rng.integers(0, N_CLASSES, size=3000))
    assert macro_f1(truth, pred) == pytest.approx(reference_macro(truth, pred))
