import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from fresh_core import anatomy, band, profile_shift  # noqa: E402

from ringtrace.scoring import per_class_f1  # noqa: E402


def test_band_clips_and_bins():
    b = band(np.array([-5.0, 0.0, 2.9, 3.0, 99.0]), [0, 3, 6])
    assert b.tolist() == [0, 0, 0, 1, 1]


def test_anatomy_pockets_sum_to_loss():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.choice(12, n, p=[0.6] + [0.4 / 11] * 11)
    pred = np.where(rng.random(n) < 0.6, y, rng.integers(0, 12, n))
    gen = np.where(rng.random(n) < 0.5, y, 0)
    anchor = rng.integers(0, 3, n)
    tmpl = rng.integers(0, 2, n)
    fb = rng.random(n) < 0.1
    rep = anatomy(y, pred, gen, anchor, tmpl, fb)
    f1 = per_class_f1(y, pred)
    for c in range(1, 12):
        assert abs(sum(rep["stake"][c].values()) - (1 - f1[c]) / 12) < 1e-12
    assert abs(rep["macro"] - f1.mean()) < 1e-12


def test_anatomy_counts_pockets():
    y = np.array([4, 4, 4, 0, 5])
    pred = np.array([0, 0, 5, 4, 5])
    gen = np.array([0, 4, 4, 0, 5])
    z = np.zeros(5, dtype=int)
    rep = anatomy(y, pred, gen, z, np.array([0, 2, 2, 0, 0]), z.astype(bool))
    s = rep["stake"][4]
    denom = 0 + 3 + 1  # TP 0, FN 3, FP 1
    assert s["a_ordinary"] == 1 / denom / 12
    assert s["b_evidence_legit"] == 1 / denom / 12
    assert s["c_evidence_wrongtype"] == 1 / denom / 12
    assert s["d_legit_called"] == 1 / denom / 12
    assert rep["wrongtype_pairs"][4] == {5: 1}
    assert rep["evidence_legit_by"][4] == {"tmpl2": 1}


def test_profile_shift_identical_is_zero():
    b = np.array([0, 1, 0, 1])
    y = np.array([1, 1, 1, 1])
    tr = np.array([True, True, False, False])
    rep = profile_shift(b, y, y, tr, ~tr, 2)
    assert rep[1]["tv_oct_oof"] == 0.0
