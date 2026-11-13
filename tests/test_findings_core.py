import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from core import (  # noqa: E402
    disagreement,
    fraud_pass_share,
    headroom,
    one_row_gain,
    slice_scores,
)

from ringtrace.scoring import macro_f1  # noqa: E402


def _data(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.choice(12, n, p=[0.8] + [0.2 / 11] * 11)
    a = np.where(rng.random(n) < 0.7, y, rng.integers(0, 12, n))
    b = a.copy()
    b[:30] = (b[:30] + 1) % 12
    return y, a, b


def test_slice_scores_deterministic_and_complementary():
    y, a, b = _data()
    s1 = slice_scores(y, [a, b], 0.3, 20, seed=3)
    s2 = slice_scores(y, [a, b], 0.3, 20, seed=3)
    assert np.array_equal(s1["public"], s2["public"])
    # identical files give identical slice scores
    s = slice_scores(y, [a, a], 0.3, 5)
    assert np.allclose(s["public"][:, 0], s["public"][:, 1])
    # the slice mean sits near the full score
    assert abs(s1["public"][:, 0].mean() - macro_f1(y, a)) < 0.03


def test_one_row_gain_matches_direct():
    n_true = np.full(12, 100.0)
    n_pred = np.full(12, 80.0)
    f1 = np.full(12, 0.5)
    g = one_row_gain(n_true, n_pred, f1)
    tp = 0.5 * 180 / 2
    assert np.allclose(g, (2 * (tp + 1) / 181 - 0.5) / 12)


def test_headroom_ceiling_is_gen_labels():
    clean = np.array([0, 1, 1, 2, 2, 0])
    gen = np.array([0, 1, 0, 2, 2, 0])
    h = headroom(clean, gen, clean, k=3)
    assert np.allclose(h["now"], 1.0)
    assert h["ordinary_share"][1] == 0.5 and h["ordinary_share"][2] == 0.0
    assert h["ceiling"][1] < 1.0


def test_disagreement_counts():
    ref = np.array([0, 0, 1, 2, 2])
    other = np.array([0, 1, 1, 0, 1])
    truth = np.array([0, 1, 1, 2, 0])
    d = disagreement(ref, other, truth, k=3)
    assert d["rows"] == 3
    assert d["truth_other"] == 1 and d["truth_ref"] == 1 and d["truth_neither"] == 1
    assert sum(c["rows"] for c in d["cells"]) == 3


def test_fraud_pass_share():
    rows = np.r_[np.ones(618, bool), np.zeros(382, bool)]
    assert abs(fraud_pass_share(rows)["share"] - 1.0) < 1e-9
    rows = np.r_[np.ones(150, bool), np.zeros(850, bool)]
    assert abs(fraud_pass_share(rows)["share"]) < 1e-9
    assert fraud_pass_share(np.zeros(0, bool))["rows"] == 0
