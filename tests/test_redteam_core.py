import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from redteam_core import cell_scan, gap_noise, per_100k  # noqa: E402


def test_gap_noise_identical_files_have_zero_error():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    p = rng.integers(0, 3, 500)
    r = gap_noise(y, p, [p.copy()], n_pub=150, draws=20, k=3)[0]
    assert r["full_gap"] == 0.0
    assert r["sd_gap_error"] == 0.0
    assert r["p_exceed"]["0.004"] == 0.0


def test_gap_noise_different_files_have_positive_error():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 2000)
    r = gap_noise(y, y.copy(), [rng.integers(0, 3, 2000)], n_pub=600, draws=50, k=3)[0]
    assert r["full_gap"] < 0
    assert r["sd_gap_error"] > 0


def test_cell_scan_flags_pure_cell_called_legit_in_october():
    n = 400
    f = {"a": np.repeat([0, 1], n // 2), "b": np.tile([0, 1], n // 2)}
    y = np.zeros(n, dtype=np.int64)
    y[f["a"] == 1] = 3  # cell a=1 is all fraud
    tr = np.arange(n) < 300
    te = ~tr
    legit = np.zeros(n, dtype=bool)
    legit[te & (f["a"] == 1)] = True  # October calls it legit, train never does
    cells = cell_scan(f, y, legit, tr, te, order=1, min_train=10)
    assert len(cells) == 1
    assert cells[0]["cell"] == {"a": 1}
    assert cells[0]["oct_called_legit"] == int((te & (f["a"] == 1)).sum())
    assert cells[0]["oct_expected_legit"] == 0.0


def test_per_100k():
    calls = np.array([0, 0, 1, 2])
    out = per_100k(calls, np.array([True, True, True, False]), k=3)
    assert np.allclose(out, [2e5 / 3, 1e5 / 3, 0])
