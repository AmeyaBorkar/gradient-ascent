import numpy as np

from ringtrace import gate
from ringtrace.classes import CLASSES


def test_change_pairs_counts_only_changed_pairs():
    a = np.array([0, 0, 1, 2, 2, 2])
    b = np.array([0, 1, 1, 0, 0, 3])
    pairs = gate.change_pairs(a, b)
    assert pairs == {
        f"{CLASSES[0]}->{CLASSES[1]}": 1,
        f"{CLASSES[2]}->{CLASSES[0]}": 2,
        f"{CLASSES[2]}->{CLASSES[3]}": 1,
    }
    assert sum(pairs.values()) == int((a != b).sum())


def test_mean_prob_weights_runs(tmp_path):
    rng = np.random.default_rng(1)
    arrs = {}
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        arrs[name] = rng.random((5, 12)).astype(np.float32)
        np.save(tmp_path / name / "oof.npy", arrs[name])
    got = gate.mean_prob(gate.parse_blend("a*3+b"), tmp_path, "oof")
    want = 0.75 * arrs["a"].astype(np.float64) + 0.25 * arrs["b"].astype(np.float64)
    assert np.allclose(got, want)


def test_month_bootstrap_self_is_zero():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 400)
    groups = rng.integers(0, 50, 400)
    r = gate.month_bootstrap(groups, y, y, y, draws=50)
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0
