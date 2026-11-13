"""The event bootstrap finds a real gain and calls identical predictions a tie."""

from __future__ import annotations

import numpy as np

from ringtrace.compare import bootstrap


def test_bootstrap_gain_and_tie():
    rng = np.random.default_rng(0)
    n = 4000
    truth = np.where(rng.random(n) < 0.1, rng.integers(1, 12, n), 0)
    period = rng.integers(7, 10, n)
    groups = np.arange(n) // 4
    weak = np.where(rng.random(n) < 0.5, truth, 0)
    strong = np.where(rng.random(n) < 0.9, truth, 0)
    gain = bootstrap(groups, truth, period, weak, strong, draws=300)
    assert gain["verdict"] == "challenger" and gain["delta"] > 0.1
    same = bootstrap(groups, truth, period, weak, weak, draws=300)
    assert same["verdict"] == "tie" and same["delta"] == 0.0
