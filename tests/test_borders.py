"""Border posteriors on a synthetic fraud phase."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace import bordershift
from ringtrace.evidence import borders, refined
from ringtrace.evidence.phones import K


def phase(seed: int = 0, zone: int = 40):
    """Eleven zones of ``zone`` numbers; each class uses its numbers at its own density and 60 %
    of its phones carry its own anchor."""
    rng = np.random.default_rng(seed)
    true = np.repeat(np.arange(K), zone)
    dens = 1.0 / (1.0 + refined.GAP_MU)
    used = rng.random(len(true)) < dens[true]
    used[0] = True
    at = np.flatnonzero(used)
    counts = np.zeros((len(at), K + 1))
    for i, x in enumerate(at):
        counts[i, true[x] + 1 if rng.random() < 0.6 else 0] = 1
    return at + 1000, counts, 1000, 1000 + len(true) - 1, true[used]


def test_posteriors_are_distributions():
    nums, counts, lo, hi, _ = phase()
    got = borders.decode(nums, counts, lo, hi)
    assert got["post"].shape == (len(nums), K)
    assert np.allclose(got["post"].sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(got["cut"].sum(axis=1), 1.0, atol=1e-6)
    assert got["cut"].shape == (K - 1, hi - lo + 2)


def test_a_right_length_prior_places_more_phones():
    nums, counts, lo, hi, truth = phase()
    flat = borders.decode(nums, counts, lo, hi, None, None)["post"].argmax(axis=1)
    even = borders.decode(nums, counts, lo, hi, np.full(K, 1.0 / K), 0.3)["post"].argmax(axis=1)
    assert (even != truth).sum() <= (flat != truth).sum()
    assert (even != truth).mean() < 0.1


def test_zones_run_in_class_order():
    nums, counts, lo, hi, _ = phase(seed=3)
    mode = borders.decode(nums, counts, lo, hi)["post"].argmax(axis=1)
    assert (np.diff(mode) >= 0).all()


def test_gap_mass_counts_the_phones_below_the_border():
    at = np.array([0, 3, 4, 9])
    cut = np.zeros(12)
    cut[4] = 0.25  # zone starts at number 4: phones 0 and 3 lie below
    cut[5] = 0.75  # zone starts at number 5: phones 0, 3 and 4 lie below
    mass = bordershift.gap_mass(cut, at)
    assert mass.tolist() == [0.0, 0.0, 0.25, 0.75, 0.0]


def test_border_shift_rejects_an_unknown_border():
    with pytest.raises(ValueError):
        bordershift.border_shift(np.zeros(3), None, None, border=K - 1)
