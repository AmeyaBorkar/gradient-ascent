import numpy as np

from ringtrace import octshape


def test_keep_units_keeps_units_whole_and_is_seeded():
    units = np.repeat(np.arange(500), 3)
    a = octshape.keep_units(units, 101, 7)
    assert np.array_equal(a, octshape.keep_units(units, 101, 7))
    assert all(len(set(a[units == u])) == 1 for u in range(500))
    assert not np.array_equal(a, octshape.keep_units(units, 102, 7))
    assert not np.array_equal(a, octshape.keep_units(units, 101, 8))
    assert 0.40 < a.mean() < 0.60


def test_blank_rows_share():
    cand = np.arange(10_000)
    b = octshape.blank_rows(cand, 101, 7)
    assert 0.41 < len(b) / len(cand) < 0.47
    assert np.array_equal(b, octshape.blank_rows(cand, 101, 7))


def test_scale_amounts_only_touches_given_rows():
    amt = np.array([100.0, 1000.0, 2500.0])
    out = octshape.scale_amounts(amt, np.array([0, 2]))
    assert out[1] == 1000.0
    assert out[0] == 116.6 and out[2] == 2915.0
    assert amt[0] == 100.0


def test_event_and_ring_mix():
    ev = np.array([5, 5, 5, -1, 9, 12, 12])
    shape = np.array([2, 2, 2, 0, 1, 3, 3])
    size = np.array([3, 3, 3, 0, 1, 2, 2])
    mask = np.ones(7, dtype=bool)
    m = octshape.event_mix(ev, mask)
    assert m["rows_per_event"] == 2.0
    assert np.isclose(m["per_100k"], 1e5 * 3 / 7)
    r = octshape.ring_mix(ev, shape, size, mask)
    assert np.isclose(r["size1"], 1e5 / 7)
    assert np.isclose(r["size2_3"], 2e5 / 7)
    assert r["size9p"] == 0.0
    assert np.isclose(r["mean_size"], 2.0)
