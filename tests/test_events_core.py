import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from events_core import (  # noqa: E402
    complete_by_link,
    evidence_share,
    f1_ceiling,
    link_cells,
    nearest_anchor,
    partial_events,
)


def test_partial_events_counts_whole_none_and_partial():
    event = np.array([0, 0, 0, 1, 1, 2, 2, 2, -1, -1])
    called = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
    rep = partial_events(event, called)
    assert rep == {
        "events": 3,
        "rows": 8,
        "whole": 1,
        "none": 1,
        "partial": 1,
        "legit_in_partial": 2,
    }


def test_partial_events_without_events():
    rep = partial_events(np.full(4, -1), np.ones(4, dtype=bool))
    assert rep["events"] == 0 and rep["partial"] == 0


def _brute(q_key, q_ts, a_key, a_ts, a_cls, direction):
    gap = np.full(len(q_key), np.inf)
    cls = np.full(len(q_key), -1)
    for i, (k, t) in enumerate(zip(q_key, q_ts, strict=True)):
        best = (np.inf, -1, np.inf)
        for kk, tt, cc in zip(a_key, a_ts, a_cls, strict=True):
            if kk != k:
                continue
            d = t - tt
            if direction == "backward" and d < 0:
                continue
            if direction == "forward" and d > 0:
                continue
            cand = (abs(d), tt, cc)
            if cand[0] < best[0] or (cand[0] == best[0] and tt < best[2]):
                best = (cand[0], cc, tt)
        gap[i], cls[i] = best[0], best[1]
    return gap, cls


@pytest.mark.parametrize("direction", ["backward", "forward", "both"])
def test_nearest_anchor_matches_brute_force(direction):
    rng = np.random.default_rng(3)
    a_key = rng.integers(0, 6, 40)
    a_ts = rng.integers(0, 1000, 40)
    a_cls = rng.integers(1, 12, 40)
    q_key = rng.integers(0, 8, 60)
    q_ts = rng.integers(0, 1000, 60)
    gap, cls = nearest_anchor(q_key, q_ts, a_key, a_ts, a_cls, direction)
    bgap, bcls = _brute(q_key, q_ts, a_key, a_ts, a_cls, direction)
    assert np.array_equal(gap, bgap)
    # classes agree wherever the nearest anchor is unique in time
    for i in range(len(q_key)):
        same = (a_key == q_key[i]) & (np.abs(q_ts[i] - a_ts) == bgap[i])
        if np.isfinite(bgap[i]) and len(np.unique(a_ts[same])) == 1 and same.sum() == 1:
            assert cls[i] == bcls[i]


def test_nearest_anchor_empty_and_bad_inputs():
    gap, cls = nearest_anchor(
        np.array([1]),
        np.array([5]),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
    )
    assert np.isinf(gap[0]) and cls[0] == -1
    with pytest.raises(ValueError):
        nearest_anchor(
            np.array([1]),
            np.array([5]),
            np.array([1]),
            np.array([5]),
            np.array([2]),
            direction="sideways",
        )
    with pytest.raises(ValueError):
        nearest_anchor(np.array([-1]), np.array([5]), np.array([1]), np.array([5]), np.array([2]))


def test_link_cells_precision():
    gap = np.array([10.0, 100.0, 5000.0, np.inf])
    acls = np.array([3, 3, 4, -1])
    label = np.array([3, 0, 4, 2])
    rep = link_cells(gap, acls, {"1m": 60, "1h": 3600, "2h": 7200}, label)
    assert rep["1m"] == {"linked": 1, "fraud": 1.0, "same": 1.0}
    assert rep["1h"]["linked"] == 2 and rep["1h"]["same"] == 0.5
    assert rep["2h"]["linked"] == 3 and rep["2h"]["fraud"] == pytest.approx(2 / 3)
    assert np.isnan(
        link_cells(np.array([np.inf]), np.array([-1]), {"w": 1}, np.array([0]))["w"]["fraud"]
    )


def test_f1_ceiling_counts_ordinary_rows_as_misses():
    # class 1: 2 generator rows, 2 ordinary rows (cleaned fraud, gen legit)
    gen = np.array([1, 1, 0, 0, 0, 2])
    cleaned = np.array([1, 1, 1, 1, 0, 2])
    rep = f1_ceiling(gen, cleaned, unreported=0.0, legit_f1=1.0)
    assert rep["per_class"][1]["f1"] == pytest.approx(2 * 2 / (2 * 2 + 2))
    assert rep["per_class"][2]["f1"] == 1.0
    assert rep["macro"] == pytest.approx((1.0 + 2 / 3 + 1.0) / 3)
    half = f1_ceiling(gen, cleaned, unreported=0.5, legit_f1=1.0)
    assert half["per_class"][1]["ordinary_rows"] == pytest.approx(4.0)


def test_evidence_share_per_period():
    mask = np.array([1, 1, 1, 0, 1], dtype=bool)
    evid = np.array([1, 0, 1, 1, 0], dtype=bool)
    period = np.array([7, 7, 8, 8, 8])
    rep = evidence_share(mask, evid, period)
    assert rep == {7: {"rows": 2, "evidence_share": 0.5}, 8: {"rows": 2, "evidence_share": 0.5}}


def test_complete_by_link_stays_inside_period_and_window():
    calls = np.array([3, 0, 0, 0, 5, 0])
    period = np.array([7, 7, 7, 8, 8, 8])
    key = np.array([1, 1, 1, 1, 2, 2])
    ts = np.array([100, 150, 5000, 160, 100, 130])
    ok = np.array([True, True, True, True, True, False])
    out = complete_by_link(calls, period, ok, key, key, ts, "both", 60)
    # row 1: same key, 50 s from the class-3 anchor -> 3; row 2: too far; row 3: other period
    # (no anchor with key 1 in period 8); row 5: not allowed by ``ok``
    assert out.tolist() == [3, 3, 0, 0, 5, 0]
