"""Fresh bursts and capped rings on synthetic events."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace import burstsim, ringsize

LEGIT, MULE, BOT, QR, REFUND, COLLU = 0, 1, 2, 7, 8, 11


def burst_frame(events: list[dict]) -> tuple[pl.DataFrame, np.ndarray]:
    """One mule-shaped burst per dict: ``rows``, youngest ``sim``, ``phase`` (a phase phone)."""
    ev, sim, block, call = [], [], [], []
    for e, spec in enumerate(events):
        n = spec["rows"]
        ev += [e] * n
        sim += [spec["sim"]] + [500.0] * (n - 1)
        block += [2 if spec.get("phase") else 0] + [0] * (n - 1)
        call += [spec.get("first", MULE)] + [MULE] * (n - 1)
    n = len(ev)
    empty = {f"{p}_shape": pl.Series([0] * n, dtype=pl.Int8) for p in ("drn", "col", "rng")}
    none = {f"{p}_event": pl.Series([None] * n, dtype=pl.Int64) for p in ("drn", "col", "rng")}
    led = pl.DataFrame(
        {
            "is_test": [True] * n,
            "mf_shape": pl.Series([2] * n, dtype=pl.Int8),
            "mf_event": pl.Series(ev, dtype=pl.Int64),
            **empty,
            **none,
            "ctr_block": pl.Series(block, dtype=pl.Int8),
            "sim_age_hours": sim,
        }
    )
    return led, np.array(call, dtype=np.int64)


def test_fresh_bursts_need_size_a_fresh_sim_and_no_phase_phone():
    led, pred = burst_frame(
        [
            {"rows": 20, "sim": 10.0, "first": LEGIT},  # fires; the legit call stays
            {"rows": 20, "sim": 100.0},  # SIM too old
            {"rows": 20, "sim": 10.0, "phase": True},  # the counter names this one
            {"rows": 12, "sim": 10.0},  # too small at 20 rows
        ]
    )
    out = burstsim.fresh_bursts(pred, None, led, log=lambda *_: None)
    ev = led["mf_event"].to_numpy()
    assert out[ev == 0].tolist() == [LEGIT] + [BOT] * 19
    assert (out[ev != 0] == pred[ev != 0]).all()
    small = burstsim.fresh_bursts(pred, None, led, log=lambda *_: None, min_rows=12)
    assert (small[ev == 3] == BOT).all() and (small[ev == 1] == MULE).all()


def ring_frame() -> tuple[pl.DataFrame, np.ndarray]:
    # event 0: test, 8 rows; event 1: test, 7 rows; event 2: train, 8 rows; event 3: test, 8 rows
    # of another shape
    sizes = [8, 7, 8, 8]
    ev = np.repeat(np.arange(4), sizes)
    led = pl.DataFrame(
        {
            "is_test": np.isin(ev, (0, 1, 3)),
            "rng_shape": pl.Series(np.where(ev == 3, 3, 2), dtype=pl.Int8),
            "rng_event": pl.Series(ev, dtype=pl.Int64),
        }
    )
    pred = np.full(len(ev), REFUND, dtype=np.int64)
    pred[0] = QR
    return led, pred


def test_capped_rings_move_test_rings_at_the_cap_only():
    led, pred = ring_frame()
    out = ringsize.capped_rings(pred, None, led, log=lambda *_: None)
    ev = led["rng_event"].to_numpy()
    assert out[ev == 0].tolist() == [QR] + [COLLU] * 7
    assert (out[ev != 0] == REFUND).all()
    both = ringsize.capped_rings(pred, None, led, log=lambda *_: None, only_test=False)
    assert (both[ev == 2] == COLLU).all() and (both[ev == 1] == REFUND).all()


def test_fresh_bursts_upper_size_cut():
    led, pred = burst_frame([{"rows": 12, "sim": 10.0}, {"rows": 15, "sim": 10.0}])
    out = burstsim.fresh_bursts(pred, None, led, log=lambda *_: None, min_rows=12, max_rows=14)
    ev = led["mf_event"].to_numpy()
    assert (out[ev == 0] == BOT).all() and (out[ev == 1] == MULE).all()


def test_fresh_bursts_lower_sim_cut():
    led, pred = burst_frame([{"rows": 12, "sim": 48.0}, {"rows": 15, "sim": 44.0}])
    out = burstsim.fresh_bursts(
        pred, None, led, log=lambda *_: None, min_rows=12, min_sim_hours=45.0
    )
    ev = led["mf_event"].to_numpy()
    assert (out[ev == 0] == BOT).all() and (out[ev == 1] == MULE).all()
