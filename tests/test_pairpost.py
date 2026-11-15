"""Pair-posterior template typing on synthetic mule-burst events."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ringtrace import pairpost, rules

LEGIT, MULE, BOT, CHAIN, SIM, ATO = range(6)
K = 11


def frame(events: list[list[dict]]) -> tuple[pl.DataFrame, np.ndarray]:
    """One mf2 (mule burst) event per list; a row is ``{"post": {class: p}, "call": c}`` on a
    fraud-phase phone, or ``{"call": c}`` on an ordinary phone. Phones are distinct unless a row
    names ``"phone"``."""
    rows = [(e, r) for e, ev in enumerate(events) for r in ev]
    n = len(rows)
    post = np.full((n, K), np.nan)
    block = np.zeros(n, dtype=np.int8)
    for i, (_, r) in enumerate(rows):
        if "post" in r:
            post[i] = 0.0
            for c, p in r["post"].items():
                post[i, c - 1] = p
            block[i] = 2
    known = block == 2
    seg = np.where(known, np.nan_to_num(post).argmax(axis=1).astype(float), np.nan)
    conf = np.where(known, np.nan_to_num(post).max(axis=1), np.nan)
    led = pl.DataFrame(
        {
            "is_test": [False] * n,
            "device": [r.get("phone", 500_000 + i) for i, (_, r) in enumerate(rows)],
            "mf_shape": pl.Series([r.get("shape", 2) for _, r in rows], dtype=pl.Int8),
            "mf_event": pl.Series([e for e, _ in rows], dtype=pl.Int64),
            **{f"{p}_shape": pl.Series([0] * n, dtype=pl.Int8) for p in ("drn", "col", "rng")},
            **{f"{p}_event": pl.Series([None] * n, dtype=pl.Int64) for p in ("drn", "col", "rng")},
            "ctr_block": pl.Series(block, dtype=pl.Int8),
            "ctr_seg": seg,
            "ctr_conf": conf,
            "ctr_dev_rows": pl.Series([1] * n, dtype=pl.Int32),
            **{f"ctr_p{k:02d}": post[:, k] for k in range(K)},
        }
    )
    return led, np.array([r["call"] for _, r in rows], dtype=np.int64)


def run(events, **params) -> np.ndarray:
    led, pred = frame(events)
    return pairpost.pair_templates(pred, None, led, log=lambda *_: None, **params)


def test_pair_share_conditions_on_the_pair():
    post = np.zeros((2, K))
    post[0, [MULE - 1, BOT - 1, CHAIN - 1]] = [0.05, 0.55, 0.40]
    post[1, CHAIN - 1] = 1.0
    mass, share = pairpost.pair_share(post, (MULE, BOT))
    assert mass == pytest.approx([0.60, 0.0])
    assert share == pytest.approx([0.55 / 0.60, 0.5])


def test_an_unconfident_mode_inside_the_pair_types_the_event():
    # bot 0.55 against layering 0.40: no anchor at confidence 0.8, but 0.92 inside the pair
    ev = [
        {"post": {MULE: 0.05, BOT: 0.55, CHAIN: 0.40}, "call": MULE},
        {"call": MULE},
        {"call": BOT},
    ]
    led, pred = frame([ev])
    old = rules.anchored_templates(pred, None, led, log=lambda *_: None)
    assert old.tolist() == [MULE, MULE, MULE]
    assert run([ev]).tolist() == [BOT, BOT, BOT]


def test_a_phone_mostly_outside_the_pair_has_no_vote():
    ev = [{"post": {MULE: 0.05, BOT: 0.30, CHAIN: 0.65}, "call": BOT}, {"call": BOT}]
    assert run([ev]).tolist() == [MULE, MULE]
    assert run([ev], min_mass=0.2).tolist() == [BOT, BOT]


def test_phones_of_one_event_add_their_log_odds_once_each():
    one = [{"post": {MULE: 0.3, BOT: 0.7}, "call": MULE}, {"call": MULE}]
    two = [
        {"post": {MULE: 0.3, BOT: 0.7}, "call": MULE},
        {"post": {MULE: 0.3, BOT: 0.7}, "call": MULE},
        {"call": MULE},
    ]
    same_phone = [
        {"post": {MULE: 0.3, BOT: 0.7}, "call": MULE, "phone": 7},
        {"post": {MULE: 0.3, BOT: 0.7}, "call": MULE, "phone": 7},
    ]
    assert run([one]).tolist() == [MULE, MULE]
    assert run([two]).tolist() == [BOT, BOT, BOT]
    assert run([same_phone]).tolist() == [MULE, MULE]


def test_a_confident_first_class_and_other_calls_stay():
    ev = [{"post": {MULE: 0.9, BOT: 0.1}, "call": BOT}, {"call": LEGIT}, {"call": SIM}]
    assert run([ev]).tolist() == [MULE, LEGIT, SIM]


def test_templates_limits_the_pair_reading():
    ev = [{"post": {MULE: 0.05, BOT: 0.55, CHAIN: 0.40}, "call": MULE}, {"call": MULE}]
    assert run([ev], templates=[2]).tolist() == [MULE, MULE]
    with pytest.raises(ValueError):
        run([ev], templates=[9])


def test_matches_the_anchored_rule_where_the_anchor_is_confident():
    ev = [{"post": {MULE: 0.1, BOT: 0.9}, "call": MULE}, {"call": MULE}, {"call": BOT}]
    led, pred = frame([ev])
    old = rules.anchored_templates(pred, None, led, log=lambda *_: None)
    assert run([ev]).tolist() == old.tolist() == [BOT, BOT, BOT]
