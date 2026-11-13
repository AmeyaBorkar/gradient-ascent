"""The squeeze candidates on synthetic frames: event types, anchor rescue, rank blend, greedy."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace import rules, squeeze

LEGIT, MULE, BOT, CHAIN, SIM, ATO = range(6)
EV = ("bot", "drn", "mf", "col", "rng", "rep")


def _frame(events: dict[str, list], n: int, **extra) -> pl.DataFrame:
    cols = {f"{p}_event": pl.Series(events.get(p, [None] * n), dtype=pl.Int64) for p in EV}
    cols.update({f"{p}_shape": pl.Series([0] * n, dtype=pl.Int8) for p in EV})
    cols.update(extra)
    return pl.DataFrame({"is_test": [False] * n, **cols})


def test_label_free_events_priority():
    led = _frame({"bot": [0, None, None], "mf": [5, 5, None]}, 3)
    ev = rules.label_free_events(led)
    assert ev[0] == 10_000_000  # bursts first
    assert ev[1] == 5 + 30_000_000
    assert ev[2] == -1


def test_event_types_unifies_fraud_calls_only():
    led = _frame({"drn": [0, 0, 0, 0, None]}, 5)
    pred = np.array([SIM, ATO, ATO, LEGIT, ATO])
    proba = np.full((5, 12), 0.01)
    proba[0, SIM] = 0.9
    proba[1, ATO] = 0.3
    proba[2, ATO] = 0.3
    out = rules.event_types(pred, proba, led, log=lambda *_: None)
    # summed sim 0.92 beats summed takeover 0.62: all three fraud calls go sim
    assert out.tolist() == [SIM, SIM, SIM, LEGIT, ATO]


def test_anchor_rescue_only_turns_legit_calls():
    n = 4
    led = _frame({}, n).with_columns(
        pl.Series("drn_shape", [3, 3, 0, 0], dtype=pl.Int8),
        *(
            pl.Series(c, [0] * n, dtype=pl.Int8)
            for c in ("mrk_p2p_category", "mrk_qr_p2p_nocat", "mrk_step025", "mrk_step035")
        ),
        pl.Series("mrk_bal102", [0, 0, 1, 0], dtype=pl.Int8),
    )
    pred = np.array([LEGIT, SIM, LEGIT, LEGIT])
    out = rules.anchor_rescue(pred, np.zeros((n, 12)), led)
    assert out.tolist() == [ATO, SIM, MULE, LEGIT]


def test_rank_mean_keeps_values_and_order():
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(12), size=50)
    period = np.repeat([7, 8], 25)
    out = squeeze.rank_mean([a, a], [0.5, 0.5], period)
    assert np.allclose(out.sum(1), 1.0)
    # identical members: ranks map back onto the same values
    assert np.allclose(out, a / a.sum(1, keepdims=True))


def test_greedy_with_replacement_and_stop():
    target = np.array([0.5, 0.25, 0.25])

    def evaluate(c):
        w = np.array(c) / sum(c)
        return -float(np.abs(w - target).sum())

    counts, path = squeeze.greedy(evaluate, 3, steps=10)
    assert counts.tolist() == [2, 1, 1]
    assert all(b > a for a, b in zip(path, path[1:], strict=False))


def test_pair_counts():
    a = np.array([0, 0, 4, 5, 5])
    b = np.array([0, 4, 4, 4, 4])
    assert squeeze.pair_counts(a, b) == {"legit -> sim_swap": 1, "account_takeover -> sim_swap": 2}
