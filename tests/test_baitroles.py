"""Operator roles, the unanswered-bait rule and the first-time PAY rule on synthetic rows."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ringtrace import baitroles, rules
from ringtrace.classes import CODE
from ringtrace.evidence import requests as rq

PHISH = CODE["phishing_collect"]
H = 3600

# (period, ts, sender, receiver, to_person, txn_type, device, amount, new_payee, col_role,
#  col_shape, rng_shape, call)
ROWS = [
    # 0  account 100 receives a QR to a person in July: an operator act
    (7, 0, 1, 100, True, "QR", 1, 500.0, 1, 0, 0, 0, 7),
    # 1  account 101 receives a just-under collect in August
    (8, 0, 2, 101, True, "COLLECT", 2, 1999.0, 1, rq.REQUEST, rq.U, 0, PHISH),
    # 2  account 102 sends a recognised bait in September
    (9, 0, 102, 3, True, "PAY", 102, 2.0, 1, rq.BAIT, rq.N1, 0, PHISH),
    # 3  account 103 receives an answer in July
    (7, 0, 4, 103, True, "PAY", 4, 3000.0, 0, rq.ANSWER, rq.A1S, 0, 9),
    # 4  account 104 sends from a bot-pool phone in August
    (8, 0, 104, 20_000_001, False, "PAY", 420_500, 99.0, 0, 0, 0, 0, 2),
    # 5  account 105 pays a round amount inside a merchant ring in September
    (9, 0, 105, 20_000_002, False, "PAY", 105, 2500.0, 0, 0, 0, 1, 7),
    # 6  a lone unanswered Rs 10 of operator 100 in August, called legit: becomes phishing
    (8, 10 * H, 100, 5, True, "PAY", 100, 10.0, 1, rq.BAIT, rq.N1X, 0, 0),
    # 7  a lone Rs 10 of operator 100 in July: its only role is in July itself, untouched
    (7, 20 * H, 100, 6, True, "PAY", 100, 10.0, 1, rq.BAIT, rq.N1X, 0, 0),
    # 8  October lone Rs 10 of operator 104 (bot sender in August): becomes phishing
    (10, 0, 104, 7, True, "PAY", 104, 10.0, 1, rq.BAIT, rq.N1X, 0, 0),
    # 9  October lone Rs 10 of a plain account: untouched
    (10, 0, 300, 8, True, "PAY", 300, 10.0, 1, rq.BAIT, rq.N1X, 0, 0),
    # 10 October lone Rs 10 of operator 105, already called fraud: untouched
    (10, 0, 105, 9, True, "PAY", 105, 10.0, 1, rq.BAIT, rq.N1X, 0, 1),
    # 11 Rs 3.5 from operator 101 in July, no payment back: a band bait
    (7, 0, 101, 10, True, "PAY", 101, 3.5, 1, 0, 0, 0, 0),
    # 12 Rs 4 from operator 102 in July, paid back within 72 h (row 13): not a band bait
    (7, 100 * H, 102, 11, True, "PAY", 102, 4.0, 1, 0, 0, 0, 0),
    (7, 130 * H, 11, 102, True, "PAY", 11, 800.0, 0, 0, 0, 0, 0),
    # 14-19 first-time PAY candidates: only 14 qualifies (and 19 is called fraud already)
    (8, 0, 400, 12, True, "PAY", 400, 1001.0, 1, 0, 0, 0, 0),
    (8, 0, 401, 13, True, "PAY", 401, 1000.0, 1, 0, 0, 0, 0),
    (8, 0, 402, 14, True, "PAY", 402, 999.0, 1, 0, 0, 0, 0),
    (8, 0, 403, 15, True, "PAY", 403, 1001.5, 1, 0, 0, 0, 0),
    (8, 0, 404, 16, True, "PAY", 404, 1001.0, 0, 0, 0, 0, 0),
    (8, 0, 405, 17, True, "PAY", 405, 1003.0, 1, 0, 0, 0, 3),
    # 20 a first-time COLLECT and 21 a merchant PAY of the same kind: no
    (8, 0, 406, 18, True, "COLLECT", 406, 1003.0, 1, 0, 0, 0, 0),
    (8, 0, 407, 20_000_003, False, "PAY", 407, 1003.0, 1, 0, 0, 0, 0),
    # 22 a first-time PAY to operator 101 (collect receiver in August) in September, 23 in August
    (9, 0, 408, 101, True, "PAY", 408, 1234.0, 1, 0, 0, 0, 0),
    (8, 0, 409, 101, True, "PAY", 409, 1234.0, 1, 0, 0, 0, 0),
]
NAMES = (
    "period",
    "ts",
    "sender",
    "receiver",
    "to_person",
    "txn_type",
    "device",
    "amount",
    "new_payee",
    "col_role",
    "col_shape",
    "rng_shape",
)


def ledger() -> tuple[pl.DataFrame, np.ndarray]:
    cols = {k: [r[i] for r in ROWS] for i, k in enumerate(NAMES)}
    led = pl.DataFrame(cols).with_columns(
        pl.col("period").cast(pl.Int8),
        pl.col("col_role").cast(pl.Int8),
        pl.col("col_shape").cast(pl.Int8),
        pl.col("rng_shape").cast(pl.Int8),
        (pl.col("period") == 10).alias("is_test"),
    )
    return led, np.array([r[-1] for r in ROWS], dtype=np.int64)


def test_operator_roles_lists_every_act():
    led, _ = ledger()
    acts = baitroles.operator_roles(baitroles.columns(led))
    got = {(a, p, r) for a, p, r in acts.rows()}
    assert (100, 7, "qr_receiver") in got
    assert (101, 8, "collect_receiver") in got
    assert (102, 9, "bait_sender") in got
    assert (103, 7, "answer_receiver") in got
    assert (104, 8, "bot_sender") in got
    assert (105, 9, "round_payer") in got
    # a lone unanswered Rs 10 is not an act, and nothing else is
    assert len(got) == 6


def test_other_month_excludes_the_own_month_and_october_roles():
    acts = pl.DataFrame(
        {"acct": [1, 2], "period": pl.Series([7, 10], dtype=pl.Int8), "role": ["bot_sender"] * 2}
    )
    hit = baitroles.other_month(np.array([1, 1, 1, 2, 2, 3]), np.array([7, 8, 10, 8, 10, 8]), acts)
    assert hit.tolist() == [False, True, True, False, False, False]
    only_qr = baitroles.other_month(np.array([1]), np.array([8]), acts, roles=("qr_receiver",))
    assert only_qr.tolist() == [False]


def test_small_unanswered_default_and_band():
    led, _ = ledger()
    cols = baitroles.columns(led)
    assert np.flatnonzero(baitroles.small_unanswered(cols)).tolist() == [6, 7, 8, 9, 10]
    band = np.flatnonzero(baitroles.small_unanswered(cols, band=True)).tolist()
    assert band == [2, 6, 7, 8, 9, 10, 11]  # the unanswered N1 bait of row 2 is in the band
    assert 12 not in band  # paid back within 72 h


def test_bait_roles_rule_moves_only_legit_calls_of_operators():
    led, pred = ledger()
    out = rules.apply(["bait_roles"], pred, np.zeros((len(pred), 12)), led, log=lambda *_: None)
    changed = np.flatnonzero(out != pred).tolist()
    assert changed == [6, 8]
    assert (out[changed] == PHISH).all()
    band = rules.apply([{"name": "bait_roles", "band": True}], pred, None, led, log=lambda *_: None)
    assert np.flatnonzero(band != pred).tolist() == [6, 8, 11]


def test_bait_roles_respects_the_role_list():
    led, pred = ledger()
    out = baitroles.bait_roles(pred, None, led, log=lambda *_: None, roles=("qr_receiver",))
    assert np.flatnonzero(out != pred).tolist() == [6]


def test_first_pay_rule():
    led, pred = ledger()
    out = rules.apply(["first_pay"], pred, None, led, log=lambda *_: None)
    assert np.flatnonzero(out != pred).tolist() == [14, 22, 23]
    assert (out[[14, 22, 23]] == PHISH).all()
    assert out[19] == 3


def test_first_pay_to_an_operator():
    led, pred = ledger()
    rule = {"name": "first_pay", "operator_receiver": True}
    out = rules.apply([rule], pred, None, led, log=lambda *_: None)
    assert np.flatnonzero(out != pred).tolist() == [22]
    out = rules.apply([{**rule, "own_month": True}], pred, None, led, log=lambda *_: None)
    assert np.flatnonzero(out != pred).tolist() == [22, 23]


def test_bait_roles_own_month():
    led, pred = ledger()
    out = baitroles.bait_roles(pred, None, led, log=lambda *_: None, own_month=True)
    assert np.flatnonzero(out != pred).tolist() == [6, 7, 8]


def test_unknown_role_moves_nothing():
    led, pred = ledger()
    out = baitroles.bait_roles(pred, None, led, log=lambda *_: None, roles=("nobody",))
    assert np.array_equal(out, pred)


@pytest.mark.parametrize("months", [(7, 8, 9), (8, 9)])
def test_other_month_is_deterministic(months):
    acts = pl.DataFrame(
        {"acct": [5, 5], "period": pl.Series([8, 9], dtype=pl.Int8), "role": ["qr_receiver"] * 2}
    )
    q = (np.array([5, 5, 5]), np.array([8, 9, 10]))
    a = baitroles.other_month(*q, acts, months=months)
    b = baitroles.other_month(*q, acts.reverse(), months=months)
    assert np.array_equal(a, b)
    assert a.tolist() == [True, True, True]
