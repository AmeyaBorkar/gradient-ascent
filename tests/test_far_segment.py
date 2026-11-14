"""The far-segment rule: confident fraud-phase rows whose fraud call is far from the segment."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace import rules
from ringtrace.classes import CODE

QR, COLLUSION, MULE = CODE["qr_swap"], CODE["merchant_collusion"], CODE["mule_account"]

# block, segment class, confidence, call
ROWS = [
    (2, COLLUSION, 0.95, QR),  # 0 far: takes merchant_collusion
    (2, COLLUSION, 0.95, CODE["loan_app_extortion"]),  # 1 a neighbour in label-list order
    (2, COLLUSION, 0.95, CODE["refund_abuse"]),  # 2 the collusion zone is written as refund
    (2, COLLUSION, 0.85, QR),  # 3 not confident enough
    (2, COLLUSION, 0.95, 0),  # 4 called legit: not this rule's business
    (3, COLLUSION, 0.95, QR),  # 5 a legit-block phone
    (2, 0, 0.95, QR),  # 6 no segment
    (2, MULE, 0.95, CODE["layering_chain"]),  # 7 the mule zone is written as layering
    (2, MULE, 0.95, CODE["account_takeover"]),  # 8 far: takes mule_account
    (2, CODE["fake_cashback"], 0.95, CODE["phishing_collect"]),  # 9 cashback written as phishing
]


def _patch(monkeypatch):
    block, seg, conf, call = (np.array([r[i] for r in ROWS]) for i in range(4))
    monkeypatch.setattr(
        rules,
        "_counter",
        lambda led: {
            "block": block,
            "seg_class": seg,
            "conf": conf.astype(float),
            "dev_rows": np.ones(len(ROWS)),
        },
    )
    led = pl.DataFrame({"is_test": [True] * len(ROWS)})
    return led, call.astype(np.int64)


def test_far_segment_moves_only_far_confident_fraud_calls(monkeypatch):
    led, pred = _patch(monkeypatch)
    out = rules.far_segment(pred, None, led, log=lambda *_: None)
    assert np.flatnonzero(out != pred).tolist() == [0, 8]
    assert out[0] == COLLUSION and out[8] == MULE


def test_far_segment_parameters(monkeypatch):
    led, pred = _patch(monkeypatch)
    loose = rules.far_segment(pred, None, led, log=lambda *_: None, min_conf=0.8)
    assert np.flatnonzero(loose != pred).tolist() == [0, 3, 8]
    # with a higher mark weight only a zone's main writing fits: refund in the collusion zone,
    # layering in the mule zone and phishing in the cashback zone become far
    strict = rules.far_segment(pred, None, led, log=lambda *_: None, min_weight=0.45)
    assert np.flatnonzero(strict != pred).tolist() == [0, 2, 7, 8, 9]


def test_segment_fits_table():
    fits = rules.segment_fits()
    assert fits[0].all()
    assert np.flatnonzero(fits[QR]).tolist() == [6, 7, 8]
    assert np.flatnonzero(fits[COLLUSION]).tolist() == [8, 10, 11]
    assert np.flatnonzero(fits[MULE]).tolist() == [0, 1, 2, 3]


def test_far_segment_is_registered_and_deterministic(monkeypatch):
    led, pred = _patch(monkeypatch)
    a = rules.apply(["far_segment"], pred, None, led, log=lambda *_: None)
    b = rules.apply(["far_segment"], pred, None, led, log=lambda *_: None)
    assert "far_segment" in rules.RULES
    assert np.array_equal(a, b)
