"""The legit top-up: rows called legit by the rules and plain decode take a likely fraud class."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ringtrace import rules
from ringtrace.classes import CODE

QR, PHISH, REFUND = CODE["qr_swap"], CODE["phishing_collect"], CODE["refund_abuse"]

# call, plain decode, top fraud class, its probability
ROWS = [
    (0, 0, QR, 0.2),  # 0 moves to qr_swap
    (0, 0, QR, 0.1),  # 1 below the threshold
    (0, PHISH, QR, 0.3),  # 2 the plain decode calls it fraud: a rule made it legit, keep it
    (3, 0, QR, 0.3),  # 3 called fraud already
    (0, 0, PHISH, 0.2),  # 4 phishing: only with the phishing top-up
    (0, 0, REFUND, 0.3),  # 5 a class not topped up
    (0, 0, QR, 0.125),  # 6 exactly at the threshold
]


def _rows():
    proba = np.zeros((len(ROWS), 12))
    for i, (_, _, k, q) in enumerate(ROWS):
        proba[i, k] = q
        proba[i, 0] = 1.0 - q
    pred = np.array([r[0] for r in ROWS], dtype=np.int64)
    plain = np.array([r[1] for r in ROWS], dtype=np.int64)
    led = pl.DataFrame({"is_test": [False, True] * 3 + [True], "period": [7, 10] * 3 + [10]})
    return pred, proba, plain, led


def test_legit_topup_qr_and_phishing(monkeypatch):
    pred, proba, plain, led = _rows()
    monkeypatch.setattr(rules, "plain_calls", lambda p, led, fraud_scale: plain)
    qr = rules.legit_topup(
        pred, proba, led, log=lambda *_: None, classes={"qr_swap": 0.125}, fraud_scale=1.0
    )
    assert np.flatnonzero(qr != pred).tolist() == [0, 6]
    assert (qr[[0, 6]] == QR).all()
    both = rules.legit_topup(
        pred,
        proba,
        led,
        log=lambda *_: None,
        classes={"qr_swap": 0.125, "phishing_collect": 0.125},
        fraud_scale=1.0,
    )
    assert np.flatnonzero(both != pred).tolist() == [0, 4, 6]
    assert both[4] == PHISH


def test_legit_topup_reads_the_scaled_probability(monkeypatch):
    pred, proba, plain, led = _rows()
    monkeypatch.setattr(rules, "plain_calls", lambda p, led, fraud_scale: plain)
    # row 1: q = (0.1 / 0.5) / (0.9 + 0.2) = 0.18 >= 0.125 once the fraud column is scaled
    out = rules.legit_topup(
        pred, proba, led, log=lambda *_: None, classes={"qr_swap": 0.125}, fraud_scale=0.5
    )
    assert np.flatnonzero(out != pred).tolist() == [0, 1, 6]


def test_legit_topup_needs_fraud_classes():
    pred, proba, _, led = _rows()
    with pytest.raises(ValueError):
        rules.legit_topup(pred, proba, led, log=lambda *_: None)
    with pytest.raises(ValueError):
        rules.legit_topup(pred, proba, led, log=lambda *_: None, classes={"legit": 0.1})
    with pytest.raises(KeyError):
        rules.legit_topup(pred, proba, led, log=lambda *_: None, classes={"nonsense": 0.1})


def test_plain_calls_decodes_each_train_month_and_october_together(monkeypatch):
    import ringtrace.decode

    seen = []

    def fake(p, fraud_scale):
        seen.append(fraud_scale)
        return np.full(len(p), len(p))

    monkeypatch.setattr(ringtrace.decode, "fixed_point", fake)
    led = pl.DataFrame(
        {
            "is_test": [False, False, False, True, False, False, True, False],
            "period": [7, 7, 8, 10, 9, 9, 10, 9],
        }
    )
    out = rules.plain_calls(np.ones((8, 12)), led, fraud_scale=0.8)
    assert out.tolist() == [2, 2, 1, 2, 3, 3, 2, 3]
    assert seen == [0.8] * 4


def test_scaled_proba():
    q = rules.scaled_proba(np.array([[0.5, 0.25, 0.25]]), fraud_scale=0.5)
    assert np.allclose(q, 1 / 3)


def test_legit_topup_is_registered():
    assert "legit_topup" in rules.RULES
