"""Signature masks and the signature rescue rule on a synthetic frame."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ringtrace import rules, signatures
from ringtrace.evidence import anchors

N = 6


def frame() -> pl.DataFrame:
    cols = {
        "txn_type": ["PAY", "QR", "QR", "PAY", "PAY", "QR"],
        "receiver_type": ["PERSON", "MERCHANT", "PERSON", "PERSON", "MERCHANT", "MERCHANT"],
        "qr_vpa_mismatch": [0.0, None, None, None, None, 0.0],
        "amount": [10.0, 50.0, 60.0, 150_000.0, 20.0, 30.0],
        "is_test": [False, True, True, False, True, True],
    }
    for m in signatures.MARKS:
        cols[f"mrk_{m}"] = [False] * N
    cols["mrk_bal110"] = [False, False, False, False, True, False]
    for p in signatures.FAMILY_SHAPES:
        cols[f"{p}_shape"] = [0] * N
        cols[f"{p}_event"] = [-1] * N
    cols["drn_shape"] = [0, 0, 0, 0, 1, 0]
    cols["drn_event"] = [-1, -1, -1, -1, 4, -1]
    cols["ctr_block"] = [0, 0, 0, 0, 2, 0]
    cols["ctr_seg"] = [None, None, None, None, 3.0, None]
    cols["ctr_conf"] = [None, None, None, None, 0.95, None]
    return pl.DataFrame(cols)


def test_raw_signatures():
    cat = signatures.catalogue(frame())
    assert cat["mark:nonqr_check"].tolist() == [True, False, False, False, False, False]
    assert cat["mark:qr_merchant_blank"].tolist() == [False, True, False, False, False, False]
    assert cat["big_p2p"].tolist() == [False, False, False, True, False, False]
    assert cat["drn:shape1"][4] and cat["drn:any"].sum() == 1
    assert cat["tmpl:drain1"][4] and cat["ctr:fraud_block_sure"][4]
    assert cat["and:mark:bal110&ctr:fraud_block"].tolist() == [False] * 4 + [True, False]


def test_mask_conjunction_and_unknown():
    led = frame()
    assert signatures.mask(led, "mark:bal110&drn:any").tolist() == [False] * 4 + [True, False]
    union = signatures.mask(led, "mark:nonqr_check|drn:any")
    assert union.tolist() == [True, False, False, False, True, False]
    with pytest.raises(KeyError):
        signatures.mask(led, "no:such")


def test_signature_rescue_only_flips_legit_calls():
    led = frame()
    pred = np.array([0, 0, 0, 0, 0, 3])
    proba = np.full((N, 12), 0.01)
    proba[:, 0] = 0.9
    proba[:, anchors.SIM] = 0.05
    out = rules.apply(
        [{"name": "signature_rescue", "signature": "drn:any"}], pred, proba, led, log=lambda m: None
    )
    assert out.tolist() == [0, 0, 0, 0, anchors.SIM, 3]
    out = rules.apply(
        [{"name": "signature_rescue", "signature": "mark:nonqr_check"}],
        pred,
        proba,
        led,
        log=lambda m: None,
    )
    assert out.tolist() == [anchors.SIM, 0, 0, 0, 0, 3]


def test_event_completion_fills_mostly_fraud_events():
    # mule events (mf shape 2): event 0 has 3 of 4 rows called fraud, event 4 has 1 of 3;
    # event 7 is split across train and October and must not join
    led = pl.DataFrame(
        {
            "mf_event": [0, 0, 0, 0, 4, 4, 4, 7, 7],
            "mf_shape": [2] * 9,
            "is_test": [False] * 8 + [True],
        }
    )
    pred = np.array([1, 1, 2, 0, 1, 0, 0, 1, 0])
    out = rules.apply(
        [{"name": "event_completion", "family": "mule", "q": 0.67}],
        pred,
        np.zeros((9, 12)),
        led,
        log=lambda m: None,
    )
    assert out.tolist() == [1, 1, 2, 1, 1, 0, 0, 1, 0]
    hit, _ = rules.completion_targets(pred, led, "chain", 0.5)
    assert not hit.any()
