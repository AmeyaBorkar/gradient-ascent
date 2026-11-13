"""Template and counter rules on synthetic template events."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace import rules
from ringtrace.evidence import anchors

LEGIT, MULE, BOT, CHAIN, SIM, ATO, PHISH, QR, REFUND, CASHB = range(10)
NAN = float("nan")

# one row per tuple: (template family, shape, event, ctr_block, ctr_seg, ctr_conf, call, is_test)
ROWS = [
    # 0-2  mf1 event 0, no anchor: pair calls take layering, the legit call stays
    ("mf", 1, 0, 0, NAN, NAN, MULE, False),
    ("mf", 1, 0, 0, NAN, NAN, CHAIN, False),
    ("mf", 1, 0, 0, NAN, NAN, LEGIT, False),
    # 3-5  mf1 event 3, row 3 on a fraud-block phone in the mule segment: the event goes mule
    ("mf", 1, 3, 2, 0.0, 0.95, CHAIN, False),
    ("mf", 1, 3, 0, NAN, NAN, CHAIN, False),
    ("mf", 1, 3, 0, NAN, NAN, MULE, False),
    # 6-7  drain1 event 6 anchored on the takeover segment (test split)
    ("drn", 1, 6, 2, 4.0, 0.85, SIM, True),
    ("drn", 1, 6, 0, NAN, NAN, SIM, True),
    # 8-9  drain1 event 8: segment in the pair but not confident -> template class
    ("drn", 1, 8, 2, 4.0, 0.5, ATO, False),
    ("drn", 1, 8, 0, NAN, NAN, ATO, False),
    # 10-11 mf1 event 10: segment outside the pair is no anchor; an off-pair call is untouched
    ("mf", 1, 10, 2, 5.0, 0.99, MULE, False),
    ("mf", 1, 10, 0, NAN, NAN, REFUND, False),
    # 12-13 col7 event 12, no anchor: the minority call cashback wins
    ("col", 7, 12, 0, NAN, NAN, PHISH, True),
    ("col", 7, 12, 0, NAN, NAN, CASHB, True),
    # 14 col1, called cashback -> phishing
    ("col", 1, 14, 0, NAN, NAN, CASHB, False),
    # 15 rng1 anchored on refund segment
    ("rng", 1, 15, 2, 7.0, 0.9, QR, False),
    # 16-18 mf2/mf3 burst event 16, called bot -> mule
    ("mf", 2, 16, 0, NAN, NAN, BOT, False),
    ("mf", 3, 16, 0, NAN, NAN, BOT, False),
    ("mf", 2, 16, 0, NAN, NAN, MULE, False),
    # 19 not in any template (mf4 fan-in): never touched
    ("mf", 4, 19, 2, 0.0, 0.99, CHAIN, False),
    # 20-23 counter rows: legit call in the fraud block, trusted segment, untrusted segment, legit
    # block one-row phone called fraud
    (None, 0, -1, 2, 2.0, 0.5, LEGIT, False),
    (None, 0, -1, 2, 0.0, 0.95, QR, True),
    (None, 0, -1, 2, 8.0, 0.99, PHISH, False),
    (None, 0, -1, 3, NAN, NAN, MULE, True),
]


def frame() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    n = len(ROWS)
    cols: dict[str, list] = {}
    for p in ("drn", "mf", "col", "rng"):
        cols[f"{p}_shape"] = [s if f == p else 0 for f, s, *_ in ROWS]
        cols[f"{p}_event"] = [e if f == p else None for f, _, e, *_ in ROWS]
    led = pl.DataFrame(
        {
            "is_test": [r[7] for r in ROWS],
            **{
                k: pl.Series(v, dtype=pl.Int8 if k.endswith("shape") else pl.Int64)
                for k, v in cols.items()
            },
            "ctr_block": pl.Series([r[3] for r in ROWS], dtype=pl.Int8),
            "ctr_seg": [r[4] for r in ROWS],
            "ctr_conf": [r[5] for r in ROWS],
            "ctr_dev_rows": pl.Series([1] * n, dtype=pl.Int32),
        }
    )
    pred = np.array([r[6] for r in ROWS], dtype=np.int64)
    proba = np.full((n, 12), 0.01)
    proba[:, 0] = 0.5
    proba[:, CHAIN] = 0.2  # best fraud class of every row
    return led, pred, proba


def test_templates():
    led, _, _ = frame()
    tmpl, ev = anchors.templates(led)
    assert tmpl.tolist()[:20] == [1] * 6 + [2] * 4 + [1] * 2 + [4] * 2 + [5, 6, 3, 3, 3, 0]
    assert ev[3] == 3 and ev[19] == -1


def test_anchored_templates():
    led, pred, proba = frame()
    out = rules.apply(["anchored_templates"], pred, proba, led, log=lambda m: None)
    want = pred.copy()
    want[[0, 1]] = CHAIN  # no anchor -> template class; legit row 2 untouched
    want[[3, 4, 5]] = MULE  # anchor keeps mule and propagates it
    want[[6, 7]] = ATO  # takeover anchor propagates through the drain event
    want[[8, 9]] = SIM  # conf 0.5 is no anchor
    want[10] = CHAIN  # phishing segment is no anchor for the chain/mule pair
    want[[12, 13]] = CASHB
    want[14] = PHISH
    want[15] = REFUND
    want[[16, 17, 18]] = MULE
    assert out.tolist() == want.tolist()


def test_plain_templates():
    led, pred, proba = frame()
    out = rules.apply(["plain_templates"], pred, proba, led, log=lambda m: None)
    assert out[[0, 1, 3, 4, 5, 10]].tolist() == [CHAIN] * 6
    assert out[[6, 7, 8, 9]].tolist() == [SIM] * 4
    assert out[15] == QR and out[2] == LEGIT and out[11] == REFUND and out[19] == CHAIN


def test_counter_and_legit_block():
    led, pred, proba = frame()
    lines: list[str] = []
    out = rules.apply(
        [{"name": "counter", "min_conf": 0.9}, "legit_block"], pred, proba, led, log=lines.append
    )
    assert out[20] == CHAIN  # called legit in the fraud block -> model's best fraud class
    assert out[21] == MULE  # confident trusted segment sets the type
    assert out[22] == PHISH  # cashback segment is not trusted: model's type kept
    assert out[23] == LEGIT  # one-row legit-block phone
    assert out[3] == MULE and out[19] == MULE  # confident mule segment in the fraud block
    assert out[10] == PHISH  # confident phishing segment
    assert out[6] == SIM and out[8] == ATO  # takeover segment below min_conf: calls kept
    assert out[15] == QR  # refund segment is not trusted
    assert "[rules] legit_block: changed train 0 / test 1 rows" in lines
    assert "[rules] counter: changed train 4 / test 1 rows" in lines


def test_nan_never_triggers():
    led, pred, proba = frame()
    led = led.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("ctr_seg"),
        pl.lit(None, dtype=pl.Float64).alias("ctr_conf"),
    )
    out = rules.apply(["anchored_templates"], pred, proba, led, log=lambda m: None)
    plain = rules.apply(["plain_templates"], pred, proba, led, log=lambda m: None)
    assert out.tolist() == plain.tolist()
