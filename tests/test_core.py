"""Synthetic checks of the core numeric pieces: windows, scoring, decoding, silver labels."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ringtrace.decode import fixed_point
from ringtrace.scoring import macro_f1, per_class_f1
from ringtrace.windows import Timeline


def test_timeline_counts_and_gaps():
    key = np.array([1, 1, 1, 2, 2, 1])
    ts = np.array([0, 50, 400, 10, 20, 1000])
    tl = Timeline(key, ts)
    assert tl.count(100, 0).tolist() == [1, 2, 1, 1, 2, 1]
    assert tl.count(0, 400).tolist() == [3, 2, 1, 2, 1, 1]
    gp = tl.gap(-1)
    assert np.isnan(gp[0]) and gp[1] == 50 and gp[2] == 350 and gp[5] == 600
    gn = tl.gap(+1)
    assert gn[0] == 50 and np.isnan(gn[5]) and gn[3] == 10
    tot = tl.total(np.array([1.0, 2, 4, 8, 16, 32]), 1000, 0)
    assert tot.tolist() == [1, 3, 7, 8, 24, 39]


def test_macro_f1_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 12, 500)
    p = np.where(rng.random(500) < 0.6, y, rng.integers(0, 12, 500))
    ours = macro_f1(y, p)
    ref = sk.f1_score(y, p, average="macro", labels=list(range(12)), zero_division=0)
    assert ours == pytest.approx(ref)
    assert per_class_f1(y, y).min() == 1.0


def test_fixed_point_calls_confident_rare_rows():
    rng = np.random.default_rng(1)
    n = 5000
    proba = np.zeros((n, 12))
    proba[:, 0] = 0.98
    proba[:, 1:] = 0.02 / 11
    # 40 rows clearly class 3, 40 rows at 0.4 for class 5 (worth calling for F1), noise elsewhere
    proba[:40] = 0.0
    proba[:40, 3] = 0.95
    proba[:40, 0] = 0.05
    proba[40:80] = 0.0
    proba[40:80, 5] = 0.40
    proba[40:80, 0] = 0.60
    proba += rng.random(proba.shape) * 1e-4
    proba /= proba.sum(1, keepdims=True)
    pred = fixed_point(proba, fraud_scale=1.0)
    assert (pred[:40] == 3).all()
    assert (pred[40:80] == 5).all()  # argmax would say legit; F1 says call them
    assert (pred[80:] == 0).all()


def _toy_ledger(n_train: int, n_test: int) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    n = n_train + n_test
    y = np.zeros(n, dtype=np.int64)
    y[:300] = 3
    y[n_train:] = -1
    return pl.DataFrame({
        "is_test": np.r_[np.zeros(n_train, bool), np.ones(n_test, bool)],
        "y": y,
        "app_version": np.where(rng.random(n) < 0.3, 3.5, 4.5),
        "ip_risk_score": rng.random(n) * 0.5,
    })


def test_silver_relabels_confident_unreported_rows():
    from ringtrace import silver

    n_train, n_test = 4000, 100
    led = _toy_ledger(n_train, n_test)
    teacher = np.zeros((n_train, 12), dtype=np.float32)
    teacher[:, 0] = 0.999
    teacher[:, 3] = 0.001
    # rows 0-299 labelled chain and rows 300-339 labelled legit both look like chain events
    teacher[:340, 0] = 0.01
    teacher[:340, 3] = 0.99
    events = np.full(n_train + n_test, -1)
    events[:340] = np.arange(340) // 34  # ten events, each with 30 labelled + 4 unlabelled rows
    out = silver.build(led, teacher, events, log=lambda *_: None)
    lab = out["label"][:n_train]
    assert (lab[300:340] == 3).all()
    assert (lab[340:] == 0).all()
    assert (lab[:300] == 3).all()


def test_silver_cleans_only_the_masked_rows():
    from ringtrace import silver

    n_train, n_test = 4000, 100
    led = _toy_ledger(n_train, n_test)
    mask = np.zeros(n_train + n_test, dtype=bool)
    mask[:3000] = True  # e.g. July and August only
    teacher = np.zeros((3000, 12), dtype=np.float32)
    teacher[:, 0] = 0.999
    teacher[:, 3] = 0.001
    teacher[:340, 0] = 0.01
    teacher[:340, 3] = 0.99
    events = np.full(n_train + n_test, -1)
    events[:340] = np.arange(340) // 34
    out = silver.build(led, teacher, events, log=lambda *_: None, mask=mask)
    assert (out["label"][300:340] == 3).all()
    assert (out["label"][3000:] == -1).all()
