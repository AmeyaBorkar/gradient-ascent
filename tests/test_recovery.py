"""The recovery stage on synthetic arrays: it only retypes legit calls, finds a separable hidden
class out of fold, and is deterministic."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.recovery import role_flags, stage


def _world(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 6000
    period = np.repeat(np.array([7, 8, 9, 10]), n // 4)
    is_test = period == 10
    target = np.zeros(n, dtype=np.int64)
    hidden = rng.random(n) < 0.04
    target[hidden] = 6  # a fraud class the calls missed
    target[is_test] = -1
    proba = np.full((n, 12), 0.01, dtype=np.float32)
    proba[:, 0] = 0.89
    flag = (hidden | (rng.random(n) < 0.002)).astype(np.float32)  # the role signal
    fields = np.column_stack([rng.normal(size=n), flag]).astype(np.float32)
    pred = np.zeros(n, dtype=np.int64)
    pred[rng.random(n) < 0.01] = 3  # some existing fraud calls

    def flags_for(rows, months):
        return np.column_stack([flag[rows], np.zeros(len(rows))]).astype(np.float32)

    return pred, proba, target, period, is_test, fields, flags_for, hidden


def test_stage_retypes_only_legit_calls_and_finds_the_hidden_class():
    pred, proba, target, period, is_test, fields, flags_for, hidden = _world()
    out = stage(
        pred,
        proba,
        target,
        period,
        is_test,
        fields,
        flags_for,
        rounds=60,
        threads=1,
        log=lambda *_: None,
    )
    changed = out != pred
    assert np.all(pred[changed] == 0)
    assert np.all(out[pred != 0] == pred[pred != 0])
    train = ~is_test
    hit = changed & train
    assert hit.sum() > 20
    assert (out[hit] == 6).mean() > 0.9
    assert hidden[hit].mean() > 0.7


def test_stage_is_deterministic():
    args = _world(1)[:7]
    a = stage(*args, rounds=40, threads=2, log=lambda *_: None)
    b = stage(*args, rounds=40, threads=2, log=lambda *_: None)
    assert np.array_equal(a, b)


def test_role_flags_respect_months_and_person_receivers():
    roles = pl.DataFrame(
        {
            "acct": [1, 2, 2],
            "period": [7, 8, 9],
            "role": ["qr_receiver", "bait_sender", "qr_receiver"],
        }
    )
    names = ["bait_sender", "qr_receiver"]
    sender = np.array([1, 2, 3])
    receiver = np.array([2, 1, 1])
    to_person = np.array([True, True, False])
    f = role_flags(sender, receiver, to_person, roles, names, months=[8, 9])
    # sender columns: bait_sender, qr_receiver; receiver columns: bait_sender, qr_receiver
    assert f.tolist() == [[0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]
