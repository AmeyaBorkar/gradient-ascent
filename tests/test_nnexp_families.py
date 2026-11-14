"""The screening driver's own-fit variants: names, entity inputs and the unseen-id guard."""

import numpy as np
import polars as pl

from ringtrace import nn_entity, nnexp


def test_tag_of_prefixes_the_job_and_names_single_seeds():
    assert nnexp.tag_of("selftrain", (0,)) == "selftrain-sd0"
    assert nnexp.tag_of("selftrain", (0,), "wildrun") == "wildrun-selftrain-sd0"
    assert nnexp.tag_of("base", (0, 1, 2), "wildrun") == "wildrun-base"


def test_unseen_ids_go_to_the_unknown_row():
    ids_tr = np.array([[1, 10], [2, 10], [3, 11]], dtype=np.int64)
    ids = np.array([[1, 12], [9, 10], [3, 11]], dtype=np.int64)
    out = nn_entity.unseen_to_unknown(ids_tr, ids, buckets=64)
    seen = nn_entity._hash_columns(ids, 64)
    assert out[0, 0] == seen[0, 0] and out[0, 1] == 64
    assert out[1, 0] == 64 and out[1, 1] == seen[1, 1]
    assert np.array_equal(out[2], seen[2])


def test_unseen_ids_predict_alike_given_the_same_features():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0.5).astype(np.int64)
    ids = rng.integers(0, 50, size=(n, 2))
    spec = nn_entity.EntitySpec(seeds=(0,), epochs=1, k=2, width=16, blocks=1, device="cpu")
    spec.buckets = 128
    Xo = np.repeat(X[:1], 2, axis=0)
    ids_o = np.array([[1000, 2000], [3000, 4000]], dtype=np.int64)
    (p,) = nn_entity.fit_predict_entity(X, ids, y, None, [Xo], [ids_o], spec, log=lambda *_: None)
    assert np.allclose(p[0], p[1])


def test_entity_inputs_pool_strictly_earlier_rows_of_each_entity():
    led = pl.DataFrame(
        {
            "sender": [1, 1, 1, 2],
            "receiver": [5, 5, 6, 5],
            "device": [7, 7, 7, 8],
            "ts": [10, 20, 20, 30],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "new_payee": [1, 0, 0, 1],
            "pin_retries": [0, 0, 0, 0],
            "login_to_txn_sec": [5, 5, 5, 5],
            "sim_age_hours": [1.0, 1.0, 1.0, 1.0],
            "device_age_days": [1.0, 1.0, 1.0, 1.0],
            "balance_before": [1.0, 1.0, 1.0, 1.0],
            "to_person": [True, True, False, True],
        }
    )
    ids, pooled = nnexp.entity_inputs(led)
    assert ids.shape == (4, 3)
    k = len(nnexp.ENTITY_POOL_COLS) + 1
    by_sender, by_receiver = pooled[:, :k], pooled[:, k:]
    # sender 1: row 0 has no past; rows 1 and 2 (tied at ts 20) both see only row 0
    assert by_sender[0, -1] == 0 and np.isnan(by_sender[0, 0])
    assert by_sender[1, -1] == 1 and by_sender[1, 0] == 100.0
    assert by_sender[2, -1] == 1 and by_sender[2, 0] == 100.0
    # receiver 5: rows 0, 1, 3; row 3 sees rows 0 and 1
    assert by_receiver[3, -1] == 2 and by_receiver[3, 0] == 150.0


def test_every_family_is_wired():
    assert set(nnexp.FAMILIES) == {
        "selftrain",
        "entity",
        "entitypool",
        "entityemb",
        "tabr",
        "shift",
        "metric",
        "gentarget",
    }


def test_domain_weights_upweight_rows_that_look_like_the_target():
    from ringtrace import nn_shift

    rng = np.random.default_rng(0)
    X_src = rng.normal(size=(4000, 3))
    X_tgt = rng.normal(size=(2000, 3))
    X_tgt[:, 0] += 1.0
    w, info = nn_shift.domain_weights(
        X_src, X_tgt, rounds=30, sample=4000, threads=1, log=lambda *_: None
    )
    assert w.shape == (4000,) and np.isclose(w.mean(), 1.0)
    assert info["auc"] > 0.6
    hi, lo = X_src[:, 0] > 1.0, X_src[:, 0] < -1.0
    assert w[hi].mean() > 2 * w[lo].mean()
    w2, _ = nn_shift.domain_weights(
        X_src, X_tgt, rounds=30, sample=4000, threads=1, log=lambda *_: None
    )
    assert np.array_equal(w, w2)


def test_domain_weights_are_flat_without_shift():
    from ringtrace import nn_shift

    rng = np.random.default_rng(1)
    X_src = rng.normal(size=(4000, 3))
    X_tgt = rng.normal(size=(4000, 3))
    w, info = nn_shift.domain_weights(
        X_src, X_tgt, rounds=20, sample=8000, threads=1, log=lambda *_: None
    )
    assert info["auc"] < 0.6
    assert info["ess_share"] > 0.8


def test_weighted_fit_with_unit_weights_runs_on_cpu():
    from ringtrace import nn_shift, tabm

    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(np.int64) * 3
    spec = tabm.TabMSpec(seeds=(0,), epochs=1, k=2, width=16, blocks=1, device="cpu")
    (p,) = nn_shift.fit_predict_weighted(X, y, np.ones(300), [X[:10]], spec, log=lambda *_: None)
    assert p.shape == (10, 12) and np.allclose(p.sum(axis=1), 1.0)


def test_table_rows_report_each_cell_and_the_worst_target():
    def cell(dg, dc, dn, calls):
        r = {t: {"delta": d} for t, d in (("given", dg), ("cleaned", dc), ("gen", dn))}
        return {**r, "changed_calls": calls}

    report = {
        "variant": "v",
        "cells": {
            "sep": {"fraud_only": cell(1e-4, 0.0, 2e-4, 3), "alone": cell(-1e-4, 0, 0, 9)},
            "s13": {"fraud_only": cell(0.0, 0.0, 0.0, 0), "alone": cell(0, 0, 0, 0)},
        },
    }
    fraud, alone = nnexp.table_rows(report)
    assert fraud == "| v | fraud-only | +1.0 / +0.0 / +2.0 (3) | +0.0 / +0.0 / +0.0 (0) | yes |"
    assert alone.endswith("| no |") and "-1.0 / +0.0 / +0.0 (9)" in alone


def test_cleaned_target_returns_the_requested_label_set(monkeypatch):
    from types import SimpleNamespace

    from ringtrace import octlike, silver, store, teacher

    led = pl.DataFrame({"y": [0, 3, 5, 0], "period": [7, 8, 9, 9]})
    frame = SimpleNamespace(
        led=led, fit=np.array([True, True, False, False]), period=led["period"].to_numpy()
    )
    feats = pl.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b_event": [1, 1, 2, 2]})
    seen = {}
    monkeypatch.setattr(store, "frame", lambda fams: feats)

    def fake_oof(X, y, period):
        seen["oof"] = (X.shape, list(period))
        return np.zeros((len(y), 12))

    monkeypatch.setattr(teacher, "teacher_oof", fake_oof)
    monkeypatch.setattr(teacher, "union_events", lambda n: np.full(n, -1))

    def fake_build(led, proba, events, log=print, mask=None):
        seen["mask"] = list(mask)
        return {
            "label": np.array([0, 3, -1, -1]),
            "structured": np.array([0, 0, -1, -1]),
            "stats": {},
        }

    monkeypatch.setattr(silver, "build", fake_build)
    lab, _ = octlike._cleaned_target(frame, print)
    gen, _ = octlike._cleaned_target(frame, print, which="structured")
    assert list(lab[:2]) == [0, 3] and list(gen[:2]) == [0, 0]
    assert seen["mask"] == [True, True, False, False]
    assert seen["oof"] == ((2, 1), [7, 8])


def test_column_auc_gap_flags_only_the_drifting_column():
    from ringtrace import nn_shift

    rng = np.random.default_rng(3)
    a = rng.normal(size=(5000, 3))
    b = rng.normal(size=(5000, 3))
    b[:, 1] += 2.0  # a calendar-like column
    b[:1000, 2] = np.nan  # missingness moves too
    gap = nn_shift.column_auc_gap(a, b, sample=5000)
    assert gap[0] < 0.03 and gap[1] > 0.4 and gap[2] > 0.05


def test_run_model_builds_the_tabr_member_from_its_recipe():
    from ringtrace import run, tabr

    recipe = run.load_recipe("recipes/wildrun-tabr.json")
    spec, fit_fn = run.model(recipe)
    assert isinstance(spec, tabr.TabRSpec) and fit_fn is tabr.fit_predict
    assert spec.seeds == (0, 1, 2) and spec.epochs == 10


def test_mean_of_averages_variants_cell_by_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(nnexp, "_dir", lambda: tmp_path)
    rows = np.array([3, 5, 9])
    np.savez(tmp_path / "a_s13.npz", rows=rows, prob=np.full((3, 12), 0.1, dtype=np.float32))
    np.savez(tmp_path / "b_s13.npz", rows=rows, prob=np.full((3, 12), 0.3, dtype=np.float32))
    nnexp.mean_of("ab", ["a", "b"], ["s13"], log=lambda *_: None)
    z = np.load(tmp_path / "ab_s13.npz")
    assert np.array_equal(z["rows"], rows) and np.allclose(z["prob"], 0.2)
    np.savez(tmp_path / "c_s13.npz", rows=rows[::-1], prob=np.zeros((3, 12), dtype=np.float32))
    import pytest

    with pytest.raises(ValueError):
        nnexp.mean_of("ac", ["a", "c"], ["s13"], log=lambda *_: None)


def test_side_outputs_carry_the_job_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(nnexp, "_dir", lambda: tmp_path)
    monkeypatch.setitem(nnexp._JOB, "prefix", "wildrun")
    assert nnexp._side("shift-columns.json") == tmp_path / "wildrun-shift-columns.json"
    monkeypatch.setitem(nnexp._JOB, "prefix", "")
    assert nnexp._side("shift-columns.json") == tmp_path / "shift-columns.json"
