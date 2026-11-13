"""The blend on synthetic member runs: shape and alignment checks, averaging, scoring, the
validated submission and the rows changed against each member."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from ringtrace import blend as bl
from ringtrace.classes import CLASSES, N_CLASSES
from ringtrace.decode import decode_by_period
from ringtrace.scoring import macro_f1

N_TRAIN, N_TEST = 600, 200


def _ledger(rng):
    period = np.r_[np.repeat([7, 8, 9], N_TRAIN // 3), np.full(N_TEST, 10)]
    y = np.r_[rng.integers(0, N_CLASSES, N_TRAIN), np.full(N_TEST, -1)]
    return pl.DataFrame(
        {
            "txn_id": [f"TX{i:05d}" for i in range(N_TRAIN + N_TEST)],
            "is_test": period == 10,
            "period": period,
            "y": y,
        }
    )


def _proba(rng, y, n, noise):
    logits = rng.normal(0, noise, (n, N_CLASSES))
    if y is not None:
        logits[np.arange(n), y] += 2.0
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def _member(runs, name, led, rng, noise, reported=True):
    y = led["y"].to_numpy()[:N_TRAIN]
    oof, test = _proba(rng, y, N_TRAIN, noise), _proba(rng, None, N_TEST, noise)
    d = runs / name
    d.mkdir(parents=True)
    np.save(d / "oof.npy", oof.astype(np.float32))
    np.save(d / "test.npy", test.astype(np.float32))
    if reported:
        pred = decode_by_period(oof.astype(np.float32), led["period"].to_numpy()[:N_TRAIN])
        per = [
            macro_f1(
                y[led["period"].to_numpy()[:N_TRAIN] == m],
                pred[led["period"].to_numpy()[:N_TRAIN] == m],
            )
            for m in (7, 8, 9)
        ]
        (d / "summary.json").write_text(
            json.dumps({"fixed_point": {"given": {"months": per}}, "submission_sha256": "x"})
        )
    return oof, test


def _setup(tmp_path, n=3):
    rng = np.random.default_rng(7)
    led = _ledger(rng)
    runs = tmp_path / "runs"
    arrays = [_member(runs, f"m{i}", led, rng, 1.0 + 0.3 * i) for i in range(n)]
    sample = tmp_path / "sample.csv"
    ids = led["txn_id"].to_numpy()[N_TRAIN:]
    pl.DataFrame({"id": ids[::-1], "label": ["legit"] * N_TEST}).write_csv(sample)
    recipe = {
        "name": "b",
        "members": [
            {"name": f"m{i}", "recipe": "r", "run": f"m{i}", "weight": 1 / n} for i in range(n)
        ],
        "decoder": {"fraud_scale": 0.875},
    }
    targets = {
        "given": led["y"].to_numpy()[:N_TRAIN],
        "gen": np.where(rng.random(N_TRAIN) < 0.1, 0, led["y"].to_numpy()[:N_TRAIN]),
    }
    return led, runs, arrays, sample, recipe, targets


def test_blend_end_to_end(tmp_path):
    led, runs, arrays, sample, recipe, targets = _setup(tmp_path)
    out = tmp_path / "out"
    rep = bl.blend(recipe, led, runs, targets, sample, out, log=lambda m: None)
    mean_oof = np.mean([a[0].astype(np.float32) for a in arrays], axis=0)
    assert np.allclose(np.load(out / "oof.npy"), mean_oof, atol=1e-6)
    assert rep["submission_problems"] == []
    sub = pl.read_csv(out / "submission.csv")
    expected_ids = pl.read_csv(sample)["id"]
    assert (sub["id"] == expected_ids).all()
    assert set(sub["label"].to_list()) <= set(CLASSES)
    assert (
        rep["submission_sha256"]
        == hashlib.sha256((out / "submission.csv").read_bytes()).hexdigest()
    )
    assert sum(rep["test_calls"].values()) == N_TEST
    assert set(rep["changed_vs_members"]) == {"m0", "m1", "m2"}
    assert set(rep["final"]) == {"given", "gen"}
    assert len(rep["final"]["given"]["months"]) == 3
    # the blend of noisy members beats the noisiest one
    assert (
        rep["final"]["given"]["mean"]
        > rep["members"]["m2"]["scores"]["fixed_point"]["given"]["mean"]
    )
    # deterministic
    rep2 = bl.blend(recipe, led, runs, targets, sample, tmp_path / "out2", log=lambda m: None)
    assert rep2["submission_sha256"] == rep["submission_sha256"]


def test_single_member_blend_reproduces_member(tmp_path):
    led, runs, _, sample, recipe, targets = _setup(tmp_path, n=1)
    rep = bl.blend(recipe, led, runs, targets, sample, tmp_path / "o", log=lambda m: None)
    assert rep["changed_vs_members"]["m0"] == {"train": 0, "october": 0}


def test_misaligned_member_is_rejected(tmp_path):
    led, runs, _, sample, recipe, targets = _setup(tmp_path)
    oof = np.load(runs / "m1" / "oof.npy")
    np.save(runs / "m1" / "oof.npy", oof[np.random.default_rng(0).permutation(N_TRAIN)])
    with pytest.raises(bl.BlendError, match="not aligned"):
        bl.blend(recipe, led, runs, targets, sample, tmp_path / "o", log=lambda m: None)


def test_bad_shapes_and_weights(tmp_path):
    led, runs, _, sample, recipe, targets = _setup(tmp_path)
    np.save(runs / "m0" / "test.npy", np.full((N_TEST - 1, N_CLASSES), 1 / N_CLASSES))
    with pytest.raises(bl.BlendError, match="shape"):
        bl.blend(recipe, led, runs, targets, sample, tmp_path / "o", log=lambda m: None)
    path = tmp_path / "r.json"
    recipe["members"][0]["weight"] = 0.5
    path.write_text(json.dumps(recipe))
    with pytest.raises(bl.BlendError, match="sum to 1"):
        bl.load_recipe(path)


def test_members_listing(tmp_path, capsys):
    led, runs, _, sample, recipe, targets = _setup(tmp_path)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(recipe))
    assert bl.main([str(path), "--members"]) == 0
    assert capsys.readouterr().out.split() == ["r", "m0", "r", "m1", "r", "m2"]


def test_shipped_recipe_is_an_equal_five_way_blend():
    recipe = bl.load_recipe("recipes/blend.json")
    assert [m["weight"] for m in recipe["members"]] == [0.2] * 5
    assert recipe["decoder"] == {"fraud_scale": 0.875}
    assert recipe["rules"] == ["anchored_templates"]
