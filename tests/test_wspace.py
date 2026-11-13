"""Weight-space methods on synthetic data (CPU, a tiny network)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ringtrace import tabm, wspace  # noqa: E402
from ringtrace.classes import N_CLASSES  # noqa: E402


def tiny_spec(**kw) -> tabm.TabMSpec:
    base = dict(
        seeds=(0,), k=4, width=16, blocks=2, d_embed=4, n_freq=6, epochs=2, batch=64, device="cpu"
    )
    return tabm.TabMSpec(**{**base, **kw})


def toy(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 1.0, 1 + (X[:, 2] > 0).astype(np.int64), 0)
    return X, y.astype(np.int64)


def test_lerp_ends_and_extrapolation():
    a = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])}
    b = {"w": torch.tensor([3.0, 2.0]), "b": torch.tensor([1.0])}
    assert torch.equal(wspace.lerp(a, b, 0.0)["w"], a["w"])
    assert torch.equal(wspace.lerp(a, b, 1.0)["w"], b["w"])
    assert torch.allclose(wspace.lerp(a, b, 2.0)["w"], torch.tensor([5.0, 2.0]))
    assert wspace.distance(a, b) == pytest.approx(np.sqrt(5.0))
    assert torch.allclose(wspace.mean_state([a, b])["w"], torch.tensor([2.0, 2.0]))


def test_add_vectors_matches_single_lerp():
    a = {"w": torch.tensor([1.0, 2.0])}
    b = {"w": torch.tensor([3.0, 5.0])}
    c = {"w": torch.tensor([0.0, 2.0])}
    assert torch.allclose(wspace.add_vectors(a, [b], [1.5])["w"], wspace.lerp(a, b, 1.5)["w"])
    both = wspace.add_vectors(a, [b, c], [1.0, 1.0])["w"]
    assert torch.allclose(both, torch.tensor([2.0, 5.0]))
    assert wspace.cosine(a, b, b) == pytest.approx(1.0)


def test_barrier_is_zero_on_a_line_and_positive_on_a_bump():
    assert wspace.barrier([1.0, 1.5, 2.0, 2.5, 3.0]) == pytest.approx(0.0)
    assert wspace.barrier([1.0, 1.0, 1.4, 1.0, 1.0]) == pytest.approx(0.4)


def test_fraud_only_keeps_legit_and_shares_the_rest():
    p = np.array([0.0, 0.3, 1.0])
    out = wspace.fraud_only(p)
    assert out.shape == (3, 12)
    assert out.dtype == np.float64
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.array_equal(out[:, 0], p)
    rest = ((1.0 - p) / 11)[:, None]
    assert np.allclose(out[:, 1:], rest)
    assert np.allclose(out[:, 1], out[:, 11])


def test_load_arms_merges_files_and_rejects_a_row_mismatch(tmp_path):
    rows = np.arange(4, dtype=np.int64)
    test_rows = np.arange(10, 14, dtype=np.int64)
    base = np.linspace(0.1, 0.4, 4).astype(np.float32)
    np.savez(
        tmp_path / "arms.npz",
        rows=rows,
        test_rows=test_rows,
        base=base,
        base__test=np.array([0.2, 0.4, 0.6, 0.8], np.float32),
        ta1=np.full(4, 0.3, np.float32),
    )
    np.savez(
        tmp_path / "rebasin.npz",
        rows=rows.copy(),
        test_rows=test_rows.copy(),
        rb_mean=np.full(4, 0.7, np.float32),
        rb_mean__test=np.full(4, 0.1, np.float32),
    )
    loaded = wspace.load_arms(tmp_path)
    assert np.array_equal(loaded["rows"], rows)
    assert np.array_equal(loaded["test_rows"], test_rows)
    assert set(loaded["arms"]) == {"base", "ta1", "rb_mean"}
    assert set(loaded["test"]) == {"base", "rb_mean"}
    assert np.allclose(loaded["arms"]["base"], base)
    assert np.allclose(loaded["test"]["rb_mean"], np.full(4, 0.1))
    np.savez(tmp_path / "rebasin.npz", rows=rows + 1, rb_mean=np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="rows"):
        wspace.load_arms(tmp_path)


def test_arm_matrix_updates_held_out_rows_and_the_october_mean():
    base = np.full((10, 12), 0.25)
    te = np.array([7, 8, 9])
    arm = "ta1"

    def pack(rows, p, test_rows=None, p_test=None):
        loaded = {
            "rows": np.asarray(rows),
            "test_rows": None if test_rows is None else np.asarray(test_rows),
            "arms": {arm: np.asarray(p, dtype=np.float64)},
            "test": {},
        }
        if p_test is not None:
            loaded["test"][arm] = np.asarray(p_test, dtype=np.float64)
        return loaded

    folds = {7: pack([0, 1], [0.2, 0.4]), 8: pack([2, 3], [0.6, 0.8])}
    out = wspace.arm_matrix(base, te, folds, arm)
    assert np.allclose(base, 0.25)
    assert np.allclose(out[[0, 1, 2, 3], 0], [0.2, 0.4, 0.6, 0.8])
    assert np.allclose(out[0, 1:], (1.0 - 0.2) / 11)
    assert np.allclose(out[4:10], 0.25)

    tests = {
        7: np.array([0.2, 0.0, 1.0]),
        8: np.array([0.0, 0.5, 0.5]),
        9: np.array([1.0, 0.5, 0.0]),
    }
    partial = {
        7: pack([0, 1], [0.2, 0.4], te, tests[7]),
        8: pack([2, 3], [0.6, 0.8], te, tests[8]),
        9: pack([4, 5], [1.0, 0.0]),
    }
    held = wspace.arm_matrix(base, te, partial, arm)
    assert np.allclose(held[[4, 5], 0], [1.0, 0.0])
    assert np.allclose(held[te], 0.25)

    full = {
        7: pack([0, 1], [0.2, 0.4], te, tests[7]),
        8: pack([2, 3], [0.6, 0.8], te, tests[8]),
        9: pack([4, 5], [1.0, 0.0], te, tests[9]),
    }
    out = wspace.arm_matrix(base, te, full, arm)
    assert np.allclose(out[te], wspace.fraud_only((tests[7] + tests[8] + tests[9]) / 3))
    assert not np.allclose(out[te, 0], tests[7])
    full[9] = pack([4, 5], [1.0, 0.0], np.array([0, 1, 2]), tests[9])
    with pytest.raises(ValueError, match="test_rows"):
        wspace.arm_matrix(base, te, full, arm)
    assert np.allclose(base, 0.25)


def test_arm_recipe_checks_and_stored_points_at_tabm():
    from ringtrace import final

    recipe = wspace.arm_recipe("ta1.5")
    assert final.check_recipe(recipe)["name"] == "wspace-ta1.5"
    neural = recipe["members"][-1]
    assert neural == {
        "name": "tabm",
        "run": "wspace:ta1.5",
        "variant": "fold",
        "cells": "wspace-ta1.5",
        "role": "fraud",
    }
    assert recipe["weights"]["mode"] == "fixed"
    assert recipe["weights"]["values"]["tabm"] == 0.5
    trees = [(m["name"], m["run"], m["cells"]) for m in recipe["members"][:-1]]
    assert trees == list(wspace.TREES)
    assert [recipe["weights"]["values"][m["name"]] for m in recipe["members"][:-1]] == [0.1] * 5
    assert all(m["role"] == "full" and m["variant"] == "fold" for m in recipe["members"][:-1])
    stored = wspace.arm_recipe("stored")
    assert stored["members"][-1]["run"] == "tabm"
    assert stored["members"][-1]["cells"] == "tabm"
    assert stored["members"][-1]["role"] == "fraud"
    assert stored["weights"]["values"]["tabm"] == 0.5


def test_write_cell_stores_fraud_only_probabilities(tmp_path):
    rows = np.array([3, 4, 9])
    p = np.array([0.0, 0.5, 1.0])
    wspace.write_cell(tmp_path, "ta1", 13, rows, p)
    with np.load(tmp_path / "wspace-ta1_s13_r0.npz") as blob:
        assert np.array_equal(blob["rows"], rows)
        assert blob["prob"].dtype == np.float32
        assert blob["prob"].shape == (3, 12)
        assert np.allclose(blob["prob"], wspace.fraud_only(p))


def test_arm_bench_member_returns_the_array_it_was_given(tmp_path):
    import polars as pl

    n = 5
    led = pl.DataFrame({"is_test": [False, False, False, True, True], "period": [7, 8, 9, 10, 10]})
    bench = wspace.ArmBench(led, {}, np.arange(n), tmp_path)
    assert bench.arm_arrays == {}
    arr = np.full((n, N_CLASSES), 0.3)
    bench.arm_arrays["x"] = arr
    assert bench.member({"run": "wspace:x", "variant": "fold"}) is arr


def test_same_seed_gives_the_same_start_and_fine_tune_moves_the_weights():
    X, y = toy()
    spec = tiny_spec()
    Z = tabm.transform(tabm.fit_prep(X, 50), X)
    y_clean = y.copy()
    y_clean[:20] = 1
    recent = np.arange(len(y)) >= 300
    ends = wspace.endpoints(Z, y, y_clean, recent, spec, seed=3, log=lambda *_: None)
    again = wspace.state(tabm.train_one(Z, y, spec, 3, log=lambda *_: None))
    assert wspace.distance(ends["g"], again) == 0.0  # deterministic, so g and c share a start
    assert 0 < wspace.distance(ends["g"], ends["c"]) < wspace.norm(ends["g"])
    assert wspace.distance(ends["g"], ends["r"]) > 0
    arms = wspace.arm_weights(ends)
    assert wspace.distance(arms["ta1"], ends["c"]) < 1e-5
    assert wspace.distance(arms["ti1"], ends["r"]) < 1e-5
    p = wspace.proba(arms["ta1.5"], Z, spec)
    assert p.shape == (len(Z), N_CLASSES)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    losses = wspace.path_losses(ends["g"], ends["c"], Z, y, spec)
    assert len(losses) == len(wspace.PATH) and np.isfinite(losses).all()


def test_fine_tune_rejects_class_weights():
    X, y = toy(200)
    spec = tiny_spec(class_weight_power=0.5)
    Z = tabm.transform(tabm.fit_prep(X, 50), X)
    sd = wspace.state(tabm.build_model(Z.shape[1], spec))
    with pytest.raises(ValueError):
        wspace.fine_tune(sd, Z, y, spec, 0, log=lambda *_: None)
