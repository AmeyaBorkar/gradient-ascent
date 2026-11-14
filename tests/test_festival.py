"""Festival scaling: roundness, the ordinary-row mask, the installed ledger, the run argv."""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

from ringtrace import festival, store

FACTOR = 1.166


@pytest.fixture(autouse=True)
def _reset_store():
    store.use_frame(None)
    yield
    store.use_frame(None)


def _mini():
    """Seven rows covering ordinary, fraud-routine, other months and the scored month."""
    led = pl.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(7)],
            "amount": np.array([1000, 2000, 250, 3000, 4000, 80, 500], dtype=np.float64),
            "balance_before": np.array([100, 200, 300, 400, 500, 600, 700], dtype=np.float64),
            "is_test": np.array([False, False, False, False, True, False, False]),
            "period": np.array([7, 7, 8, 9, 10, 7, 8], dtype=np.int8),
            "y": np.array([0, 3, 0, 1, 0, 0, 0], dtype=np.int8),
        }
    )
    # 0 ordinary, non-zero fraud routine, -1 outside the train rows the routine labelled.
    structured = np.array([0, 4, 0, 0, 0, 6, -1], dtype=np.int64)
    return led, structured


def _patch(monkeypatch, led, structured):
    monkeypatch.setattr("ringtrace.tables.ledger", lambda: led)

    def fake_digest(path, n=16):
        return "abcdef"

    monkeypatch.setattr("ringtrace.tables.file_digest", fake_digest)

    def fake_load(frame, which="label"):
        if which != "structured" or structured is None:
            return None
        if len(frame) != len(structured):
            return None
        return structured

    monkeypatch.setattr("ringtrace.silver.load", fake_load)


def _roundness_counts(amount: np.ndarray) -> dict[str, int]:
    """Nested shares: multiple of 100, of 10, whole rupee, and the rest."""
    cents = np.rint(np.asarray(amount, dtype=np.float64) * 100.0).astype(np.int64)
    whole = cents % 100 == 0
    rupees = cents // 100
    return {
        "100": int(np.count_nonzero(whole & (rupees % 100 == 0))),
        "10": int(np.count_nonzero(whole & (rupees % 10 == 0))),
        "1": int(np.count_nonzero(whole)),
        "other": int(np.count_nonzero(~whole)),
    }


def test_scale_rows_scales_only_the_mask_and_keeps_the_input():
    weird = np.nextafter(np.float64(50.0), np.float64(0.0))
    weird_bal = np.nextafter(np.float64(7.0), np.float64(20.0))
    amount = np.array([200.0, 250.0, 13.0, 10.4, weird, 80.0, 0.01, 0.0], dtype=np.float64)
    balance = np.array([1000.0, 10.004, 1.0, 0.0, weird_bal, 50.5, 5.0, 8.0], dtype=np.float64)
    mask = np.array([True, True, True, True, False, False, True, True])
    led = pl.DataFrame(
        {
            "amount": amount,
            "balance_before": balance,
            "y": np.arange(len(amount), dtype=np.int8),
            "is_test": np.array([False, False, False, False, True, False, False, False]),
            "txn_id": [f"t{i}" for i in range(len(amount))],
        }
    )
    snap_amt = led["amount"].to_numpy().copy()
    snap_bal = led["balance_before"].to_numpy().copy()
    out = festival.scale_rows(led, mask, FACTOR, 1.02)
    again = festival.scale_rows(led, mask, FACTOR, 1.02)

    assert out is not led
    assert out.columns == led.columns
    assert out.schema["amount"] == pl.Float64
    assert out.schema["balance_before"] == pl.Float64
    assert led["amount"].to_numpy().tobytes() == snap_amt.tobytes()
    assert led["balance_before"].to_numpy().tobytes() == snap_bal.tobytes()
    assert out["amount"].to_numpy().tobytes() == again["amount"].to_numpy().tobytes()
    assert out["y"].equals(led["y"])
    assert out["txn_id"].equals(led["txn_id"])
    assert out["is_test"].equals(led["is_test"])
    got = out["amount"].to_numpy()
    assert got[~mask].tobytes() == snap_amt[~mask].tobytes()
    assert out["balance_before"].to_numpy()[~mask].tobytes() == snap_bal[~mask].tobytes()
    # 200*1.166 = 233.2 -> 200 or 300 (unit 100); 250 -> 291.5 -> 290, because 300 would be a
    # multiple of 100; 13 -> 15 or 16; 10.40 -> 12.12 or 12.13; 0.01 -> 0.01 or 0.02;
    # 0 is a multiple of 100 and the floor is 100.
    cents = np.rint(got * 100.0).astype(np.int64)
    allowed = [{20000, 30000}, {29000}, {1500, 1600}, {1212, 1213}, {1, 2}, {10000}]
    assert all(c in a for c, a in zip(cents[mask].tolist(), allowed, strict=True))
    other_seed = festival.scale_rows(led, mask, FACTOR, 1.02, seed=1)["amount"].to_numpy()
    assert all(
        c in a
        for c, a in zip(np.rint(other_seed[mask] * 100).astype(int).tolist(), allowed, strict=True)
    )
    bal_cents = np.rint(out["balance_before"].to_numpy() * 100.0).astype(np.int64)
    assert bal_cents[mask].tolist() == [102000, 1020, 102, 0, 510, 816]

    floor = pl.DataFrame({"amount": np.array([0.01, 4.0]), "balance_before": np.array([1.0, 2.0])})
    floored = festival.scale_rows(floor, np.array([True, True]), 0.1)
    assert np.rint(floored["amount"].to_numpy() * 100.0).astype(np.int64).tolist() == [1, 100]
    # Default balance factor is 1, still rounded to the paisa: 12.341 -> 12.34, 12.34 stays.
    grid = pl.DataFrame(
        {
            "amount": np.array([100.0, 100.0]),
            "balance_before": np.array([12.34, 12.341]),
        }
    )
    held = festival.scale_rows(grid, np.array([True, True]), 1.0)
    assert np.rint(held["balance_before"].to_numpy() * 100.0).astype(np.int64).tolist() == [
        1234,
        1234,
    ]
    assert np.rint(held["amount"].to_numpy() * 100.0).astype(np.int64).tolist() == [10000, 10000]


def test_scale_rows_never_moves_a_row_into_a_rounder_class():
    # 90 (unit 10) -> 104.94 lies between 100, a multiple of 100, and 110: always 110.
    # 9 (unit 1) -> 10.494 lies between 10 and 11: always 11.
    # 85.76 (paise) -> 99.996 lies between 99.99 and the whole rupee 100.00: always 99.99.
    # 300 (unit 100) -> 349.8 goes to 300 or 400 and stays a multiple of 100.
    amount = np.array([90.0, 9.0, 85.76, 300.0], dtype=np.float64)
    led = pl.DataFrame({"amount": amount, "balance_before": np.ones(len(amount))})
    for seed in range(5):
        out = festival.scale_rows(led, np.ones(len(amount), dtype=bool), FACTOR, seed=seed)
        cents = np.rint(out["amount"].to_numpy() * 100.0).astype(np.int64)
        assert cents[:3].tolist() == [11000, 1100, 9999]
        assert cents[3] in (30000, 40000)

    # Small multiples of 100 scale on average too: 100, 200 and 300 do not stay where they are.
    hundreds = np.repeat(np.array([100.0, 200.0, 300.0]), 4000)
    frame = pl.DataFrame({"amount": hundreds, "balance_before": np.ones(len(hundreds))})
    up = festival.scale_rows(frame, np.ones(len(hundreds), dtype=bool), FACTOR)["amount"]
    assert np.all(np.rint(up.to_numpy() * 100).astype(np.int64) % 10000 == 0)
    assert up.to_numpy().mean() / hundreds.mean() == pytest.approx(FACTOR, rel=0.01)

    # A realistic grid: every class share is kept exactly, rounding hits included.
    rng = np.random.default_rng(3)
    paise = rng.integers(100, 500_000, size=20_000)
    paise[:5000] = (paise[:5000] // 1000) * 1000  # multiples of 10 (and some of 100)
    paise[5000:5600] = (paise[5000:5600] // 10000 + 1) * 10000  # multiples of 100
    paise[5600:6000] = (paise[5600:6000] // 100) * 100  # whole rupees
    grid = paise.astype(np.float64) / 100.0
    frame = pl.DataFrame({"amount": grid, "balance_before": np.ones(len(grid))})
    scaled = festival.scale_rows(frame, np.ones(len(grid), dtype=bool), FACTOR)
    got = scaled["amount"].to_numpy()
    assert _roundness_counts(got) == _roundness_counts(grid)
    assert got.mean() / grid.mean() == pytest.approx(FACTOR, rel=0.005)


def test_scale_rows_keeps_roundness_shares_and_the_mean_ratio():
    # Residues sit well clear of a coarser boundary after *1.166, so float noise cannot change
    # the class. A plain multiply would leave every class; unit rounding puts them back.
    # Masked rows are the measured mix: 2.5% of 100, 25% of 10, 27% whole-rupee, rest other.
    amount_100 = 10000.0 + 100.0 * np.arange(1, 26)
    amount_10 = 50000.0 * np.arange(1, 226) + 30.0
    amount_1 = 50000.0 * np.arange(1, 21) + 3.0
    amount_other = 50000.0 * np.arange(1, 731) + (337 / 100)
    hold_100 = 10000.0 + 100.0 * np.arange(26, 36)
    hold_10 = 50000.0 * np.arange(226, 236) + 30.0
    hold_1 = 50000.0 * np.arange(21, 31) + 3.0
    hold_other = 50000.0 * np.arange(731, 741) + (337 / 100)
    amount = np.concatenate(
        [amount_100, amount_10, amount_1, amount_other, hold_100, hold_10, hold_1, hold_other]
    ).astype(np.float64)
    mask = np.concatenate(
        [
            np.ones(1000, dtype=bool),
            np.zeros(40, dtype=bool),
        ]
    )
    rng = np.random.default_rng(0)
    order = rng.permutation(len(amount))
    amount, mask = amount[order], mask[order]
    balance = np.full(len(amount), 1000.0, dtype=np.float64)
    balance[~mask] = np.nextafter(np.float64(1000.0), np.float64(0.0))
    led = pl.DataFrame(
        {
            "amount": amount,
            "balance_before": balance,
            "y": np.zeros(len(amount), dtype=np.int8),
            "note": ["r"] * len(amount),
        }
    )
    snap_amt = led["amount"].to_numpy().copy()
    snap_bal = led["balance_before"].to_numpy().copy()
    known = {"100": 25, "10": 250, "1": 270, "other": 730}
    assert _roundness_counts(snap_amt[mask]) == known

    out = festival.scale_rows(led, mask, FACTOR, 1.02)
    got = out["amount"].to_numpy()
    assert led["amount"].to_numpy().tobytes() == snap_amt.tobytes()
    assert got[~mask].tobytes() == snap_amt[~mask].tobytes()
    assert out["balance_before"].to_numpy()[~mask].tobytes() == snap_bal[~mask].tobytes()
    assert out["y"].equals(led["y"])
    assert out["note"].equals(led["note"])
    assert _roundness_counts(got[mask]) == known
    assert _roundness_counts(got) == _roundness_counts(snap_amt)
    ratio = got[mask].mean() / snap_amt[mask].mean()
    assert ratio == pytest.approx(FACTOR, rel=0.01)
    scaled_bal = np.rint(out["balance_before"].to_numpy()[mask] * 100.0)
    assert np.array_equal(scaled_bal, np.full(int(mask.sum()), 102000.0))


def test_festival_mask_excludes_test_fraud_and_other_periods():
    led, structured = _mini()
    only_july = festival.festival_mask(led, structured, [7])
    assert only_july.dtype == bool
    assert only_july.tolist() == [True, False, False, False, False, False, False]
    # Period 9 row is fraud-labelled (y=1) but structured 0, so it is ordinary and eligible.
    assert festival.festival_mask(led, structured, [9]).tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert festival.festival_mask(led, structured, [7, 8]).tolist() == [
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    # The scored month is period 10 and structured 0; it still stays out.
    assert not festival.festival_mask(led, structured, [10]).any()


def test_install_scaled_ledger_and_key(monkeypatch):
    led, structured = _mini()
    _patch(monkeypatch, led, structured)
    snap = led["amount"].to_numpy().copy()
    store.use_frame(None)
    base_key = store._ledger_key()

    summary = festival.install((8, 7), FACTOR, 1.02)
    got = store.ledger()
    mask = festival.festival_mask(led, structured, (7, 8))
    expect = festival.scale_rows(led, mask, FACTOR, 1.02)

    assert summary["key"] == store._ledger_key()
    assert store._ledger_key() != base_key
    assert re.fullmatch(
        rf"fest-{re.escape(base_key)}-a1\.166-b1\.02-p7-8-[0-9a-f]{{12}}-[0-9a-f]{{12}}",
        summary["key"],
    )
    assert summary["periods"] == [7, 8]
    assert summary["rows_scaled"] == {7: 1, 8: 1}
    assert summary["mean_amount_ratio"] == pytest.approx(
        expect["amount"].to_numpy()[mask].sum() / snap[mask].sum()
    )
    assert got["amount"].to_numpy().tobytes() == expect["amount"].to_numpy().tobytes()
    assert (
        got["balance_before"].to_numpy().tobytes() == expect["balance_before"].to_numpy().tobytes()
    )
    assert got["amount"].to_numpy()[mask].tobytes() != snap[mask].tobytes()
    scored = led["is_test"].to_numpy()
    fraud = structured != 0
    assert got["amount"].to_numpy()[scored].tobytes() == snap[scored].tobytes()
    assert got["amount"].to_numpy()[fraud].tobytes() == snap[fraud].tobytes()
    assert got["y"].equals(led["y"])
    assert got["txn_id"].equals(led["txn_id"])
    assert got["is_test"].equals(led["is_test"])
    assert got["period"].equals(led["period"])
    assert led["amount"].to_numpy().tobytes() == snap.tobytes()

    again = festival.install((7, 8), FACTOR, 1.02)
    assert again["key"] == summary["key"]
    assert store.ledger()["amount"].to_numpy().tobytes() == got["amount"].to_numpy().tobytes()

    empty = festival.install((6,))
    assert empty["rows_scaled"] == {6: 0}
    assert empty["mean_amount_ratio"] is None
    assert empty["key"] != summary["key"]
    assert store._ledger_key() != base_key
    # Period 7 was scaled above; starting over from the base leaves it unscaled this time.
    assert store.ledger()["amount"].to_numpy().tobytes() == snap.tobytes()


def test_install_requires_structured_labels(monkeypatch):
    led, _structured = _mini()
    _patch(monkeypatch, led, None)
    with pytest.raises(RuntimeError, match="structured"):
        festival.install((7,))


def test_run_forwards_argv(monkeypatch):
    led, structured = _mini()
    _patch(monkeypatch, led, structured)
    seen = []

    def fake_main(argv=None):
        seen.append(list(argv))
        assert store._KEY is not None and str(store._KEY).startswith("fest-")
        return {"ok": True}

    monkeypatch.setattr("ringtrace.run.main", fake_main)
    result = festival.main(["run", "recipes/champion.json", "--periods", "7,8,9", "--folds", "9"])
    assert result == {"ok": True}
    assert seen[0] == [
        "recipes/champion.json",
        "--name",
        "champion-fest-7-8-9",
        "--folds",
        "9",
    ]
    # Install ran first: September's ordinary row moved, the scored month did not.
    assert store.ledger()["amount"].to_numpy()[3] != led["amount"].to_numpy()[3]
    assert store.ledger()["amount"].to_numpy()[4] == led["amount"].to_numpy()[4]

    festival.main(["run", "recipes/champion.json", "--periods", "9,7"])
    assert seen[1] == ["recipes/champion.json", "--name", "champion-fest-7-9"]

    festival.main(
        [
            "run",
            "recipes/lgbm.json",
            "--periods",
            "8",
            "--name",
            "custom",
            "--amount-factor",
            "1.5",
            "--balance-factor",
            "1.1",
            "--folds",
            "8",
        ]
    )
    assert seen[2] == ["recipes/lgbm.json", "--name", "custom", "--folds", "8"]
    mask = festival.festival_mask(led, structured, (8,))
    expect = festival.scale_rows(led, mask, 1.5, 1.1)
    got = store.ledger()
    assert got["amount"].to_numpy().tobytes() == expect["amount"].to_numpy().tobytes()
    assert (
        got["balance_before"].to_numpy().tobytes() == expect["balance_before"].to_numpy().tobytes()
    )
    # The earlier install had scaled July; this one started from the base ledger again.
    assert got["amount"].to_numpy()[0] == led["amount"].to_numpy()[0]


def test_show_prints_summary_without_running(monkeypatch, capsys):
    led, structured = _mini()
    _patch(monkeypatch, led, structured)
    called = []
    monkeypatch.setattr("ringtrace.run.main", lambda argv=None: called.append(argv))
    summary = festival.main(["show", "--periods", "7,8"])
    assert called == []
    text = capsys.readouterr().out
    assert "mean_amount_ratio" in text
    assert '"7": 1' in text
    assert summary["rows_scaled"] == {7: 1, 8: 1}
    assert "a1.166-b1.02-p7-8-" in summary["key"]
