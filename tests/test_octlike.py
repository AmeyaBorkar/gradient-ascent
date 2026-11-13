import numpy as np
import polars as pl
import pytest

from ringtrace import octlike

H = 3600


def _ledger(rows):
    """rows: (sender, receiver, to_person, device, ts, y)."""
    s, r, p, d, t, y = map(np.array, zip(*rows, strict=True))
    return pl.DataFrame(
        {
            "sender": s,
            "receiver": r,
            "to_person": p.astype(bool),
            "device": d,
            "ts": t.astype(np.int64),
            "y": y,
        }
    )


def _units(led, events=(), **kw):
    return octlike.september_units(
        led, [np.asarray(e) for e in events], np.arange(led.height), **kw
    )


def test_label_unit_takes_neighbours_not_strangers():
    led = _ledger(
        [
            (1, 50, 1, 100, 0, 3),  # fraud
            (1, 60, 1, 101, 1800, 0),  # same sender within 1 h -> joined
            (1, 70, 1, 102, 3 * H, 0),  # same sender, too late
            (2, 50, 1, 103, 48 * H, 0),  # same receiving person within 72 h -> joined
            (50, 80, 1, 104, 10 * H, 0),  # receiver passes money on within 24 h -> joined
            (9, 99, 1, 105, 0, 0),  # unrelated
        ]
    )
    u = _units(led)
    assert u[0] == u[1] == u[3] == u[4]
    assert len({u[0], u[2], u[5]}) == 3


def test_legit_rows_alone_are_not_joined():
    led = _ledger([(1, 50, 1, 100, 0, 0), (1, 51, 1, 100, 10, 0)])
    u = _units(led)
    assert u[0] != u[1]


def test_pair_and_device_windows():
    led = _ledger(
        [
            (1, 50, 1, 100, 0, 2),
            (1, 50, 1, 200, 6 * 24 * H, 0),  # same pair within 7 days
            (7, 66, 0, 100, 1200, 0),  # same device within 1 h
            (8, 66, 0, 300, 1200, 0),  # same merchant: not a person, not joined
        ]
    )
    u = _units(led)
    assert u[0] == u[1] == u[2]
    assert u[3] != u[0]


def test_events_join_across_modules_and_with_label_units():
    # merchant-ring style rows: different senders, merchant receiver, not label-adjacent
    led = _ledger(
        [
            (1, 10_000_001, 0, 100, 0, 8),
            (2, 10_000_001, 0, 101, 10, 8),
            (3, 10_000_001, 0, 102, 20, 8),
            (3, 77, 1, 102, 30, 0),  # same sender within 1 h of a fraud row
            (4, 5, 1, 103, 9 * 24 * H, 0),
        ]
    )
    ev_a = [5, 5, -1, -1, -1]
    ev_b = [-1, 9, 9, -1, -1]
    u = _units(led, [ev_a, ev_b])
    assert u[0] == u[1] == u[2] == u[3]
    assert u[4] != u[0]
    only_labels = _units(led, [ev_a, ev_b], with_events=False)
    assert only_labels[0] != only_labels[1]  # label units alone cut the ring


def test_thinning_keeps_units_whole_and_is_deterministic():
    rng = np.random.default_rng(0)
    units = rng.integers(0, 3000, 20_000)
    a, b = octlike.thin(units, 13), octlike.thin(units, 13)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, octlike.thin(units, 14))
    for u in np.unique(units)[:200]:
        assert len(set(a[units == u])) == 1
    share = np.mean([a[units == u][0] for u in np.unique(units)])
    assert share == pytest.approx(octlike.KEEP, abs=0.03)


def test_qr_blank_is_seeded():
    cand = np.arange(10_000)
    a = octlike.qr_blank(cand, 13)
    assert np.array_equal(a, octlike.qr_blank(cand, 13))
    assert len(a) / len(cand) == pytest.approx(octlike.QR_BLANK, abs=0.02)


def test_units_deterministic_on_random_ledger():
    rng = np.random.default_rng(1)
    n = 3000
    led = _ledger(
        list(
            zip(
                rng.integers(0, 300, n),
                rng.integers(0, 300, n),
                rng.random(n) < 0.8,
                rng.integers(0, 400, n),
                np.sort(rng.integers(0, 30 * 24 * H, n)),
                np.where(rng.random(n) < 0.05, 1, 0),
                strict=True,
            )
        )
    )
    ev = np.where(rng.random(n) < 0.02, rng.integers(0, 20, n), -1)
    u1, u2 = _units(led, [ev]), _units(led, [ev])
    assert np.array_equal(u1, u2)
    # every event lies inside one unit
    for e in np.unique(ev[ev >= 0]):
        assert len(set(u1[ev == e])) == 1


def test_parse_blend():
    assert octlike.parse_blend("a") == [("a", 1.0)]
    assert octlike.parse_blend("a+b") == [("a", 0.5), ("b", 0.5)]
    assert octlike.parse_blend("a*3+b*1") == [("a", 0.75), ("b", 0.25)]


def test_bootstrap_self_is_zero_and_better_is_positive():
    rng = np.random.default_rng(2)
    n = 4000
    truth = rng.integers(0, 12, n)
    units = rng.integers(0, 1500, n)
    pred = np.where(rng.random(n) < 0.6, truth, rng.integers(0, 12, n))
    same = octlike.bootstrap(units, truth, pred, pred, draws=200)
    assert same == {"delta": 0.0, "lo": 0.0, "hi": 0.0}
    better = np.where(rng.random(n) < 0.5, truth, pred)
    res = octlike.bootstrap(units, truth, pred, better, draws=200)
    assert res["delta"] > 0 and res["lo"] > 0
    assert res == octlike.bootstrap(units, truth, pred, better, draws=200)


def test_train_target_modes_only_touch_fit_rows():
    rng = np.random.default_rng(0)
    n = 60
    y = np.zeros(n, dtype=np.int64)
    ev = np.full(n, -1, dtype=np.int64)
    ev[:10] = 0
    y[:7] = 3  # an event mostly labelled 3, three legit rows inside it
    ev[30:40] = 1
    y[30:37] = 3  # the same pattern outside the fit rows
    fit_rows = np.zeros(n, dtype=bool)
    fit_rows[:20] = True
    events = [ev, np.full(n, -1, dtype=np.int64)]
    assert np.array_equal(octlike.train_target({}, y, fit_rows), y)
    for mode in ("event_clean", "event_l2"):
        out = octlike.train_target({"train_on": mode}, y, fit_rows, events=lambda: events)
        assert np.array_equal(out[~fit_rows], y[~fit_rows])
        assert (out[7:10] == 3).all()
    cl = rng.integers(0, 12, n)
    out = octlike.train_target({"train_on": "cleaned"}, y, fit_rows, cleaned=lambda: cl)
    assert np.array_equal(out[fit_rows], cl[fit_rows])
    assert np.array_equal(out[~fit_rows], y[~fit_rows])
    with pytest.raises(ValueError):
        octlike.train_target({"train_on": "nonsense"}, y, fit_rows)


def test_catboost_gpu_ram_cap_from_environment(monkeypatch):
    from ringtrace.cascade import CatBoostSpec

    monkeypatch.delenv("RINGTRACE_CATBOOST_GPU_RAM", raising=False)
    assert "gpu_ram_part" not in CatBoostSpec().params(0, "Logloss")
    monkeypatch.setenv("RINGTRACE_CATBOOST_GPU_RAM", "0.4")
    assert CatBoostSpec().params(0, "Logloss")["gpu_ram_part"] == 0.4
    assert "gpu_ram_part" not in CatBoostSpec(task_type="CPU").params(0, "Logloss")
