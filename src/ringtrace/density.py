"""The device counter at October's density: a held-out month's counter rebuilt as its own split.

    python -m ringtrace.density check [--seeds 13,14,15]
    python -m ringtrace.density cells champion:0 tabm:0 [--seeds 13,14,15] [--tag -nopos]
    python -m ringtrace.density lomo <recipe> --name <run>

Production computes the counter on October alone: October's counter pass wrote only October's
rows, at October's half density, so its fraud phase holds about 410 phones and its legit block is
dense. The train split's pass wrote July, August and September together, interleaved number by
number (1,448 month runs over 2,240 fraud-phase phones; 30,013 runs over 45,177 legit-block
numbers). The October-like frames held September's counter at its value on that shared,
full-density pass, so every counter column of a thinned September came from a fit on all three
months and their 2,240 phones, and September's positions sat on the same phase as the fit months'.

``alone`` rebuilds one train month's counter the way production builds October's:

* the month's fraud-phase and legit-block phones are renumbered as a pass of their own: each
  phone keeps the run of unused numbers just below it on the shared line (its own gap), the
  other months' phones and gaps are taken out, and so are the phones of every unit the thinning
  dropped, so the fraud phase keeps its occupancy (about 0.41) and the legit block stays dense;
  the one gap between the fraud phase and the legit block is kept at the shared line's width;
* the renumbered month is its own split (the other train months are split 0 and set only the
  bot-block edge), and the HMM decodes its fraud phase from its own anchors, read on the frame.

``honest_phones(frame)`` is the frame's counter family with September's rows rebuilt this way;
``lomo_phones()`` gives every train month its own counter (full density), so a held-out month's
counter is built the way October's is. ``cells`` fits a recipe on July + August of each frame once
and predicts September twice, with the held counter and with the rebuilt one, so the two cells
differ only in September's counter columns. ``lomo`` runs the month folds and predicts each
held-out month with its standalone counter as well. Only aggregates are printed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import paths
from ringtrace.evidence import phones
from ringtrace.tables import HOME_DEVICES

PASS_BLOCKS = (2, 3)  # fraud phase, legit block
BASE = 1_000_000  # renumbered phones live above every real device number
CTR = phones.PREFIX + "_"


@dataclass
class Line:
    """The shared train pass: its numbers (sorted), each number's own gap and block."""

    nums: np.ndarray
    gaps: np.ndarray
    blocks: np.ndarray
    transition: int  # unused numbers between the fraud phase and the legit block


def shared_line(device: np.ndarray, block: np.ndarray, train: np.ndarray) -> Line:
    """The train pass from production's counter: numbers of fraud-phase and legit-block phones."""
    device = np.asarray(device, dtype=np.int64)
    sel = np.asarray(train) & np.isin(block, PASS_BLOCKS) & (device > HOME_DEVICES)
    nums, first = np.unique(device[sel], return_index=True)
    blocks = np.asarray(block)[sel][first].astype(np.int64)
    if len(nums) == 0:
        return Line(nums, np.zeros(0, np.int64), blocks, 0)
    gaps = np.concatenate([[0], np.diff(nums) - 1]).astype(np.int64)
    legit = np.flatnonzero(blocks == 3)
    fraud = np.flatnonzero(blocks == 2)
    transition = 0
    if len(legit) and len(fraud) and legit[0] > fraud[-1]:
        transition = int(gaps[legit[0]])
    return Line(nums, gaps, blocks, transition)


def renumber(line: Line, chosen: np.ndarray, base: int = BASE) -> tuple[np.ndarray, np.ndarray]:
    """New numbers for the ``chosen`` numbers of the line, as a pass of their own: each keeps its
    own gap, the first legit-block phone at least the shared fraud/legit gap. Returns (old, new)."""
    keep = np.isin(line.nums, chosen)
    old = line.nums[keep]
    if len(old) == 0:
        return old, old.copy()
    steps = 1 + line.gaps[keep]
    b = line.blocks[keep]
    switch = np.flatnonzero(np.diff(b) != 0) + 1
    steps[switch] = 1 + np.maximum(line.gaps[keep][switch], line.transition)
    steps[0] = 1
    new = base + np.cumsum(steps) - 1
    return old, new.astype(np.int64)


def alone(
    line: Line,
    device: np.ndarray,
    period: np.ndarray,
    train: np.ndarray,
    anchor: np.ndarray,
    month: int,
) -> dict[str, np.ndarray]:
    """Counter columns of one train month's rows (of the frame given by the arrays) rebuilt as
    their own split. Returns columns over the month's rows, in frame order."""
    device = np.asarray(device, dtype=np.int64)
    train = np.asarray(train, dtype=bool)
    mine = train & (period == month)
    other = train & (period != month)
    dev_m = device[mine]
    on_line = np.isin(dev_m, line.nums)
    old, new = renumber(line, np.unique(dev_m[on_line]))
    dev_new = dev_m.copy()
    dev_new[on_line] = new[np.searchsorted(old, dev_m[on_line])]
    dev = np.concatenate([device[other], dev_new])
    split = np.concatenate([np.zeros(int(other.sum()), np.int64), np.ones(len(dev_m), np.int64)])
    anc = np.concatenate([anchor[other], anchor[mine]]).astype(np.int64)
    cols = phones.counter(dev, split, anc)
    return {k: v[int(other.sum()) :] for k, v in cols.items()}


def _as_frame(cols: dict[str, np.ndarray]) -> pl.DataFrame:
    dtypes = {"block": pl.Int8, "dev_rows": pl.Int32, "dev_single": pl.Int8}
    return pl.DataFrame(
        {f"{CTR}{k}": pl.Series(v, dtype=dtypes.get(k, pl.Float64)) for k, v in cols.items()}
    )


def replace_rows(base: pl.DataFrame, rows: np.ndarray, cols: dict[str, np.ndarray]) -> pl.DataFrame:
    """``base`` (a counter family) with ``rows`` taking the new columns."""
    new = _as_frame(cols)
    out = {}
    for c in base.columns:
        a = base[c].to_numpy().copy()
        a[rows] = new[c].to_numpy().astype(a.dtype)
        out[c] = pl.Series(c, a, dtype=base[c].dtype)
    return pl.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# the store side


def _cache_dir() -> Path:
    d = paths.cache_dir() / "density"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out_dir(name: str) -> Path:
    d = paths.out_dir() / "density" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _production_line() -> Line:
    from ringtrace import store

    store.use_frame(None)
    led = store.ledger()
    ph = store.family("phones")
    return shared_line(
        led["device"].fill_null(-1).to_numpy(),
        ph["ctr_block"].to_numpy(),
        ~led["is_test"].to_numpy(),
    )


def honest_phones(seed: int, line: Line | None = None, log=print) -> pl.DataFrame:
    """The frame's counter family with September rebuilt on its own (cached per frame)."""
    from ringtrace import octlike, store
    from ringtrace.evidence import anchors

    path = _cache_dir() / f"honest-s{seed}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    line = line or _production_line()
    frame = octlike.Frame(seed)  # installs the frame with the held counter
    held = store.family("phones")
    led = frame.led
    anchor = anchors.row_anchor(led)
    train = ~led["is_test"].to_numpy()
    cols = alone(
        line,
        led["device"].fill_null(-1).to_numpy(),
        frame.period,
        train,
        anchor,
        octlike.HOLD_PERIOD,
    )
    out = replace_rows(held, np.flatnonzero(frame.hold), cols)
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    store.use_frame(None)
    log(f"[density] honest counter for frame s{seed}: {int(frame.hold.sum())} September rows")
    return out


def lomo_phones(line: Line | None = None, log=print) -> pl.DataFrame:
    """The ledger's counter family with every train month rebuilt as its own split."""
    from ringtrace import store
    from ringtrace.evidence import anchors

    path = _cache_dir() / "lomo-alone.parquet"
    if path.exists():
        return pl.read_parquet(path)
    line = line or _production_line()
    store.use_frame(None)
    led = store.ledger()
    held = store.family("phones")
    anchor = anchors.row_anchor(led)
    period = led["period"].to_numpy()
    train = ~led["is_test"].to_numpy()
    device = led["device"].fill_null(-1).to_numpy()
    out = held
    for m in (7, 8, 9):
        cols = alone(line, device, period, train, anchor, m)
        out = replace_rows(out, np.flatnonzero(train & (period == m)), cols)
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    log("[density] standalone counter for every train month")
    return out


def half_split(units: np.ndarray, draw: int, month: int) -> np.ndarray:
    """Which rows of a month fall in its first half: whole units, kept with October's keep
    rate (seeded by draw and month); the rest form the second half."""
    from ringtrace import octlike

    return octlike.thin(units, 1000 * (draw + 1) + month)


def half_phones(draw: int, line: Line | None = None, log=print) -> pl.DataFrame:
    """The ledger's counter family with every train month at October's density: the month's
    protected units are split into two halves, each half is a pass of its own (renumbered,
    its own split), and every row takes the counter of its half. October keeps production's."""
    from ringtrace import octlike, store
    from ringtrace.evidence import anchors

    path = _cache_dir() / f"half-d{draw}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    line = line or _production_line()
    store.use_frame(None)
    led = store.ledger()
    held = store.family("phones")
    anchor = anchors.row_anchor(led)
    period = led["period"].to_numpy()
    train = ~led["is_test"].to_numpy()
    device = led["device"].fill_null(-1).to_numpy()
    events = [
        store.family(f)[c].fill_null(-1).to_numpy().astype(np.int64)
        for f, c in zip(octlike.EVENT_FAMILIES, octlike.EVENT_COLUMNS, strict=True)
    ]
    out = held
    for m in (7, 8, 9):
        rows = np.flatnonzero(train & (period == m))
        others = np.flatnonzero(train & (period != m))
        first = half_split(octlike.september_units(led, events, rows), draw, m)
        for half in (first, ~first):
            idx = np.concatenate([others, rows[half]])
            cols = alone(line, device[idx], period[idx], train[idx], anchor[idx], m)
            out = replace_rows(out, rows[half], cols)
        log(
            f"[density] half-density counter d{draw}, month {m}: halves {first.sum()} / "
            f"{(~first).sum()} rows"
        )
    tmp = path.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    return out


def training_counter(kind: str, line: Line | None = None, log=print) -> pl.DataFrame | None:
    """The counter family the fit rows read: None (production), "alone" (each train month
    its own pass) or "half<d>" (each month at October's density, draw d)."""
    if kind in ("", "production"):
        return None
    if kind == "alone":
        return lomo_phones(line, log)
    if kind.startswith("half"):
        return half_phones(int(kind[4:]), line, log)
    raise ValueError(f"unknown training counter {kind!r}")


def _counts(a: np.ndarray) -> dict[int, int]:
    k, n = np.unique(a, return_counts=True)
    return {int(x): int(c) for x, c in zip(k, n, strict=True)}


def agreement(held: pl.DataFrame, honest: pl.DataFrame, rows: np.ndarray) -> dict:
    """How the rebuilt counter differs from the held one on ``rows`` (aggregates only)."""
    hb = held["ctr_block"].to_numpy()[rows]
    nb = honest["ctr_block"].to_numpy()[rows]
    hs = held["ctr_seg"].to_numpy()[rows]
    ns = honest["ctr_seg"].to_numpy()[rows]
    hp = held["ctr_pos"].to_numpy()[rows]
    np_ = honest["ctr_pos"].to_numpy()[rows]
    hc = held["ctr_conf"].to_numpy()[rows]
    nc = honest["ctr_conf"].to_numpy()[rows]
    both = (hb == 2) & (nb == 2)
    out = {
        "rows": int(len(rows)),
        "block_counts_held": _counts(hb),
        "block_counts_alone": _counts(nb),
        "block_agree": float((hb == nb).mean()),
        "fraud_phase_rows_both": int(both.sum()),
        "seg_agree": float((hs[both] == ns[both]).mean()) if both.any() else None,
        "pos_abs_diff_mean": float(np.abs(hp[both] - np_[both]).mean()) if both.any() else None,
        "pos_abs_diff_p90": float(np.quantile(np.abs(hp[both] - np_[both]), 0.9))
        if both.any()
        else None,
        "conf_mean_held": float(np.nanmean(hc[both])) if both.any() else None,
        "conf_mean_alone": float(np.nanmean(nc[both])) if both.any() else None,
        "conf_below_0.8_held": int((hc[both] < 0.8).sum()),
        "conf_below_0.8_alone": int((nc[both] < 0.8).sum()),
    }
    return out


def seg_label_agreement(cols: pl.DataFrame, y: np.ndarray, rows: np.ndarray) -> dict:
    """Share of fraud-phase rows whose segment class equals the given label (train rows)."""
    b = cols["ctr_block"].to_numpy()[rows]
    s = cols["ctr_seg"].to_numpy()[rows]
    yy = y[rows]
    fb = b == 2
    fr = fb & (yy > 0)
    return {
        "fraud_phase_rows": int(fb.sum()),
        "seg_equals_label": float((s[fb] + 1 == yy[fb]).mean()) if fb.any() else None,
        "seg_equals_fraud_label": float((s[fr] + 1 == yy[fr]).mean()) if fr.any() else None,
    }


# ---------------------------------------------------------------------------------------------
# October-mix frames: September thinned with October's class mix


# October against train, per class (calls; the bot factor applies to bot-burst units only)
MIX = {1: 1.22, 2: 0.53, 8: 1.11, 11: 0.63}
BOT_CLASS = 2


def unit_class(units: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Majority fraud label of each unit's fraud-labelled rows (ties: the lower class), 0 for a
    unit with none."""
    n = int(units.max()) + 1 if len(units) else 0
    counts = np.zeros((n, int(y.max(initial=0)) + 1), dtype=np.int64)
    fr = y > 0
    np.add.at(counts, (units[fr], y[fr]), 1)
    counts[:, 0] = 0
    cls = counts.argmax(axis=1)
    cls[counts.sum(axis=1) == 0] = 0
    return cls


def mix_keep(
    units: np.ndarray,
    y: np.ndarray,
    burst: np.ndarray,
    seed: int,
    keep: float,
    mix: dict[int, float] = MIX,
) -> np.ndarray:
    """Row mask: each unit kept whole with probability ``keep`` times its class's factor (the
    bot factor only for units holding a bot burst row)."""
    cls = unit_class(units, y)
    n = len(cls)
    has_burst = np.zeros(n, dtype=bool)
    has_burst[units[burst]] = True
    factor = np.ones(n)
    for c, f in mix.items():
        sel = cls == c
        if c == BOT_CLASS:
            sel &= has_burst
        factor[sel] = f
    rng = np.random.default_rng([seed, 0])
    kept_unit = rng.random(n) < np.minimum(keep * factor, 1.0)
    return kept_unit[units]


def build_mix_frame(seed: int, log=print) -> dict:
    """An October-like frame whose September keeps October's class mix; saved like the
    October-like frames (``octlike.Frame(seed)`` reads it)."""
    from ringtrace import octlike, store

    store.use_frame(None)
    led = store.ledger()
    period = led["period"].to_numpy()
    is_test = led["is_test"].to_numpy()
    sept = np.flatnonzero(~is_test & (period == octlike.HOLD_PERIOD))
    events = [
        store.family(f)[c].fill_null(-1).to_numpy().astype(np.int64)
        for f, c in zip(octlike.EVENT_FAMILIES, octlike.EVENT_COLUMNS, strict=True)
    ]
    units = octlike.september_units(led, events, sept)
    y = led["y"].to_numpy().astype(np.int64)[sept]
    burst = events[0][sept] >= 0
    kept = mix_keep(units, y, burst, seed, octlike.KEEP)
    keep_rows = np.sort(
        np.concatenate([np.flatnonzero(~np.isin(np.arange(led.height), sept)), sept[kept]])
    )
    qr = (led["txn_type"] == "QR").to_numpy() & led["to_person"].to_numpy()
    cand = sept[kept & qr[sept]]
    blanked = octlike.qr_blank(cand, seed)
    unit_of = np.full(led.height, -1, dtype=np.int64)
    unit_of[sept] = units
    res = {
        "seed": seed,
        "rows": keep_rows.astype(np.int64),
        "blank": blanked.astype(np.int64),
        "unit": unit_of[keep_rows],
    }
    d = paths.cache_dir() / "octlike"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"s{seed}.npz", **res)
    labels = {int(c): int((y[kept] == c).sum()) for c in range(12)}
    full = {int(c): int((y == c).sum()) for c in range(12)}
    stats = {
        "mix": {str(k): v for k, v in MIX.items()},
        "september_rows": int(len(sept)),
        "kept_september_rows": int(kept.sum()),
        "kept_share_by_label": {
            str(c): labels[c] / full[c] if full[c] else None for c in range(12)
        },
        "qr_blanked": int(len(blanked)),
        "frame_rows": int(len(keep_rows)),
    }
    (d / f"s{seed}.json").write_text(json.dumps(stats, indent=1))
    log(f"[density] mix frame s{seed}: {json.dumps(stats)}")
    return stats


# ---------------------------------------------------------------------------------------------
# fits


def _ctr_index(cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(cols) if c.startswith(CTR)]


def swap_counter(X: np.ndarray, cols: list[str], rows: np.ndarray, ph: pl.DataFrame) -> np.ndarray:
    """X[rows] with every counter column taken from ``ph`` (a counter family over the frame)."""
    out = X[rows].copy()
    for j in _ctr_index(cols):
        out[:, j] = ph[cols[j]].to_numpy()[rows].astype(np.float32)
    return out


def fit_cells(
    seeds: list[int],
    cells: list[tuple[str, list[int]]],
    recipes_dir: Path,
    tag: str = "",
    train_counter: str = "",
    log=print,
) -> dict:
    """Fit each recipe on July + August of every frame once; write September's probabilities
    with the held counter to ``density/cells-held`` and with the rebuilt one to
    ``density/cells-honest`` (cell name ``<recipe><tag>_s<seed>_r<r>.npz``). ``train_counter``
    (see ``training_counter``) replaces the fit rows' counter columns."""
    from ringtrace import eventclean, octlike, store
    from ringtrace import run as runner
    from ringtrace.scoring import macro_f1

    t0 = time.time()
    out_held, out_honest = _out_dir("cells-held"), _out_dir("cells-honest")
    first = cells[0][0] if cells else "none"
    seed_tag = "-".join(str(s) for s in seeds)
    summary_path = _out_dir("cells") / f"summary-{first}{tag}-s{seed_tag}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    line = _production_line()
    train_ph = training_counter(train_counter, line, log)
    for seed in seeds:
        honest = honest_phones(seed, line, log)
        frame = octlike.Frame(seed)
        held = store.family("phones")
        key = store._KEY

        def use(ph, frame=frame, key=key):
            store.use_frame(frame.led, key=key, fixed={"phones": ph})

        hold = np.flatnonzero(frame.hold)
        y = frame.led["y"].to_numpy().astype(np.int64)
        fit_rows = frame.fit
        for recipe_name, sets in cells:
            recipe = runner.load_recipe(recipes_dir / f"{recipe_name}.json")
            use(held)
            X, cols = runner.feature_matrix(recipe)
            Xh = swap_counter(X, cols, hold, honest)
            Xf = X[fit_rows]
            if train_ph is not None:
                # the fit months are whole in every frame: their stacked rows index the ledger
                fit_idx = np.flatnonzero(fit_rows)
                Xf = swap_counter(X, cols, fit_idx, _rows_of(train_ph, frame.stacked))
            target = octlike.train_target(
                recipe, y, fit_rows, _no_cleaned, eventclean.ledger_events
            )
            spec, fit_predict = runner.model(recipe)
            for r in sets:
                name = f"{recipe_name}{tag}_s{seed}_r{r}"
                if (out_honest / f"{name}.npz").exists() and (out_held / f"{name}.npz").exists():
                    log(f"[density] {name} exists, skipped")
                    continue
                t1 = time.time()
                spec.seeds = (3 * r, 3 * r + 1, 3 * r + 2)
                p_held, p_honest = fit_predict(
                    Xf,
                    target[fit_rows],
                    frame.period[fit_rows],
                    [X[hold], Xh],
                    spec,
                    log=lambda *_: None,
                )
                res: dict = {
                    "seeds": list(spec.seeds),
                    "features": len(cols),
                    "train_counter": train_counter or "production",
                }
                for kind, p, d, ph in (
                    ("held", p_held, out_held, held),
                    ("honest", p_honest, out_honest, honest),
                ):
                    tmp = d / f".{name}.tmp.npz"
                    np.savez(tmp, rows=frame.stacked[hold], prob=p.astype(np.float32))
                    tmp.replace(d / f"{name}.npz")
                    use(ph)
                    pred = octlike.decode(frame, p, recipe.get("decoder"), recipe.get("rules", []))
                    res[kind] = {t: macro_f1(v[frame.hold], pred) for t, v in frame.targets.items()}
                use(held)
                res["wall_s"] = time.time() - t1
                summary[name] = res
                log(
                    f"[density] {name}: "
                    + "; ".join(
                        f"{k} " + ", ".join(f"{t} {v:.4f}" for t, v in res[k].items())
                        for k in ("held", "honest")
                    )
                    + f" ({res['wall_s']:.0f}s, total {time.time() - t0:.0f}s)"
                )
                summary_path.write_text(json.dumps(summary, indent=1))
    store.use_frame(None)
    return summary


def _rows_of(ph: pl.DataFrame, stacked: np.ndarray) -> pl.DataFrame:
    """A ledger-wide counter family taken at a frame's stacked rows."""
    return ph[stacked]


def _no_cleaned():
    raise ValueError("train_on 'cleaned' is not supported here")


def lomo(recipe_path: Path, name: str, train_counter: str = "", log=print) -> dict:
    """Month folds of a recipe; each held-out month predicted with the production counter
    (``oof.npy``) and with its standalone counter (``oof_alone.npy``), October from the fold
    mean (``test.npy``). Written to ``$RINGTRACE_OUT/runs/<name>/``. ``train_counter`` (see
    ``training_counter``) replaces the fit rows' counter columns."""
    from ringtrace import eventclean, store
    from ringtrace import run as runner

    t0 = time.time()
    recipe = runner.load_recipe(recipe_path)
    out = paths.out_dir() / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    alone_ph = lomo_phones(log=log)
    train_ph = training_counter(train_counter, log=log)
    store.use_frame(None)
    led = store.ledger()
    X, cols = runner.feature_matrix(recipe)
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y = led["y"].to_numpy().astype(np.int64)
    tr = np.flatnonzero(~is_test)
    te = np.flatnonzero(is_test)
    spec, fit_fn = runner.model(recipe)
    mode = runner.EVENT_MODES.get(recipe.get("train_on", ""))
    events = eventclean.ledger_events() if mode else None
    n_cls = 12
    oof = np.zeros((len(y), n_cls))
    oof_alone = np.zeros((len(y), n_cls))
    test = np.zeros((len(te), n_cls))
    for m in runner.TRAIN_MONTHS:
        fit = tr[period[tr] != m]
        hold = tr[period[tr] == m]
        target = y
        if mode:
            mask = np.zeros(len(y), dtype=bool)
            mask[fit] = True
            target = eventclean.clean_labels(y, events, mode, mask=mask)
        t1 = time.time()
        a, b, c = fit_fn(
            X[fit] if train_ph is None else swap_counter(X, cols, fit, train_ph),
            target[fit],
            period[fit],
            [X[hold], swap_counter(X, cols, hold, alone_ph), X[te]],
            spec,
            log=lambda *_: None,
        )
        oof[hold], oof_alone[hold] = a, b
        test += c / len(runner.TRAIN_MONTHS)
        log(f"[density] {name} fold {m}: {time.time() - t1:.0f}s")
    np.save(out / "oof.npy", oof[tr].astype(np.float32))
    np.save(out / "oof_alone.npy", oof_alone[tr].astype(np.float32))
    np.save(out / "test.npy", test.astype(np.float32))
    summary = {
        "recipe": recipe,
        "features": cols,
        "train_counter": train_counter or "production",
        "runtime_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"[density] {name} done in {time.time() - t0:.0f}s")
    return summary


# ---------------------------------------------------------------------------------------------
# gates: the October-like cells under either counter, and month folds with the standalone counter


def paired_draws(units, truth, pred_a, pred_b, draws: int, seed: int) -> tuple[float, np.ndarray]:
    """Point change and Poisson-bootstrap draws of macro-F1(b) - macro-F1(a) over units."""
    import scipy.sparse as sp

    from ringtrace.compare import _counts, _macro

    _, g = np.unique(units, return_inverse=True)
    ng = int(g.max()) + 1
    ca, cb = _counts(g, truth, pred_a, ng), _counts(g, truth, pred_b, ng)
    rng = np.random.default_rng(seed)
    w = sp.csr_matrix(rng.poisson(1.0, size=(draws, ng)).astype(np.float64))
    d = _macro(np.asarray((w @ cb).todense())) - _macro(np.asarray((w @ ca).todense()))
    one = np.ones((1, ng))
    point = float(_macro(np.asarray(one @ cb))[0] - _macro(np.asarray(one @ ca))[0])
    return point, d


def cell_calls(recipe: dict, cells_dir: Path, seed: int, r: int, frame) -> np.ndarray:
    """September calls of a final recipe from its members' cells, decoded under the counter the
    store currently serves (fixed point, rules, base-rate rules)."""
    from ringtrace import final, octlike
    from ringtrace import rules as rulebook

    rows, probs = None, []
    for m in recipe["members"]:
        f = np.load(Path(cells_dir) / f"{m['cells']}_s{seed}_r{r}.npz")
        if rows is None:
            rows = f["rows"]
        elif not np.array_equal(rows, f["rows"]):
            raise ValueError(f"{m['cells']}: rows differ in s{seed} r{r}")
        probs.append(f["prob"].astype(np.float64))
    if not np.array_equal(rows, frame.stacked[frame.hold]):
        raise ValueError(f"cells s{seed} r{r} do not match the frame")
    p = final.mix(probs, final.static_weights(recipe), [m["role"] for m in recipe["members"]])
    pred = octlike.decode(frame, p, recipe["decoder"], recipe["rules"])
    if recipe["base_rate_rules"]:
        hold = np.flatnonzero(frame.hold)
        full_pred = np.zeros(frame.led.height, dtype=np.int64)
        full_pred[hold] = pred
        full_p = np.zeros((frame.led.height, p.shape[1]))
        full_p[hold] = p
        names = [
            {k: v for k, v in x.items() if k != "max_october_rows"}
            for x in recipe["base_rate_rules"]
        ]
        pred = rulebook.apply(names, full_pred, full_p, frame.led, log=lambda *_: None)[hold]
    return pred


def cell_gate(
    sides: list[tuple[dict, Path, str]],
    seeds: list[int],
    sets: list[int],
    draws: int = 2000,
    log=print,
) -> dict:
    """Side b against side a in every cell; a side is (final recipe, cells dir, counter), the
    counter "held" or "honest". Per-seed mean change, the pooled 95 % bounds of the mean change
    over the cells, and the scores of both sides."""
    from ringtrace import octlike, store
    from ringtrace.scoring import macro_f1

    (ra, da, ca), (rb, db, cb) = sides
    line = _production_line()
    cells: dict = {}
    pooled: dict = {}
    for seed in seeds:
        honest = honest_phones(seed, line, log)
        frame = octlike.Frame(seed)
        held = store.family("phones")
        key = store._KEY
        units = frame.unit[frame.hold]
        truth = {t: v[frame.hold] for t, v in frame.targets.items()}
        for r in sets:
            out = []
            for rec, d, counter in ((ra, da, ca), (rb, db, cb)):
                ph = honest if counter == "honest" else held
                store.use_frame(frame.led, key=key, fixed={"phones": ph})
                out.append(cell_calls(rec, d, seed, r, frame))
            store.use_frame(frame.led, key=key, fixed={"phones": held})
            ya, yb = out
            cell: dict = {"rows_changed": int((ya != yb).sum())}
            for t, yt in truth.items():
                point, dr = paired_draws(units, yt, ya, yb, draws, seed * 100 + r)
                pooled.setdefault(t, []).append(dr)
                cell[t] = {"a": macro_f1(yt, ya), "b": macro_f1(yt, yb), "delta": point}
            cells[(seed, r)] = cell
            log(
                f"[density] cell s{seed} r{r}: "
                + ", ".join(f"{t} {cell[t]['delta']:+.5f}" for t in truth)
                + f" ({cell['rows_changed']} rows)"
            )
    store.use_frame(None)
    tg = list(pooled)
    rep = {
        "a": [ra["name"], str(da), ca],
        "b": [rb["name"], str(db), cb],
        "per_seed": {
            t: {
                s: float(np.mean([v[t]["delta"] for (s2, _), v in cells.items() if s2 == s]))
                for s in seeds
            }
            for t in tg
        },
        "mean": {t: float(np.mean([v[t]["delta"] for v in cells.values()])) for t in tg},
        "lower_bound": {t: float(np.percentile(np.mean(pooled[t], axis=0), 2.5)) for t in tg},
        "upper_bound": {t: float(np.percentile(np.mean(pooled[t], axis=0), 97.5)) for t in tg},
        "a_mean": {t: float(np.mean([v[t]["a"] for v in cells.values()])) for t in tg},
        "b_mean": {t: float(np.mean([v[t]["b"] for v in cells.values()])) for t in tg},
        "cells": {f"s{s}_r{r}": v for (s, r), v in cells.items()},
    }
    for t in tg:
        log(
            f"[density] {t:<8} a {rep['a_mean'][t]:.4f} b {rep['b_mean'][t]:.4f} per seed "
            + " / ".join(f"{v:+.5f}" for v in rep["per_seed"][t].values())
            + f"  mean {rep['mean'][t]:+.5f} [{rep['lower_bound'][t]:+.5f}, "
            f"{rep['upper_bound'][t]:+.5f}]"
        )
    return rep


def standalone_bench(runs_dir: Path, log=print):
    """A ``final.Bench`` whose rules read every train month's standalone counter (point its
    recipes at runs that hold the standalone predictions)."""
    from ringtrace import final

    bench = final.Bench.from_store(runs_dir, None, log=log)
    alone_ph = lomo_phones(log=log)
    cols = [c for c in bench.led.columns if c.startswith(CTR)]
    bench.led = bench.led.with_columns([alone_ph[c] for c in cols])
    return bench


# ---------------------------------------------------------------------------------------------
# the check: how far the rebuilt counter moves from the held one


def check(seeds: list[int], log=print) -> dict:
    from ringtrace import octlike, store

    line = _production_line()
    report: dict = {
        "line": {
            "numbers": int(len(line.nums)),
            "fraud_phase": int((line.blocks == 2).sum()),
            "transition": line.transition,
        }
    }
    store.use_frame(None)
    led = store.ledger()
    full = store.family("phones")
    y_full = led["y"].to_numpy().astype(np.int64)
    period = led["period"].to_numpy()
    train = ~led["is_test"].to_numpy()
    alone_ph = lomo_phones(line, log)
    for m in (7, 8, 9):
        rows = np.flatnonzero(train & (period == m))
        report[f"lomo_m{m}"] = {
            **agreement(full, alone_ph, rows),
            "shared": seg_label_agreement(full, y_full, rows),
            "alone": seg_label_agreement(alone_ph, y_full, rows),
        }
    for seed in seeds:
        honest = honest_phones(seed, line, log)
        frame = octlike.Frame(seed)
        held = store.family("phones")
        hold = np.flatnonzero(frame.hold)
        yf = frame.targets["given"]
        report[f"s{seed}"] = {
            **agreement(held, honest, hold),
            "held": seg_label_agreement(held, yf, hold),
            "alone": seg_label_agreement(honest, yf, hold),
        }
    store.use_frame(None)
    te = np.flatnonzero(~train)
    b = full["ctr_block"].to_numpy()[te]
    report["october"] = {
        "block_counts": _counts(b),
        "conf_mean": float(np.nanmean(full["ctr_conf"].to_numpy()[te][b == 2])),
    }
    path = _out_dir("check") / "check.json"
    path.write_text(json.dumps(report, indent=1))
    log(json.dumps(report, indent=1))
    return report


def prep_frame(seed: int, log=print) -> None:
    """Build a frame's feature families and its rebuilt counter once, before parallel fits."""
    from ringtrace import octlike, store

    honest_phones(seed, log=log)
    octlike.Frame(seed)
    for name in store.FAMILIES:
        if name != octlike.HELD_FAMILY:
            store.family(name)
    store.use_frame(None)
    log(f"[density] frame s{seed} ready")


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None):
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.density")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--seeds", default="13,14,15")
    mx = sub.add_parser("mixframes", help="October-mix frames (new thinning seeds)")
    mx.add_argument("--seeds", default="23,24,25")
    pf = sub.add_parser("prepframes", help="build every family and the rebuilt counter")
    pf.add_argument("--seeds", default="23,24,25")
    f = sub.add_parser("cells")
    f.add_argument("cells", nargs="+", help="recipe:seedset,seedset ...")
    f.add_argument("--seeds", default="13,14,15")
    f.add_argument("--recipes", default="recipes")
    f.add_argument("--tag", default="")
    f.add_argument("--train-counter", default="", help="production, alone or half<d>")
    lo = sub.add_parser("lomo")
    lo.add_argument("recipe")
    lo.add_argument("--name", required=True)
    lo.add_argument("--train-counter", default="", help="production, alone or half<d>")
    g = sub.add_parser("cellgate", help="side b against side a on the October-like cells")
    g.add_argument("a", help="recipe.json:cells_dir:held|honest")
    g.add_argument("b", help="recipe.json:cells_dir:held|honest")
    g.add_argument("--seeds", default="13,14,15")
    g.add_argument("--sets", default="0")
    g.add_argument("--draws", type=int, default=2000)
    g.add_argument("--report", required=True)
    t = sub.add_parser("table", help="final.table on the month folds, standalone counter rules")
    t.add_argument("recipes", nargs="+")
    t.add_argument("--draws", type=int, default=2000)
    t.add_argument("--report", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return check(_ints(args.seeds))
    if args.cmd == "prepframes":
        return [prep_frame(sd) for sd in _ints(args.seeds)]
    if args.cmd == "mixframes":
        if set(_ints(args.seeds)) & {13, 14, 15}:
            raise ValueError("seeds 13-15 are the October-like frames; pick others")
        return [build_mix_frame(sd) for sd in _ints(args.seeds)]
    if args.cmd == "cells":
        cells = []
        for spec in args.cells:
            name, _, sets = spec.partition(":")
            cells.append((name, _ints(sets) if sets else [0]))
        return fit_cells(_ints(args.seeds), cells, Path(args.recipes), args.tag, args.train_counter)
    if args.cmd == "lomo":
        return lomo(Path(args.recipe), args.name, args.train_counter)
    from ringtrace import final

    if args.cmd == "cellgate":
        sides = []
        for spec in (args.a, args.b):
            rec, d, counter = spec.rsplit(":", 2)
            if counter not in ("held", "honest"):
                raise ValueError(f"counter must be held or honest: {spec}")
            sides.append((final.load_recipe(rec), Path(d), counter))
        rep = cell_gate(sides, _ints(args.seeds), _ints(args.sets), args.draws)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=1))
        return rep
    bench = standalone_bench(paths.out_dir() / "runs")
    recipes = [final.load_recipe(p) for p in args.recipes]
    rep = final.table(bench, recipes[0], recipes[1:], args.draws, 1000)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rep, indent=1, default=final._json_default))
    return rep


if __name__ == "__main__":
    main()
