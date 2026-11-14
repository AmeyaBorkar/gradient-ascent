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


def install_honest(frame, honest: pl.DataFrame) -> None:
    """Serve the frame's ledger with the rebuilt counter (other families as cached)."""
    from ringtrace import store

    store.use_frame(frame.led, key=store._KEY, fixed={"phones": honest})


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
    log=print,
) -> dict:
    """Fit each recipe on July + August of every frame once; write September's probabilities
    with the held counter to ``density/cells-held`` and with the rebuilt one to
    ``density/cells-honest`` (cell name ``<recipe><tag>_s<seed>_r<r>.npz``)."""
    from ringtrace import eventclean, octlike, store
    from ringtrace import run as runner
    from ringtrace.scoring import macro_f1

    t0 = time.time()
    out_held, out_honest = _out_dir("cells-held"), _out_dir("cells-honest")
    summary_path = _out_dir("cells") / f"summary{tag}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    line = _production_line()
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
                    X[fit_rows],
                    target[fit_rows],
                    frame.period[fit_rows],
                    [X[hold], Xh],
                    spec,
                    log=lambda *_: None,
                )
                res: dict = {"seeds": list(spec.seeds), "features": len(cols)}
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


def _no_cleaned():
    raise ValueError("train_on 'cleaned' is not supported here")


def lomo(recipe_path: Path, name: str, log=print) -> dict:
    """Month folds of a recipe; each held-out month predicted with the production counter
    (``oof.npy``) and with its standalone counter (``oof_alone.npy``), October from the fold
    mean (``test.npy``). Written to ``$RINGTRACE_OUT/runs/<name>/``."""
    from ringtrace import eventclean, store
    from ringtrace import run as runner

    t0 = time.time()
    recipe = runner.load_recipe(recipe_path)
    out = paths.out_dir() / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    alone_ph = lomo_phones(log=log)
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
            X[fit],
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
    summary = {"recipe": recipe, "features": cols, "runtime_s": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"[density] {name} done in {time.time() - t0:.0f}s")
    return summary


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


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None):
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.density")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--seeds", default="13,14,15")
    f = sub.add_parser("cells")
    f.add_argument("cells", nargs="+", help="recipe:seedset,seedset ...")
    f.add_argument("--seeds", default="13,14,15")
    f.add_argument("--recipes", default="recipes")
    f.add_argument("--tag", default="")
    lo = sub.add_parser("lomo")
    lo.add_argument("recipe")
    lo.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return check(_ints(args.seeds))
    if args.cmd == "cells":
        cells = []
        for spec in args.cells:
            name, _, sets = spec.partition(":")
            cells.append((name, _ints(sets) if sets else [0]))
        return fit_cells(_ints(args.seeds), cells, Path(args.recipes), args.tag)
    return lomo(Path(args.recipe), args.name)


if __name__ == "__main__":
    main()
