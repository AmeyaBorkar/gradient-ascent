"""October-like mode: fit on July + August of a thinned frame, predict its September.

A frame directory (e.g. ``~/data-oct/s13``) holds a ``train.csv`` whose September is thinned to
October's density, ``rows.csv`` (each line's row in the original train-then-test stacking) and
links to the original test, accounts and merchants files.

    # once, with the normal environment: the unthinned device counter, keyed by stacked row
    python -m ringtrace.octlike export-counter
    # per frame: cells in order, e.g. the given-label champion for seed sets 0-2, then cleaned
    python -m ringtrace.octlike run ~/data-oct/s13 champion:0,1,2 champion-cleantrain:0

Protocol of each cell (recipe, seed set r):

* every feature family is built on the frame's own ledger, except the device counter, whose
  columns come from the unthinned build joined by stacked row (thinning would break the block
  finder);
* the cascade is fitted on July and August only (its screening score is cross-fitted between
  those two months), with cascade seeds 3r, 3r + 1, 3r + 2;
* for a recipe with ``train_on = cleaned``, the cleaned labels are built on the frame from July and
  August only: the teacher scores July with a model fitted on August and vice versa;
* September's pre-decoder probabilities go to ``<out>/<prefix>_s<seed>_r<r>.npz`` (``rows`` =
  stacked row, ``prob`` = n x 12 in ``CLASSES`` order); September is never trained on.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

COUNTER_NAME = "counter_unthinned.parquet"
FIT_PERIODS = (7, 8)
HOLD_PERIOD = 9
PREFIX = {"champion": "champion", "champion-cleantrain": "cleantrain"}


def _root() -> Path:
    return Path(os.environ.get("RINGTRACE_OCTLIKE_CACHE", "~/cache/octlike")).expanduser()


def export_counter() -> None:
    from ringtrace import store

    led = store.ledger()
    ctr = store.family("phones")
    assert ctr.height == led.height
    _root().mkdir(parents=True, exist_ok=True)
    target = _root() / COUNTER_NAME
    stacked = pl.Series("stacked_row", led["row"].to_numpy().astype(np.int64))
    ctr.with_columns(stacked).write_parquet(target)
    print(f"[octlike] counter: {ctr.width} columns, {ctr.height} rows -> {target}")


def _features(
    recipe: dict, led: pl.DataFrame, stacked: np.ndarray, log
) -> tuple[np.ndarray, list[str]]:
    """The recipe's feature matrix on the frame, with the counter taken from the unthinned build."""
    from ringtrace import store

    drop = set(recipe.get("drop", []))

    def keep(c: str) -> bool:
        return c not in drop and not c.startswith("_") and not c.endswith("_event")

    blocks, names = [], []
    for fam in recipe["families"]:
        if fam == "phones":
            ctr = pl.read_parquet(_root() / COUNTER_NAME)
            assert (ctr["stacked_row"].to_numpy() == np.arange(ctr.height)).all()
            cols = [c for c in ctr.columns if c != "stacked_row" and keep(c)]
            blocks.append(ctr.select(cols).to_numpy().astype(np.float32)[stacked])
        else:
            df = store.family(fam)
            cols = [c for c in df.columns if keep(c)]
            blocks.append(df.select(cols).to_numpy().astype(np.float32))
        names += cols
    X = np.hstack(blocks)
    log(f"[octlike] {X.shape[1]} features ({time.time() - T0:.0f}s)")
    return X, names


def _cleaned_target(led: pl.DataFrame, fit: np.ndarray, log) -> tuple[np.ndarray, dict]:
    """Cleaned labels for the fit rows (July, August) from a teacher that never sees September."""
    from ringtrace import silver, store, teacher

    feats = store.frame(["context", "marks"])
    cols = [c for c in feats.columns if not c.endswith("_event")]
    Xt = feats.select(cols).to_numpy().astype(np.float32)
    y = led["y"].to_numpy().astype(np.int64)
    period = led["period"].to_numpy()
    proba = teacher.teacher_oof(Xt[fit], y[fit], period[fit])
    events = teacher.union_events(led.height)
    res = silver.build(led, proba, events, log=log, mask=fit)
    return res["label"], res["stats"]


T0 = time.time()


def run(frame: Path, cells: list[tuple[str, list[int]]], out: Path, recipes_dir: Path) -> None:
    global T0
    T0 = time.time()
    os.environ["RINGTRACE_DATA"] = str(frame)
    os.environ.setdefault("RINGTRACE_CACHE", str(_root() / frame.name))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES is not 1")
    from ringtrace import run as runner
    from ringtrace import store
    from ringtrace.cascade import fit_predict
    from ringtrace.decode import decode_by_period
    from ringtrace.scoring import macro_f1

    seed = int(json.loads((frame / "frame.json").read_text())["seed"])
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    n_train = int((~is_test).sum())
    frame_rows = pl.read_csv(frame / "rows.csv")["stacked_row"].to_numpy().astype(np.int64)
    assert len(frame_rows) == n_train, (len(frame_rows), n_train)
    ctr_rows = pl.scan_parquet(_root() / COUNTER_NAME).select(pl.len()).collect().item()
    n_test = led.height - n_train
    stacked = np.concatenate([frame_rows, np.arange(ctr_rows - n_test, ctr_rows)])
    period = led["period"].to_numpy()
    y = led["y"].to_numpy().astype(np.int64)
    fit = ~is_test & np.isin(period, FIT_PERIODS)
    hold = ~is_test & (period == HOLD_PERIOD)
    log(
        f"[octlike] frame s{seed}: ledger {led.height} rows, fit {fit.sum()}, "
        f"September {hold.sum()}, test {n_test}"
    )
    summary_path = out / f"summary_s{seed}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    X_cache: dict[str, np.ndarray] = {}
    for recipe_name, seed_sets in cells:
        recipe = runner.load_recipe(recipes_dir / f"{recipe_name}.json")
        key = json.dumps([recipe["families"], recipe.get("drop", [])])
        if key not in X_cache:
            X_cache.clear()
            X_cache[key], _ = _features(recipe, led, stacked, log)
        X = X_cache[key]
        target = y.copy()
        if recipe.get("train_on") == "cleaned":
            cleaned, stats = _cleaned_target(led, fit, log)
            target[fit] = cleaned[fit]
            summary[f"{PREFIX[recipe_name]}_cleaning"] = stats
            log(
                f"[octlike] cleaned labels on July+August: {stats['changed']} rows changed "
                f"({time.time() - T0:.0f}s)"
            )
        base = runner.cascade_spec(recipe)
        for r in seed_sets:
            name = f"{PREFIX[recipe_name]}_s{seed}_r{r}"
            target_file = out / f"{name}.npz"
            if target_file.exists():
                log(f"[octlike] {name} exists, skipped")
                continue
            base.seeds = (3 * r, 3 * r + 1, 3 * r + 2)
            (prob,) = fit_predict(X[fit], target[fit], period[fit], [X[hold]], base, log=log)
            tmp = out / f".{name}.tmp.npz"
            np.savez(tmp, rows=stacked[hold].astype(np.int64), prob=prob.astype(np.float32))
            tmp.replace(target_file)
            pred = decode_by_period(prob, period[hold], **recipe.get("decoder", {}))
            score = macro_f1(y[hold], pred)
            arg = macro_f1(y[hold], prob.argmax(1))
            summary[name] = {
                "given_fixed_point": score,
                "given_argmax": arg,
                "seeds": list(base.seeds),
                "rows": int(hold.sum()),
            }
            log(
                f"[octlike] {name}: September given macro-F1 fixed point {score:.4f} "
                f"(argmax {arg:.4f}) ({time.time() - T0:.0f}s)"
            )
            summary_path.write_text(json.dumps(summary, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export-counter")
    r = sub.add_parser("run")
    r.add_argument("frame")
    r.add_argument("cells", nargs="+", help="recipe:seedset,seedset ...")
    r.add_argument("--out", default="~/outputs/octlike")
    r.add_argument("--recipes", default="recipes")
    args = ap.parse_args(argv)
    if args.cmd == "export-counter":
        export_counter()
        return
    cells = []
    for c in args.cells:
        name, sets = c.split(":")
        cells.append((name, [int(x) for x in sets.split(",")]))
    run(Path(args.frame).expanduser(), cells, Path(args.out).expanduser(), Path(args.recipes))


if __name__ == "__main__":
    main()
