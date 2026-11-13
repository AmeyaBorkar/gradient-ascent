"""Training-technique experiments for TabM: one fit on July + August per variant and cell.

    python -m ringtrace.nnexp teacher [--cells sep,s13,s14,s15]
    python -m ringtrace.nnexp fit <variant> [--cells sep,s13,s14,s15] [--seeds 0,1,2]
    python -m ringtrace.nnexp score <variant> [--base base] [--cells ...]

Cells: ``sep`` is the normal ledger with September held out (the leave-one-month-out fold 9);
``s13``..``s15`` are the October-like frames (``ringtrace.octlike``). Every variant is
``recipes/tabm.json`` plus the spec changes in ``VARIANTS``. September's probabilities go to
``$RINGTRACE_OUT/nntech/<variant>_<cell>.npz`` (``rows``, ``prob``); seeds other than 0-2 add
``-sd<seeds>`` to the name, which ``score`` then takes as the variant name.

``teacher`` cross-fits the champion recipe between July and August (fit one, predict the
other) on each cell's fit rows: out-of-fold soft targets that never see September.

``score`` judges a variant against the base twice, with the same seeds: TabM alone, and
fraud-only TabM in the tree blend (legit column from TabM at 1/2 plus the trees at 1/2, fraud
types from the trees, see ``blend.fraud_split_from``). Trees: the five-way blend's fold-9
probabilities on ``sep``; on the October-like cells, the tree members whose r0 cells exist.
Calls: fixed point, then the template rule and the blank merchant QR rule. Aggregates only.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import paths

VARIANTS: dict[str, dict] = {
    "base": {},
    "kd30": {"distill_alpha": 0.3},
    "kd70": {"distill_alpha": 0.7},
    "pre": {"pretrain_epochs": 3},
    "qraug": {"qr_blank_p": 0.44},
    "mixup": {"mixup_alpha": 0.2},
    "noise": {"noise_std": 0.05},
    "ema": {"ema_decay": 0.999},
    "swa": {"swa_last": 3},
    "logitadj": {"loss": "logit_adjust", "logit_tau": 1.0},
    "focal": {"loss": "focal", "focal_gamma": 2.0},
    "smooth": {"smooth_fraud": 0.03},
}
TREES_SEP = ("blend-champion-m1", "lgbm", "catboost", "clean-event", "clean-l2")
TREES_OCT = ("champion", "lgbm", "catboost", "clean-event", "clean-l2")
RULES = ["anchored_templates", "blank_merchant_qr"]
FIT_PERIODS = (7, 8)


def variant_spec(name: str, recipe: dict):
    """The TabM spec of a variant: the recipe's, then ``VARIANTS[name]`` (``a+b`` combines)."""
    from ringtrace import run

    spec, _ = run.model(recipe)
    for part in name.split("+"):
        spec = replace(spec, **VARIANTS[part])
    return spec


def _dir() -> Path:
    d = paths.out_dir() / "nntech"
    d.mkdir(parents=True, exist_ok=True)
    return d


class SepCell:
    """The normal ledger, fit on July + August, September held out."""

    def __init__(self):
        from ringtrace import silver, store

        store.use_frame(None)
        led = store.ledger()
        self.led = led
        self.period = led["period"].to_numpy()
        is_test = led["is_test"].to_numpy()
        self.fit = ~is_test & np.isin(self.period, FIT_PERIODS)
        self.hold = ~is_test & (self.period == 9)
        self.stacked = np.arange(led.height)
        self.unit = np.arange(led.height)
        self.targets = {"given": led["y"].to_numpy().astype(np.int64)}
        for tname, which in (("cleaned", "label"), ("gen", "structured")):
            arr = silver.load(led, which)
            if arr is not None:
                self.targets[tname] = arr.astype(np.int64)


def open_cell(name: str):
    from ringtrace.octlike import Frame

    return SepCell() if name == "sep" else Frame(int(name.lstrip("s")))


def qr_person(led: pl.DataFrame) -> np.ndarray:
    """QR payments to a person whose check is filled (the rows October blanks)."""
    qr = (led["txn_type"] == "QR").to_numpy() & led["to_person"].to_numpy()
    return qr & led["qr_vpa_mismatch"].fill_nan(None).is_not_null().to_numpy()


def blanked_matrix(recipe: dict) -> np.ndarray:
    """The cell's feature matrix with the QR check blanked on every QR-to-person row (every
    family rebuilt on that ledger under its own cache key), then the cell's ledger restored."""
    from ringtrace import run, store

    led = store.ledger()
    saved = (led, store._KEY, dict(store._FIXED))
    blank = pl.Series(qr_person(led))
    led_b = led.with_columns(
        pl.when(blank).then(None).otherwise(pl.col("qr_vpa_mismatch")).alias("qr_vpa_mismatch")
    )
    store.use_frame(led_b, key=f"{store._ledger_key()}-qrblank", fixed=saved[2])
    try:
        X, _ = run.feature_matrix(recipe)
    finally:
        store.use_frame(*saved)
    return X


def cross_fit_teacher(X: np.ndarray, y: np.ndarray, period: np.ndarray, recipe: dict, log):
    """Out-of-fold probabilities between the fit months: fit on all but one, predict it."""
    from ringtrace import run

    spec, fit_predict = run.model(recipe)
    out = np.zeros((len(y), 12), dtype=np.float64)
    months = np.unique(period)
    for m in months:
        tr, ho = period != m, period == m
        (p,) = fit_predict(X[tr], y[tr], period[tr], [X[ho]], spec, log=log)
        out[ho] = p
    return out


def teacher(cells: list[str], log=print) -> None:
    from ringtrace import run

    recipe = run.load_recipe(Path("recipes") / "champion.json")
    for c in cells:
        path = _dir() / f"teacher_{c}.npy"
        if path.exists():
            log(f"[nnexp] {path.name} exists, skipped")
            continue
        t0 = time.time()
        cell = open_cell(c)
        X, _ = run.feature_matrix(recipe)
        y = cell.targets["given"]
        p = cross_fit_teacher(X[cell.fit], y[cell.fit], cell.period[cell.fit], recipe, log)
        np.save(path, p.astype(np.float32))
        log(f"[nnexp] teacher {c}: {int(cell.fit.sum())} rows ({time.time() - t0:.0f}s)")


def fit(variant: str, cells: list[str], seeds: tuple[int, ...], log=print) -> None:
    from ringtrace import run, tabm

    recipe = run.load_recipe(Path("recipes") / "tabm.json")
    fit_predict = tabm.fit_predict
    if variant == "realmlp":  # the second neural family, on the same features
        from ringtrace import realmlp

        spec, fit_predict = realmlp.RealMLPSpec(seeds=seeds), realmlp.fit_predict
    else:
        spec = replace(variant_spec(variant, recipe), seeds=seeds)
    tag = variant if seeds == (0, 1, 2) else f"{variant}-sd{''.join(map(str, seeds))}"
    for c in cells:
        path = _dir() / f"{tag}_{c}.npz"
        if path.exists():
            log(f"[nnexp] {path.name} exists, skipped")
            continue
        t0 = time.time()
        cell = open_cell(c)
        X, _ = run.feature_matrix(recipe)
        y = cell.targets["given"]
        ex = tabm.Extras()
        if getattr(spec, "distill_alpha", 0) > 0:
            ex.teacher = np.load(_dir() / f"teacher_{c}.npy")
        if getattr(spec, "qr_blank_p", 0) > 0:
            ex.X_blank = blanked_matrix(recipe)[cell.fit]
            ex.blank_rows = np.flatnonzero(qr_person(cell.led)[cell.fit])
            log(f"[nnexp] {c}: {len(ex.blank_rows)} QR-to-person fit rows can be blanked")
        if getattr(spec, "pretrain_epochs", 0) > 0:
            ex.X_unlabelled = X[~cell.fit]  # September and October: inputs only
        (prob,) = fit_predict(
            X[cell.fit],
            y[cell.fit],
            cell.period[cell.fit],
            [X[cell.hold]],
            spec,
            log=log,
            extras=ex,
        )
        np.savez(path, rows=cell.stacked[cell.hold], prob=prob.astype(np.float32))
        log(f"[nnexp] {tag} {c} done ({time.time() - t0:.0f}s)")


def tree_prob(cell_name: str, cell) -> np.ndarray:
    """Equal-weight mean of the tree members' September probabilities in this cell."""
    if cell_name == "sep":
        from ringtrace.blend import blend_arrays

        tr = np.flatnonzero(~cell.led["is_test"].to_numpy())
        oof = blend_arrays([(n, 1 / len(TREES_SEP)) for n in TREES_SEP])
        return oof[np.isin(tr, np.flatnonzero(cell.hold))]
    seed = int(cell_name.lstrip("s"))
    probs = []
    for n in TREES_OCT:
        f = paths.out_dir() / "octlike" / f"{n}_s{seed}_r0.npz"
        if f.exists():
            z = np.load(f)
            if not np.array_equal(z["rows"], cell.stacked[cell.hold]):
                raise ValueError(f"{n}: rows differ in cell {cell_name}")
            probs.append(z["prob"].astype(np.float64))
    if not probs:
        raise FileNotFoundError(f"no tree cells for {cell_name}")
    return np.mean(probs, axis=0)


def fraud_only(tabm_prob: np.ndarray, trees: np.ndarray) -> np.ndarray:
    from ringtrace.blend import fraud_split_from

    return fraud_split_from(trees, 0.5 * tabm_prob + 0.5 * trees)


def _decode(cell, prob: np.ndarray) -> np.ndarray:
    from ringtrace.octlike import decode

    return decode(cell, prob, None, RULES)


def score(variant: str, base: str, cells: list[str], draws: int = 1000, log=print) -> dict:
    from ringtrace.octlike import bootstrap
    from ringtrace.scoring import macro_f1

    report: dict = {"variant": variant, "base": base, "cells": {}}
    for c in cells:
        cell = open_cell(c)
        load = [np.load(_dir() / f"{v}_{c}.npz") for v in (base, variant)]
        for z in load:
            if not np.array_equal(z["rows"], cell.stacked[cell.hold]):
                raise ValueError(f"cell {c}: rows differ")
        pa, pb = (z["prob"].astype(np.float64) for z in load)
        trees = tree_prob(c, cell)
        res: dict = {}
        units = cell.unit[cell.hold]
        for role, (qa, qb) in {
            "alone": (pa, pb),
            "fraud_only": (fraud_only(pa, trees), fraud_only(pb, trees)),
        }.items():
            ya, yb = _decode(cell, qa), _decode(cell, qb)
            for t, truth in cell.targets.items():
                yt = truth[cell.hold]
                r = {"base": macro_f1(yt, ya), "variant": macro_f1(yt, yb)}
                if c != "sep":
                    r.update(bootstrap(units, yt, ya, yb, draws, seed=int(c[1:])))
                else:
                    r["delta"] = r["variant"] - r["base"]
                res.setdefault(role, {})[t] = r
            res[role]["changed_calls"] = int((ya != yb).sum())
        report["cells"][c] = res
        log(
            f"[nnexp] {variant} vs {base} {c}: "
            + "  ".join(
                f"{role} "
                + " ".join(f"{t} {res[role][t]['delta']:+.5f}" for t in cell.targets)
                + f" ({res[role]['changed_calls']} calls)"
                for role in res
            )
        )
    (_dir() / f"score_{variant}_vs_{base}.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> None:
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.nnexp")
    ap.add_argument("cmd", choices=["teacher", "fit", "score"])
    ap.add_argument("variant", nargs="?", default="base")
    ap.add_argument("--cells", default="sep,s13,s14,s15")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--base", default="base")
    args = ap.parse_args(argv)
    cells = [c for c in args.cells.split(",") if c]
    if args.cmd == "teacher":
        teacher(cells)
    elif args.cmd == "fit":
        for v in args.variant.split(","):
            fit(v, cells, tuple(int(s) for s in args.seeds.split(",")))
    else:
        for v in args.variant.split(","):
            score(v, args.base, cells)


if __name__ == "__main__":
    main()
