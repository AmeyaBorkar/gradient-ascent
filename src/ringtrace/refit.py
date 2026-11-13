"""More training data: the learning curve, fold mean against refit, and refit October predictions.

    python -m ringtrace.refit curve recipes/champion.json
    python -m ringtrace.refit october recipes/<member>.json

``curve`` (screening off in every arm, so one-month and two-month fits stay comparable): for each
held-out month, fit the cascade on each other month alone ("one"), average those two models'
probabilities ("avg_one", the fold-mean analogue), and fit on both months at the base rounds
("two") and at 1.5x the rounds ("two_x1.5"). Every arm is decoded as the run decodes (fixed point
per month, then the recipe's rules) and scored on the given, cleaned and gen-only labels. Writes
``runs/<prefix>-curve/summary.json``.

``october``: fit once on every train row (screening cross-fitted over the three months as usual),
with the booster rounds scaled by (train rows / mean fold train rows), and predict October. Writes
``runs/<member>-refit/`` (refit ``test.npy`` and the member's ``oof.npy`` copied unchanged) and
``runs/<member>-hedge/`` (the 1:1 mean of the fold-mean and refit October probabilities).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace

import numpy as np

from ringtrace import paths, store
from ringtrace.cascade import fit_predict
from ringtrace.classes import CLASSES
from ringtrace.decode import decode_by_period
from ringtrace.run import EVENT_MODES, TRAIN_MONTHS, cascade_spec, feature_matrix, load_recipe
from ringtrace.scoring import macro_f1, per_class_f1

PREFIX = "refit"


def scaled_rounds(base: int, n_all: int, fold_sizes: list[int]) -> int:
    """Rounds for a fit on ``n_all`` rows, scaled from ``base`` at the mean fold size."""
    return int(round(base * n_all / float(np.mean(fold_sizes))))


def hedge(fold_mean: np.ndarray, refit: np.ndarray) -> np.ndarray:
    """1:1 mean of two probability matrices, renormalised per row."""
    h = 0.5 * (np.asarray(fold_mean, np.float64) + np.asarray(refit, np.float64))
    return h / h.sum(axis=1, keepdims=True)


def argmax_changes(a: np.ndarray, b: np.ndarray) -> dict:
    """Aggregate count of rows whose argmax differs, and the class counts under each."""
    pa, pb = a.argmax(1), b.argmax(1)
    k = a.shape[1]
    return {
        "changed": int((pa != pb).sum()),
        "rows": len(pa),
        "counts_a": np.bincount(pa, minlength=k).tolist(),
        "counts_b": np.bincount(pb, minlength=k).tolist(),
    }


def decode_rows(
    proba: np.ndarray, rows: np.ndarray, period: np.ndarray, recipe: dict, led, n: int, log=print
) -> np.ndarray:
    """Decode ``proba`` (one row per index in ``rows``) as the run does: fixed point per month,
    then the recipe's rules, which act on each row from label-free columns only."""
    dec = recipe.get("decoder", {"fraud_scale": 0.875})
    pred = decode_by_period(proba, period[rows], **dec)
    rules = recipe.get("rules", [])
    if not rules:
        return pred
    from ringtrace import rules as rulebook

    full_pred = np.zeros(n, dtype=np.int64)
    full_proba = np.zeros((n, proba.shape[1]))
    full_proba[:, 0] = 1.0
    full_pred[rows] = pred
    full_proba[rows] = proba
    return rulebook.apply(rules, full_pred, full_proba, led, log=log)[rows]


def _targets(led, rows: np.ndarray) -> dict[str, np.ndarray]:
    from ringtrace import silver

    out = {"given": led["y"].to_numpy().astype(np.int64)[rows]}
    for name, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            out[name] = arr[rows]
    return out


def train_labels(recipe: dict, led, mask: np.ndarray, log=print) -> np.ndarray:
    """The recipe's training labels, with only ``mask`` rows voting (event-cleaned recipes)."""
    y = led["y"].to_numpy().astype(np.int64)
    on = recipe.get("train_on")
    if on in (None, "given"):
        return y
    if on == "cleaned":
        from ringtrace import silver

        out = y.copy()
        out[mask] = silver.load(led, "label")[mask]
        return out
    if on in EVENT_MODES:
        from ringtrace import eventclean

        out = eventclean.clean_labels(y, eventclean.ledger_events(), EVENT_MODES[on], mask=mask)
        log(f"  {EVENT_MODES[on]} labels: {int((out != y)[mask].sum())} rows changed")
        return out
    raise ValueError(f"unknown train_on {on!r}")


def curve(recipe: dict, log=print) -> dict:
    t0 = time.time()
    led = store.ledger()
    X, _ = feature_matrix(recipe)
    period = led["period"].to_numpy()
    is_test = led["is_test"].to_numpy()
    n = led.height
    spec = replace(cascade_spec(recipe), use_screen=False)
    base = spec.booster.rounds
    long_spec = replace(spec, booster=replace(spec.booster, rounds=int(round(1.5 * base))))
    report: dict = {"recipe": recipe["name"], "base_rounds": base, "months": {}}
    for m in TRAIN_MONTHS:
        hold = np.flatnonzero(~is_test & (period == m))
        others = [o for o in TRAIN_MONTHS if o != m]
        targets = _targets(led, hold)
        probs: dict[str, np.ndarray] = {}
        for o in others:
            fit = np.flatnonzero(~is_test & (period == o))
            y = train_labels(recipe, led, np.isin(np.arange(n), fit), log=log)
            log(f"hold {m}: one month {o} ({len(fit)} rows)")
            probs[f"one_{o}"] = fit_predict(X[fit], y[fit], period[fit], [X[hold]], spec, log=log)[
                0
            ]
        fit = np.flatnonzero(~is_test & np.isin(period, others))
        y = train_labels(recipe, led, np.isin(np.arange(n), fit), log=log)
        log(f"hold {m}: two months ({len(fit)} rows)")
        probs["two"] = fit_predict(X[fit], y[fit], period[fit], [X[hold]], spec, log=log)[0]
        probs["two_x1.5"] = fit_predict(X[fit], y[fit], period[fit], [X[hold]], long_spec, log=log)[
            0
        ]
        probs["avg_one"] = 0.5 * (probs[f"one_{others[0]}"] + probs[f"one_{others[1]}"])
        month: dict = {}
        for arm, p in probs.items():
            pred = decode_rows(p, hold, period, recipe, led, n, log=lambda *_: None)
            month[arm] = {
                t: {
                    "macro": macro_f1(yt, pred),
                    "per_class": dict(
                        zip(CLASSES, map(float, per_class_f1(yt, pred)), strict=True)
                    ),
                }
                for t, yt in targets.items()
            }
            log(
                f"  hold {m} {arm:<9} "
                + "  ".join(f"{t} {v['macro']:.4f}" for t, v in month[arm].items())
            )
        report["months"][str(m)] = month
    report["runtime_s"] = time.time() - t0
    out = paths.out_dir() / "runs" / f"{PREFIX}-curve"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    log(f"done in {report['runtime_s']:.0f}s -> {out}")
    return report


def october(recipe: dict, log=print) -> dict:
    t0 = time.time()
    name = recipe["name"]
    runs = paths.out_dir() / "runs"
    fold_dir = runs / name
    fold_test = np.load(fold_dir / "test.npy")
    led = store.ledger()
    X, _ = feature_matrix(recipe)
    period = led["period"].to_numpy()
    is_test = led["is_test"].to_numpy()
    tr = np.flatnonzero(~is_test)
    te = np.flatnonzero(is_test)
    folds = [int((period[tr] != m).sum()) for m in TRAIN_MONTHS]
    spec = cascade_spec(recipe)
    rounds = scaled_rounds(spec.booster.rounds, len(tr), folds)
    spec = replace(spec, booster=replace(spec.booster, rounds=rounds))
    if spec.stages == "catboost":  # CatBoost stages count their own iterations
        its = scaled_rounds(spec.catboost.iterations, len(tr), folds)
        spec = replace(spec, catboost=replace(spec.catboost, iterations=its))
        log(f"[{name}] CatBoost stages: {its} iterations")
    log(f"[{name}] refit on {len(tr)} rows, {rounds} rounds (folds {folds})")
    y = train_labels(recipe, led, ~is_test, log=log)
    refit = fit_predict(X[tr], y[tr], period[tr], [X[te]], spec, log=log)[0]
    hedged = hedge(fold_test, refit)
    report = {"recipe": name, "rounds": rounds, "fold_rows": folds, "train_rows": len(tr)}
    oof = fold_dir / "oof.npy"
    n = led.height
    calls = {"fold_mean": decode_rows(fold_test, te, period, recipe, led, n, log=lambda *_: None)}
    for kind, p in (("refit", refit), ("hedge", hedged)):
        d = runs / f"{name}-{kind}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "test.npy", p.astype(np.float32))
        shutil.copyfile(oof, d / "oof.npy")
        calls[kind] = decode_rows(p, te, period, recipe, led, n, log=lambda *_: None)
        report[kind] = {
            "argmax_vs_fold_mean": argmax_changes(fold_test, p),
            "decoded_changed": int((calls[kind] != calls["fold_mean"]).sum()),
            "decoded_counts": np.bincount(calls[kind], minlength=len(CLASSES)).tolist(),
        }
        log(
            f"  {kind}: argmax changed {report[kind]['argmax_vs_fold_mean']['changed']}, "
            f"decoded calls changed {report[kind]['decoded_changed']} of {len(te)}"
        )
    report["fold_mean_decoded_counts"] = np.bincount(
        calls["fold_mean"], minlength=len(CLASSES)
    ).tolist()
    report["runtime_s"] = time.time() - t0
    for kind in ("refit", "hedge"):
        (runs / f"{name}-{kind}" / "summary.json").write_text(json.dumps(report, indent=1))
    log(f"done in {report['runtime_s']:.0f}s")
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("curve", "october"))
    ap.add_argument("recipe")
    ap.add_argument("--gpu-ram-part", type=float, default=None, help="cap CatBoost GPU memory")
    args = ap.parse_args(argv)
    recipe = load_recipe(args.recipe)
    if args.gpu_ram_part is not None:
        recipe.setdefault("cascade", {}).setdefault("catboost", {})["gpu_ram_part"] = (
            args.gpu_ram_part
        )
    return curve(recipe) if args.mode == "curve" else october(recipe)


if __name__ == "__main__":
    main()
