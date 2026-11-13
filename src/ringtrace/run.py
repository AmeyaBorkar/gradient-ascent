"""Run one recipe end to end: features -> month folds -> cascade -> decoder -> rules -> scores.

    python -m ringtrace.run recipes/<name>.json [--no-test] [--name <run>]

Writes under ``$RINGTRACE_OUT/runs/<name>/``: ``oof.npy`` (train-row probabilities from the
month folds), ``test.npy`` (October probabilities, mean of the fold models), ``summary.json``
(per-month scores on the given and cleaned labels, stage by stage) and ``submission.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import paths, store
from ringtrace.cascade import CascadeSpec, CatBoostSpec, booster_spec, fit_predict
from ringtrace.classes import CLASSES
from ringtrace.decode import decode_by_period
from ringtrace.scoring import macro_f1, per_class_f1

TRAIN_MONTHS = (7, 8, 9)
EVENT_MODES = {"event_clean": "clean", "event_l2": "l2"}


def load_recipe(path: str | Path) -> dict:
    with open(path) as fh:
        recipe = json.load(fh)
    recipe.setdefault("name", Path(path).stem)
    return recipe


def cascade_spec(recipe: dict) -> CascadeSpec:
    c = dict(recipe.get("cascade", {}))
    booster = booster_spec(c.pop("booster", {}))
    if "screen" in c:
        c["screen"] = booster_spec(c["screen"])
    if "catboost" in c:
        c["catboost"] = CatBoostSpec(**c["catboost"])
    if "seeds" in c:
        c["seeds"] = tuple(c["seeds"])
    return CascadeSpec(booster=booster, **c)


def model(recipe: dict):
    """(spec, fit_predict) of the recipe's model: the cascade, or ``"model": "tabm"``."""
    if recipe.get("model") == "tabm":
        from ringtrace import tabm

        t = dict(recipe.get("tabm", {}))
        if "seeds" in t:
            t["seeds"] = tuple(t["seeds"])
        return tabm.TabMSpec(**t), tabm.fit_predict
    return cascade_spec(recipe), fit_predict


def feature_matrix(recipe: dict) -> tuple[np.ndarray, list[str]]:
    feats = store.frame(recipe["families"])
    drop = set(recipe.get("drop", []))
    cols = [
        c
        for c in feats.columns
        if c not in drop and not c.startswith("_") and not c.endswith("_event")
    ]
    X = feats.select(cols).to_numpy().astype(np.float32)
    return X, cols


def cleaned_labels(led: pl.DataFrame) -> np.ndarray | None:
    from ringtrace import silver

    return silver.load(led, "label")


def score_block(
    name: str,
    pred: np.ndarray,
    targets: dict[str, np.ndarray],
    period: np.ndarray,
    report: dict,
    log=print,
    months: tuple[int, ...] = TRAIN_MONTHS,
) -> None:
    line = [f"{name:<14}"]
    for tname, y in targets.items():
        per = []
        for m in months:
            sel = period == m
            per.append(macro_f1(y[sel], pred[sel]))
        report.setdefault(name, {})[tname] = {"months": per, "mean": float(np.mean(per))}
        line.append(f"{tname} " + " / ".join(f"{s:.4f}" for s in per) + f" ({np.mean(per):.4f})")
    log("  ".join(line))


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--no-test", action="store_true", help="skip the October submission")
    ap.add_argument(
        "--without",
        default="",
        help="comma list of families to leave out (ablation); renames the run",
    )
    ap.add_argument("--rules", default=None, help="comma list replacing the recipe's rules")
    ap.add_argument("--name", default=None, help="run name replacing the recipe's")
    ap.add_argument(
        "--folds",
        default="",
        help="comma list of held-out months to run (default all); renames the run",
    )
    args = ap.parse_args(argv)
    recipe = load_recipe(args.recipe)
    if args.name:
        recipe["name"] = args.name
    if args.without:
        gone = [f for f in args.without.split(",") if f]
        recipe["families"] = [f for f in recipe["families"] if f not in gone]
        recipe["name"] = f"{recipe['name']}-without-{'-'.join(gone)}"
    if args.rules is not None:
        recipe["rules"] = [r for r in args.rules.split(",") if r]
    folds = tuple(int(m) for m in args.folds.split(",") if m) or TRAIN_MONTHS
    if folds != TRAIN_MONTHS:
        recipe["name"] = f"{recipe['name']}-fold-{'-'.join(map(str, folds))}"
    t0 = time.time()
    out = paths.out_dir() / "runs" / recipe["name"]
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    led = store.ledger()
    X, cols = feature_matrix(recipe)
    log(f"[{recipe['name']}] {X.shape[1]} features, {X.shape[0]} rows ({time.time() - t0:.0f}s)")
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y = led["y"].to_numpy().astype(np.int64)
    tr_idx = np.flatnonzero(~is_test)
    te_idx = np.flatnonzero(is_test)
    spec, fit_fn = model(recipe)
    train_target = y.copy()
    if recipe.get("train_on") == "cleaned":
        cl = cleaned_labels(led)
        train_target[tr_idx] = cl[tr_idx]
    event_mode = EVENT_MODES.get(recipe.get("train_on", ""))
    if recipe.get("train_on") not in (None, "given", "cleaned") and event_mode is None:
        raise ValueError(f"unknown train_on {recipe['train_on']!r}")
    events = None
    if event_mode is not None:
        from ringtrace import eventclean

        events = eventclean.ledger_events()
        whole = eventclean.clean_labels(y, events, event_mode, mask=~is_test)
        log(
            f"[{event_mode}] all train months: {int((whole != y)[tr_idx].sum())} rows changed, "
            f"{int(((whole != y) & (y == 0))[tr_idx].sum())} legit -> fraud"
        )

    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
    test_sum = np.zeros((len(te_idx), len(CLASSES)), dtype=np.float64)
    for m in folds:
        fit = tr_idx[period[tr_idx] != m]
        hold = tr_idx[period[tr_idx] == m]
        log(f"fold {m}: train {len(fit)} rows, hold {len(hold)}")
        if events is not None:
            # only the fit months' labels vote, so the held-out month never shapes them
            fit_mask = np.zeros(len(y), dtype=bool)
            fit_mask[fit] = True
            train_target = eventclean.clean_labels(y, events, event_mode, mask=fit_mask)
            log(f"  {event_mode} labels: {int((train_target[fit] != y[fit]).sum())} rows changed")
        outs = [X[hold]] + ([] if args.no_test else [X[te_idx]])
        res = fit_fn(X[fit], train_target[fit], period[fit], outs, spec, log=log)
        oof[hold] = res[0]
        if not args.no_test:
            test_sum += res[1]
    test_proba = test_sum / len(folds)

    targets = {"given": y[tr_idx]}
    from ringtrace import silver

    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr[tr_idx]
    report: dict = {"recipe": recipe, "features": cols, "model": asdict(spec)}
    per_tr = period[tr_idx]
    p_tr = oof[tr_idx]
    score_block("argmax", p_tr.argmax(1), targets, per_tr, report, log, folds)
    dec = recipe.get("decoder", {"fraud_scale": 0.875})
    pred_tr = decode_by_period(p_tr, per_tr, **dec)
    score_block("fixed_point", pred_tr, targets, per_tr, report, log, folds)

    rules = recipe.get("rules", [])
    pred_all = np.zeros(len(y), dtype=np.int64)
    pred_all[tr_idx] = pred_tr
    if not args.no_test:
        pred_all[te_idx] = decode_by_period(test_proba, period[te_idx], **dec)
    if rules:
        from ringtrace import rules as rulebook

        proba_all = oof.copy()
        proba_all[te_idx] = test_proba
        pred_all = rulebook.apply(rules, pred_all, proba_all, led, log=log)
        score_block("rules", pred_all[tr_idx], targets, per_tr, report, log, folds)

    final_name = "rules" if rules else "fixed_point"
    for tname, yt in targets.items():
        f1s = per_class_f1(yt, pred_all[tr_idx])
        report.setdefault("per_class", {})[tname] = dict(zip(CLASSES, map(float, f1s), strict=True))
    report["final"] = report[final_name]
    np.save(out / "oof.npy", oof[tr_idx].astype(np.float32))
    if not args.no_test:
        np.save(out / "test.npy", test_proba.astype(np.float32))
        sub = pl.DataFrame(
            {
                "id": led["txn_id"].to_numpy()[te_idx],
                "label": np.asarray(CLASSES)[pred_all[te_idx]],
            }
        )
        sub_path = out / "submission.csv"
        sub.write_csv(sub_path)
        report["submission_sha256"] = hashlib.sha256(sub_path.read_bytes()).hexdigest()
        counts = sub.group_by("label").len().sort("label")
        report["test_calls"] = dict(
            zip(counts["label"].to_list(), counts["len"].to_list(), strict=True)
        )
        log(f"test calls: {report['test_calls']}")
    report["runtime_s"] = time.time() - t0
    report["log"] = log_lines
    (out / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"done in {report['runtime_s']:.0f}s -> {out}")
    return report


if __name__ == "__main__":
    main()
