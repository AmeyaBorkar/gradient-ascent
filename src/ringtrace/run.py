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
from ringtrace.mix import class_mix_weights, em_priors, ratios_from_names
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


def class_mix_ratios(recipe: dict) -> dict[int, float] | None:
    """Fraud-class code to ratio from ``class_mix``, or None when the key is absent.

    The weights are applied inside the cascade's stage fits, so any other model is rejected.
    An empty mapping is the unweighted fit.
    """
    if "class_mix" not in recipe or recipe["class_mix"] is None:
        return None
    if recipe.get("model") not in (None, "cascade"):
        raise ValueError(
            f"class_mix applies only to the cascade, not to model {recipe.get('model')!r}"
        )
    raw = recipe["class_mix"]
    if not isinstance(raw, dict):
        raise ValueError("class_mix must be a mapping of fraud class name to a positive ratio")
    ratios = ratios_from_names(raw)
    return ratios or None


def sample_weight_for(
    recipe: dict, y: np.ndarray, events: np.ndarray | None = None
) -> np.ndarray | None:
    """Per-row cascade weights for the training target ``y``.

    None when the recipe has no class mix. Callers leave ``sample_weight`` off the fit
    in that case, so a recipe without the key stays on today's fit path. ``events``, when
    given, is one event id per row (negative for none); every row of an event then shares
    the weight of the event's majority label.
    """
    ratios = class_mix_ratios(recipe)
    if not ratios:
        return None
    return class_mix_weights(y, ratios, events)


def label_shift_on(recipe: dict) -> bool:
    """True when the recipe corrects each decoded block for label shift before decoding."""
    if "label_shift_em" not in recipe:
        return False
    flag = recipe["label_shift_em"]
    if not isinstance(flag, bool):
        raise ValueError(f"label_shift_em must be true or false, got {flag!r}")
    return flag


def reference_priors(
    oof_held: np.ndarray, period_held: np.ndarray, folds: tuple[int, ...]
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Reference class shares for the prior correction, from out-of-fold probabilities.

    Month m's reference is the mean out-of-fold probability of the other held-out months, so
    it reflects what the cascade predicts on months it did not train on, with the training
    class mix. With a single held-out month there is no other month: its own mean is used,
    which makes the correction a no-op for it. The scored month's reference is the mean over
    every held-out month.
    """
    oof_held = np.asarray(oof_held, dtype=np.float64)
    period_held = np.asarray(period_held)
    whole = oof_held.mean(axis=0)
    out: dict[int, np.ndarray] = {}
    for m in folds:
        other = period_held != m
        out[int(m)] = oof_held[other].mean(axis=0) if other.any() else oof_held[~other].mean(0)
    return out, whole


def apply_label_shift(
    proba: np.ndarray,
    period: np.ndarray,
    train_priors: np.ndarray | dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[str, dict[str, list[float]]]]:
    """EM-adjust each period block of ``proba``.

    ``train_priors`` is the class shares of the training rows behind the block: one vector
    shared by every block, or a mapping from period to that block's vector. Returns the
    adjusted probabilities, in the same row order, and for each period the training prior
    and the estimated prior. ``main`` writes that mapping to ``summary.json`` under
    ``label_shift_priors``.
    """
    proba = np.asarray(proba, dtype=np.float64)
    period = np.asarray(period)
    if len(proba) != len(period):
        raise ValueError(f"probabilities have {len(proba)} rows, periods have {len(period)}")
    out = proba.copy()
    recorded: dict[str, dict[str, list[float]]] = {}
    shared = train_priors if isinstance(train_priors, np.ndarray) else None
    for p in np.unique(period):
        key = int(p)
        if shared is None:
            try:
                prior = train_priors[key]
            except KeyError as exc:
                raise KeyError(f"no training prior for period {key}") from exc
        else:
            prior = shared
        sel = period == p
        q, adj = em_priors(out[sel], prior)
        out[sel] = adj
        recorded[str(key)] = {
            "train": [float(v) for v in np.asarray(prior, dtype=np.float64)],
            "estimated": [float(v) for v in q],
        }
    return out, recorded


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


def check_proba(p: np.ndarray, what: str) -> None:
    """Every row must be a finite distribution before it is decoded or saved."""
    bad = ~np.isfinite(p).all(axis=1) | (np.abs(p.sum(axis=1) - 1) > 1e-4)
    if bad.any():
        raise ValueError(f"{what}: {int(bad.sum())} rows are not finite or do not sum to 1")


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
    # Reject a bad class mix or label-shift flag before any run directory is created.
    mix_ratios = class_mix_ratios(recipe)
    shift_on = label_shift_on(recipe)
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
    if mix_ratios:
        shown = ", ".join(
            f"{CLASSES[code]}={ratio:g}" for code, ratio in sorted(mix_ratios.items())
        )
        log(f"class-mix weights: {shown}")
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
    whole = None
    if event_mode is not None:
        from ringtrace import eventclean

        events = eventclean.ledger_events()
        whole = eventclean.clean_labels(y, events, event_mode, mask=~is_test)
        log(
            f"[{event_mode}] all train months: {int((whole != y)[tr_idx].sum())} rows changed, "
            f"{int(((whole != y) & (y == 0))[tr_idx].sum())} legit -> fraud"
        )
    # The prior correction needs the class shares the model is calibrated to. The cascade's
    # stage one treats ordinary-looking fraud rows as legit, so the label shares overstate every
    # fraud class. The reference is therefore the mean out-of-fold probability of the training
    # months (see ``reference_priors``), taken after the fold loop.

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
        weight = sample_weight_for(recipe, train_target[fit])
        if weight is None:
            res = fit_fn(X[fit], train_target[fit], period[fit], outs, spec, log=log)
        else:
            res = fit_fn(
                X[fit],
                train_target[fit],
                period[fit],
                outs,
                spec,
                log=log,
                sample_weight=weight,
            )
        oof[hold] = res[0]
        if not args.no_test:
            test_sum += res[1]
    test_proba = test_sum / len(folds)
    label_shift_priors = None
    if shift_on:
        # Held-out month m was scored by the model trained on the other months.
        held_rows = np.flatnonzero(~is_test & np.isin(period, folds))
        fold_priors, scored_prior = reference_priors(oof[held_rows], period[held_rows], folds)
        held_adj, label_shift_priors = apply_label_shift(
            oof[held_rows], period[held_rows], fold_priors
        )
        oof[held_rows] = held_adj
        if not args.no_test:
            te_adj, te_priors = apply_label_shift(test_proba, period[te_idx], scored_prior)
            test_proba = te_adj
            label_shift_priors.update(te_priors)
        for key, rec in label_shift_priors.items():
            est = " ".join(f"{v:.4f}" for v in rec["estimated"])
            log(f"  label-shift period {key}: estimated prior {est}")

    targets = {"given": y[tr_idx]}
    from ringtrace import silver

    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr[tr_idx]
    report: dict = {"recipe": recipe, "features": cols, "model": asdict(spec)}
    if label_shift_priors is not None:
        report["label_shift_priors"] = label_shift_priors
    per_tr = period[tr_idx]
    p_tr = oof[tr_idx]
    score_block("argmax", p_tr.argmax(1), targets, per_tr, report, log, folds)
    dec = recipe.get("decoder", {"fraud_scale": 0.875})
    held = np.isin(per_tr, folds)
    check_proba(p_tr[held], "held-out rows")
    pred_tr = np.zeros(len(tr_idx), dtype=np.int64)  # months outside --folds are never scored
    pred_tr[held] = decode_by_period(p_tr[held], per_tr[held], **dec)
    score_block("fixed_point", pred_tr, targets, per_tr, report, log, folds)

    rules = recipe.get("rules", [])
    pred_all = np.zeros(len(y), dtype=np.int64)
    pred_all[tr_idx] = pred_tr
    if not args.no_test:
        check_proba(test_proba, "October rows")
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
