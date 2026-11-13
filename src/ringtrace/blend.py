"""Score a weighted average of finished runs' month-fold probabilities, decoded like a run:

    python -m ringtrace.blend champion+tabm [--months 9] [--rules anchored_templates]
    python -m ringtrace.blend recipes/blend-tabmF-bq.json --write <october.csv>

A blend recipe (JSON) lists ``members`` ({"run", "weight", optional "test_run": the run whose
October array replaces the member's, e.g. a refit}) and its ``decoder`` and ``rules``, which
replace ``--scale`` and ``--rules``. With a ``split`` list as well, the blend is fraud-only: the
legit column is the ``split`` average and the fraud types are the ``members`` average rescaled to
its fraud mass (``fraud_split_from``); the result is rounded to float32, as a saved run is.

``a+b`` weighs the runs equally, ``a*0.6+b*0.4`` fixes the weights (the gate's syntax). Runs
are read from ``$RINGTRACE_OUT/runs/<name>/oof.npy``; a run made with ``--folds`` holds zeros
outside its folds, so pass ``--months`` to score only months every run covers.
"""

from __future__ import annotations

import argparse

import numpy as np

from ringtrace import paths, silver, store
from ringtrace.decode import decode_by_period
from ringtrace.octlike import parse_blend
from ringtrace.run import TRAIN_MONTHS, check_proba, score_block


def blend_arrays(blend: list[tuple[str, float]], which: str = "oof") -> np.ndarray:
    out = 0.0
    for name, w in blend:
        f = paths.out_dir() / "runs" / name / f"{which}.npy"
        out = out + w * np.load(f).astype(np.float64)
    return out


def recipe_arrays(recipe: dict, which: str = "oof") -> np.ndarray:
    """A blend recipe's probabilities for the train rows (``oof``) or October (``test``)."""

    def avg(members):
        total = sum(m["weight"] for m in members)
        parts = []
        for m in members:
            run = m.get("test_run", m["run"]) if which == "test" else m["run"]
            parts.append((run, m["weight"] / total))
        return blend_arrays(parts, which)

    prob = avg(recipe["members"])
    if "split" in recipe:
        prob = fraud_split_from(prob, avg(recipe["split"])).astype(np.float32).astype(np.float64)
    return prob


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m ringtrace.blend")
    ap.add_argument("blend")
    ap.add_argument("--months", default=",".join(map(str, TRAIN_MONTHS)))
    ap.add_argument("--rules", default="anchored_templates")
    ap.add_argument("--scale", type=float, default=0.875)
    ap.add_argument("--write", default="", help="also decode October and write this csv")
    args = ap.parse_args(argv)
    months = tuple(int(m) for m in args.months.split(",") if m)
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    tr_idx = np.flatnonzero(~is_test)
    period = led["period"].to_numpy()[tr_idx]
    if args.blend.endswith(".json"):
        import json

        with open(args.blend) as fh:
            recipe = json.load(fh)
        args.scale = recipe.get("decoder", {}).get("fraud_scale", args.scale)
        args.rules = ",".join(recipe.get("rules", []))

        def arrays(which):
            return recipe_arrays(recipe, which)
    else:
        blend = parse_blend(args.blend)

        def arrays(which):
            return blend_arrays(blend, which)

    prob = arrays("oof")
    te_idx = np.flatnonzero(is_test)
    sel = np.isin(period, months)
    pred = np.zeros(len(tr_idx), dtype=np.int64)
    pred[sel] = decode_by_period(prob[sel], period[sel], fraud_scale=args.scale)
    targets = {"given": led["y"].to_numpy()[tr_idx]}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr[tr_idx]
    report: dict = {}
    score_block("fixed_point", pred, targets, period, report, months=months)
    rules = [r for r in args.rules.split(",") if r]
    if rules:
        from ringtrace import rules as rulebook

        full_pred = np.zeros(led.height, dtype=np.int64)
        full_prob = np.zeros((led.height, prob.shape[1]))
        full_pred[tr_idx] = pred
        full_prob[tr_idx] = prob
        if args.write:
            test = arrays("test")
            check_proba(test, "October rows")
            full_prob[te_idx] = test
            full_pred[te_idx] = decode_by_period(
                test, led["period"].to_numpy()[te_idx], fraud_scale=args.scale
            )
        full_pred = rulebook.apply(rules, full_pred, full_prob, led, log=lambda *_: None)
        score_block("rules", full_pred[tr_idx], targets, period, report, months=months)
        if args.write:
            write_submission(led, te_idx, full_pred[te_idx], args.write)
    return report


def write_submission(led, te_idx: np.ndarray, pred: np.ndarray, path: str) -> None:
    """October calls in the sample's id order, validated against the sample."""
    import hashlib
    from pathlib import Path

    import polars as pl

    from ringtrace.build import validate
    from ringtrace.classes import CLASSES

    sub = pl.DataFrame({"id": led["txn_id"].to_numpy()[te_idx], "label": np.asarray(CLASSES)[pred]})
    sample_path = paths.data_dir() / "sample_submission.csv"
    sample = pl.read_csv(sample_path, infer_schema=False).select("id")
    sub = sample.join(sub.cast(pl.String), on="id", how="left", maintain_order="left")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.write_csv(out)
    problems = validate(out, sample_path)
    if problems:
        raise ValueError("invalid submission: " + "; ".join(problems))
    print(f"VALID {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")



def fraud_split_from(types: np.ndarray, split: np.ndarray) -> np.ndarray:
    """``types``'s distribution over the fraud classes, rescaled to ``split``'s fraud mass
    (the legit column comes from ``split``)."""
    out = np.empty_like(types, dtype=np.float64)
    out[:, 0] = split[:, 0]
    mass = types[:, 1:].sum(axis=1, keepdims=True)
    share = np.divide(types[:, 1:], mass, out=np.full_like(types[:, 1:], 1 / 11), where=mass > 0)
    out[:, 1:] = share * (1 - split[:, :1])
    return out


def save_fraud_split(types_spec: str, split_spec: str, name: str) -> None:
    """Write run ``name`` (oof + test) from two blends, see ``fraud_split_from``."""
    out = paths.out_dir() / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    for which in ("oof", "test"):
        p = fraud_split_from(
            blend_arrays(parse_blend(types_spec), which),
            blend_arrays(parse_blend(split_spec), which),
        )
        check_proba(p, name)
        np.save(out / f"{which}.npy", p.astype(np.float32))


if __name__ == "__main__":
    main()
