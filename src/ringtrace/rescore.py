"""Re-decode and re-score a finished run from its saved probabilities, without refitting:

    python -m ringtrace.rescore <run-name> [--rules anchored_templates,...] [--scale 0.875]

Useful when the silver labels or a rule changed after the run finished.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ringtrace import paths, silver, store
from ringtrace.classes import CLASSES
from ringtrace.decode import decode_by_period
from ringtrace.run import score_block
from ringtrace.scoring import per_class_f1


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rules", default=None, help="comma list; default = the recipe's rules")
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args(argv)
    out = paths.out_dir() / "runs" / args.run
    summary = json.loads((out / "summary.json").read_text())
    recipe = summary["recipe"]
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr_idx, te_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    proba = np.zeros((led.height, len(CLASSES)))
    proba[tr_idx] = np.load(out / "oof.npy")
    if (out / "test.npy").exists():
        proba[te_idx] = np.load(out / "test.npy")
    dec = dict(recipe.get("decoder", {"fraud_scale": 0.875}))
    if args.scale is not None:
        dec["fraud_scale"] = args.scale
    targets = {"given": led["y"].to_numpy()[tr_idx]}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr[tr_idx]
    report: dict = {}
    pred = np.zeros(led.height, dtype=np.int64)
    pred[tr_idx] = decode_by_period(proba[tr_idx], period[tr_idx], **dec)
    pred[te_idx] = decode_by_period(proba[te_idx], period[te_idx], **dec)
    score_block("fixed_point", pred[tr_idx], targets, period[tr_idx], report)
    rules = (
        recipe.get("rules", []) if args.rules is None else [r for r in args.rules.split(",") if r]
    )
    if rules:
        from ringtrace import rules as rulebook

        pred = rulebook.apply(rules, pred, proba, led)
        score_block("rules", pred[tr_idx], targets, period[tr_idx], report)
    for tname, yt in targets.items():
        report.setdefault("per_class", {})[tname] = dict(
            zip(CLASSES, np.round(per_class_f1(yt, pred[tr_idx]), 4).tolist(), strict=True)
        )
    print(json.dumps(report.get("per_class", {}).get("cleaned", {})))
    return report


if __name__ == "__main__":
    main()
