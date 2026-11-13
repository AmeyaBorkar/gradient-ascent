"""Score a weighted average of finished runs' month-fold probabilities, decoded like a run:

    python -m ringtrace.blend champion+tabm [--months 9] [--rules anchored_templates]

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
from ringtrace.run import TRAIN_MONTHS, score_block


def blend_oof(blend: list[tuple[str, float]]) -> np.ndarray:
    out = 0.0
    for name, w in blend:
        out = out + w * np.load(paths.out_dir() / "runs" / name / "oof.npy").astype(np.float64)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m ringtrace.blend")
    ap.add_argument("blend")
    ap.add_argument("--months", default=",".join(map(str, TRAIN_MONTHS)))
    ap.add_argument("--rules", default="anchored_templates")
    ap.add_argument("--scale", type=float, default=0.875)
    args = ap.parse_args(argv)
    months = tuple(int(m) for m in args.months.split(",") if m)
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    tr_idx = np.flatnonzero(~is_test)
    period = led["period"].to_numpy()[tr_idx]
    prob = blend_oof(parse_blend(args.blend))
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
        full_pred = rulebook.apply(rules, full_pred, full_prob, led, log=lambda *_: None)
        score_block("rules", full_pred[tr_idx], targets, period, report, months=months)
    return report


if __name__ == "__main__":
    main()
