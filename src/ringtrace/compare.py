"""Paired bootstrap over events between two finished runs:

    python -m ringtrace.compare <base-run> <challenger-run> [--target cleaned] [--draws 2000]

Rows of one generator event are correlated, so the resampling unit is the event (the first
family that places a row in an event; a row outside every event is its own unit). Each draw gives
every unit a Poisson(1) weight; per-class true-positive, predicted and true counts of each run
are then one sparse matrix product away, and macro-F1 is computed per held-out month and averaged.
The verdict is "challenger" only when the whole 5-95 % interval of the gain is above zero.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.sparse as sp

from ringtrace import paths, silver, store
from ringtrace.classes import CLASSES
from ringtrace.decode import decode_by_period

K = len(CLASSES)


def decoded(run: str, led) -> np.ndarray:
    """Train-row calls of a finished run, re-decoded exactly as its recipe says."""
    out = paths.out_dir() / "runs" / run
    recipe = json.loads((out / "summary.json").read_text())["recipe"]
    is_test = led["is_test"].to_numpy()
    tr = np.flatnonzero(~is_test)
    period = led["period"].to_numpy()
    proba = np.zeros((led.height, K))
    proba[tr] = np.load(out / "oof.npy")
    pred = np.zeros(led.height, dtype=np.int64)
    pred[tr] = decode_by_period(
        proba[tr], period[tr], **recipe.get("decoder", {"fraud_scale": 0.875})
    )
    if recipe.get("rules"):
        from ringtrace import rules

        pred = rules.apply(recipe["rules"], pred, proba, led, log=lambda *_: None)
    return pred[tr]


def _counts(groups: np.ndarray, truth: np.ndarray, pred: np.ndarray, n_groups: int):
    """Sparse (groups x 3K) matrix: per group, TP, predicted and true counts per class."""
    rows = np.concatenate([groups[truth == pred], groups, groups])
    cols = np.concatenate([truth[truth == pred], K + pred, 2 * K + truth])
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(n_groups, 3 * K))


def _macro(c: np.ndarray) -> np.ndarray:
    tp, pr, tr = c[:, :K], c[:, K : 2 * K], c[:, 2 * K :]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(pr + tr > 0, 2 * tp / (pr + tr), 0.0)
    return f1.mean(axis=1)


def bootstrap(groups, truth, period, pred_a, pred_b, draws=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    _, g = np.unique(groups, return_inverse=True)
    deltas = np.zeros(draws)
    point = 0.0
    months = np.unique(period)
    for m in months:
        s = period == m
        _, gm = np.unique(g[s], return_inverse=True)
        ng = gm.max() + 1
        ca = _counts(gm, truth[s], pred_a[s], ng)
        cb = _counts(gm, truth[s], pred_b[s], ng)
        w = rng.poisson(1.0, size=(draws, ng)).astype(np.float64)
        da = _macro(np.asarray(w @ ca))
        db = _macro(np.asarray(w @ cb))
        deltas += (db - da) / len(months)
        one = np.ones((1, ng))
        point += float((_macro(np.asarray(one @ cb)) - _macro(np.asarray(one @ ca)))[0]) / len(
            months
        )
    lo, hi = np.percentile(deltas, [5, 95])
    return {
        "delta": point,
        "p05": float(lo),
        "p95": float(hi),
        "verdict": "challenger" if lo > 0 else ("base" if hi < 0 else "tie"),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("challenger")
    ap.add_argument("--target", default="both", choices=["given", "cleaned", "both"])
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args(argv)
    from ringtrace.teacher import union_events

    led = store.ledger()
    tr = np.flatnonzero(~led["is_test"].to_numpy())
    period = led["period"].to_numpy()[tr]
    ev = union_events(led.height)[tr]
    groups = np.where(ev >= 0, ev, -1 - np.arange(len(tr)))
    pa, pb = decoded(args.base, led), decoded(args.challenger, led)
    targets = {"given": led["y"].to_numpy()[tr]}
    cl = silver.load(led, "label")
    if cl is not None:
        targets["cleaned"] = cl[tr]
    out = {}
    for name, y in targets.items():
        if args.target not in ("both", name):
            continue
        out[name] = bootstrap(groups, y.astype(np.int64), period, pa, pb, args.draws)
        r = out[name]
        print(
            f"{args.challenger} vs {args.base} [{name}]: {r['delta']:+.4f} "
            f"[{r['p05']:+.4f}, {r['p95']:+.4f}] -> {r['verdict']}"
        )
    return out


if __name__ == "__main__":
    main()
