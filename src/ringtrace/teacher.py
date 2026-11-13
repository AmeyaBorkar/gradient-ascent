"""Build and freeze the silver labels:  python -m ringtrace.teacher

The teacher is deliberately not the scored model: one plain multiclass booster on the relational
context and the row marks (no event families, no device counter), fitted out of fold by month on
the labels as given, with legit rows subsampled. Its only job is to say which rows look
generator-made; the cleaning itself is in ``ringtrace.silver``.
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import numpy as np
import xgboost as xgb

from ringtrace import silver, store, threads
from ringtrace.classes import N_CLASSES

EVENT_FAMILIES = ("bursts", "drains", "flows", "requests", "merchant", "repeats")
EVENT_COLUMNS = {"bursts": "bot_event", "drains": "drn_event", "flows": "mf_event",
                 "requests": "col_event", "merchant": "rng_event", "repeats": "rep_event"}


def available(name: str) -> bool:
    return importlib.util.find_spec(store.FAMILIES[name]) is not None


def union_events(n: int) -> np.ndarray:
    """One event id per row: the first family (in a fixed precedence) that places it in an event."""
    out = np.full(n, -1, dtype=np.int64)
    for i, fam in enumerate(EVENT_FAMILIES):
        if not available(fam):
            continue
        ev = store.family(fam)[EVENT_COLUMNS[fam]].fill_null(-1).to_numpy().astype(np.int64)
        take = (out < 0) & (ev >= 0)
        out[take] = ev[take] + (i + 1) * 10_000_000
    return out


def teacher_oof(X: np.ndarray, y: np.ndarray, period: np.ndarray, seed: int = 7,
                rounds: int = 250, legit_keep: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    params = {"tree_method": "hist", "device": "cuda", "objective": "multi:softprob",
              "num_class": N_CLASSES, "eta": 0.1, "max_depth": 7, "subsample": 0.8,
              "colsample_bytree": 0.6, "min_child_weight": 3, "lambda": 2.0, "seed": seed,
              "nthread": threads.budget(), "verbosity": 0}
    for m in np.unique(period):
        fit = period != m
        keep = fit & ((y > 0) | (rng.random(len(y)) < legit_keep))
        w = np.where(y[keep] > 0, 1.0, 1.0 / legit_keep)
        d = xgb.QuantileDMatrix(X[keep], label=y[keep], weight=w)
        model = xgb.train(params, d, rounds)
        out[period == m] = model.inplace_predict(X[period == m])
    # undo the legit subsampling's prior shift: the weights already restore it
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="context,marks")
    ap.add_argument("--oof-only", action="store_true", help="fit and cache the teacher only")
    args = ap.parse_args(argv)
    t0 = time.time()
    led = store.ledger()
    fams = [f for f in args.families.split(",") if available(f)]
    feats = store.frame(fams)
    cols = [c for c in feats.columns if not c.endswith("_event")]
    X_all = feats.select(cols).to_numpy().astype(np.float32)
    train = ~led["is_test"].to_numpy()
    y = led["y"].to_numpy()[train].astype(np.int64)
    period = led["period"].to_numpy()[train]
    print(f"[teacher] families {fams}, {len(cols)} columns", flush=True)
    cached = silver._dir() / f"teacher_oof_{'-'.join(fams)}.npy"
    if cached.exists():
        proba = np.load(cached)
    else:
        proba = teacher_oof(X_all[train], y, period)
        np.save(cached, proba)
    if args.oof_only:
        return
    from ringtrace.scoring import macro_f1

    for m in np.unique(period):
        s = period == m
        print(f"[teacher] month {m}: argmax macro-F1 {macro_f1(y[s], proba[s].argmax(1)):.4f}")
    events = union_events(led.height)
    print(f"[teacher] rows in an event: {(events[train] >= 0).sum()}", flush=True)
    result = silver.build(led, proba, events)
    silver.save(result)
    print(f"[teacher] silver labels frozen in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
