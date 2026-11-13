"""Where a challenger blend's calls differ from a base blend, and what each kind of change is worth:

    python -m ringtrace.decompose <base> <challenger> [--rules anchored_templates,...]

Both are decoded like a run (fixed point 0.875 per month, then the rules) on the month-fold
probabilities. Changed train rows are split into legit->fraud adds, fraud->legit drops and type
switches (also by class pair). Per category: count, precision on each target (the challenger's
call is right) and its macro-F1 contribution per month (the base with only that category's
changes applied, minus the base). Aggregates only.
"""

from __future__ import annotations

import argparse

import numpy as np

from ringtrace import silver, store
from ringtrace.blend import blend_arrays
from ringtrace.classes import CLASSES
from ringtrace.decode import decode_by_period
from ringtrace.octlike import parse_blend
from ringtrace.run import TRAIN_MONTHS
from ringtrace.scoring import macro_f1


def decode_all(prob_tr, prob_te, led, rules, scale=0.875) -> np.ndarray:
    """Calls for every ledger row: fixed point per month, then the rules over train and test."""
    from ringtrace import rules as rulebook

    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    pred = np.zeros(led.height, dtype=np.int64)
    full = np.zeros((led.height, prob_tr.shape[1]))
    full[tr], full[te] = prob_tr, prob_te
    pred[tr] = decode_by_period(prob_tr, period[tr], fraud_scale=scale)
    pred[te] = decode_by_period(prob_te, period[te], fraud_scale=scale)
    return rulebook.apply(rules, pred, full, led, log=lambda *_: None) if rules else pred


def categories(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """Masks of the changed rows a -> b by kind, and type switches by class pair."""
    out = {
        "legit->fraud": (a == 0) & (b > 0),
        "fraud->legit": (a > 0) & (b == 0),
        "type switch": (a > 0) & (b > 0) & (a != b),
    }
    sw = out["type switch"]
    for f, t in sorted(set(zip(a[sw].tolist(), b[sw].tolist(), strict=True))):
        out[f"  {CLASSES[f]}->{CLASSES[t]}"] = sw & (a == f) & (b == t)
    return out


def decompose(a, b, targets, period, months=TRAIN_MONTHS) -> list[dict]:
    rows = []
    for name, m in categories(a, b).items():
        r = {"category": name, "n": int(m.sum())}
        for t, y in targets.items():
            r[f"prec_{t}"] = float((b[m] == y[m]).mean()) if m.any() else float("nan")
            mixed = np.where(m, b, a)
            r[f"d_{t}"] = [
                macro_f1(y[period == mo], mixed[period == mo])
                - macro_f1(y[period == mo], a[period == mo])
                for mo in months
            ]
        rows.append(r)
    return rows


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(prog="python -m ringtrace.decompose")
    ap.add_argument("base")
    ap.add_argument("challenger")
    ap.add_argument("--rules", default="anchored_templates")
    args = ap.parse_args(argv)
    led = store.ledger()
    tr = np.flatnonzero(~led["is_test"].to_numpy())
    rules = [r for r in args.rules.split(",") if r]
    preds = []
    for spec in (args.base, args.challenger):
        bl = parse_blend(spec)
        preds.append(decode_all(blend_arrays(bl), blend_arrays(bl, "test"), led, rules)[tr])
    targets = {"given": led["y"].to_numpy()[tr]}
    for t, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[t] = arr[tr]
    rows = decompose(preds[0], preds[1], targets, led["period"].to_numpy()[tr])
    for r in rows:
        print(
            f"{r['category']:<44} n {r['n']:>4}  "
            + "  ".join(
                f"{t} prec {r[f'prec_{t}']:.2f} d " + "/".join(f"{x:+.5f}" for x in r[f"d_{t}"])
                for t in targets
            )
        )
    return rows


if __name__ == "__main__":
    main()
