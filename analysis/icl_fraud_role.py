"""The in-context member in the fraud-only role, leave-one-month-out (aggregates only).

    python analysis/icl_fraud_role.py [member-run]

Fraud mass = w_tabm x TabM + w_icl x member + 0.1 x each of the five trees; the type split is the
trees' mean; legit = 1 - fraud mass. Decoded like the blend (fixed point 0.875 per month, then the
anchored template rule) and compared with the w_icl = 0 role by the paired event bootstrap.
"""

from __future__ import annotations

import sys

import numpy as np

from ringtrace import blend, compare, paths, store
from ringtrace.icl import compose
from ringtrace.teacher import union_events

TREES = ["blend-champion-m1", "lgbm", "catboost", "clean-event", "clean-l2"]
RULES = ["anchored_templates"]


def main(member: str = "icl-tabpfn-v2") -> None:
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr = np.flatnonzero(~is_test)
    runs = paths.out_dir() / "runs"
    oof = {r: np.load(runs / r / "oof.npy").astype(np.float64) for r in [*TREES, "tabm", member]}
    trees = np.mean([oof[r] for r in TREES], axis=0)
    fraud = {r: 1 - oof[r][:, 0] for r in oof}
    from ringtrace.icl import _targets

    targets = _targets(led, tr)
    ev = union_events(led.height)[tr]
    groups = np.where(ev >= 0, ev, -1 - np.arange(len(tr)))
    preds = {}
    for w_icl in (0.0, 0.125, 0.25, 0.5):
        pf = (0.5 - w_icl) * fraud["tabm"] + w_icl * fraud[member]
        pf += 0.1 * sum(fraud[r] for r in TREES)
        full = np.zeros((led.height, trees.shape[1]))
        full[~is_test] = compose(pf, trees)
        full[is_test, 0] = 1.0  # October is not scored here
        calls = blend.decode_all(full, period, is_test, {"fraud_scale": 0.875}, RULES, led,
                                 log=lambda *_: None)  # fmt: skip
        pred = calls["rules"][tr]
        preds[w_icl] = pred
        sc = blend.month_scores(pred, targets, period[tr])
        print(f"w_icl {w_icl:.3f}: {blend._line(sc)}")
        if w_icl > 0:
            for t in ("given", "cleaned"):
                r = compare.bootstrap(groups, targets[t].astype(np.int64), period[tr],
                                      preds[0.0], pred)  # fmt: skip
                print(f"   vs w_icl 0 [{t}]: {r['delta']:+.4f} [{r['p05']:+.4f}, {r['p95']:+.4f}]")


if __name__ == "__main__":
    main(*sys.argv[1:])
