"""Blend-against-blend checks on finished runs, and aggregate call changes.

    python -m ringtrace.gate lomo <base> <challenger> [--draws 1000]
    python -m ringtrace.gate october <base> <challenger> [--out NAME]

A blend is written like the October-like scorer's: ``a+b`` (equal weights) or ``a*0.6+b*0.4``,
where each name is a run directory under ``$RINGTRACE_OUT/runs``.

``lomo`` averages the runs' ``oof.npy`` per blend, decodes each held-out month on its own (fixed
point, then the rules, as the blend recipe does) and reports macro-F1 per month on the given,
cleaned and gen-only targets, with a paired Poisson bootstrap over generator events per month
(95 % bounds).

``october`` does the same with ``test.npy``, decodes October, writes the challenger's submission
under ``runs/<out>/`` and reports its sha256 and the count of rows per (base call, challenger
call) pair. No row is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ringtrace.classes import CLASSES, N_CLASSES

DRAWS = 1000
DECODER = {"fraud_scale": 0.875}
RULES = ["anchored_templates"]


def parse_blend(spec: str) -> list[tuple[str, float]]:
    from ringtrace.octlike import parse_blend as parse

    return parse(spec)


def mean_prob(blend: list[tuple[str, float]], runs: Path, which: str) -> np.ndarray:
    out = 0.0
    for name, w in blend:
        out = out + w * np.load(runs / name / f"{which}.npy").astype(np.float64)
    return out


def change_pairs(a: np.ndarray, b: np.ndarray, k: int = N_CLASSES) -> dict[str, int]:
    """Rows per (call in a -> call in b) pair, changed pairs only, keyed ``from->to``."""
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (np.asarray(a, np.int64), np.asarray(b, np.int64)), 1)
    return {
        f"{CLASSES[i]}->{CLASSES[j]}": int(m[i, j])
        for i in range(k)
        for j in range(k)
        if i != j and m[i, j]
    }


def month_bootstrap(groups, truth, pred_a, pred_b, draws=DRAWS, seed=0) -> dict:
    """Paired Poisson bootstrap of macro-F1(b) - macro-F1(a) over groups: point, 95 % bounds."""
    from ringtrace.octlike import bootstrap

    return bootstrap(groups, truth, pred_a, pred_b, draws, seed)


def _decode(proba: np.ndarray, rows: np.ndarray, led) -> np.ndarray:
    from ringtrace import rules
    from ringtrace.decode import decode_by_period

    period = led["period"].to_numpy()
    full = np.zeros((led.height, proba.shape[1]))
    full[:, 0] = 1.0
    full[rows] = proba
    pred = np.zeros(led.height, dtype=np.int64)
    pred[rows] = decode_by_period(proba, period[rows], **DECODER)
    return rules.apply(RULES, pred, full, led, log=lambda *_: None)[rows]


def lomo(base: str, challenger: str, draws: int = DRAWS, log=print) -> dict:
    from ringtrace import paths, silver, store
    from ringtrace.scoring import macro_f1
    from ringtrace.teacher import union_events

    led = store.ledger()
    runs = paths.out_dir() / "runs"
    tr = np.flatnonzero(~led["is_test"].to_numpy())
    period = led["period"].to_numpy()[tr]
    ev = union_events(led.height)[tr]
    groups = np.where(ev >= 0, ev, -1 - np.arange(len(tr)))
    pa = _decode(mean_prob(parse_blend(base), runs, "oof"), tr, led)
    pb = _decode(mean_prob(parse_blend(challenger), runs, "oof"), tr, led)
    targets = {"given": led["y"].to_numpy().astype(np.int64)[tr]}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr.astype(np.int64)[tr]
    rep = {"base": base, "challenger": challenger, "targets": {}}
    for t, y in targets.items():
        months = {}
        for m in np.unique(period):
            s = period == m
            bs = month_bootstrap(groups[s], y[s], pa[s], pb[s], draws, seed=int(m))
            months[int(m)] = {"base": macro_f1(y[s], pa[s]), "challenger": macro_f1(y[s], pb[s])}
            months[int(m)].update(bs)
        rep["targets"][t] = months
        log(
            f"[lomo] {t:<8} "
            + "  ".join(
                f"m{m} {v['base']:.4f}->{v['challenger']:.4f} {v['delta']:+.4f} "
                f"[{v['lo']:+.4f}, {v['hi']:+.4f}]"
                for m, v in months.items()
            )
        )
    rep["changed_rows"] = int((pa != pb).sum())
    return rep


def october(base: str, challenger: str, out: str, log=print) -> dict:
    from ringtrace import paths, store
    from ringtrace.blend import write_submission

    led = store.ledger()
    runs = paths.out_dir() / "runs"
    te = np.flatnonzero(led["is_test"].to_numpy())
    pa = _decode(mean_prob(parse_blend(base), runs, "test"), te, led)
    pb = _decode(mean_prob(parse_blend(challenger), runs, "test"), te, led)
    d = runs / out
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "test.npy", mean_prob(parse_blend(challenger), runs, "test").astype(np.float32))
    sub = write_submission(
        d / "submission.csv",
        led["txn_id"].to_numpy()[te],
        pb,
        paths.data_dir() / "sample_submission.csv",
    )
    rep = {
        "base": base,
        "challenger": challenger,
        "sha256": sub["sha256"],
        "problems": sub["problems"],
        "changed": int((pa != pb).sum()),
        "pairs": change_pairs(pa, pb),
        "calls_base": np.bincount(pa, minlength=N_CLASSES).tolist(),
        "calls_challenger": np.bincount(pb, minlength=N_CLASSES).tolist(),
    }
    (d / "summary.json").write_text(json.dumps(rep, indent=1))
    log(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> dict:
    from ringtrace import paths, threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lo = sub.add_parser("lomo")
    lo.add_argument("base")
    lo.add_argument("challenger")
    lo.add_argument("--draws", type=int, default=DRAWS)
    lo.add_argument("--out", default="gate-lomo.json")
    oc = sub.add_parser("october")
    oc.add_argument("base")
    oc.add_argument("challenger")
    oc.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "lomo":
        rep = lomo(args.base, args.challenger, args.draws)
        (paths.out_dir() / args.out).write_text(json.dumps(rep, indent=1))
        return rep
    return october(args.base, args.challenger, args.out)


if __name__ == "__main__":
    main()
