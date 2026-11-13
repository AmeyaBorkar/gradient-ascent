"""Where a challenger's macro-F1 gain over a base comes from: per-class F1, error pockets,
disagreements and calibration, on the October-like cells and on the month folds.

    python -m ringtrace.tabmwhy [--base champion] [--challenger tabm] [--sets 0]

Pockets (generator rows = rows whose gen-only label is fraud):

* (b) generator rows called legit;
* (c) generator rows called another fraud type than their cleaned label, by template;
* (d) rows whose label is legit called fraud.

Only aggregates are printed and saved, to ``$RINGTRACE_OUT/tabmwhy/<challenger>_vs_<base>.json``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ringtrace import paths
from ringtrace.classes import CLASSES, N_CLASSES
from ringtrace.scoring import per_class_f1

P_BINS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 0.99, 1.0)
TEMPLATE_NONE = "none"


def class_deltas(truth: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    fa, fb = per_class_f1(truth, pred_a), per_class_f1(truth, pred_b)
    return {
        c: {"base": float(a), "challenger": float(b), "delta": float(b - a)}
        for c, a, b in zip(CLASSES, fa, fb, strict=True)
    }


def pockets(
    pred: np.ndarray,
    label: np.ndarray,
    gen: np.ndarray,
    legit_truth: np.ndarray,
    template: np.ndarray,
) -> dict:
    """Row counts of pockets (b), (c) (by template and (label, call) pair) and (d)."""
    is_gen = gen > 0
    b = is_gen & (pred == 0)
    c = is_gen & (pred > 0) & (label > 0) & (pred != label)
    d = (legit_truth == 0) & (pred > 0)
    by = {}
    for t, lab, p in zip(template[c], label[c], pred[c], strict=True):
        key = f"{t}: {CLASSES[lab]} -> {CLASSES[p]}"
        by[key] = by.get(key, 0) + 1
    return {"b": int(b.sum()), "c": int(c.sum()), "d": int(d.sum()), "c_by": by}


def disagreements(
    pred_a: np.ndarray, pred_b: np.ndarray, truth: np.ndarray, template: np.ndarray
) -> dict:
    """Rows where the calls differ: who is right, by (base call, challenger call) and template."""
    diff = pred_a != pred_b
    out: dict = {"n": int(diff.sum()), "pairs": {}, "templates": {}}
    for key_of, slot in (
        (lambda i: f"{CLASSES[pred_a[i]]} -> {CLASSES[pred_b[i]]}", "pairs"),
        (lambda i: str(template[i]), "templates"),
    ):
        for i in np.flatnonzero(diff):
            k = key_of(i)
            cell = out[slot].setdefault(k, {"n": 0, "base_right": 0, "challenger_right": 0})
            cell["n"] += 1
            cell["base_right"] += int(pred_a[i] == truth[i])
            cell["challenger_right"] += int(pred_b[i] == truth[i])
    return out


def reliability(prob: np.ndarray, truth: np.ndarray, bins=P_BINS) -> dict:
    """P(fraud) = 1 - P(legit) in bins: rows, mean P(fraud), observed fraud rate."""
    pf = 1 - prob[:, 0]
    fraud = truth > 0
    idx = np.clip(np.searchsorted(bins, pf, side="right") - 1, 0, len(bins) - 2)
    rows = []
    for j in range(len(bins) - 1):
        m = idx == j
        rows.append(
            {
                "lo": bins[j],
                "hi": bins[j + 1],
                "n": int(m.sum()),
                "mean_p": float(pf[m].mean()) if m.any() else None,
                "rate": float(fraud[m].mean()) if m.any() else None,
            }
        )
    mid = (pf > 0.05) & (pf < 0.95)
    return {
        "bins": rows,
        "mid_rows": int(mid.sum()),
        "mid_mass": float(pf[mid].sum()),
        "fraud_mass": float(pf.sum()),
        "brier": float(np.mean((pf - fraud) ** 2)),
    }


def type_accuracy(prob: np.ndarray, truth: np.ndarray, template: np.ndarray) -> dict:
    """On cleaned-fraud rows: share whose top fraud type is the label, per template."""
    f = truth > 0
    top = 1 + prob[:, 1:].argmax(1)
    out = {}
    for t in np.unique(template[f]):
        m = f & (template == t)
        out[str(t)] = {"n": int(m.sum()), "acc": float((top[m] == truth[m]).mean())}
    return out


def compare(pa, pb, ya, yb, targets: dict, template: np.ndarray) -> dict:
    gen = targets.get("gen", np.zeros_like(ya))
    cl = targets["cleaned"]
    tname = np.array([f"{t}" for t in template])
    return {
        "per_class": {t: class_deltas(v, ya, yb) for t, v in targets.items()},
        "pockets": {
            tg: {
                "base": pockets(ya, targets[tg], gen, targets[tg], tname),
                "challenger": pockets(yb, targets[tg], gen, targets[tg], tname),
            }
            for tg in ("given", "cleaned")
        },
        "disagree": disagreements(ya, yb, cl, tname),
        "reliability": {"base": reliability(pa, cl), "challenger": reliability(pb, cl)},
        "type_acc": {
            "base": type_accuracy(pa, cl, tname),
            "challenger": type_accuracy(pb, cl, tname),
        },
    }


def _template_names(led) -> np.ndarray:
    from ringtrace.evidence.anchors import TEMPLATE_NAMES, template_of

    t = template_of(led)
    names = np.array([TEMPLATE_NONE] + [TEMPLATE_NAMES[i] for i in range(1, 7)])
    return names[t]


def octlike_part(base: str, challenger: str, sets: list[int]) -> dict:
    from ringtrace import octlike, store

    res = {}
    for seed in octlike.SEEDS:
        frame = octlike.Frame(seed)
        tmpl = _template_names(frame.led)[frame.hold]
        targets = {t: v[frame.hold] for t, v in frame.targets.items()}
        for r in sets:
            _, pa = octlike.cell_prob([(base, 1.0)], seed, r)
            _, pb = octlike.cell_prob([(challenger, 1.0)], seed, r)
            ya, yb = octlike.decode(frame, pa), octlike.decode(frame, pb)
            res[f"s{seed}_r{r}"] = compare(pa, pb, ya, yb, targets, tmpl)
        store.use_frame(None)
    return res


def lomo_part(base: str, challenger: str) -> dict:
    from ringtrace import silver, store
    from ringtrace.compare import decoded

    store.use_frame(None)
    led = store.ledger()
    tr = np.flatnonzero(~led["is_test"].to_numpy())
    period = led["period"].to_numpy()[tr]
    targets = {"given": led["y"].to_numpy().astype(np.int64)[tr]}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        targets[tname] = silver.load(led, which).astype(np.int64)[tr]
    tmpl = _template_names(led)[tr]
    out_dir = paths.out_dir() / "runs"
    pa = np.load(out_dir / base / "oof.npy").astype(np.float64)
    pb = np.load(out_dir / challenger / "oof.npy").astype(np.float64)
    ya, yb = decoded(base, led), decoded(challenger, led)
    res = {}
    for m in (7, 8, 9):
        s = period == m
        res[f"m{m}"] = compare(
            pa[s], pb[s], ya[s], yb[s], {t: v[s] for t, v in targets.items()}, tmpl[s]
        )
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m ringtrace.tabmwhy")
    ap.add_argument("--base", default="champion")
    ap.add_argument("--challenger", default="tabm")
    ap.add_argument("--sets", default="0")
    args = ap.parse_args(argv)
    sets = [int(s) for s in args.sets.split(",")]
    rep = {
        "octlike": octlike_part(args.base, args.challenger, sets),
        "lomo": lomo_part(args.base, args.challenger),
    }
    out = paths.out_dir() / "tabmwhy"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.challenger}_vs_{args.base}.json"
    path.write_text(json.dumps(rep, indent=1))
    print(f"wrote {path}")
    return rep


assert len(CLASSES) == N_CLASSES

if __name__ == "__main__":
    main()
