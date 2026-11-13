"""An offline stand-in for the clean test labels ("A-clean"), and macro-F1 under it.

The test labels count every fraud-pass row as fraud, ordinary-looking rows included. The cleaned
train labels recover the unreported generator-made rows, but the unreported *ordinary* rows look
legit and stay legit: about 3,482 train rows. They cannot be found row by row (nothing separates
them from legit traffic), so the proxy scores their expectation: every cleaned-legit row of a
month is hidden fraud of class k with the same probability ``h_k``, where the month's hidden count
of class k is shared out in proportion to that month's ordinary-looking class-k labels.

The expected confusion matrix is then exact: a row with cleaned label 0 and call c adds ``h_k`` to
cell (k, c) and ``1 - sum h`` to cell (0, c). Macro-F1 is read off the expected matrix.

    python -m ringtrace.aclean rank        # the public files' train-month analogues, per month
    python -m ringtrace.aclean sweep       # decoder settings on the fused fraud mass
    python -m ringtrace.aclean mass        # fraud-mass bands and per-class reliability
    python -m ringtrace.aclean rare        # rows just below each rare class's cut-off
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ringtrace.classes import CLASSES, N_CLASSES

HIDDEN_TOTAL = 3482  # train fraud-pass rows that the cleaned labels leave legit


def hidden_counts(
    cleaned: np.ndarray, structured: np.ndarray, period: np.ndarray, total: float = HIDDEN_TOTAL
) -> dict[int, np.ndarray]:
    """Expected hidden fraud rows per month and class (length-k vectors, legit = 0), shared out
    in proportion to the ordinary-looking labels (cleaned fraud, structured legit)."""
    ordinary = (cleaned > 0) & (structured == 0)
    out = {}
    grand = ordinary.sum()
    for m in np.unique(period):
        sel = ordinary & (period == m)
        out[int(m)] = np.bincount(cleaned[sel], minlength=N_CLASSES).astype(float) * total / grand
    return out


def expected_confusion(
    cleaned: np.ndarray, pred: np.ndarray, hidden: np.ndarray, k: int = N_CLASSES
) -> np.ndarray:
    """Expected confusion (truth x call) when ``hidden[c]`` of the cleaned-legit rows are, at
    random, class c."""
    cm = np.bincount(cleaned * k + pred, minlength=k * k).reshape(k, k).astype(float)
    legit_calls = cm[0].copy()
    h = hidden / max(legit_calls.sum(), 1.0)
    h[0] = 0.0
    cm[0] -= h.sum() * legit_calls
    cm += h[:, None] * legit_calls[None, :]
    return cm


def f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def aclean_f1(cleaned: np.ndarray, pred: np.ndarray, hidden: np.ndarray) -> float:
    return float(f1_from_confusion(expected_confusion(cleaned, pred, hidden)).mean())


def per_month(
    pred: np.ndarray, targets: dict, period: np.ndarray, hidden: dict, months=(7, 8, 9)
) -> dict[str, list[float]]:
    """Macro-F1 per month on every target, plus ``aclean`` from the cleaned one."""
    from ringtrace.scoring import macro_f1

    out = {}
    for t, y in targets.items():
        out[t] = [macro_f1(y[period == m], pred[period == m]) for m in months]
    y = targets["cleaned"]
    out["aclean"] = [aclean_f1(y[period == m], pred[period == m], hidden[m]) for m in months]
    return out


def scaled_rows(proba: np.ndarray, fraud_scale: float) -> np.ndarray:
    """Rows as the decoder sees them: fraud columns divided by ``fraud_scale``, renormalised."""
    p = np.asarray(proba, dtype=np.float64).copy()
    p[:, 1:] /= fraud_scale
    return p / p.sum(axis=1, keepdims=True)


def band_table(mass: np.ndarray, edges=(0.0, 0.01, 0.05, 0.2, 0.5, 0.95, 1.0001)) -> np.ndarray:
    """Rows per fraud-mass band (edges are left-closed)."""
    return np.histogram(mass, bins=np.asarray(edges))[0]


def call_margin(
    proba: np.ndarray,
    pred: np.ndarray,
    k: int,
    fraud_scale: float = 0.875,
    hidden_factor: float = 1.0,
) -> np.ndarray:
    """For rows not called k: their class-k probability over the fixed point's admission cut
    F_k / 2 (above 1 they would be admitted if nothing else competed)."""
    p = scaled_rows(proba, fraud_scale)
    n = p[:, k].sum() * (hidden_factor if k else 1.0)
    called = pred == k
    tp = p[called, k].sum()
    f = 2 * tp / max(n + called.sum(), 1e-12)
    return p[:, k] / max(f / 2, 1e-12)


# ----------------------------------------------------------------------------------- CLI
MEMBERS = ["blend-champion-m1", "lgbm", "catboost", "clean-event", "clean-l2"]


def _setup():
    from ringtrace import silver, store

    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    tr = np.flatnonzero(~is_test)
    period = led["period"].to_numpy()[tr]
    cleaned = silver.load(led, "label")[tr]
    targets = {
        "given": led["y"].to_numpy()[tr],
        "cleaned": cleaned,
        "gen": silver.load(led, "structured")[tr],
    }
    hidden = hidden_counts(cleaned, targets["gen"], period)
    return led, tr, period, targets, hidden


def _arrays(spec):
    from ringtrace.blend import blend_arrays, recipe_arrays

    if isinstance(spec, dict):
        return recipe_arrays(spec, "oof"), recipe_arrays(spec, "test")
    return blend_arrays(spec, "oof"), blend_arrays(spec, "test")


def five_way():
    return [(m, 0.2) for m in MEMBERS]


def tabm_full():
    return [("tabm", 0.5)] + [(m, 0.1) for m in MEMBERS]


def tabm_fraud_only() -> dict:
    return {
        "members": [{"run": m, "weight": 1} for m in MEMBERS],
        "split": [{"run": r, "weight": w} for r, w in tabm_full()],
    }


def calls(led, oof, test, rules, template_class=None, **dec) -> np.ndarray:
    """Every ledger row's call: fixed point per month (``dec``), then ``rules``. A
    ``template_class`` dict overrides the template rule's class per template id."""
    from ringtrace import rules as rulebook
    from ringtrace.decode import decode_by_period
    from ringtrace.evidence import anchors

    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    pred = np.zeros(led.height, dtype=np.int64)
    full = np.zeros((led.height, oof.shape[1]))
    full[tr], full[te] = oof, test
    pred[tr] = decode_by_period(oof, period[tr], **dec)
    pred[te] = decode_by_period(test, period[te], **dec)
    saved = dict(anchors.TEMPLATE_CLASS)
    try:
        if template_class:
            anchors.TEMPLATE_CLASS.update(template_class)
        return rulebook.apply(rules, pred, full, led, log=lambda *_: None)
    finally:
        anchors.TEMPLATE_CLASS.clear()
        anchors.TEMPLATE_CLASS.update(saved)


def _fmt(scores: dict) -> str:
    return "  ".join(
        f"{t} " + " / ".join(f"{s:.4f}" for s in v) + f" ({np.mean(v):.4f})"
        for t, v in scores.items()
    )


def cmd_rank(args) -> dict:
    from ringtrace.evidence.anchors import PHISH

    led, tr, period, targets, hidden = _setup()
    rule_a = ["anchored_templates"]
    rule_ab = ["anchored_templates", "blank_merchant_qr"]
    files = {
        "champion": ([("champion", 1.0)], rule_a, None, 0.5470),
        "five-way": (five_way(), rule_a, None, 0.5474),
        "blend-blankqr": (five_way(), rule_ab, None, 0.5486),
        "blend-col7-phishing": (five_way(), rule_ab, {4: PHISH}, 0.5408),
        "tabm-full-bq": (tabm_full(), rule_ab, None, 0.5483),
        "tabmF-bq": (tabm_fraud_only(), rule_ab, None, 0.5499),
    }
    out = {}
    for name, (spec, rules, tc, public) in files.items():
        oof, test = _arrays(spec)
        pred = calls(led, oof, test, rules, template_class=tc, fraud_scale=0.875)[tr]
        sc = per_month(pred, targets, period, hidden)
        out[name] = {"public": public, **sc}
        print(f"{name:<20} public {public:.4f}  {_fmt(sc)}", flush=True)
    return out


def fused(which: str = "oof"):
    from ringtrace.blend import recipe_arrays

    return recipe_arrays(tabm_fraud_only(), which)


def cmd_mass(args) -> dict:
    from ringtrace.blend import blend_arrays

    led, tr, period, targets, hidden = _setup()
    edges = (0.0, 0.01, 0.05, 0.2, 0.5, 0.95, 1.0001)
    out = {}
    for name, oof in (("five-way", blend_arrays(five_way())), ("fused", fused())):
        mass = 1 - oof[:, 0]
        row = {"bands_per_100k": (band_table(mass, edges) / len(mass) * 1e5).round(1).tolist()}
        mid = (mass >= 0.05) & (mass < 0.95)
        row["mid_share_rows"] = float(mid.mean())
        row["mid_share_mass"] = float(mass[mid].sum() / mass.sum())
        rel = {}
        for t in ("given", "cleaned"):
            y = targets[t]
            b = np.digitize(mass, edges[1:-1])
            rel[t] = [
                float((y[b == i] > 0).mean()) if (b == i).any() else None
                for i in range(len(edges) - 1)
            ]
        row["fraud_rate_by_band"] = rel
        # per class: top-class probability bands against cleaned precision of that class
        cls = {}
        pk_edges = (0.05, 0.2, 0.5, 0.8, 0.95, 1.0001)
        for k in range(1, N_CLASSES):
            pk = oof[:, k]
            r = []
            for lo, hi in zip(pk_edges[:-1], pk_edges[1:], strict=True):
                s = (pk >= lo) & (pk < hi)
                r.append(
                    [
                        int(s.sum()),
                        float(oof[s, k].mean()) if s.any() else None,
                        float((targets["cleaned"][s] == k).mean()) if s.any() else None,
                    ]
                )
            cls[CLASSES[k]] = r
        row["class_reliability"] = cls
        out[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "class_reliability"}))
        for c, r in cls.items():
            print(f"   {c:<20}", "  ".join(f"n {n:>5} p {_r(p)} prec {_r(q)}" for n, p, q in r))
    return out


def _r(x):
    return x if x is None else round(x, 2)


def sweep_grid():
    grid = []
    for hf in (1.0, 1.5, 2.0, 2.7):
        for fs in (0.8, 0.875, 1.0):
            grid.append({"hidden_factor": hf, "fraud_scale": fs, "rare_only": False})
    for hf in (1.5, 2.0, 2.7):
        grid.append({"hidden_factor": hf, "fraud_scale": 0.875, "rare_only": True})
    return grid


def cmd_sweep(args) -> dict:
    led, tr, period, targets, hidden = _setup()
    oof, test = fused(), fused("test")
    rules = ["anchored_templates", "blank_merchant_qr"]
    base = None
    out = []
    grid = sweep_grid()
    incumbent = {"hidden_factor": 1.0, "fraud_scale": 0.875, "rare_only": False}
    grid.remove(incumbent)
    for g in [incumbent, *grid]:  # deltas are against the incumbent decoder
        hf = g["hidden_factor"]
        if g["rare_only"]:
            hf = np.ones(N_CLASSES)
            hf[rare_classes(oof)] = g["hidden_factor"]
        pred_all = calls(led, oof, test, rules, fraud_scale=g["fraud_scale"], hidden_factor=hf)
        pred = pred_all[tr]
        sc = per_month(pred, targets, period, hidden)
        te_calls = np.bincount(pred_all[~np.isin(np.arange(led.height), tr)], minlength=N_CLASSES)
        if base is None:
            base = (sc, pred_all)
        delta = {t: [a - b for a, b in zip(sc[t], base[0][t], strict=True)] for t in sc}
        n_oct = int((pred_all != base[1])[led["is_test"].to_numpy()].sum())
        rec = {
            **g,
            "scores": sc,
            "delta": delta,
            "october_changed": n_oct,
            "october_fraud_calls": int(te_calls[1:].sum()),
        }
        out.append(rec)
        print(
            f"hf {g['hidden_factor']:<4} fs {g['fraud_scale']:<5} rare {g['rare_only']!s:<5} "
            f"Oct changed {n_oct:>4} fraud {rec['october_fraud_calls']}  "
            + "  ".join(f"{t} " + "/".join(f"{x:+.5f}" for x in v) for t, v in delta.items()),
            flush=True,
        )
    return out


def rare_classes(oof: np.ndarray, n: int = 4) -> np.ndarray:
    """The ``n`` classes with the least probability mass (label-free)."""
    return 1 + np.argsort(oof[:, 1:].sum(axis=0))[:n]


RARE = ("loan_app_extortion", "merchant_collusion", "fake_cashback", "velocity_bot", "refund_abuse")


def cmd_rare(args) -> dict:
    from ringtrace.classes import CODE

    led, tr, period, targets, hidden = _setup()
    is_test = led["is_test"].to_numpy()
    te = np.flatnonzero(is_test)
    oof, test = fused(), fused("test")
    rules = ["anchored_templates", "blank_merchant_qr"]
    pred_all = calls(led, oof, test, rules, fraud_scale=0.875)
    ptr, pte = pred_all[tr], pred_all[te]
    bands = ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0))  # of the cut F_k / 2
    out = {}
    for name in RARE:
        k = CODE[name]
        rec = {"bands": [], "months": {}}
        mtr = np.zeros(len(tr))
        for m in (7, 8, 9):
            s = period == m
            mtr[s] = call_margin(oof[s], ptr[s], k)
        mte = call_margin(test, pte, k)
        y = targets["cleaned"]
        cm_base = {
            m: expected_confusion(y[period == m], ptr[period == m], hidden[m]) for m in (7, 8, 9)
        }
        for lo, hi in bands:
            btr = (ptr != k) & (mtr >= lo) & (mtr < hi)
            bte = (pte != k) & (mte >= lo) & (mte < hi)
            gains = []
            for m in (7, 8, 9):
                s = period == m
                new = ptr[s].copy()
                new[btr[s]] = k
                cm = expected_confusion(y[s], new, hidden[m])
                gains.append(
                    float(f1_from_confusion(cm).mean() - f1_from_confusion(cm_base[m]).mean())
                )
            prec = {
                t: float((targets[t][btr] == k).mean()) if btr.any() else None
                for t in ("given", "cleaned")
            }
            from_legit = float((ptr[btr] == 0).mean()) if btr.any() else None
            # A-clean precision adds the hidden share on cleaned-legit rows
            hid = np.mean([hidden[m][k] / max((y[period == m] == 0).sum(), 1) for m in (7, 8, 9)])
            prec["aclean"] = (
                None if not btr.any() else prec["cleaned"] + float((y[btr] == 0).mean()) * hid
            )
            f1_k = float(np.mean([f1_from_confusion(cm_base[m])[k] for m in (7, 8, 9)]))
            rec["bands"].append(
                {
                    "band": [lo, hi],
                    "train_rows": int(btr.sum()),
                    "october_rows": int(bte.sum()),
                    "precision": prec,
                    "from_legit": from_legit,
                    "aclean_gain_months": gains,
                    "class_f1_aclean": f1_k,
                }
            )
            print(
                f"{name:<20} band {lo:.2f}-{hi:.2f} of F/2  train {int(btr.sum()):>4} "
                f"Oct {int(bte.sum()):>3}  prec given {prec['given']} cleaned "
                f"{prec['cleaned']} aclean {prec['aclean']}  F1/2 {f1_k / 2:.3f}  "
                f"gain " + "/".join(f"{g:+.5f}" for g in gains),
                flush=True,
            )
        out[name] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ringtrace.aclean")
    ap.add_argument("cmd", choices=["rank", "mass", "sweep", "rare"])
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    res = {"rank": cmd_rank, "mass": cmd_mass, "sweep": cmd_sweep, "rare": cmd_rare}[args.cmd](args)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    return res


if __name__ == "__main__":
    main()
