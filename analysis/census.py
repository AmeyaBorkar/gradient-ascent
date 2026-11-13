"""October census: label-free units, expected against actual October calls (aggregates only).

    python analysis/census.py table   [--recipe recipes/blend-blankqr.json] [--sub <csv>]
    python analysis/census.py explain --unit <name> --cls <code>

``table`` rebuilds the blend's out-of-fold calls from its member runs, counts every label-free
census unit (template, event-family shape, counter zone and segment, mark, anchor, payment kind)
on train and October, and compares October's actual calls (from ``--sub``, default the blend's
own decode) with the calls expected from the out-of-fold rates. ``explain`` compares the inputs
of one unit's October rows with its train rows. October rows are only ever counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from census_core import FRAUD, move, top, unit_table  # noqa: E402

from ringtrace import paths, silver, store  # noqa: E402
from ringtrace.blend import decode_all, load_recipe, weighted_mean  # noqa: E402
from ringtrace.classes import CLASSES, CODE, N_CLASSES  # noqa: E402
from ringtrace.evidence import anchors  # noqa: E402

FAMILIES = ("bot", "drn", "mf", "col", "rng", "rep")
NUMERIC = (
    "amount",
    "sim_age_hours",
    "device_age_days",
    "sender_account_age_days",
    "new_payee",
    "pin_retries",
    "login_to_txn_sec",
    "balance_before",
    "app_version",
    "ip_risk_score",
    "qr_vpa_mismatch",
)


def name_of(c: int) -> str:
    return "any_fraud" if c == FRAUD else CLASSES[c]


def blend_proba(recipe: dict, led: pl.DataFrame) -> np.ndarray:
    te = led["is_test"].to_numpy()
    runs = paths.out_dir() / "runs"
    oofs, tests = [], []
    for m in recipe["members"]:
        oofs.append(np.load(runs / m["run"] / "oof.npy").astype(np.float64))
        tests.append(np.load(runs / m["run"] / "test.npy").astype(np.float64))
    w = [float(m["weight"]) for m in recipe["members"]]
    p = np.zeros((led.height, N_CLASSES))
    p[~te], p[te] = weighted_mean(oofs, w), weighted_mean(tests, w)
    return p


def calls_of(recipe: dict, led: pl.DataFrame, proba: np.ndarray) -> np.ndarray:
    return decode_all(
        proba,
        led["period"].to_numpy(),
        led["is_test"].to_numpy(),
        recipe.get("decoder", {"fraud_scale": 0.875}),
        recipe.get("rules", []),
        led,
        log=lambda m: None,
    )["rules" if recipe.get("rules") else "fixed_point"]


def sub_calls(path: Path, led: pl.DataFrame) -> np.ndarray:
    te = led["is_test"].to_numpy()
    sub = pl.read_csv(path)
    ids = pl.DataFrame({"id": led["txn_id"].to_numpy()[te]})
    lab = ids.join(sub, on="id", how="left", maintain_order="left")["label"].to_list()
    out = np.zeros(led.height, dtype=np.int64)
    out[te] = [CODE[x] for x in lab]
    return out


def census_units(led: pl.DataFrame) -> dict[str, np.ndarray]:
    """Every label-free unit, as a ledger-length mask."""
    u: dict[str, np.ndarray] = {}
    anchor = anchors.row_anchor(led).astype(np.int64)
    tmpl = anchors.template_of(led).astype(np.int64)
    for c in range(N_CLASSES):
        u[f"anchor={CLASSES[c] if c else 'none'}"] = anchor == c
    for t, nm in anchors.TEMPLATE_NAMES.items():
        u[f"tmpl={nm}"] = tmpl == t
    fam = anchors.family_columns(led, [f"{p}_{k}" for p in FAMILIES for k in ("shape", "event")])
    for p in FAMILIES:
        shape = fam[f"{p}_shape"].fill_null(0).to_numpy().astype(np.int64)
        ev = fam[f"{p}_event"].fill_null(-1).to_numpy()
        u[f"{p}:event"] = ev >= 0
        for s in np.unique(shape[ev >= 0]):
            u[f"{p}:shape{int(s)}"] = (shape == s) & (ev >= 0)
    ctr = anchors.family_columns(led, ["ctr_block", "ctr_seg", "ctr_conf", "ctr_dev_rows"])
    block = ctr["ctr_block"].fill_null(0).to_numpy().astype(np.int64)
    seg = ctr["ctr_seg"].cast(pl.Float64).fill_null(-1).fill_nan(-1).to_numpy().astype(np.int64)
    conf = ctr["ctr_conf"].cast(pl.Float64).fill_null(-1).to_numpy()
    for b in np.unique(block):
        u[f"ctr:block{b}"] = block == b
        for s in np.unique(seg[block == b]):
            if s >= 0:
                u[f"ctr:block{b}:seg{CLASSES[s + 1]}"] = (block == b) & (seg == s)
    for lo, hi in ((0.0, 0.8), (0.8, 0.9), (0.9, 1.01)):
        u[f"ctr:block2:conf{lo:.1f}-{min(hi, 1):.1f}"] = (block == 2) & (conf >= lo) & (conf < hi)
    u["ctr:block2:single_row"] = (block == 2) & (ctr["ctr_dev_rows"].fill_null(0).to_numpy() == 1)
    mrk = store.family("marks", verbose=False)
    for col in mrk.columns:
        if col.endswith("_n") or col in ("mrk_count",):
            continue
        u[col] = mrk[col].fill_null(0).to_numpy() > 0
    kind = (led["txn_type"].fill_null("?") + "/" + led["receiver_type"].fill_null("?")).to_numpy()
    for k in np.unique(kind):
        u[f"pay:{k}"] = kind == k
    u["pay:QR/MERCHANT:blank_check"] = (kind == "QR/MERCHANT") & led[
        "qr_vpa_mismatch"
    ].is_null().to_numpy()
    return u


def fine_cells(led: pl.DataFrame) -> np.ndarray:
    """A label-free partition: anchor x template x first family shape x counter block (and
    segment on fraud-block phones) x payment kind x blank QR check x any mark. Returns a string
    key per row."""
    anchor = anchors.row_anchor(led).astype(np.int64)
    tmpl = anchors.template_of(led).astype(np.int64)
    fam = anchors.family_columns(led, [f"{p}_{k}" for p in FAMILIES for k in ("shape", "event")])
    first = np.full(led.height, "-", dtype=object)
    for p in FAMILIES:
        shape = fam[f"{p}_shape"].fill_null(0).to_numpy().astype(np.int64)
        ev = fam[f"{p}_event"].fill_null(-1).to_numpy()
        hit = (first == "-") & (ev >= 0)
        first[hit] = [f"{p}{s}" for s in shape[hit]]
    ctr = anchors.family_columns(led, ["ctr_block", "ctr_seg"])
    block = ctr["ctr_block"].fill_null(0).to_numpy().astype(np.int64)
    seg = ctr["ctr_seg"].cast(pl.Float64).fill_null(-1).fill_nan(-1).to_numpy().astype(np.int64)
    seg = np.where(block == 2, seg, -1)
    mrk = anchors.family_columns(led, ["mrk_any"])["mrk_any"].fill_null(0).to_numpy() > 0
    kind = (led["txn_type"].fill_null("?") + "/" + led["receiver_type"].fill_null("?")).to_numpy()
    blank = led["qr_vpa_mismatch"].is_null().to_numpy() & (led["txn_type"] == "QR").to_numpy()
    return np.array(
        [
            f"a{a}|t{t}|{f}|b{b}s{g}|{k}{'|blank' if bl else ''}{'|mrk' if m else ''}"
            for a, t, f, b, g, k, bl, m in zip(
                anchor, tmpl, first, block, seg, kind, blank, mrk, strict=True
            )
        ],
        dtype=object,
    )


def cmd_cells(a) -> dict:
    """Composition-controlled census: expected calls per class from the fine cells, and the
    October rows in cells that train holds fewer than ``min_train`` times."""
    led, proba, calls, clean = load_all(a.recipe, a.sub)
    te = led["is_test"].to_numpy()
    tr = ~te
    key = fine_cells(led)
    uk, inv = np.unique(key, return_inverse=True)
    units = {}
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uk) + 1))
    for i, k in enumerate(uk):
        m = np.zeros(led.height, bool)
        m[order[bounds[i] : bounds[i + 1]]] = True
        units[str(k)] = m
    rows = unit_table(units, tr, te, calls, calls, clean, N_CLASSES, min_train=a.min_train)
    covered = np.zeros(led.height, bool)
    for r in rows:
        if r["cls"] == FRAUD:
            covered |= units[r["unit"]]
    unseen = te & ~covered
    print(
        f"[cells] {len(uk)} cells; October rows in cells with < {a.min_train} train rows: "
        f"{int(unseen.sum())}, called {np.bincount(calls[unseen], minlength=N_CLASSES).tolist()}"
    )
    thin = [
        (str(k), int((units[str(k)] & tr).sum()), int((units[str(k)] & te).sum()))
        for k in uk
        if (units[str(k)] & te).sum() and (units[str(k)] & tr).sum() < a.min_train
    ]
    for k, ntr, nte in sorted(thin, key=lambda x: -x[2])[:20]:
        m = units[k]
        print(
            f"  thin cell {k:<52} train {ntr:>3} "
            f"(clean {np.bincount(clean[m & tr], minlength=N_CLASSES).tolist()})"
            f" October {nte:>4} called {np.bincount(calls[m & te], minlength=N_CLASSES).tolist()}"
        )
    exp = np.zeros(N_CLASSES)
    act = np.zeros(N_CLASSES)
    for r in rows:
        if r["cls"] >= 0:
            exp[r["cls"]] += r["expected"]
            act[r["cls"]] += r["actual"]
    print("[cells] per class over covered rows: class, expected, actual, ratio")
    for c in range(N_CLASSES):
        print(f"  {CLASSES[c]:<20}{exp[c]:>10.1f}{act[c]:>8.0f}{act[c] / max(exp[c], 1e-9):>8.3f}")
    for title, sign in (("DEFICITS", -1), ("SURPLUSES", 1)):
        print(f"== cell {title}")
        for r in top(rows, a.top, sign, min_gap=2.0):
            print(
                f"{r['unit']:<52}{name_of(r['cls']):<20}{r['n_train']:>8}{r['n_oct']:>6}"
                f"{100 * r['oof_rate']:>7.2f}{100 * r['clean_rate']:>7.2f}"
                f"{r['oof_precision']:>7.3f}{r['expected']:>7.1f}{r['actual']:>5}{r['z']:>7.2f}"
            )
    out = paths.out_dir() / "census"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells.json").write_text(json.dumps(rows, indent=0))
    return {"rows": len(rows)}


def load_all(recipe_path: str, sub: str | None):
    recipe = load_recipe(recipe_path)
    led = store.ledger()
    proba = blend_proba(recipe, led)
    calls = calls_of(recipe, led, proba)
    te = led["is_test"].to_numpy()
    if sub:
        actual = sub_calls(Path(sub), led)
        print(
            f"[census] own decode vs {Path(sub).name}: "
            f"{int((actual[te] != calls[te]).sum())} differ"
        )
        calls = np.where(te, actual, calls)
    clean = silver.load(led, "label").astype(np.int64)
    return led, proba, calls, clean


def cmd_table(a) -> dict:
    led, proba, calls, clean = load_all(a.recipe, a.sub)
    te = led["is_test"].to_numpy()
    tr = ~te
    units = census_units(led)
    rows = unit_table(units, tr, te, calls, calls, clean, N_CLASSES)
    # known effects: classes whose total October calls already depart for an explained reason
    print(f"[census] {len(units)} units, {len(rows)} (unit, class) cells")
    for title, sign in (("DEFICITS", -1), ("SURPLUSES", 1)):
        print(f"== {title} (actual - expected)/sqrt(expected), |gap| >= 3 rows")
        print(
            f"{'unit':<34}{'class':<20}{'n_tr':>9}{'n_oct':>7}{'oof%':>8}{'clean%':>8}"
            f"{'prec':>7}{'exp':>8}{'act':>6}{'z':>7}"
        )
        for r in top(rows, a.top, sign):
            print(
                f"{r['unit']:<34}{name_of(r['cls']):<20}{r['n_train']:>9}{r['n_oct']:>7}"
                f"{100 * r['oof_rate']:>8.2f}{100 * r['clean_rate']:>8.2f}"
                f"{r['oof_precision']:>7.3f}{r['expected']:>8.1f}{r['actual']:>6}{r['z']:>7.2f}"
            )
    out = paths.out_dir() / "census"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps(rows, indent=0))
    return {"rows": len(rows)}


def cmd_explain(a) -> dict:
    led, proba, calls, clean = load_all(a.recipe, a.sub)
    te = led["is_test"].to_numpy()
    u = census_units(led)[a.unit]
    c = a.cls
    ts = led["timestamp"]
    feats = {k: led[k].cast(pl.Float64).to_numpy() for k in NUMERIC}
    feats["hour"] = (ts.dt.hour() + ts.dt.minute() / 60).to_numpy().astype(float)
    feats[f"p_{CLASSES[c]}"] = proba[:, c]
    feats["p_legit"] = proba[:, 0]
    trs = u & ~te & (calls == c)
    oct_hit = u & te & (calls == c)
    oct_miss = u & te & (calls != c)
    print(
        f"[explain] {a.unit} / {CLASSES[c]}: train called {int(trs.sum())}, "
        f"October called {int(oct_hit.sum())}, October not called {int(oct_miss.sum())}"
    )
    went = np.bincount(calls[oct_miss], minlength=N_CLASSES)
    print(
        "  October not-called rows go to:",
        {CLASSES[k]: int(v) for k, v in enumerate(went) if v},
    )
    rep = {}
    for k, v in feats.items():
        r = {"train_called": move(v[trs], v[oct_miss]), "oct_called": move(v[oct_hit], v[oct_miss])}
        rep[k] = r
        t = r["train_called"]
        if t["train_mean"] is None:
            continue
        print(
            f"  {k:<26} train-called mean {t['train_mean']:>12.3f}  October-missed mean "
            f"{t['oct_mean']:>12.3f}  shift {t['shift_sd']:+.2f} sd"
        )
    for k in NUMERIC:
        v = led[k].is_null().to_numpy()
        print(
            f"  {k:<26} null share train-called {v[trs].mean():.3f}  October-missed "
            f"{v[oct_miss].mean() if oct_miss.any() else float('nan'):.3f}"
        )
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "cells", "explain"])
    ap.add_argument("--recipe", default="recipes/blend-blankqr.json")
    ap.add_argument("--sub", default=None)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--unit", default="")
    ap.add_argument("--min-train", type=int, default=20)
    ap.add_argument("--cls", type=int, default=1)
    a = ap.parse_args(argv)
    {"table": cmd_table, "cells": cmd_cells, "explain": cmd_explain}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
