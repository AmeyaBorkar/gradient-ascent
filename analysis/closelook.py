"""A close look at the best public file: per-class anatomy, call volume, look-alike pairs, October.

    python analysis/closelook.py prep     # rebuild blend-tabmF-bq from arrays, cache arrays
    python analysis/closelook.py anatomy  # TP / FP / FN per class and month, three targets
    python analysis/closelook.py volume   # per-class volume scaling, nested over the months
    python analysis/closelook.py pairs    # look-alike templates by counter hold and size
    python analysis/closelook.py october  # October calls and mass per class against train

The best file is the recipe ``recipes/final-half.json`` (five tree members for the type split, the
fraud mass from TabM 1/2 and each tree 1/10, the fixed point at 0.875, the anchored template rule,
then ``blank_merchant_qr``). ``prep`` checks that the rebuilt October calls equal the uploaded
file on every row. October rows are only ever counted.

The anchored template rule and ``blank_merchant_qr`` read only each row's own call, so ``prep``
stores their per-row targets once and the other commands re-apply them in a vectorised step
(``fast_rules``); ``prep`` checks that this reproduces the recipe's calls on every row.

Outputs go to ``$RINGTRACE_OUT/closelook/`` (``prep.npz``, ``rules.npz`` and one JSON per
command).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from closelook_core import (  # noqa: E402
    class_table,
    fixed_point_scaled,
    hidden_counts,
    macro,
)

from ringtrace import paths  # noqa: E402
from ringtrace.classes import CLASSES, N_CLASSES  # noqa: E402

MONTHS = (7, 8, 9)
TARGETS = ("given", "cleaned", "aclean", "gen")
RECIPE = "recipes/final-half.json"
BEST_FILE = "blend-tabmF-bq.csv"


def out_dir() -> Path:
    d = paths.out_dir() / "closelook"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tabm_run() -> str:
    """The saved TabM run (3 seeds, given labels) that the recipe calls ``final-tabm``."""
    import os

    return os.environ.get("CLOSELOOK_TABM_RUN", "tabm")


def best_recipe(path: str = RECIPE) -> dict:
    from ringtrace.final import load_recipe

    r = load_recipe(path)
    for m in r["members"]:
        if m["run"] == "final-tabm":
            m["run"] = tabm_run()
    return r


def bench(cells: Path | None = None):
    from ringtrace.final import Bench

    cells = cells or paths.out_dir() / "lens-cells"
    return Bench.from_store(paths.out_dir() / "runs", cells)


def mixed_proba(b, recipe: dict) -> np.ndarray:
    from ringtrace.final import TRAIN_MONTHS, mix

    arrays = [b.member(m) for m in recipe["members"]]
    roles = [m["role"] for m in recipe["members"]]
    w = b.weights(recipe)
    proba = np.zeros((b.led.height, N_CLASSES))
    for k in (*TRAIN_MONTHS, "october"):
        rows = b.te if k == "october" else b.tr[b.period[b.tr] == k]
        proba[rows] = mix([a[rows] for a in arrays], w[k], roles)
    return proba


def read_calls(path: Path, ids: np.ndarray) -> np.ndarray:
    import polars as pl

    f = pl.read_csv(path, infer_schema=False)
    pos = {c: i for i, c in enumerate(CLASSES)}
    lab = dict(zip(f["id"].to_list(), f["label"].to_list(), strict=True))
    return np.array([pos[lab[str(i)]] for i in ids], dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# fast rules: the anchored template rule and blank_merchant_qr only read each row's own call


def rule_targets(led, proba: np.ndarray, tmpl: np.ndarray) -> dict:
    """Per row: the class the anchored rule gives a template row whose call is in the template's
    pair, and the blank-QR mask with its most likely fraud class."""
    from ringtrace import rules as rulebook
    from ringtrace.evidence.anchors import TEMPLATE_PAIR

    seed = np.zeros(led.height, dtype=np.int64)
    pair0 = np.zeros(7, dtype=np.int64)
    pair1 = np.zeros(7, dtype=np.int64)
    for t, (a, b) in TEMPLATE_PAIR.items():
        pair0[t], pair1[t] = a, b
    seed[tmpl > 0] = pair0[tmpl[tmpl > 0]]
    target = rulebook.apply(["anchored_templates"], seed, proba, led, log=lambda *_: None)
    qr = (
        (led["txn_type"] == "QR").to_numpy()
        & (led["receiver_type"] == "MERCHANT").to_numpy()
        & led["qr_vpa_mismatch"].is_null().to_numpy()
    )
    return {
        "tmpl": tmpl,
        "pair0": pair0,
        "pair1": pair1,
        "target": target,
        "qr": qr,
        "qr_class": proba[:, 1:].argmax(axis=1) + 1,
    }


def fast_rules(pred: np.ndarray, rt: dict, rows: np.ndarray | None = None) -> np.ndarray:
    """``anchored_templates`` then ``blank_merchant_qr`` on ``pred`` (aligned to ``rows``)."""
    sel = slice(None) if rows is None else rows
    tmpl, target = rt["tmpl"][sel], rt["target"][sel]
    out = pred.copy()
    inpair = (tmpl > 0) & ((pred == rt["pair0"][tmpl]) | (pred == rt["pair1"][tmpl]))
    out[inpair] = target[inpair]
    q = rt["qr"][sel] & (out == 0)
    out[q] = rt["qr_class"][sel][q]
    return out


# ---------------------------------------------------------------------------------------------
# prep


def cmd_prep() -> dict:
    from ringtrace.decode import fixed_point
    from ringtrace.evidence import anchors
    from ringtrace.rules import _counter, label_free_events

    t0 = time.time()
    b = bench()
    recipe = best_recipe()
    proba = mixed_proba(b, recipe)
    calls = b.calls(recipe)["pred"]
    fp = np.zeros(b.led.height, dtype=np.int64)
    for k in (*MONTHS, "october"):
        rows = b.te if k == "october" else b.tr[b.period[b.tr] == k]
        fp[rows] = fixed_point(proba[rows], **recipe["decoder"])
    ids = b.led["txn_id"].to_numpy()[b.te]
    up = read_calls(paths.out_dir() / "candidates" / BEST_FILE, ids)
    tmpl, tev = anchors.templates(b.led)
    tmpl = np.asarray(tmpl, dtype=np.int64)
    c = _counter(b.led)
    rt = rule_targets(b.led, proba, tmpl)
    np.savez_compressed(out_dir() / "rules.npz", **rt)
    np.savez_compressed(
        out_dir() / "prep.npz",
        proba=proba,
        calls=calls,
        fixed=fp,
        is_test=b.is_test,
        period=b.period,
        given=b.targets["given"],
        cleaned=b.targets["cleaned"],
        gen=b.targets["gen"],
        tmpl=tmpl,
        tmpl_event=np.asarray(tev, dtype=np.int64),
        anchor=anchors.row_anchor(b.led).astype(np.int64),
        ctr_block=c["block"],
        ctr_seg=c["seg_class"],
        ctr_conf=c["conf"],
        events=label_free_events(b.led),
        groups=b.groups,
    )
    rep = {
        "recipe": recipe["name"],
        "tabm_run": tabm_run(),
        "october_rows_differing_from_upload": int((up != calls[b.te]).sum()),
        "fast_rules_rows_differing": int((fast_rules(fp, rt) != calls).sum()),
        "scores": b.lomo(recipe),
        "wall_s": time.time() - t0,
    }
    (out_dir() / "prep.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


def load_prep() -> dict:
    z = np.load(out_dir() / "prep.npz")
    return {k: z[k] for k in z.files}


def _ctx() -> dict:
    """The cached arrays plus the per-month index sets and the A-clean hidden counts."""
    d = load_prep()
    z = np.load(out_dir() / "rules.npz")
    d["rt"] = {k: z[k] for k in z.files}
    tr = ~d["is_test"]
    d["per_tr"] = d["period"][tr]
    d["rows"] = {m: np.flatnonzero(d["period"] == m) for m in MONTHS}
    d["rows"]["oct"] = np.flatnonzero(d["is_test"])
    d["hid"] = hidden_counts(d["cleaned"], d["gen"], d["per_tr"])
    return d


def month_targets(d: dict, m: int, pred_m: np.ndarray) -> dict[str, float]:
    s = d["per_tr"] == m
    return {
        "given": macro(d["given"][s], pred_m),
        "cleaned": macro(d["cleaned"][s], pred_m),
        "aclean": macro(d["cleaned"][s], pred_m, d["hid"][m]),
        "gen": macro(d["gen"][s], pred_m),
    }


def pair_counts(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    ch = a != b
    keys, n = np.unique(a[ch] * N_CLASSES + b[ch], return_counts=True)
    return {
        f"{CLASSES[k // N_CLASSES]} -> {CLASSES[k % N_CLASSES]}": int(c)
        for k, c in zip(keys, n, strict=True)
    }


# ---------------------------------------------------------------------------------------------
# anatomy


def cmd_anatomy() -> dict:
    d = _ctx()
    calls = d["calls"][~d["is_test"]]
    rep = {}
    for m in MONTHS:
        s = d["per_tr"] == m
        rep[str(m)] = {
            "given": class_table(d["given"][s], calls[s]),
            "cleaned": class_table(d["cleaned"][s], calls[s]),
            "aclean": class_table(d["cleaned"][s], calls[s], d["hid"][m]),
            "gen": class_table(d["gen"][s], calls[s]),
        }
    (out_dir() / "anatomy.json").write_text(json.dumps(rep, indent=1))
    for t in ("given", "cleaned", "aclean", "gen"):
        print(f"== {t}: per month true / calls / TP / precision / recall / F1")
        for c in range(N_CLASSES):
            cells = []
            for m in MONTHS:
                r = rep[str(m)][t][str(c)]
                cells.append(
                    f"{r['true']:7.0f} {r['calls']:6.0f} {r['tp']:6.0f} "
                    f"{r['precision']:.2f} {r['recall']:.2f} {r['f1']:.3f}"
                )
            print(f"{CLASSES[c][:18]:<19}| " + " | ".join(cells))
        print("macro", [round(rep[str(m)][t]["macro"], 5) for m in MONTHS])
    return rep


# ---------------------------------------------------------------------------------------------
# volume

VOLUME_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.15, 1.4, 2.0)


def cmd_volume() -> dict:
    """Each class's admission multiplier alone, per month; then a nested choice per class
    (chosen on two months by the A-clean change, kept only if no fit month loses on any target)
    scored on the third, and the October rows of the choice made on July + August."""
    d = _ctx()
    base = {m: month_targets(d, m, d["calls"][d["rows"][m]]) for m in MONTHS}

    def run(mult: np.ndarray, months=(*MONTHS, "oct")) -> dict:
        out = {}
        for m in months:
            r = d["rows"][m]
            p = fast_rules(fixed_point_scaled(d["proba"][r], mult), d["rt"], r)
            if m == "oct":
                out[m] = pair_counts(d["calls"][r], p)
            else:
                s = month_targets(d, m, p)
                out[m] = {t: s[t] - base[m][t] for t in s}
                out[m]["rows"] = int((p != d["calls"][r]).sum())
        return out

    sweep = {}
    for k in range(1, N_CLASSES):
        for mu in VOLUME_GRID:
            mv = np.ones(N_CLASSES)
            mv[k] = mu
            sweep[f"{k}_{mu}"] = res = run(mv)
            print(
                f"{CLASSES[k][:16]:<16} m={mu:<4} "
                + " | ".join(
                    f"{m}: n {res[m]['rows']:3d} "
                    + " ".join(f"{t[0]} {res[m][t] * 1e5:+6.1f}" for t in TARGETS)
                    for m in MONTHS
                )
                + f" | oct {sum(res['oct'].values())}",
                flush=True,
            )
    nested = {}
    for h in (*MONTHS, "oct"):
        fit = [m for m in MONTHS if m != h] if h != "oct" else [7, 8]
        mv = np.ones(N_CLASSES)
        for k in range(1, N_CLASSES):
            ok = [
                mu
                for mu in VOLUME_GRID
                if all(sweep[f"{k}_{mu}"][m][t] >= 0 for m in fit for t in TARGETS)
            ]
            if not ok:
                continue
            best = max(ok, key=lambda mu, k=k: sum(sweep[f"{k}_{mu}"][m]["aclean"] for m in fit))
            if sum(sweep[f"{k}_{best}"][m]["aclean"] for m in fit) > 0:
                mv[k] = best
        res = run(mv, months=(h,))[h]
        nested[str(h)] = {"mult": {CLASSES[k]: mv[k] for k in range(1, N_CLASSES) if mv[k] != 1}}
        nested[str(h)]["pairs" if h == "oct" else "change"] = res
        print(f"nested, held {h}: {nested[str(h)]}")
    rep = {"sweep": sweep, "nested": nested}
    (out_dir() / "volume.json").write_text(json.dumps(rep, indent=1))
    return rep


# ---------------------------------------------------------------------------------------------
# look-alike pairs


def event_key(d: dict) -> np.ndarray:
    ev = np.where(d["tmpl_event"] >= 0, d["tmpl_event"], 0)
    return d["tmpl"].astype(np.int64) * 10**9 + ev


def held_by_counter(d: dict) -> np.ndarray:
    """Template rows whose template event holds a counter anchor of the pair (the anchored rule
    sets their class from the counter)."""
    from ringtrace.evidence.anchors import TEMPLATE_PAIR

    tmpl = d["tmpl"]
    anch = np.zeros(len(tmpl), dtype=bool)
    for t, pair in TEMPLATE_PAIR.items():
        anch |= (
            (tmpl == t)
            & (d["ctr_block"] == 2)
            & np.isin(d["ctr_seg"], pair)
            & (d["ctr_conf"] >= 0.8)
        )
    key = event_key(d)
    return (tmpl > 0) & np.isin(key, np.unique(key[anch]))


def cmd_pairs() -> dict:
    """Every look-alike template: truth against call by counter hold and event size (train),
    October call counts, the within-pool AUC of the blend's pair score, and a nested split that
    gives the top-scored events of the unheld pool to the partner class."""
    from sklearn.metrics import roc_auc_score

    from ringtrace.evidence.anchors import TEMPLATE_NAMES, TEMPLATE_PAIR

    d = _ctx()
    tr = ~d["is_test"]
    tmpl, calls, proba, per = d["tmpl"], d["calls"], d["proba"], d["period"]
    key = event_key(d)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    size = np.where(tmpl > 0, np.minimum(cnt[inv], 6), 0)
    held = held_by_counter(d)
    cleaned = np.full(len(tmpl), -1)
    cleaned[tr] = d["cleaned"]
    base = {m: month_targets(d, m, calls[d["rows"][m]]) for m in MONTHS}
    rep = {}
    for t, (a, b) in TEMPLATE_PAIR.items():
        name = TEMPLATE_NAMES[t]
        cells = []
        for h in (True, False):
            for sz in range(1, 7):
                s = (tmpl == t) & (held == h) & (size == sz)
                st, so = s & tr, s & d["is_test"]
                if not s.any():
                    continue
                y, c, co = cleaned[st], calls[st], calls[so]
                cells.append(
                    {
                        "held": bool(h),
                        "size": sz,
                        "train": int(st.sum()),
                        "truth": {
                            "a": int((y == a).sum()),
                            "b": int((y == b).sum()),
                            "other": int(((y > 0) & (y != a) & (y != b)).sum()),
                            "legit": int((y == 0).sum()),
                        },
                        "calls": {"a": int((c == a).sum()), "b": int((c == b).sum())},
                        "october": {
                            "rows": int(so.sum()),
                            "a": int((co == a).sum()),
                            "b": int((co == b).sum()),
                        },
                    }
                )
        score = proba[:, b] / np.maximum(proba[:, a] + proba[:, b], 1e-12)
        pool = (tmpl == t) & ~held & (calls == a)
        _, pinv = np.unique(key[pool], return_inverse=True)
        sev = np.bincount(pinv, weights=score[pool]) / np.bincount(pinv)
        s_ev = np.full(len(tmpl), -1.0)
        s_ev[np.flatnonzero(pool)] = sev[pinv]
        auc = {}
        for m in MONTHS:
            s = pool & tr & (per == m) & np.isin(cleaned, (a, b))
            if s.sum() and 0 < (cleaned[s] == b).mean() < 1:
                auc[m] = float(roc_auc_score(cleaned[s] == b, s_ev[s]))
        split = {}
        for q in (0.8, 0.9, 0.95, 0.98):
            split[q] = {}
            for m in (*MONTHS, "oct"):
                rows = d["is_test"] if m == "oct" else per == m
                sel = pool & rows
                if not sel.any():
                    split[q][m] = 0 if m == "oct" else dict.fromkeys(TARGETS, 0.0)
                    continue
                flip = sel & (s_ev > np.quantile(s_ev[sel], q))
                new = calls.copy()
                new[flip] = b
                if m == "oct":
                    split[q][m] = int(flip.sum())
                else:
                    sc = month_targets(d, m, new[d["rows"][m]])
                    split[q][m] = {x: sc[x] - base[m][x] for x in TARGETS}
        nested = {}
        for h in MONTHS:
            fit = [m for m in MONTHS if m != h]
            gain = {q: sum(split[q][m]["cleaned"] for m in fit) for q in split}
            q = max(gain, key=gain.get)
            nested[h] = {"q": q, "change": split[q][h]} if gain[q] > 0 else {"q": None}
        rep[name] = {
            "pair": [CLASSES[a], CLASSES[b]],
            "cells": cells,
            "auc": auc,
            "split": {str(q): v for q, v in split.items()},
            "nested": nested,
        }
        print(f"== {name} ({CLASSES[a]} / {CLASSES[b]}), pool AUC {auc}")
        for c in cells:
            tt = c["truth"]
            share = tt["a"] / max(tt["a"] + tt["b"], 1)
            print(
                f"  held={int(c['held'])} size={c['size']}: train {c['train']} truth {tt} "
                f"(a share {share:.2f}) calls {c['calls']} October {c['october']}"
            )
        for q, v in split.items():
            print(
                f"  split top {1 - q:.2f}: cleaned "
                + " / ".join(f"{v[m]['cleaned'] * 1e5:+.1f}" for m in MONTHS)
                + f" (1e-5), October rows {v['oct']}"
            )
        print("  nested split:", nested)
    (out_dir() / "pairs.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


# ---------------------------------------------------------------------------------------------
# October aggregates


def cmd_october() -> dict:
    """Calls and probability mass per class per 100k rows, train months against October, the
    generator-only label rate on train for reference, and the September-scaled call count.
    Aggregates only."""
    d = _ctx()
    tr = ~d["is_test"]
    te = d["is_test"]
    per, calls, proba = d["per_tr"], d["calls"], d["proba"]
    ctr, ptr = calls[tr], proba[tr]
    n_oct = int(te.sum())
    rep = {}
    for c in range(N_CLASSES):
        r = {}
        for m in MONTHS:
            s = per == m
            n = s.sum()
            r[str(m)] = {
                "calls": int((ctr[s] == c).sum()),
                "calls_100k": float((ctr[s] == c).sum() / n * 1e5),
                "psum_100k": float(ptr[s][:, c].sum() / n * 1e5),
                "cleaned_100k": float((d["cleaned"][s] == c).sum() / n * 1e5),
                "gen_100k": float((d["gen"][s] == c).sum() / n * 1e5),
            }
        r["oct"] = {
            "calls": int((calls[te] == c).sum()),
            "calls_100k": float((calls[te] == c).sum() / n_oct * 1e5),
            "psum_100k": float(proba[te][:, c].sum() / n_oct * 1e5),
            "sep_scaled_calls": float(r["9"]["calls_100k"] * n_oct / 1e5),
            "train_mean_scaled_calls": float(
                np.mean([r[str(m)]["calls_100k"] for m in MONTHS]) * n_oct / 1e5
            ),
        }
        rep[CLASSES[c]] = r
        print(
            f"{CLASSES[c][:18]:<19} calls/100k "
            + " ".join(f"{r[str(m)]['calls_100k']:7.1f}" for m in MONTHS)
            + f" | Oct {r['oct']['calls_100k']:7.1f} ({r['oct']['calls']} vs Sep-scaled "
            f"{r['oct']['sep_scaled_calls']:.0f}) | p-sum/100k "
            + " ".join(f"{r[str(m)]['psum_100k']:7.1f}" for m in MONTHS)
            + f" | Oct {r['oct']['psum_100k']:7.1f} | gen/100k "
            + " ".join(f"{r[str(m)]['gen_100k']:6.1f}" for m in MONTHS)
        )
    # label-free anchors: per anchor class, rows per 100k and the share called that class / fraud
    anchor = d["anchor"]
    print("anchor class: rows/100k Jul Aug Sep Oct | share called the anchor class | called fraud")
    for c in range(1, N_CLASSES):
        r = {}
        for m in (*MONTHS, "oct"):
            rows = d["rows"][m]
            s = anchor[rows] == c
            r[str(m)] = {
                "rows_100k": float(s.sum() / len(rows) * 1e5),
                "called_class": float((calls[rows][s] == c).mean()) if s.any() else 0.0,
                "called_fraud": float((calls[rows][s] > 0).mean()) if s.any() else 0.0,
            }
        rep[CLASSES[c]]["anchor"] = r
        keys = (*map(str, MONTHS), "oct")
        print(
            f"{CLASSES[c][:18]:<19} "
            + " ".join(f"{r[k]['rows_100k']:6.1f}" for k in keys)
            + " | "
            + " ".join(f"{r[k]['called_class']:.3f}" for k in keys)
            + " | "
            + " ".join(f"{r[k]['called_fraud']:.3f}" for k in keys)
        )
    (out_dir() / "october.json").write_text(json.dumps(rep, indent=1))
    return rep


# ---------------------------------------------------------------------------------------------
# the bar and the candidate files


def cmd_gate(recipes: list[str], report: str = "gate") -> dict:
    """Every recipe against the best file's recipe on the full bar (``ringtrace.final.table``:
    per-month change on every target, event bootstrap, the nine October-like cells), plus the
    A-clean change per month."""
    from ringtrace.final import _json_default, table

    b = bench()
    base = best_recipe()
    chals = [best_recipe(p) for p in recipes]
    path = out_dir() / f"{report}.json"

    def save(rep):
        path.write_text(json.dumps(rep, indent=1, default=_json_default))

    rep = table(b, base, chals, save=save)
    d = _ctx()
    ptr = b.period[b.tr]
    a = b.calls(base)["pred"][b.tr]
    for c in chals:
        p = b.calls(c)["pred"][b.tr]
        ch = []
        for m in MONTHS:
            s = ptr == m
            ch.append(
                macro(d["cleaned"][s], p[s], d["hid"][m])
                - macro(d["cleaned"][s], a[s], d["hid"][m])
            )
        rep["challengers"][c["name"]]["aclean_change"] = ch
        print(f"[closelook] {c['name']}: A-clean change " + " / ".join(f"{x:+.5f}" for x in ch))
    save(rep)
    return rep


def cmd_build(recipes: list[str]) -> dict:
    """October files for ``recipes`` into ``$RINGTRACE_OUT/candidates/<name>.csv``, validated
    against the sample submission, with the rows changed against the best file."""
    import shutil

    from ringtrace.final import build

    b = bench()
    out = out_dir() / "build"
    reps = {}
    for p in recipes:
        r = best_recipe(p)
        reps[r["name"]] = rep = build(
            b,
            r,
            out,
            paths.data_dir() / "sample_submission.csv",
            paths.out_dir() / "candidates" / BEST_FILE,
        )
        if not rep["problems"]:
            shutil.copyfile(
                out / f"{r['name']}.csv", paths.out_dir() / "candidates" / f"{r['name']}.csv"
            )
    return reps


COMMANDS = {
    "prep": cmd_prep,
    "anatomy": cmd_anatomy,
    "volume": cmd_volume,
    "pairs": cmd_pairs,
    "october": cmd_october,
    "gate": cmd_gate,
    "build": cmd_build,
}


def main(argv=None) -> None:
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python analysis/closelook.py")
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    ap.add_argument("recipes", nargs="*", help="recipes for gate / build")
    ap.add_argument("--report", default="gate", help="report name for gate")
    a = ap.parse_args(argv)
    if a.cmd == "gate":
        cmd_gate(a.recipes, a.report)
    elif a.cmd == "build":
        cmd_build(a.recipes)
    else:
        COMMANDS[a.cmd]()


if __name__ == "__main__":
    main()
