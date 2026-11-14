"""Unanswered small baits and the sender's operator roles: what the best file does, and two rules.

    python analysis/bait.py measure                  # the best file on small P2P baits (counts)
    python analysis/bait.py validate                 # leave-one-month-out bar of every variant
    python analysis/bait.py octlike [VARIANT ...]    # the nine October-like cells
    python analysis/bait.py thinned [VARIANT ...]    # fresh thinning seeds of September
    python analysis/bait.py build VARIANT            # candidates/bait-<VARIANT>.csv

``measure`` tabulates the bait sessions of the requests family (shapes N1 to A2) per month: the
target classes on train, the best file's calls on train and October, and how often a bait's
sender holds a label-free operator role (``ringtrace.baitroles``) in another month.

``validate`` scores each variant on the best file's out-of-fold calls (``prep.npz``, ledger
order): the change of macro-F1 per held-out month on the given, cleaned and gen-only targets, the
precision of the moved rows against phishing_collect's break-even (half its F1) per month, and a
paired event bootstrap (2,000 draws) of the mean change. A train month takes its roles from the
other two train months, October from all three. ``history`` is a control that replaces the roles
with "the sender sent a fraud-labelled row in another train month".

``octlike`` re-decodes the best recipe in the nine October-like cells (September thinned to
October's density by protected units and fitted on July + August) and applies each variant on
top: change per cell, precision per cell, and the pooled 95 % lower bound (2,000 draws).
``thinned`` repeats the thinning with fresh seeds on the out-of-fold calls.

``build`` applies a variant to the best file's October calls and writes the candidate in the
sample submission's order. October rows are only ever counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bait_core import by_month, move_table, other_month_label_acts, passes_months  # noqa: E402

from ringtrace import baitroles, paths, store  # noqa: E402
from ringtrace import rules as rulebook  # noqa: E402
from ringtrace.classes import CLASSES, CODE, N_CLASSES  # noqa: E402
from ringtrace.evidence import requests as rq  # noqa: E402

PHISH = CODE["phishing_collect"]
MONTHS = (7, 8, 9)
ALL_MONTHS = [7, 8, 9, 10]
TARGETS = ("given", "cleaned", "gen")
DRAWS = 2000
BEST_FILE = "blend-tabmF-bq.csv"
BEST_RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "final-half.json"
SHAPE_NAMES = {
    rq.N1: "N1 lone 1/2/5",
    rq.N1X: "N1X lone 10",
    rq.N2: "N2 2+ unanswered",
    rq.A1S: "A1S answered <4h",
    rq.A1L: "A1L answered 4-72h",
    rq.A2: "A2 2+ answered",
}
VARIANTS: dict[str, list] = {
    "roles": [{"name": "bait_roles"}],
    "roles-band": [{"name": "bait_roles", "band": True}],
    **{f"roles-{r}": [{"name": "bait_roles", "roles": [r]}] for r in baitroles.ROLES},
    "roles-all4": [{"name": "bait_roles", "months": ALL_MONTHS, "own_month": True}],
    "pay": [{"name": "first_pay"}],
    "pay-roles": [{"name": "first_pay", "operator_receiver": True}],
    "pay-roles-all4": [
        {"name": "first_pay", "operator_receiver": True, "months": ALL_MONTHS, "own_month": True}
    ],
    "pay-roles-500": [{"name": "first_pay", "operator_receiver": True, "min_amount": 500.0}],
    "combo": [{"name": "bait_roles"}, {"name": "first_pay", "operator_receiver": True}],
}
CONTROLS = ("history",)


def out_dir() -> Path:
    d = paths.out_dir() / "bait"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _quiet(*_):
    return None


def load_prep(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def ledger() -> pl.DataFrame:
    store.use_frame(None)
    return store.ledger()


def history_calls(led: pl.DataFrame, calls: np.ndarray, given: np.ndarray) -> np.ndarray:
    """The control: a lone unanswered Rs 10 called legit becomes phishing_collect when its sender
    sent a fraud-labelled row (given labels) in another train month."""
    cols = baitroles.columns(led)
    ntr = len(given)
    snd = cols["sender"].to_numpy()
    per = cols["period"].to_numpy()
    acct, p = other_month_label_acts(snd[:ntr], per[:ntr], given > 0)
    acts = pl.DataFrame({"acct": acct, "period": p.astype(np.int8), "role": "labelled"})
    cand = np.flatnonzero(baitroles.small_unanswered(cols) & (calls == 0))
    out = calls.copy()
    hit = baitroles.other_month(snd[cand], per[cand], acts, roles=("labelled",))
    out[cand[hit]] = PHISH
    return out


def variant_calls(name: str, led: pl.DataFrame, d: dict) -> np.ndarray:
    if name == "history":
        return history_calls(led, d["calls"], d["given"])
    return rulebook.apply(VARIANTS[name], d["calls"], d["proba"], led, log=_quiet)


def moves(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    diff = a != b
    keys, n = np.unique(a[diff] * N_CLASSES + b[diff], return_counts=True)
    return {
        f"{CLASSES[k // N_CLASSES]} -> {CLASSES[k % N_CLASSES]}": int(c)
        for k, c in zip(keys, n, strict=True)
    }


# ---------------------------------------------------------------------------------------------
# measure


def cmd_measure(prep: Path) -> dict:
    d = load_prep(prep)
    led = ledger()
    cols = baitroles.columns(led)
    calls, period, is_test = d["calls"], d["period"], d["is_test"]
    ntr = len(d["given"])
    role = cols["col_role"].fill_null(0).to_numpy()
    shape = cols["col_shape"].fill_null(0).to_numpy()
    bait = role == rq.BAIT
    acts = baitroles.operator_roles(cols)
    snd = cols["sender"].to_numpy()
    has_role = np.zeros(len(calls), dtype=bool)
    idx = np.flatnonzero(bait)
    has_role[idx] = baitroles.other_month(snd[idx], period[idx], acts)
    out: dict = {"train": {}, "october": {}, "role_cover": {}}
    for s, nm in SHAPE_NAMES.items():
        rows = {}
        for m in MONTHS:
            r = np.flatnonzero(bait & (shape == s) & (period == m) & ~is_test)
            c = d["cleaned"][r]
            rows[m] = {
                "rows": len(r),
                "cleaned_phishing": int((c == PHISH).sum()),
                "cleaned_cashback": int((c == CODE["fake_cashback"]).sum()),
                "cleaned_other_fraud": int(((c > 0) & (c != PHISH) & (c != 9)).sum()),
                "gen_phishing": int((d["gen"][r] == PHISH).sum()),
                "called_fraud": int((calls[r] > 0).sum()),
                "called_phishing": int((calls[r] == PHISH).sum()),
                "called_legit_cleaned_phishing": int(((calls[r] == 0) & (c == PHISH)).sum()),
                "sender_role_other_month": int(has_role[r].sum()),
                "role_and_called_legit": int((has_role[r] & (calls[r] == 0)).sum()),
            }
        out["train"][nm] = rows
        r = np.flatnonzero(bait & (shape == s) & is_test)
        out["october"][nm] = {
            "rows": len(r),
            "called_fraud": int((calls[r] > 0).sum()),
            "called_phishing": int((calls[r] == PHISH).sum()),
            "sender_role_other_month": int(has_role[r].sum()),
            "role_and_called_legit": int((has_role[r] & (calls[r] == 0)).sum()),
        }
    # role coverage per role: senders of recognised baits against senders of lone Rs 10s
    lone = bait & (shape == rq.N1X) & ~is_test
    real = bait & np.isin(shape, baitroles.RECOGNISED_BAITS) & ~is_test
    clean = np.full(len(calls), -1)
    clean[:ntr] = d["cleaned"]
    for r_name in (*baitroles.ROLES, "any"):
        use = baitroles.ROLES if r_name == "any" else (r_name,)
        cover = {}
        for nm, m in (
            ("recognised bait", real),
            ("lone 10, cleaned phishing", lone & (clean == PHISH)),
            ("lone 10, cleaned legit", lone & (clean == 0)),
        ):
            i = np.flatnonzero(m)
            h = baitroles.other_month(snd[i], period[i], acts, roles=use)
            cover[nm] = {"rows": len(i), "with_role": int(h.sum()), "share": float(h.mean())}
        out["role_cover"][r_name] = cover
    # October roles from a two-month window (August + September), the train-month view
    i = np.flatnonzero(bait & (shape == rq.N1X) & is_test)
    h2 = baitroles.other_month(snd[i], period[i], acts, months=(8, 9))
    out["october_lone10_two_month_window"] = {
        "with_role": int(h2.sum()),
        "with_role_called_legit": int((h2 & (calls[i] == 0)).sum()),
    }
    # the band: small P2P PAY that are not bait amounts
    p2p_pay = cols["to_person"].fill_null(False).to_numpy() & (cols["txn_type"] == "PAY").to_numpy()
    amt = cols["amount"].to_numpy()
    odd = p2p_pay & (amt >= 1) & (amt <= 10) & (role == 0)
    out["band_non_bait_amounts"] = {
        "train_rows": int((odd & ~is_test).sum()),
        "train_cleaned_fraud": int((clean[odd & ~is_test] > 0).sum()),
        "train_called_fraud": int((calls[odd & ~is_test] > 0).sum()),
        "october_rows": int((odd & is_test).sum()),
        "october_called_fraud": int((calls[odd & is_test] > 0).sum()),
    }
    (out_dir() / "measure.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------------------------
# validate


def lomo(name: str, led: pl.DataFrame, d: dict, boot) -> dict:
    new = variant_calls(name, led, d)
    base = d["calls"]
    tr = ~d["is_test"]
    ptr = d["period"][tr]
    targets = {t: d[t] for t in TARGETS}
    table = by_month(targets, base[tr], new[tr], ptr, PHISH, MONTHS)
    bs = boot.compare(new[tr])
    return {
        "months": table,
        "passes_months": passes_months(table),
        "bootstrap": bs,
        "train_rows_changed": int((new[tr] != base[tr]).sum()),
        "october_rows_changed": int((new[~tr] != base[~tr]).sum()),
        "october_moves": moves(base[~tr], new[~tr]),
    }


def fmt(name: str, r: dict) -> str:
    parts = [f"{name:<26}"]
    for t in ("given", "cleaned", "gen"):
        g = " / ".join(f"{row['gain']:+.5f}" for row in r["months"][t])
        parts.append(f"{t} {g}")
    for t in ("given", "cleaned"):
        b = r["bootstrap"][t]
        parts.append(f"boot {t} {b['delta']:+.5f} [{b['lo']:+.5f}, {b['hi']:+.5f}]")
    prec = " / ".join(
        f"{row['right']}/{row['moved']} (be {row['break_even']:.3f})"
        for row in r["months"]["cleaned"]
    )
    parts.append(f"cleaned precision {prec}")
    parts.append(f"rows {r['train_rows_changed']} / Oct {r['october_rows_changed']}")
    parts.append("PASS" if r["passes_months"] else "FAIL")
    return "  ".join(parts)


def cmd_validate(prep: Path, names: list[str], report: str = "validate") -> dict:
    from ringtrace.pairboot import PairedBoot

    t0 = time.time()
    d = load_prep(prep)
    led = ledger()
    tr = ~d["is_test"]
    targets = {t: d[t] for t in TARGETS}
    boot = PairedBoot(d["groups"], targets, d["calls"][tr], d["period"][tr], draws=DRAWS)
    out = {}
    for name in names:
        out[name] = r = lomo(name, led, d, boot)
        print(fmt(name, r), flush=True)
    out["wall_s"] = time.time() - t0
    (out_dir() / f"{report}.json").write_text(json.dumps(out, indent=1, default=_np))
    return out


def _np(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------------------------
# October-like cells


def cmd_octlike(
    names: list[str], cells_dir: Path, report: str = "octlike", seeds=(13, 14, 15), sets=(0, 1, 2)
) -> dict:
    """Each variant on top of the best recipe's calls in the nine October-like cells."""
    from ringtrace import octlike as ol
    from ringtrace.final import load_recipe, mix, static_weights
    from ringtrace.pairboot import PairedBoot

    t0 = time.time()
    recipe = load_recipe(BEST_RECIPE)
    w = static_weights(recipe)
    roles = [m["role"] for m in recipe["members"]]
    res = {n: {"cells": {}} for n in names}
    pooled = {n: {t: [] for t in TARGETS} for n in names}
    try:
        for seed in seeds:
            frame = ol.Frame(seed)
            hold = np.flatnonzero(frame.hold)
            units = frame.unit[frame.hold]
            truth = {t: y[frame.hold] for t, y in frame.targets.items()}
            for r in sets:
                probs, rows = [], None
                for m in recipe["members"]:
                    f = np.load(cells_dir / f"{m['cells']}_s{seed}_r{r}.npz")
                    rows = f["rows"] if rows is None else rows
                    if not np.array_equal(f["rows"], frame.stacked[frame.hold]):
                        raise ValueError(f"{m['cells']} s{seed} r{r}: rows differ from the frame")
                    probs.append(f["prob"].astype(np.float64))
                p = mix(probs, w, roles)
                pred = ol.decode(frame, p, recipe["decoder"], recipe["rules"])
                full_pred = np.zeros(frame.led.height, dtype=np.int64)
                full_pred[hold] = pred
                full_p = np.zeros((frame.led.height, N_CLASSES))
                full_p[hold] = p
                brr = [
                    {k: v for k, v in x.items() if k != "max_october_rows"}
                    for x in recipe["base_rate_rules"]
                ]
                full_pred = rulebook.apply(brr, full_pred, full_p, frame.led, log=_quiet)
                ya = full_pred[hold]
                boot = PairedBoot(units, truth, ya, draws=DRAWS, seed=seed * 100)
                for n in names:
                    yb = rulebook.apply(VARIANTS[n], full_pred, full_p, frame.led, log=_quiet)[hold]
                    bs = boot.compare(yb, keep_draws=True)
                    cell = {"rows_changed": int((ya != yb).sum())}
                    for t in TARGETS:
                        pooled[n][t].append(bs[t].pop("draws"))
                        cell[t] = {**move_table(truth[t], ya, yb, PHISH), **bs[t]}
                    res[n]["cells"][f"s{seed}_r{r}"] = cell
                print(f"[bait] cell s{seed} r{r} done ({time.time() - t0:.0f}s)", flush=True)
    finally:
        store.use_frame(None)
    for n in names:
        c = res[n]["cells"]
        res[n]["mean"] = {t: float(np.mean([v[t]["delta"] for v in c.values()])) for t in TARGETS}
        res[n]["lower_bound"] = {
            t: float(np.percentile(np.mean(pooled[n][t], axis=0), 2.5)) for t in TARGETS
        }
        res[n]["per_seed"] = {
            t: {
                s: float(np.mean([v[t]["delta"] for k, v in c.items() if k.startswith(f"s{s}_")]))
                for s in seeds
            }
            for t in TARGETS
        }
        res[n]["moved_right"] = {
            t: [
                int(sum(v[t]["moved"] for v in c.values())),
                int(sum(v[t]["right"] for v in c.values())),
            ]
            for t in TARGETS
        }
        print(
            f"[bait] {n}: octlike mean given {res[n]['mean']['given']:+.5f} "
            f"(lb {res[n]['lower_bound']['given']:+.5f}), cleaned {res[n]['mean']['cleaned']:+.5f} "
            f"(lb {res[n]['lower_bound']['cleaned']:+.5f}); moved / right (cleaned) "
            f"{res[n]['moved_right']['cleaned']}",
            flush=True,
        )
    res["wall_s"] = time.time() - t0
    (out_dir() / f"{report}.json").write_text(json.dumps(res, indent=1, default=_np))
    return res


def cmd_thinned(prep: Path, names: list[str], seeds: list[int], report: str = "thinned") -> dict:
    """Fresh thinning seeds of September (protected units as in ``ringtrace.octlike``) on the
    out-of-fold calls: change per seed and the pooled 95 % lower bound."""
    from ringtrace import octlike as ol
    from ringtrace.pairboot import PairedBoot

    t0 = time.time()
    d = load_prep(prep)
    led = ledger()
    tr = ~d["is_test"]
    sept = np.flatnonzero(tr & (d["period"] == 9))
    events = [
        store.family(f, verbose=False)[c].fill_null(-1).to_numpy().astype(np.int64)
        for f, c in zip(ol.EVENT_FAMILIES, ol.EVENT_COLUMNS, strict=True)
    ]
    units = ol.september_units(led, events, sept)
    news = {n: variant_calls(n, led, d) for n in names}
    res = {n: {"seeds": {}} for n in names}
    pooled = {n: {t: [] for t in TARGETS} for n in names}
    for s in seeds:
        kept = ol.thin(units, s)
        rows = sept[kept]
        truth = {t: d[t][rows] for t in TARGETS}
        ya = d["calls"][rows]
        boot = PairedBoot(units[kept], truth, ya, draws=DRAWS, seed=s * 100)
        for n in names:
            yb = news[n][rows]
            bs = boot.compare(yb, keep_draws=True)
            cell = {"rows_changed": int((ya != yb).sum())}
            for t in TARGETS:
                pooled[n][t].append(bs[t].pop("draws"))
                cell[t] = {**move_table(truth[t], ya, yb, PHISH), **bs[t]}
            res[n]["seeds"][str(s)] = cell
    for n in names:
        c = res[n]["seeds"]
        res[n]["mean"] = {t: float(np.mean([v[t]["delta"] for v in c.values()])) for t in TARGETS}
        res[n]["positive_seeds"] = {
            t: int(sum(v[t]["delta"] > 0 for v in c.values())) for t in TARGETS
        }
        res[n]["lower_bound"] = {
            t: float(np.percentile(np.mean(pooled[n][t], axis=0), 2.5)) for t in TARGETS
        }
        print(
            f"[bait] {n}: {len(seeds)} fresh seeds, mean given {res[n]['mean']['given']:+.5f} "
            f"(lb {res[n]['lower_bound']['given']:+.5f}), cleaned {res[n]['mean']['cleaned']:+.5f} "
            f"(lb {res[n]['lower_bound']['cleaned']:+.5f}), positive seeds (cleaned) "
            f"{res[n]['positive_seeds']['cleaned']}",
            flush=True,
        )
    res["wall_s"] = time.time() - t0
    (out_dir() / f"{report}.json").write_text(json.dumps(res, indent=1, default=_np))
    return res


# ---------------------------------------------------------------------------------------------
# build


def cmd_build(name: str, best: Path, out_path: Path | None) -> dict:
    sample = pl.read_csv(paths.data_dir() / "sample_submission.csv", columns=["id"])
    sub = pl.read_csv(best, infer_schema=False)
    led = ledger()
    ids = led.select("txn_id", "is_test").with_row_index("i")
    pos = {c: k for k, c in enumerate(CLASSES)}
    calls = np.zeros(led.height, dtype=np.int64)
    te = ids.filter(pl.col("is_test")).join(sub.rename({"id": "txn_id"}), on="txn_id", how="left")
    if te["label"].null_count():
        raise ValueError("best file misses October ids")
    ti = te["i"].to_numpy()
    calls[ti] = [pos[x] for x in te["label"].to_list()]
    proba = np.zeros((led.height, N_CLASSES))  # the variants read no probabilities
    new = rulebook.apply(VARIANTS[name], calls, proba, led, log=_quiet)
    lab = np.array(CLASSES)[new[ti]]
    out = sample.join(pl.DataFrame({"id": te["txn_id"], "label": lab}), on="id", how="left")
    if out.height != sample.height or out["label"].null_count():
        raise ValueError("candidate does not cover the sample submission")
    if not set(out["label"].unique().to_list()) <= set(CLASSES):
        raise ValueError("candidate holds an unknown label")
    out_path = out_path or paths.out_dir() / "candidates" / f"bait-{name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    old = sample.join(sub, on="id", how="left")["label"]
    if not np.array_equal(out["id"].to_numpy(), sample["id"].to_numpy()):
        raise ValueError("candidate is not in the sample's id order")
    moved = out.with_columns(old.alias("old")).filter(pl.col("label") != pl.col("old"))
    rep = {
        "file": str(out_path),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "rows": out.height,
        "rows_changed": moved.height,
        "moves": moved.group_by("old", "label").len().sort("len", descending=True).rows(),
        "against": best.name,
        "rules": VARIANTS[name],
    }
    out_path.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python analysis/bait.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_prep = paths.out_dir() / "closelook" / "prep.npz"
    m = sub.add_parser("measure")
    m.add_argument("--prep", type=Path, default=default_prep)
    v = sub.add_parser("validate")
    v.add_argument("names", nargs="*", default=[*VARIANTS, *CONTROLS])
    v.add_argument("--prep", type=Path, default=default_prep)
    v.add_argument("--report", default="validate")
    o = sub.add_parser("octlike")
    o.add_argument("names", nargs="*", default=["roles", "pay"])
    o.add_argument("--cells", type=Path, default=paths.out_dir() / "lens-cells")
    o.add_argument("--report", default="octlike")
    t = sub.add_parser("thinned")
    t.add_argument("names", nargs="*", default=["roles", "pay"])
    t.add_argument("--prep", type=Path, default=default_prep)
    t.add_argument("--seeds", default="101-120")
    t.add_argument("--report", default="thinned")
    b = sub.add_parser("build")
    b.add_argument("name", choices=sorted(VARIANTS))
    b.add_argument("--best", type=Path, default=paths.out_dir() / "candidates" / BEST_FILE)
    b.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        cmd_measure(a.prep)
    elif a.cmd == "validate":
        cmd_validate(a.prep, a.names, a.report)
    elif a.cmd == "octlike":
        cmd_octlike(a.names, a.cells, a.report)
    elif a.cmd == "thinned":
        lo, _, hi = a.seeds.partition("-")
        cmd_thinned(a.prep, a.names, list(range(int(lo), int(hi or lo) + 1)), a.report)
    else:
        cmd_build(a.name, a.best, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
