"""Event completion survey for the best file: do our calls leave parts of fraud events uncalled,
and do rows linked to fraud calls carry fraud? October is only ever counted.

    python analysis/events.py all        # every survey below, written to $RINGTRACE_OUT/events/
    python analysis/events.py partial    # label-free evidence events called whole / not / in part
    python analysis/events.py links      # legit-called rows linked to fraud calls, per kind/window
    python analysis/events.py rules      # completion rules over those links, per month and target
    python analysis/events.py fraudblock # evidence share of fraud-pass rows on new devices
    python analysis/events.py mf23       # mule-shaped fan-in template events per month and label
    python analysis/events.py ceiling    # clean-label F1 of calling exactly the generator rows

Input: the best file's cached arrays (``$RINGTRACE_OUT/closelook/prep.npz``: out-of-fold calls on
the train months and the October calls of ``blend-tabmF-bq``, the label-free events, templates and
counter) and the cached ledger. Link kinds, from a legit-called query row to the nearest
fraud-called anchor row of the same month:

* ``sender`` / ``device``: same sender / same device, either side in time;
* ``recv``: same person receiver, either side;
* ``onward``: the query's sender is the anchor's person receiver, query after the anchor
  (a mule's or a chain's onward transfer);
* ``back``: the query pays the anchor's sender, query after the anchor (a cashback payback).

Train queries leave out generator-made rows (``gen > 0``), so the precision is that of rows a
completion rule would add.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from events_core import (  # noqa: E402
    complete_by_link,
    evidence_share,
    f1_ceiling,
    link_cells,
    nearest_anchor,
    partial_events,
)

from ringtrace import paths, tables  # noqa: E402
from ringtrace.classes import CLASSES  # noqa: E402

MONTHS = (7, 8, 9)
OCTOBER = 10
FAMILIES = {1: "bot", 2: "drain", 3: "flow", 4: "collect", 5: "ring", 6: "repeat"}
WINDOWS = {"10m": 600, "1h": 3600, "24h": 86400, "7d": 7 * 86400, "31d": 31 * 86400}
KINDS = {
    "sender": ("sender", "sender", "both"),
    "device": ("device", "device", "both"),
    "recv": ("receiver", "receiver", "both"),
    "onward": ("sender", "receiver", "backward"),
    "back": ("receiver", "sender", "backward"),
}
LEDGER_COLS = ["ts", "sender", "receiver", "device", "to_person", "app_version", "ip_risk_score"]


def out_dir() -> Path:
    d = paths.out_dir() / "events"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load() -> dict:
    z = np.load(paths.out_dir() / "closelook" / "prep.npz")
    keys = (
        "calls",
        "events",
        "is_test",
        "period",
        "given",
        "cleaned",
        "gen",
        "tmpl",
        "tmpl_event",
        "anchor",
        "ctr_block",
    )
    d = {k: z[k] for k in keys}
    names = ("train", "test", "accounts", "merchants")
    key = "-".join(tables.file_digest(paths.data_dir() / f"{n}.csv", 8) for n in names)
    target = paths.cache_dir() / "tables" / f"ledger-{key}.parquet"
    led = pl.read_parquet(target, columns=LEDGER_COLS) if target.exists() else tables.ledger()
    if led.height != len(d["calls"]):
        raise ValueError(f"ledger has {led.height} rows, arrays {len(d['calls'])}")
    d["led"] = led.select(LEDGER_COLS)
    n, ntr = led.height, len(d["given"])
    pad = np.full(n - ntr, -1, dtype=np.int64)
    for k in ("given", "cleaned", "gen"):
        d[k + "_full"] = np.r_[d[k], pad]
    return d


def cmd_partial(d: dict) -> dict:
    ev, calls, period = d["events"], d["calls"], d["period"]
    fam = np.where(ev >= 0, ev // 10_000_000, 0)
    rep = {}
    for p in (*MONTHS, OCTOBER):
        s = period == p
        cell = {"all": partial_events(np.where(s, ev, -1), calls > 0)}
        for f, name in FAMILIES.items():
            cell[name] = partial_events(np.where(s & (fam == f), ev, -1), calls > 0)
        rep[p] = cell
        a = cell["all"]
        print(
            f"period {p}: events {a['events']}, whole {a['whole']}, none {a['none']}, "
            f"partial {a['partial']} (legit rows in them {a['legit_in_partial']})"
        )
    return rep


def cmd_links(d: dict) -> dict:
    led, calls, period = d["led"], d["calls"], d["period"]
    ts = led["ts"].to_numpy().astype(np.int64)
    person = led["to_person"].to_numpy()
    cols = {c: led[c].to_numpy().astype(np.int64) for c in ("sender", "receiver", "device")}
    rep = {}
    for p in (*MONTHS, OCTOBER):
        s = period == p
        anc = s & (calls > 0)
        qry = s & (calls == 0) & (d["gen_full"] <= 0)
        cell = {"rows": int(s.sum())}
        for kind, (qk, ak, direction) in KINDS.items():
            a = anc & person if ak == "receiver" else anc
            q = qry & person if qk == "receiver" else qry
            gap, acls = nearest_anchor(
                cols[qk][q], ts[q], cols[ak][a], ts[a], calls[a], direction=direction
            )
            label = d["cleaned_full"][q] if p != OCTOBER else None
            cell[kind] = link_cells(gap, acls, WINDOWS, label)
            if p != OCTOBER:
                given = link_cells(gap, acls, WINDOWS, d["given_full"][q])
                for w in WINDOWS:
                    cell[kind][w]["fraud_given"] = given[w]["fraud"]
                    cell[kind][w]["same_given"] = given[w]["same"]
        rep[p] = cell
    print(
        f"{'kind':7s} {'win':4s} | train per 100k, cleaned fraud / same-class share "
        f"(Jul+Aug+Sep) | October per 100k | ratio"
    )
    ntr_rows = sum(rep[m]["rows"] for m in MONTHS)
    for kind in KINDS:
        for w in WINDOWS:
            n = sum(rep[m][kind][w]["linked"] for m in MONTHS)
            fr = sum(
                rep[m][kind][w]["linked"] * np.nan_to_num(rep[m][kind][w]["fraud"]) for m in MONTHS
            )
            sm = sum(
                rep[m][kind][w]["linked"] * np.nan_to_num(rep[m][kind][w]["same"]) for m in MONTHS
            )
            tr100 = n / ntr_rows * 1e5
            oc100 = rep[OCTOBER][kind][w]["linked"] / rep[OCTOBER]["rows"] * 1e5
            print(
                f"{kind:7s} {w:4s} | {tr100:8.1f}  {fr / max(n, 1):.3f} / {sm / max(n, 1):.4f} | "
                f"{oc100:8.1f} | {oc100 / tr100 if tr100 else float('nan'):.2f}"
            )
    return rep


def cmd_fraudblock(d: dict) -> dict:
    led, period = d["led"], d["period"]
    dev = led["device"].to_numpy()
    _, inv, cnt = np.unique(dev, return_inverse=True, return_counts=True)
    single = cnt[inv] == 1
    fb = d["ctr_block"] == 2
    evid = (d["events"] >= 0) | (d["anchor"] > 0) | (d["tmpl"] > 0)
    rep = {
        "fraud_block": evidence_share(fb, evid, period),
        "fraud_block_single_row": evidence_share(fb & single, evid, period),
    }
    ntr = len(d["given"])
    av = led["app_version"].to_numpy()[:ntr]
    ip = led["ip_risk_score"].to_numpy()[:ntr]
    gen = d["gen"]
    sets = {
        "fraud-block, generator-made": fb[:ntr] & (gen > 0),
        "fraud-block, no evidence": fb[:ntr] & (gen == 0),
        "legit-block": d["ctr_block"][:ntr] == 3,
        "home phone, given legit": (d["ctr_block"][:ntr] == 0) & (d["given"] == 0),
    }
    rep["train_draws"] = {
        k: {
            "rows": int(m.sum()),
            "old_app": float((av[m] < 4).mean()),
            "ip_above_half": float((ip[m] > 0.5).mean()),
            "given_fraud": float((d["given"][m] > 0).mean()),
        }
        for k, m in sets.items()
    }
    for k, v in rep["fraud_block_single_row"].items():
        print(
            f"period {k}: single-row fraud-block rows {v['rows']}, evidence share "
            f"{v['evidence_share']:.3f}; all fraud-block {rep['fraud_block'][k]['rows']}, "
            f"{rep['fraud_block'][k]['evidence_share']:.3f}"
        )
    for k, v in rep["train_draws"].items():
        print(
            f"  {k:28s} rows {v['rows']:8d} old app {v['old_app']:.3f} ip>0.5 "
            f"{v['ip_above_half']:.3f} given fraud {v['given_fraud']:.3f}"
        )
    return rep


def cmd_mf23(d: dict) -> dict:
    tm, tev, period = d["tmpl"], d["tmpl_event"], d["period"]
    rep = {}
    for p in (*MONTHS, OCTOBER):
        s = (tm == 3) & (period == p)
        ev = tev[s]
        u, g = np.unique(ev, return_inverse=True)
        cell = {
            "events": int(len(u)),
            "rows": int(s.sum()),
            "events_per_100k": float(len(u) / (period == p).sum() * 1e5),
        }
        if p != OCTOBER:
            lab = d["cleaned_full"][s]
            votes = np.zeros((len(u), len(CLASSES)), dtype=np.int64)
            np.add.at(votes, (g, lab), 1)
            major = votes.argmax(axis=1)
            cell["by_label"] = {CLASSES[c]: int((major == c).sum()) for c in np.unique(major)}
        bot = (d["events"] >= 0) & (d["events"] // 10_000_000 == 1) & (period == p)
        cell["bot_bursts"] = int(np.unique(d["events"][bot]).size)
        cell["bot_bursts_per_100k"] = float(cell["bot_bursts"] / (period == p).sum() * 1e5)
        rep[p] = cell
        print(f"period {p}: {cell}")
    return rep


def cmd_ceiling(d: dict) -> dict:
    rep = f1_ceiling(d["gen"], d["cleaned"])
    for c, v in rep["per_class"].items():
        print(
            f"  {CLASSES[c]:20s} generator rows {v['generator_rows']:5d} ordinary "
            f"{v['ordinary_rows']:7.0f} F1 ceiling {v['f1']:.3f}"
        )
    print(f"macro ceiling on clean labels: {rep['macro']:.4f}")
    return {
        "macro": rep["macro"],
        "per_class": {CLASSES[c]: v for c, v in rep["per_class"].items()},
    }


RULES = {
    "sender-10m": ("sender", 600),
    "sender-1h": ("sender", 3600),
    "device-1h": ("device", 3600),
    "recv-1h": ("recv", 3600),
    "recv-24h": ("recv", 86400),
    "onward-1h": ("onward", 3600),
    "onward-24h": ("onward", 86400),
    "back-24h": ("back", 86400),
}


def cmd_rules(d: dict) -> dict:
    """Leave-one-month-out effect of each completion rule on the best file's calls."""
    from ringtrace.scoring import per_class_f1

    led, calls, period = d["led"], d["calls"], d["period"]
    ts = led["ts"].to_numpy().astype(np.int64)
    person = led["to_person"].to_numpy()
    cols = {c: led[c].to_numpy().astype(np.int64) for c in ("sender", "receiver", "device")}
    ntr = len(d["given"])
    targets = {"given": d["given"], "cleaned": d["cleaned"], "gen": d["gen"]}
    rep = {}
    for name, (kind, window) in RULES.items():
        qk, ak, direction = KINDS[kind]
        new = complete_by_link(
            calls,
            period,
            d["gen_full"] <= 0,
            cols[qk],
            cols[ak],
            ts,
            direction,
            window,
            q_person=person if qk == "receiver" else None,
            a_person=person if ak == "receiver" else None,
        )
        cell = {
            "train_rows": int((new[:ntr] != calls[:ntr]).sum()),
            "october_rows": int((new[ntr:] != calls[ntr:]).sum()),
        }
        for t, y in targets.items():
            cell[t] = []
            for m in MONTHS:
                r = np.flatnonzero(period[:ntr] == m)
                base = float(np.mean(per_class_f1(y[r], calls[r])))
                cell[t].append(float(np.mean(per_class_f1(y[r], new[r]))) - base)
        rep[name] = cell
        print(
            f"{name:11s} rows train {cell['train_rows']:4d} / October {cell['october_rows']:3d} | "
            + " | ".join(f"{t} " + " / ".join(f"{x:+.5f}" for x in cell[t]) for t in targets)
        )
    return rep


COMMANDS = {
    "partial": cmd_partial,
    "links": cmd_links,
    "rules": cmd_rules,
    "fraudblock": cmd_fraudblock,
    "mf23": cmd_mf23,
    "ceiling": cmd_ceiling,
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    args = ap.parse_args(argv)
    t0 = time.time()
    d = load()
    names = list(COMMANDS) if args.command == "all" else [args.command]
    report = {}
    for name in names:
        print(f"== {name}")
        report[name] = COMMANDS[name](d)
    report["wall_s"] = time.time() - t0
    path = out_dir() / f"{args.command}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {path} ({report['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
