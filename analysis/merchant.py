"""October check of the merchant-side classes: mule_account, refund_abuse, merchant_collusion.

    python analysis/merchant.py [--prep PATH] [--best PATH]

Reads the best file's calls (out of fold on the train months, October) from ``--prep`` (an npz
in ledger order with ``calls``, ``period``, ``is_test``, ``given``, ``cleaned``, ``gen``; default
``$RINGTRACE_OUT/closelook/prep.npz``) and the label-free evidence families from the store.
Reports, by split period:

* calls per 100k rows of every class (and the same by calendar month, which counts the 147 train
  rows dated 1 October as October);
* each class's evidence bands: rows and calls per 100k, train precision on given / cleaned /
  gen-only labels;
* fraud-block rows per 100k by counter segment (the fraud pass's volume of each class);
* ring events (round-amount merchant chains) by size, and the counter segment of their
  fraud-block rows (train by label, October by size band);
* repeat-payer merchant chains on other amount grids (an adapted collusion ring would show here);
* five candidate trims or retypes, scored on the train months (macro-F1 change per month and
  target) with the October rows each would change.

October is only ever counted. Writes ``$RINGTRACE_OUT/merchant/report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merchant_core import (  # noqa: E402
    band_report,
    call_rates,
    event_sizes,
    passes,
    rule_effect,
    zone_votes,
)

from ringtrace import paths, store  # noqa: E402
from ringtrace.classes import CLASSES, CODE  # noqa: E402

MONTHS = (7, 8, 9)
MULE, REFUND, COLLU = CODE["mule_account"], CODE["refund_abuse"], CODE["merchant_collusion"]
RING_AMOUNTS = (1000.0, 2000.0, 2500.0, 5000.0, 10000.0)


def load(prep: Path):
    d = np.load(prep)
    led = store.ledger()
    if len(d["calls"]) != led.height:
        raise ValueError(f"{prep}: {len(d['calls'])} rows, ledger {led.height}")
    if not np.array_equal(d["period"], led["period"].to_numpy()):
        raise ValueError(f"{prep}: period differs from the ledger")
    fam = {
        "merchant": store.family("merchant", verbose=False),
        "repeats": store.family("repeats", verbose=False),
        "flows": store.family("flows", verbose=False),
        "phones": store.family("phones", verbose=False),
    }
    return d, led, fam


def check_best(d, led, best: Path) -> dict:
    """Agreement of the npz's October calls with the best file (rows in the sample's order)."""
    if not best.exists():
        return {"best_file": None}
    sub = pl.read_csv(best)
    code = sub["label"].replace_strict({c: i for i, c in enumerate(CLASSES)}, return_dtype=pl.Int64)
    te = led.filter(pl.col("is_test")).select("txn_id").with_row_index("i")
    pos = dict(zip(sub["id"].to_list(), range(sub.height), strict=True))
    order = np.array([pos[t] for t in te["txn_id"].to_list()])
    agree = int((code.to_numpy()[order] == d["calls"][d["is_test"]]).sum())
    return {"best_file": best.name, "october_rows": int(te.height), "agree": agree}


def label_full(d, name: str) -> np.ndarray:
    out = np.full(len(d["calls"]), -1, dtype=np.int64)
    out[~d["is_test"]] = d[name]
    return out


def bands(fam) -> dict[int, np.ndarray]:
    rng, rep, fl = fam["merchant"], fam["repeats"], fam["flows"]
    shape = rng["rng_shape"].to_numpy()
    size = rng["rng_size"].fill_null(0).to_numpy()
    rep_shape = rep["rep_shape"].to_numpy()
    rep_size = rep["rep_size"].fill_null(0).to_numpy()
    mf = fl["mf_shape"].to_numpy()
    snd = fl["mf_bu_senders"].fill_null(0).to_numpy()
    n = len(shape)

    refund = np.full(n, "none", dtype=object)
    refund[shape == 1] = "ring 1-3"
    refund[shape == 2] = "ring 4-7"
    refund[(shape == 2) & (size == 8)] = "ring 8"
    refund[rep_shape == 1] = np.char.add("refund chain ", np.clip(rep_size, 3, 9).astype(str))[
        rep_shape == 1
    ]

    collu = np.full(n, "none", dtype=object)
    collu[shape == 1] = "ring 1-3"
    collu[shape == 2] = "ring 4-8"
    collu[shape == 3] = "ring 9+"
    collu[shape == 4] = "QR to person"
    collu[shape == 5] = "whole-rupee chain"

    mule = np.full(n, "none", dtype=object)
    mule[mf == 1] = "chain hop"
    mule[mf == 4] = "fan-in 3-7"
    mule[mf == 3] = "onward"
    for lo, hi in ((8, 11), (12, 15), (16, 19), (20, 23), (24, 99)):
        mule[(mf == 2) & (snd >= lo) & (snd <= hi)] = f"inflow {lo}-{min(hi, 26)} senders"
    return {MULE: mule.astype(str), REFUND: refund.astype(str), COLLU: collu.astype(str)}


def ring_events(d, fam, cleaned: np.ndarray) -> dict:
    rng = fam["merchant"]
    shape = rng["rng_shape"].to_numpy()
    ev = np.where(np.isin(shape, (2, 3)), rng["rng_event"].fill_null(-1).to_numpy(), -1)
    size = event_sizes(ev)
    period = d["period"]
    ph = fam["phones"]
    block = ph["ctr_block"].to_numpy()
    seg_class = np.where(block == 2, np.nan_to_num(ph["ctr_seg"].to_numpy(), nan=-2) + 1, -1)
    seg_class = seg_class.astype(np.int64)
    first = np.zeros(len(ev), dtype=bool)
    _, idx = np.unique(ev, return_index=True)
    first[idx] = True
    first &= ev >= 0
    hist = {}
    for p in (7, 8, 9, 10):
        s = size[first & (period == p)]
        vals, cnt = np.unique(np.clip(s, 0, 16), return_counts=True)
        hist[str(p)] = {int(v): int(c) for v, c in zip(vals, cnt, strict=True)}
    # train events by their majority cleaned label, October events by size band
    lab = pl.DataFrame({"ev": ev, "y": cleaned}).filter(pl.col("ev") >= 0)
    maj = lab.group_by("ev").agg(pl.col("y").mode().min().alias("m"))
    maj_of = dict(zip(maj["ev"].to_list(), maj["m"].to_list(), strict=True))
    ev_lab = np.array([maj_of.get(e, -1) for e in ev])
    sz = np.where(size <= 7, "4-7", np.where(size == 8, "8", "9+"))
    train = (period < 10) & (ev >= 0)
    test = (period == 10) & (ev >= 0)
    votes_train = zone_votes(np.where(train, ev, -1), block == 2, seg_class, ev_lab)
    votes_oct = zone_votes(np.where(test, ev, -1), block == 2, seg_class, sz)
    held = {}
    for p in (7, 8, 9, 10):
        for b in ("4-7", "8", "9+"):
            sel = first & (period == p) & (sz == b)
            fb = pl.DataFrame({"ev": ev, "fb": block == 2}).filter(pl.col("ev") >= 0)
            has = set(fb.filter(pl.col("fb"))["ev"].unique().to_list())
            held[f"{p} {b}"] = [int(sel.sum()), int(sum(e in has for e in ev[sel]))]
    return {
        "size_hist": hist,
        "zone_train_by_label": votes_train,
        "zone_october_by_size": votes_oct,
        "events_with_fraud_block_row": held,
        "seg_agrees_with_prep": bool(
            np.array_equal(
                np.where(block == 2, seg_class, 0), np.where(block == 2, d["ctr_seg"], 0)
            )
        ),
    }


def repeat_payer_scan(led, d, cleaned) -> dict:
    """Merchant chains (24 h gap) of >= 5 PAY/QR rows from at most half as many senders, by amount
    grid: events per 100k rows per period, train label mix, October calls."""
    df = led.select("is_test", "ts", "row", "sender", "receiver", "to_person", "txn_type", "amount")
    df = df.with_columns(
        pl.Series("per", d["period"]), pl.Series("call", d["calls"]), pl.Series("y", cleaned)
    ).filter(~pl.col("to_person") & pl.col("txn_type").is_in(["PAY", "QR"]))
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    grids = {
        "ring amounts": pl.col("amount").is_in(RING_AMOUNTS),
        "multiple of 500, not ring": (cents % 50_000 == 0) & ~pl.col("amount").is_in(RING_AMOUNTS),
        "multiple of 100, not 500": (cents % 10_000 == 0) & (cents % 50_000 != 0),
        "whole rupees, not 100, >= 900": (cents % 100 == 0)
        & (cents % 10_000 != 0)
        & (pl.col("amount") >= 900),
        "not whole, >= 900": (cents % 100 != 0) & (pl.col("amount") >= 900),
    }
    rows = dict(zip(*np.unique(d["period"], return_counts=True), strict=True))
    out = {}
    for name, f in grids.items():
        c = df.filter(f).sort("is_test", "receiver", "ts", "row")
        new = (
            (pl.col("receiver") != pl.col("receiver").shift(1))
            | (pl.col("is_test") != pl.col("is_test").shift(1))
            | ((pl.col("ts") - pl.col("ts").shift(1)) > 86_400)
        ).fill_null(True)
        ev = (
            c.with_columns(new.cum_sum().alias("ch"))
            .group_by("ch")
            .agg(
                pl.col("per").first(),
                pl.len().alias("size"),
                pl.col("sender").n_unique().alias("snd"),
                (pl.col("y") == COLLU).sum().alias("collusion"),
                (pl.col("y") == REFUND).sum().alias("refund"),
                (pl.col("y") > 0).sum().alias("fraud"),
                (pl.col("call") == COLLU).sum().alias("called_collusion"),
                (pl.col("call") > 0).sum().alias("called_fraud"),
            )
            .filter((pl.col("size") >= 5) & (pl.col("size") >= 2 * pl.col("snd")))
        )
        rep = {}
        for p in (7, 8, 9, 10):
            e = ev.filter(pl.col("per") == p)
            r = {
                "events_per_100k": e.height / rows[p] * 1e5,
                "rows_per_100k": int(e["size"].sum()) / rows[p] * 1e5,
                "called_collusion": int(e["called_collusion"].sum()),
                "called_fraud": int(e["called_fraud"].sum()),
            }
            if p < 10:
                r.update({k: int(e[k].sum()) for k in ("collusion", "refund", "fraud")})
            rep[str(p)] = r
        out[name] = rep
    return out


def rules(d, fam) -> dict[str, np.ndarray]:
    rng, rep, fl = fam["merchant"], fam["repeats"], fam["flows"]
    shape = rng["rng_shape"].to_numpy()
    size = rng["rng_size"].fill_null(0).to_numpy()
    rep_shape = rep["rep_shape"].to_numpy()
    mf = fl["mf_shape"].to_numpy()
    snd = fl["mf_bu_senders"].fill_null(0).to_numpy()
    calls = d["calls"]
    out = {}
    new = calls.copy()
    new[(calls == REFUND) & (shape == 2) & (size == 8)] = COLLU
    out["ring8-collusion"] = new
    new = calls.copy()
    new[(calls == MULE) & (mf == 2) & (snd >= 24)] = 0
    out["mule-wide-trim"] = new
    new = calls.copy()
    new[(calls == REFUND) & (rep_shape != 1) & (shape != 2)] = 0
    out["refund-plain-trim"] = new
    new = calls.copy()
    new[(calls == COLLU) & ~np.isin(shape, (3, 5))] = 0
    out["collusion-plain-trim"] = new
    new = calls.copy()
    new[(calls == MULE) & (mf == 0)] = 0
    out["mule-plain-trim"] = new
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, default=paths.out_dir() / "closelook" / "prep.npz")
    ap.add_argument(
        "--best", type=Path, default=paths.out_dir() / "candidates" / "blend-tabmF-bq.csv"
    )
    a = ap.parse_args(argv)
    d, led, fam = load(a.prep)
    is_test = d["is_test"]
    period = d["period"]
    given, cleaned, gen = (label_full(d, k) for k in ("given", "cleaned", "gen"))
    train = ~is_test
    report = {"check": check_best(d, led, a.best)}
    calendar = np.where(is_test, 10, led["month"].to_numpy())
    report["call_rates_period"] = call_rates(d["calls"], period)
    report["call_rates_calendar"] = call_rates(d["calls"], calendar)
    targets = {"given": given, "cleaned": cleaned, "gen": gen}
    report["bands"] = {
        CLASSES[k]: band_report(b, d["calls"], period, k, targets, train)
        for k, b in bands(fam).items()
    }
    report["ring_events"] = ring_events(d, fam, cleaned)
    ph = fam["phones"]
    fb = ph["ctr_block"].to_numpy() == 2
    seg = np.where(fb, np.nan_to_num(ph["ctr_seg"].to_numpy(), nan=-2) + 1, 0).astype(np.int64)
    report["zone_rows_per_100k"] = call_rates(seg, period)
    report["repeat_payer_scan"] = repeat_payer_scan(led, d, cleaned)
    tt = {"given": d["given"], "cleaned": d["cleaned"], "gen": d["gen"]}
    report["rules"] = {}
    for name, new in rules(d, fam).items():
        eff = rule_effect(d["calls"][train], new[train], period[train], tt, MONTHS)
        moved = new[is_test] != d["calls"][is_test]
        report["rules"][name] = {
            "train": eff,
            "passes": passes(eff),
            "october_changed": int(moved.sum()),
        }
    out = paths.out_dir() / "merchant"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["check"]))
    print("calls per 100k by period (Jul Aug Sep Oct) | by calendar month:")
    for k in (MULE, REFUND, COLLU):
        p = [report["call_rates_period"][m][k] for m in (7, 8, 9, 10)]
        c = [report["call_rates_calendar"][m][k] for m in (7, 8, 9, 10)]
        print(
            f"  {CLASSES[k]:<20} "
            + " ".join(f"{v:6.1f}" for v in p)
            + f"  x{p[3] / np.mean(p[:3]):.2f} | "
            + " ".join(f"{v:6.1f}" for v in c)
            + f"  x{c[3] / np.mean(c[:3]):.2f}"
        )
    print("fraud-block rows per 100k by counter segment (Jul Aug Sep Oct):")
    for k in range(1, 12):
        z = [report["zone_rows_per_100k"][m][k] for m in (7, 8, 9, 10)]
        print(
            f"  {CLASSES[k]:<20} "
            + " ".join(f"{v:6.1f}" for v in z)
            + f"  x{z[3] / np.mean(z[:3]):.2f}"
        )
    for name, r in report["rules"].items():
        cells = " ".join(
            f"{m}:{r['train'][m]['given']:+.4f}/{r['train'][m]['cleaned']:+.4f}/"
            f"{r['train'][m]['gen']:+.4f}"
            for m in r["train"]
        )
        print(f"  rule {name:<22} October rows {r['october_changed']:4d} | {cells} | {r['passes']}")
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
