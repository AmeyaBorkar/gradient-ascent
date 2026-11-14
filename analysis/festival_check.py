"""Aggregates that check the festival scaling against the scored month.

For each train month, unscaled and scaled with ``ringtrace.festival.scale_rows``, and for the
scored month, it prints the round-amount shares (whole rupee, multiple of 10, multiple of 100),
the mean amount, amount quantiles per payment type and the median balance. Train months use their
ordinary-routine rows (structured silver label 0). The scored month has no such label, so its
ordinary rows are the rows a submission file calls legit (``--calls``); all rows are shown as
well. Only aggregates are printed.

    python analysis/festival_check.py --calls <submission.csv> [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import festival, paths, silver

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def round_shares(amount: np.ndarray) -> dict[str, float]:
    paise = np.rint(np.asarray(amount, dtype=np.float64) * 100.0).astype(np.int64)
    return {
        "whole_rupee": float(np.mean(paise % 100 == 0)),
        "mult_10": float(np.mean(paise % 1000 == 0)),
        "mult_100": float(np.mean(paise % 10000 == 0)),
    }


def block(amount: np.ndarray, balance: np.ndarray, txn_type: np.ndarray) -> dict:
    out = {"rows": int(len(amount)), "mean_amount": float(np.mean(amount))}
    out.update(round_shares(amount))
    out["median_balance"] = float(np.median(balance))
    paise = np.rint(np.asarray(amount, dtype=np.float64) * 100.0).astype(np.int64)
    cls = np.select(
        [paise % 10000 == 0, paise % 1000 == 0, paise % 100 == 0], ["x100", "x10", "x1"], "paise"
    )
    out["class_mean"] = {str(c): float(np.mean(amount[cls == c])) for c in np.unique(cls)}
    out["quantiles"] = {
        "all": [float(v) for v in np.quantile(amount, QUANTILES)],
    }
    for t in np.unique(txn_type):
        sel = txn_type == t
        if sel.sum() >= 200:
            out["quantiles"][str(t)] = [float(v) for v in np.quantile(amount[sel], QUANTILES)]
    return out


def ratios(a: dict, b: dict) -> dict:
    """Quantile and mean ratios a / b, per payment type present in both."""
    out = {"mean_amount": a["mean_amount"] / b["mean_amount"]}
    out["median_balance"] = a["median_balance"] / b["median_balance"]
    out["class_mean"] = {
        c: a["class_mean"][c] / b["class_mean"][c] for c in a["class_mean"] if c in b["class_mean"]
    }
    for key, qa in a["quantiles"].items():
        qb = b["quantiles"].get(key)
        if qb is not None:
            out[key] = [float(x / y) if y else None for x, y in zip(qa, qb, strict=True)]
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True, help="submission whose legit calls mark ordinary")
    ap.add_argument("--amount-factor", type=float, default=festival.DEFAULT_AMOUNT_FACTOR)
    ap.add_argument("--balance-factor", type=float, default=festival.DEFAULT_BALANCE_FACTOR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ledger = sorted((paths.cache_dir() / "tables").glob("ledger-*.parquet"))
    if len(ledger) != 1:
        raise SystemExit(f"expected one ledger file, found {len(ledger)}")
    cols = ["txn_id", "amount", "balance_before", "period", "is_test", "txn_type"]
    led = pl.read_parquet(ledger[0], columns=cols)
    structured = silver.load(led, "structured")
    if structured is None:
        raise SystemExit("structured silver labels are missing")
    periods = tuple(int(p) for p in np.unique(led.filter(~pl.col("is_test"))["period"].to_numpy()))
    mask = festival.festival_mask(led, structured, periods)
    scaled = festival.scale_rows(led, mask, args.amount_factor, args.balance_factor)

    calls = pl.read_csv(args.calls).rename({"id": "txn_id"})
    test = led.filter(pl.col("is_test")).join(calls, on="txn_id", how="left")
    if test["label"].null_count():
        raise SystemExit("the calls file does not cover every scored row")
    legit = (test["label"] == "legit").to_numpy()

    amount = led["amount"].to_numpy()
    amount_s = scaled["amount"].to_numpy()
    bal = led["balance_before"].to_numpy()
    bal_s = scaled["balance_before"].to_numpy()
    ttype = led["txn_type"].to_numpy()
    period = led["period"].to_numpy()
    report: dict = {"factors": [args.amount_factor, args.balance_factor], "months": {}}
    for p in periods:
        sel = mask & (period == p)
        report["months"][str(p)] = {
            "unscaled": block(amount[sel], bal[sel], ttype[sel]),
            "scaled": block(amount_s[sel], bal_s[sel], ttype[sel]),
        }
    t_amount = test["amount"].to_numpy()
    t_bal = test["balance_before"].to_numpy()
    t_type = test["txn_type"].to_numpy()
    octo = block(t_amount[legit], t_bal[legit], t_type[legit])
    report["october_called_legit"] = octo
    report["october_all"] = block(t_amount, t_bal, t_type)
    pooled_u = block(amount[mask], bal[mask], ttype[mask])
    pooled_s = block(amount_s[mask], bal_s[mask], ttype[mask])
    report["train_ordinary_pooled"] = {"unscaled": pooled_u, "scaled": pooled_s}
    report["ratio_october_over_unscaled"] = ratios(octo, pooled_u)
    report["ratio_october_over_scaled"] = ratios(octo, pooled_s)

    def line(name: str, b: dict) -> str:
        return (
            f"{name:28s} rows {b['rows']:8d}  mean {b['mean_amount']:9.2f}  whole "
            f"{b['whole_rupee']:.4f}  x10 {b['mult_10']:.4f}  x100 {b['mult_100']:.4f}  "
            f"bal50 {b['median_balance']:10.2f}"
        )

    for p in periods:
        m = report["months"][str(p)]
        print(line(f"month {p} ordinary unscaled", m["unscaled"]))
        print(line(f"month {p} ordinary scaled", m["scaled"]))
    print(line("train ordinary unscaled", pooled_u))
    print(line("train ordinary scaled", pooled_s))
    print(line("october called legit", octo))
    print(line("october all rows", report["october_all"]))
    for key in ("ratio_october_over_unscaled", "ratio_october_over_scaled"):
        r = report[key]
        shown = " ".join(f"{c} {v:.3f}" for c, v in r["class_mean"].items())
        print(key, f"mean {r['mean_amount']:.4f} bal50 {r['median_balance']:.4f} | {shown}")
        for t, v in r.items():
            if isinstance(v, list):
                print(f"   {t:14s} " + " ".join(f"{x:.3f}" for x in v))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
