"""Red-team diagnostics: how could another file be 0.008 above the blend on the public slice?

    python analysis/redteam.py noise   [--run fresh-blend-blankqr]
    python analysis/redteam.py census  [--run fresh-blend-blankqr]
    python analysis/redteam.py cells   [--run fresh-blend-blankqr]

``noise``: September with cleaned labels stands in for October. Slices of October's public size
(78,175 rows) are drawn, and the slice gap between the blend and materially different models
(context only, evidence without phones, the champion without phones) is compared with their
full-month gap. ``census``: calls per 100,000 rows per class, September out-of-fold against
October, beside the gen-only rate of September. ``cells``: label-free cells (crosses of discrete
row columns) that are at least 90 % fraud on cleaned train labels and that October calls legit
more often than out of fold. October enters only as counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fresh import load  # noqa: E402
from redteam_core import cell_scan, gap_noise, per_100k  # noqa: E402

from ringtrace import paths, silver  # noqa: E402
from ringtrace.blend import decode_all  # noqa: E402
from ringtrace.classes import CLASSES, N_CLASSES  # noqa: E402

OTHERS = ("context-only", "evidence-nophones", "champion-without-phones", "champion")
PUBLIC_ROWS = 78175


def decode(run: str, led) -> np.ndarray:
    is_test = led["is_test"].to_numpy()
    out = paths.out_dir() / "runs" / run
    proba = np.zeros((led.height, N_CLASSES))
    proba[~is_test] = np.load(out / "oof.npy")
    return decode_all(
        proba,
        led["period"].to_numpy(),
        is_test,
        {"fraud_scale": 0.875},
        ["anchored_templates"],
        led,
        log=lambda m: None,
    )["rules"]


def cmd_noise(run: str) -> dict:
    led, _, calls = load(run)
    sep = (~led["is_test"].to_numpy()) & (led["period"].to_numpy() == 9)
    y = silver.load(led, "label").astype(np.int64)[sep]
    others = {o: decode(o, led)[sep] for o in OTHERS}
    res = gap_noise(y, calls[sep], list(others.values()), PUBLIC_ROWS, draws=2000)
    rep = dict(zip(others, res, strict=True))
    for o, r in rep.items():
        print(f"[noise] {o:<26} {json.dumps(r)}")
    return rep


def cmd_census(run: str) -> dict:
    led, _, calls = load(run)
    te = led["is_test"].to_numpy()
    sep = (~te) & (led["period"].to_numpy() == 9)
    gen = silver.load(led, "structured").astype(np.int64)
    clean = silver.load(led, "label").astype(np.int64)
    rep = {}
    for k in range(N_CLASSES):
        rep[CLASSES[k]] = {
            "sep_gen_only": float(per_100k(gen, sep)[k]),
            "sep_cleaned": float(per_100k(clean, sep)[k]),
            "sep_calls": float(per_100k(calls, sep)[k]),
            "oct_calls": float(per_100k(calls, te)[k]),
        }
        print(f"[census] {CLASSES[k]:<20} {json.dumps(rep[CLASSES[k]])}")
    return rep


def features(led) -> dict[str, np.ndarray]:
    def col(name, fill):
        return led[name].fill_null(fill).to_numpy()

    amt = led["amount"].to_numpy()
    whole = np.isclose(amt, np.round(amt))
    a = np.round(amt).astype(np.int64)
    kind = np.select(
        [
            amt <= 10,
            whole & (a % 1000 == 0),
            whole & np.isin(a % 1000, [999, 990, 950]),
            whole & (a % 10 == 0),
            whole,
        ],
        [1, 2, 3, 4, 5],
        0,
    )
    ts = led["timestamp"]
    hour = ts.dt.hour().to_numpy()
    qr = led["qr_vpa_mismatch"]
    return {
        "txn_type": np.unique(col("txn_type", "?"), return_inverse=True)[1],
        "to_person": (led["receiver_type"] == "P2P").fill_null(False).to_numpy().astype(int),
        "qr_check": np.where(qr.is_null().to_numpy(), 2, qr.fill_null(0).to_numpy()).astype(int),
        "mcat_blank": led["merchant_category"].is_null().to_numpy().astype(int),
        "new_payee": col("new_payee", 0).astype(int),
        "pin": np.clip(col("pin_retries", 0), 0, 3).astype(int),
        "hour": np.digitize(hour, [6, 20, 22]),
        "amount_kind": kind,
        "sim_fresh": (col("sim_age_hours", 1e9) < 36).astype(int),
        "device_new": (col("device_age_days", 1e9) < 1.5).astype(int),
        "away": (col("device_city", "?") != col("s_home_city", "?")).astype(int),
        "login_fast": (col("login_to_txn_sec", 1e9) < 25).astype(int),
    }


def cmd_cells(run: str) -> dict:
    led, _, calls = load(run)
    te = led["is_test"].to_numpy()
    tr = ~te
    clean = silver.load(led, "label").astype(np.int64)
    feats = features(led)
    rep = {}
    for order in (1, 2, 3):
        cells = cell_scan(feats, clean, calls == 0, tr, te, order=order)
        rep[str(order)] = cells
        print(f"[cells] order {order}: {len(cells)} flagged")
        for c in cells[:15]:
            print("   ", json.dumps(c))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["noise", "census", "cells"])
    ap.add_argument("--run", default="fresh-blend-blankqr")
    a = ap.parse_args(argv)
    rep = {"noise": cmd_noise, "census": cmd_census, "cells": cmd_cells}[a.cmd](a.run)
    out = paths.out_dir() / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.cmd}-{a.run}.json").write_text(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
