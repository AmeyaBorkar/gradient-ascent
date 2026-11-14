"""October drift scan: does the scored month leave the train months, per evidence family?

    python analysis/drift.py --calls CALLS.npy [--out PATH]

``CALLS.npy`` is one class code per ledger row, in ledger order; train calls are out of fold.
The scan reads the ledger and the cached evidence columns and writes
``$RINGTRACE_OUT/drift/drift.json``. Every volume is a count or a rate per 100,000 rows of that
month (``period``: a train row dated October stays in September). Test rows are only ever
counted, and a test group of 1 to 4 is reported as ``<5``.

Families with events are bursts, drains, flows, requests, merchant and repeats. Marks have no
events. The device counter is not one of the thresholds this scan relaxes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_core import (  # noqa: E402
    BANDS,
    CLASS_CUTS,
    MONTHS,
    PER,
    TEST,
    TRAIN,
    _blurb,
    band_masks,
    band_table,
    call_table,
    collapse_events,
    departure,
    rank_departures,
    size_table,
    volume_table,
)

from ringtrace import paths, silver, store  # noqa: E402
from ringtrace.classes import CLASSES  # noqa: E402
from ringtrace.evidence import anchors  # noqa: E402

# family, event column, shape column, measure name -> column
FAMILIES = (
    ("bot", "bot_event", "bot_shape", {"size": "bot_size", "accounts": "bot_accounts"}),
    ("drn", "drn_event", "drn_shape", {"size": "drn_size"}),
    (
        "mf",
        "mf_event",
        "mf_shape",
        {"size": "mf_size", "bu_senders": "mf_bu_senders", "fan_senders": "mf_fan_senders"},
    ),
    ("col", "col_event", "col_shape", {"size": "col_size", "n_bait": "col_n_bait"}),
    ("rng", "rng_event", "rng_shape", {"size": "rng_size", "senders": "rng_senders"}),
    ("rep", "rep_event", "rep_shape", {"size": "rep_size"}),
)

NOTES = {
    "rate": "per 100,000 rows of that month; period keeps a train row dated October in September",
    "z": (
        "(test count - mu) / sqrt(mu); mu is the mean of the three train monthly rates "
        "times test rows"
    ),
    "chi2": (
        "sum (obs - exp)^2 / exp over bins with exp > 0; "
        "exp is the pooled train count times test_rows / train_rows; the test total is not fitted"
    ),
    "pileup": "residual z >= 2 on a bin next to a class cut (ring size 3 and 8, sender count 7)",
    "small": (
        "test counts from 1 to 4 are '<5>'; their rate and ratio are omitted; "
        "z uses the exact count"
    ),
    "near_miss": "the law holds except one threshold, relaxed on the side the law rejects",
    "omitted": "marks have no events; the device-counter block mix is not a listed threshold",
}

TOP = 15


def _plain(value):
    if isinstance(value, np.floating):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _rows_by_month(period: np.ndarray) -> dict[int, int]:
    counts = {m: 0 for m in MONTHS}
    values, n = np.unique(period, return_counts=True)
    for v, k in zip(values.tolist(), n.tolist(), strict=True):
        if int(v) in counts:
            counts[int(v)] = int(k)
    return counts


def _family_frame(
    period, gen, fam: pl.DataFrame, event: str, shape: str, measures: dict
) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "event": fam[event],
            "shape": fam[shape],
            "period": pl.Series("period", period),
            "gen": pl.Series("gen", gen),
        }
    )
    for name, col in measures.items():
        df = df.with_columns(fam[col].alias(name))
    return df.filter(
        (pl.col("shape").fill_null(0) > 0) & pl.col("event").is_not_null() & (pl.col("event") >= 0)
    )


def _shape_label(shape: int) -> str | None:
    return None if int(shape) == -1 else int(shape)


def _scan_family(name, frame, rows, names) -> tuple[list[dict], list[dict]]:
    measures = next(spec[3] for spec in FAMILIES if spec[0] == name)
    events, shapes = collapse_events(frame, tuple(measures))
    volume = []
    if events.height:
        total = volume_table(events.with_columns(pl.lit(-1).alias("shape")), rows, names)
        per = volume_table(shapes, rows, names)
        for rec in total + per:
            rec["family"] = name
            rec["shape"] = _shape_label(rec["shape"])
            volume.append(rec)
    sizes = []
    for measure in measures:
        if events.height == 0 or measure not in events.columns:
            continue
        sub = events.filter(pl.col(measure).is_not_null())
        if sub.height == 0:
            continue
        rec = size_table(
            sub[measure].to_numpy(),
            sub["period"].to_numpy(),
            sub["gen"].to_numpy(),
            rows,
            CLASS_CUTS.get((name, measure), ()),
            names,
        )
        rec["family"] = name
        rec["measure"] = measure
        sizes.append(rec)
    return volume, sizes


def _size_departure(family: str, measure: str, bin_row: dict, test_rows: int) -> dict:
    test_cell = next(c for c in bin_row["months"] if c["month"] == TEST)
    hidden = test_cell["n"] == "<5"
    train_rates = [
        c["rate"] for c in bin_row["months"] if c["month"] in TRAIN and c["rate"] is not None
    ]
    baseline = float(np.mean(train_rates)) if train_rates else 0.0
    no_baseline = bin_row["z"] is None and bin_row["novel"] and not hidden
    return {
        "kind": "size",
        "name": f"{family} {measure} {bin_row['bin']}",
        "z": bin_row["z"],
        "no_baseline": no_baseline or bool(bin_row["novel"] and bin_row["pileup"]),
        "train_rate": baseline,
        "test_rate": test_cell["rate"],
        "ratio": None,
        "test_n": test_cell["n"],
        "pileup": bin_row["pileup"],
        "expected_per_100k": None if test_rows <= 0 else PER * bin_row["expected"] / test_rows,
    }


def build_report(
    led: pl.DataFrame, calls: np.ndarray, gen: np.ndarray | None, fam: pl.DataFrame
) -> dict:
    """The whole scan from frames already aligned to the ledger. Aggregates only."""
    period = led["period"].to_numpy()
    rows = _rows_by_month(period)
    names = tuple(CLASSES)
    if gen is None:
        gen_arr = np.full(led.height, -1, dtype=np.int64)
        have_gen = False
    else:
        gen_arr = np.asarray(gen, dtype=np.int64)
        have_gen = True
    calls = np.asarray(calls)
    report: dict = {
        "rows": [{"month": m, "rows": rows[m]} for m in MONTHS],
        "have_structured": have_gen,
        "notes": NOTES,
        "calls": call_table(calls, period, rows, names),
    }
    volume: list[dict] = []
    sizes: list[dict] = []
    for name, event, shape, measures in FAMILIES:
        frame = _family_frame(period, gen_arr, fam, event, shape, measures)
        vol, sz = _scan_family(name, frame, rows, names)
        volume.extend(vol)
        sizes.extend(sz)
    report["events"] = volume
    report["sizes"] = sizes
    gen_for_bands = gen_arr if have_gen else None
    masks = band_masks(led)
    near = []
    for band, _fn in BANDS:
        near.append(
            {
                "band": band,
                "blurb": _blurb(band),
                **band_table(masks[band].to_numpy(), period, calls, gen_for_bands, rows),
            }
        )
    report["near_miss"] = near
    items = []
    for row in report["calls"]:
        items.append(departure("calls", row["name"], row))
    for ev in volume:
        label = "all" if ev["shape"] is None else str(ev["shape"])
        items.append(departure("events", f"{ev['family']} {label} events", ev["events"]))
        items.append(departure("event_rows", f"{ev['family']} {label} rows", ev["rows"]))
    test_rows = rows[TEST]
    for sz in sizes:
        for b in sz["bins"]:
            novel_kept = b["pileup"] or b["months"][-1]["n"] != "<5"
            if b["z"] is None and not (b["novel"] and novel_kept):
                continue
            items.append(_size_departure(sz["family"], sz["measure"], b, test_rows))
    for band in near:
        items.append(departure("near_miss", band["band"], band))
    report["ranked"] = rank_departures(items)
    return report


def _fmt(value, digits=2, width=8) -> str:
    if value is None:
        return f"{'-':>{width}}"
    if isinstance(value, str):
        return f"{value:>{width}}"
    return f"{value:>{width}.{digits}f}"


def _month_rates(comp: dict) -> str:
    return " ".join(_fmt(c["rate"]) for c in comp["months"])


def print_report(report: dict) -> None:
    """Compact aggregate tables. Test groups of 1 to 4 rows are already ``<5>``."""
    print("rows " + "  ".join(f"{r['month']}:{r['rows']}" for r in report["rows"]))
    if not report["have_structured"]:
        print("structured labels not loaded; generator-made shares are empty")
    print("\ncalls per 100k")
    print(f"{'class':<22} " + " ".join(f"{m:>8}" for m in MONTHS) + f" {'oct/train':>9} {'z':>8}")
    for row in report["calls"]:
        print(
            f"{row['name']:<22} {_month_rates(row)} {_fmt(row['ratio'], width=9)} {_fmt(row['z'])}"
        )
    print("\nevents per 100k (events, then rows)")
    print(f"{'family':<6} {'shape':<6} " + " ".join(f"{m:>8}" for m in MONTHS) + f" {'z':>8}")
    for ev in report["events"]:
        label = "all" if ev["shape"] is None else str(ev["shape"])
        print(
            f"{ev['family']:<6} {label:<6} {_month_rates(ev['events'])} {_fmt(ev['events']['z'])}"
        )
        print(f"{'':<6} {'rows':<6} {_month_rates(ev['rows'])} {_fmt(ev['rows']['z'])}")
    print("\nsizes (chi-square against row-scaled train; * is a pile-up next to a class cut)")
    for sz in report["sizes"]:
        chi = "-" if sz["chi2"] is None else f"{sz['chi2']:.1f}"
        p = "-" if sz["p"] is None else f"{sz['p']:.3g}"
        flagged = [b for b in sz["bins"] if b["pileup"]]
        flag = ", ".join(
            f"{b['bin']} z={b['z']:.2f}" if b["z"] is not None else f"{b['bin']} new"
            for b in flagged
        )
        extra = f"  pile-up {flag}" if flag else ""
        print(f"{sz['family']} {sz['measure']:<12} chi2={chi} df={sz['df']} p={p}{extra}")
    print("\nnear-miss per 100k")
    print(
        f"{'band':<16} "
        + " ".join(f"{m:>8}" for m in MONTHS)
        + f" {'z':>8} {'called10':>8} {'gen_train':>9}"
    )
    for band in report["near_miss"]:
        test = next(c for c in band["months"] if c["month"] == TEST)
        train_shares = [
            c["gen_fraud"]
            for c in band["months"]
            if c["month"] in TRAIN and c["gen_fraud"] is not None
        ]
        gen_mean = float(np.mean(train_shares)) if train_shares else None
        print(
            f"{band['band']:<16} {_month_rates(band)} {_fmt(band['z'])} "
            f"{_fmt(test['called'])} {_fmt(gen_mean, width=9)}"
        )
    print(f"\nlargest |z| (top {TOP})")
    print(f"{'z':>8} {'kind':<12} {'name':<28} {'train':>8} {'test':>8} {'n':>6}")
    for row in report["ranked"][:TOP]:
        z = "base" if row.get("no_baseline") and row.get("z") is None else _fmt(row.get("z"))
        print(
            f"{z:>8} {row['kind']:<12} {row['name']:<28} "
            f"{_fmt(row.get('train_rate'))} {_fmt(row.get('test_rate'))} "
            f"{_fmt(row.get('test_n'), width=6)}"
        )


def run(calls_path: Path) -> dict:
    led = store.ledger()
    calls = np.load(calls_path, allow_pickle=False)
    if calls.ndim != 1:
        raise ValueError(f"calls must be one value per ledger row, got shape {calls.shape}")
    if len(calls) != led.height:
        raise ValueError(f"calls has {len(calls)} rows, the ledger has {led.height}")
    cols = []
    for _name, event, shape, measures in FAMILIES:
        cols.extend([event, shape, *measures.values()])
    fam = anchors.family_columns(led, cols)
    gen = silver.load(led, "structured")
    return build_report(led, calls.astype(np.int64, copy=False), gen, fam)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="October drift scan (aggregates only)")
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    report = run(args.calls)
    out = args.out or (paths.out_dir() / "drift" / "drift.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=_plain) + "\n", encoding="utf-8")
    print_report(report)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
