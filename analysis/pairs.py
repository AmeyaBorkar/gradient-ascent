"""Look-alike pairs: who writes the partner label, and a takeover flip inside the own-phone drain.

    python analysis/pairs.py evidence                      # zone and size tables (train only)
    python analysis/pairs.py validate [--prep PATH]        # train-month numbers of the flip
    python analysis/pairs.py build --best CSV --out CSV    # the October file

``evidence`` shows that a partner label inside a look-alike template is the routine that wrote
the event, not label noise: in the drain and collect templates, every event with a phone from the
fraud block sits in the zone of its own label, and the partner events differ from the template's
own events in generation-time parameters (size, payee and login flags) that a label flip applied
after generation could not change.

``validate`` scores the flip on the best file's out-of-fold calls (``prep.npz`` holds them, in
ledger order, with the given / cleaned / gen-only targets) per held-out month. ``build`` applies
the same rule to the best file's October calls and writes the candidate in the sample
submission's order. October rows are only ever counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pairs_core import (  # noqa: E402
    drain_event_features,
    event_label,
    event_scores,
    flip,
    matrix,
    month_gains,
    passes_bar,
)

from ringtrace import paths, store, tables  # noqa: E402
from ringtrace.classes import CLASSES, CODE  # noqa: E402

SIM, ATO = CODE["sim_swap"], CODE["account_takeover"]
OWN_PHONE_DRAIN = 1  # drain shape 1: 0.25 steps on a fresh SIM throughout
NEW_PHONE_DRAIN = 3  # drain shape 3: 0.35 steps on one extra phone
C_REG = 1.0
THRESHOLD = 0.57
MONTHS = (7, 8, 9)
LEDGER_COLS = [
    "txn_id", "ts", "timestamp", "sender", "device", "login_to_txn_sec", "new_payee",
    "pin_retries", "sim_age_hours", "balance_before", "sender_account_age_days", "s_opened_on",
    "is_test", "period", "y",
]  # fmt: skip


def drain_rows() -> pl.DataFrame:
    """Every drain row (train and October) with the columns the event table needs."""
    led = tables.ledger().select(LEDGER_COLS).with_row_index("i")
    drn = store.family("drains", verbose=False).select("drn_event", "drn_shape")
    seg = store.family("phones", verbose=False).select("ctr_seg")
    led = pl.concat([led, drn, seg], how="horizontal").filter(pl.col("drn_event") >= 0)
    age = (pl.col("timestamp").dt.date() - pl.col("s_opened_on")).dt.total_days()
    return led.with_columns(
        (pl.col("sender_account_age_days") - age).alias("age_err"),
        pl.col("timestamp").dt.hour().alias("hour"),
    )


def own_phone_events(rows: pl.DataFrame) -> pl.DataFrame:
    """The own-phone drain events with features, given-label counts, zone flags and row lists."""
    rows = rows.filter(pl.col("drn_shape") == OWN_PHONE_DRAIN)
    feats = drain_event_features(rows)
    meta = rows.group_by("drn_event").agg(
        pl.col("period").first(),
        pl.col("is_test").first(),
        (pl.col("y") == SIM).sum().alias("n_sim"),
        (pl.col("y") == ATO).sum().alias("n_ato"),
        (pl.col("ctr_seg") == SIM - 1).any().fill_null(False).alias("sim_zone"),
        pl.col("i").alias("rows"),
    )
    return feats.join(meta, on="drn_event").sort("drn_event")


def flip_selection(events: pl.DataFrame, c: float = C_REG, threshold: float = THRESHOLD):
    """Scores and the events whose sim_swap calls become account_takeover (never an event with
    a phone in the counter's sim_swap zone)."""
    X = matrix(events)
    label = event_label(events["n_sim"].to_numpy(), events["n_ato"].to_numpy())
    is_test = events["is_test"].to_numpy()
    s = event_scores(X, label, events["period"].to_numpy(), is_test, MONTHS, c)
    select = (s >= threshold) & ~events["sim_zone"].to_numpy()
    return s, select, label


def cmd_evidence() -> dict:
    rows = drain_rows().filter(~pl.col("is_test"))
    ev = rows.group_by("drn_event").agg(
        pl.col("drn_shape").first().alias("shape"),
        pl.len().alias("size"),
        (pl.col("y") == SIM).sum().alias("n_sim"),
        (pl.col("y") == ATO).sum().alias("n_ato"),
        pl.col("ctr_seg").drop_nans().drop_nulls().max().alias("seg"),
        (pl.col("login_to_txn_sec") > 25).any().alias("login_hi"),
        (pl.col("new_payee") == 0).any().alias("old_payee"),
    )
    lab = event_label(ev["n_sim"].to_numpy(), ev["n_ato"].to_numpy())
    ev = ev.with_columns(
        pl.Series("lab", np.array(["sim", "ato", "tie"])[np.where(lab < 0, 2, lab)])
    )
    out = {}
    for shape in (OWN_PHONE_DRAIN, NEW_PHONE_DRAIN):
        e = ev.filter((pl.col("shape") == shape) & (pl.col("lab") != "tie"))
        zone = e.filter(pl.col("seg").is_not_null())
        out[f"drain{shape}"] = {
            "events": e.group_by("lab").len().sort("lab").rows(),
            "size_by_label": e.group_by("lab", "size").len().sort("lab", "size").rows(),
            "zone_by_label": zone.group_by("lab", "seg").len().sort("lab", "seg").rows(),
            "login_or_payee_break_by_label": e.group_by("lab")
            .agg((pl.col("login_hi") | pl.col("old_payee")).sum())
            .sort("lab")
            .rows(),
        }
    # collect templates: just-under collect (shape 1) and one bait answered within 4 h (shape 7)
    led = tables.ledger().select("is_test", "y")
    col = store.family("requests", verbose=False).select("col_event", "col_shape")
    seg = store.family("phones", verbose=False).select("ctr_seg")
    rows = pl.concat([led, col, seg], how="horizontal").filter(
        ~pl.col("is_test") & pl.col("col_shape").is_in([1, 7])
    )
    names = {CODE["phishing_collect"]: "phish", CODE["fake_cashback"]: "cashb"}
    names[CODE["loan_app_extortion"]] = "loan"
    rows = rows.with_columns(
        pl.col("y").replace_strict(names, default="other", return_dtype=pl.String).alias("lab")
    ).filter(pl.col("y") > 0)
    for shape in (1, 7):
        r = rows.filter(pl.col("col_shape") == shape)
        zone = r.filter(pl.col("ctr_seg").is_not_nan() & pl.col("ctr_seg").is_not_null())
        out[f"collect{shape}"] = {
            "fraud_rows_by_label": r.group_by("lab").len().sort("lab").rows(),
            "zone_rows_by_label": zone.group_by("lab", "ctr_seg")
            .len()
            .sort("lab", "ctr_seg")
            .rows(),
        }
    print(json.dumps(out, indent=1, default=str))
    return out


def cmd_validate(prep: Path) -> dict:
    d = np.load(prep)
    calls, period, is_test = d["calls"], d["period"], d["is_test"]
    events = own_phone_events(drain_rows())
    s, select, label = flip_selection(events)
    rows_of = [np.asarray(r) for r in events["rows"].to_list()]
    new = flip(calls, rows_of, select, SIM, ATO)
    tr = ~is_test
    gains = {
        t: month_gains(d[t], calls[tr], new[tr], period[tr], MONTHS)
        for t in ("given", "cleaned", "gen")
    }
    from sklearn.metrics import roc_auc_score

    te = events["is_test"].to_numpy()
    sel_tr = select & ~te & (label >= 0)
    ev_period = events["period"].to_numpy()
    auc = []
    for m in MONTHS:
        ok = ~te & (label >= 0) & (ev_period == m)
        auc.append(float(roc_auc_score(label[ok], s[ok])))
    out = {
        "event_auc_by_month": auc,
        "gains": gains,
        "passes": passes_bar(gains),
        "train_events_flipped": int((select & ~te).sum()),
        "their_takeover_share": float(label[sel_tr].mean()) if sel_tr.any() else None,
        "train_rows_changed": int((new != calls)[tr].sum()),
        "october_events_flipped": int((select & te).sum()),
        "october_rows_changed": int((new != calls)[is_test].sum()),
    }
    print(json.dumps(out, indent=1))
    return out


def cmd_build(best: Path, out_path: Path) -> dict:
    sample = pl.read_csv(paths.data_dir() / "sample_submission.csv", columns=["id"])
    sub = pl.read_csv(best, infer_schema=False)
    led = tables.ledger().select("txn_id", "is_test").with_row_index("i")
    pos = {c: k for k, c in enumerate(CLASSES)}
    calls = np.full(led.height, -1, dtype=np.int64)
    te = led.filter(pl.col("is_test")).join(sub.rename({"id": "txn_id"}), on="txn_id", how="left")
    if te["label"].null_count():
        raise ValueError("best file misses October ids")
    calls[te["i"].to_numpy()] = [pos[x] for x in te["label"].to_list()]
    events = own_phone_events(drain_rows())
    _, select, _ = flip_selection(events)
    select &= events["is_test"].to_numpy()
    rows_of = [np.asarray(r) for r in events["rows"].to_list()]
    new = flip(calls, rows_of, select, SIM, ATO)
    lab = np.array(CLASSES)[new[te["i"].to_numpy()]]
    out = sample.join(pl.DataFrame({"id": te["txn_id"], "label": lab}), on="id", how="left")
    if out.height != sample.height or out["label"].null_count():
        raise ValueError("candidate does not cover the sample submission")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path)
    old = sample.join(sub, on="id", how="left")["label"]
    moved = out.with_columns(old.alias("old")).filter(pl.col("label") != pl.col("old"))
    moves = moved.group_by("old", "label").len().sort("len", descending=True).rows()
    rep = {
        "file": str(out_path),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "rows_changed": moved.height,
        "moves": moves,
        "october_events_flipped": int(select.sum()),
    }
    print(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evidence")
    v = sub.add_parser("validate")
    v.add_argument("--prep", type=Path, default=paths.out_dir() / "closelook" / "prep.npz")
    b = sub.add_parser("build")
    b.add_argument(
        "--best", type=Path, default=paths.out_dir() / "candidates" / "blend-tabmF-bq.csv"
    )
    b.add_argument(
        "--out", type=Path, default=paths.out_dir() / "candidates" / "pairs-tmplA-flip.csv"
    )
    a = ap.parse_args(argv)
    if a.cmd == "evidence":
        cmd_evidence()
    elif a.cmd == "validate":
        cmd_validate(a.prep)
    else:
        cmd_build(a.best, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
