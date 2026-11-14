"""Command line for the audits: ``python -m ringtrace.audit <command> [options]``.

    ordinary    can a feature set tell ordinary fraud-pass rows from legit-pass rows,
                in train and in October? (label-free ground truth from the device counter)
    links       do ordinary fraud rows cluster on a sender, receiver or phone, or recur in a
                later month? (train labels)
    shift       does one period hold a sub-population that another lacks? (adversarial)
    ceiling     each month's label-free composition and the macro-F1 of a perfect caller of the
                reachable rows; with --target, the extra correct rows a score would need
    lookalike   can the sibling type inside each look-alike template be told apart?
    shuffle     write an order probe of a submission (same content, rows permuted)

Every command writes ``$RINGTRACE_OUT/audit/<command>.json`` (aggregates only) and prints a
summary. Features come from the feature store (``--families``, default ``context``) plus a few raw
row columns (``RAW``); test rows are only ever used through such aggregates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import paths, store, threads
from ringtrace.audit import ceiling as ceil
from ringtrace.audit import lookalike, probes, shuffle, stats
from ringtrace.classes import CLASSES, N_CLASSES

LAST_TRAIN_PERIOD = 9
# raw row columns added to the store families; app_version and ip_risk_score are dead in October
# and are kept here on purpose, to show what they do in each split
RAW = (
    "minute",
    "second",
    "day",
    "cents",
    "amount",
    "balance_before",
    "device_age_days",
    "sender_account_age_days",
    "app_version",
    "ip_risk_score",
)
# columns of the context family that do not depend on the month's density or calendar
STABLE = (
    "hour",
    "sim_age_hours",
    "new_payee",
    "pin_retries",
    "login_to_txn_sec",
    "acct_age_err",
    "away",
    "dest_city_match",
    "to_person",
    "amt_whole",
    "amt_mod10",
    "amt_mod100",
    "log_balance",
    "cat_txn_type",
    "cat_merchant_category",
    "cat_app",
    "cat_s_kyc_level",
    "cat_s_os",
    "cat_s_bank_code",
    "cat_m_category",
    "cat_r_kyc",
)


def raw_columns(led: pl.DataFrame) -> pl.DataFrame:
    t = pl.col("timestamp")
    return led.select(
        t.dt.minute().cast(pl.Float32).alias("minute"),
        t.dt.second().cast(pl.Float32).alias("second"),
        t.dt.day().cast(pl.Float32).alias("day"),
        ((pl.col("amount") * 100).round(0) % 100).cast(pl.Float32).alias("cents"),
        *[pl.col(c).cast(pl.Float32) for c in RAW[4:]],
    )


def features(
    led: pl.DataFrame, families: list[str], raw: bool, only: tuple[str, ...] | None = None
):
    """Feature frame (store families, then raw columns) and the indices of categorical columns."""
    parts = [store.frame(families)] if families else []
    if raw:
        parts.append(raw_columns(led))
    frame = pl.concat(parts, how="horizontal")
    if only:
        frame = frame.select([c for c in only if c in frame.columns])
    names = frame.columns
    cat = [i for i, n in enumerate(names) if n.startswith("cat_")]
    return frame, names, cat


def rows_of(frame: pl.DataFrame, idx: np.ndarray) -> np.ndarray:
    return frame[idx].to_numpy().astype(np.float32)


def _ints(s: str) -> tuple[int, ...]:
    return tuple(int(v) for v in s.split(",") if v != "")


def _write(name: str, rep: dict) -> Path:
    out = paths.out_dir() / "audit"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.json"
    target.write_text(json.dumps(rep, indent=2, default=float))
    return target


def _split_mask(led: pl.DataFrame, split: str) -> np.ndarray:
    test = led["is_test"].to_numpy()
    return test if split == "test" else ~test


# --------------------------------------------------------------------------------------------
# commands


def cmd_ordinary(args, led, grp) -> dict:
    frame, names, cat = features(led, args.families, raw=not args.no_raw)
    probe = grp["probe"].to_numpy()
    y_lab = led["y"].to_numpy()
    rep: dict = {"families": args.families, "raw": not args.no_raw, "splits": {}}
    for split in ("train", "test"):
        idx = np.flatnonzero(_split_mask(led, split) & (probe >= 0))
        y = (probe[idx] == probes.ORDINARY).astype(np.int64)
        x = rows_of(frame, idx)
        res = stats.cv_auc(
            x,
            y,
            names=names,
            categorical=cat,
            seeds=args.seeds,
            folds=args.folds,
            threads=threads.budget(),
        )
        uni = stats.univariate({n: x[:, i] for i, n in enumerate(names)}, y)
        res["univariate_top"] = uni[: args.top]
        if split == "train":
            pos = idx[y == 1]
            res["probe_fraud_label_share"] = float((y_lab[pos] > 0).mean()) if len(pos) else None
        rep["splits"][split] = res
        print(
            f"[ordinary] {split}: {res['positives']} ordinary vs {res['negatives']} legit rows, "
            f"AUC {res['auc']:.4f} (seeds {', '.join(f'{a:.4f}' for a in res['auc_seeds'])})"
        )
        for u in uni[:5]:
            stat = f"auc {u['auc']:.3f}" if u["kind"] == "rank" else f"z {u['z']:+.2f}"
            print(f"    {u['feature']:28s} {stat}  p={u['p']:.2g}")
    return rep


def cmd_links(args, led, grp) -> dict:
    kind = grp["kind"].to_numpy()
    evid = grp["evidence"].to_numpy()
    train = ~led["is_test"].to_numpy()
    y = led["y"].to_numpy()
    pop = train & ~evid & np.isin(kind, (probes.OWN, probes.BORROWED, probes.LEGIT_BLOCK))
    fraud = pop & (y > 0)
    legit = pop & (y == 0)
    ts = led["ts"].to_numpy()
    rep: dict = {
        "population": int(pop.sum()),
        "fraud": int(fraud.sum()),
        "legit": int(legit.sum()),
        "links": [],
        "history": {},
    }
    for key in ("sender", "receiver", "device"):
        k = led[key].to_numpy()
        for w in args.windows:
            hit = stats.link_share(k, ts, fraud, w)
            a, b = float(hit[fraud].mean()), float(hit[legit].mean())
            rep["links"].append(
                {
                    "key": key,
                    "window_s": w,
                    "fraud_rows": a,
                    "legit_rows": b,
                    "ratio": a / b if b > 0 else None,
                }
            )
            print(
                f"[links] {key:8s} +-{w:>7d}s: another ordinary-fraud row on the key for "
                f"{a:.4f} of fraud rows vs {b:.4f} of legit rows (ratio {a / max(b, 1e-12):.3f})"
            )
    # does a key's earlier ordinary fraud predict ordinary fraud in the last train month?
    period = led["period"].to_numpy()
    last = pop & (period == LAST_TRAIN_PERIOD)
    earlier = pop & (period < LAST_TRAIN_PERIOD)
    for key in ("sender", "receiver", "device"):
        k = led[key].to_numpy()
        seen = np.isin(k, k[earlier])
        seen_fraud = np.isin(k, k[earlier & fraud])
        bucket = np.where(seen_fraud, 2, np.where(seen, 1, 0))
        rows = {}
        for b, nm in ((0, "unseen"), (1, "seen_no_fraud"), (2, "seen_with_fraud")):
            m = last & (bucket == b)
            rows[nm] = {
                "rows": int(m.sum()),
                "fraud_rate": float(fraud[m].mean()) if m.any() else None,
            }
        rep["history"][key] = rows
        print(
            f"[links] {key:8s} history -> last month fraud rate: "
            + ", ".join(
                f"{nm} {v['fraud_rate']:.4f} (n={v['rows']})"
                for nm, v in rows.items()
                if v["fraud_rate"] is not None
            )
        )
    return rep


def cmd_shift(args, led, grp) -> dict:
    frame, names, cat = features(
        led, ["context"], raw=True, only=STABLE + ("minute", "second", "cents")
    )
    period = led["period"].to_numpy()
    kind = grp["kind"].to_numpy()
    evid = grp["evidence"].to_numpy()
    pop = ~evid & (kind == probes.KIND_NAMES.index(args.kind))
    ia = np.flatnonzero(pop & (period == args.a))
    ib = np.flatnonzero(pop & (period == args.b))
    res = stats.adversarial(
        rows_of(frame, ia),
        rows_of(frame, ib),
        names=names,
        categorical=cat,
        seeds=args.seeds,
        folds=args.folds,
        max_rows=args.max_rows,
        threads=threads.budget(),
    )
    res.update({"a": args.a, "b": args.b, "kind": args.kind, "features": names})
    print(
        f"[shift] period {args.a} vs {args.b}, {args.kind} phones without evidence, "
        f"{res['rows_per_side']} rows a side: AUC {res['auc']:.4f}; share of {args.b} "
        f"among the most {args.b}-like rows: {res['tail_share_b']}"
    )
    return res


def _reachable_class(led: pl.DataFrame, grp: pl.DataFrame) -> np.ndarray:
    """Class a label-free detector gives a row: the counter segment on fraud-phase phones, else
    the row anchor (0 = unreachable)."""
    from ringtrace.evidence import anchors

    anchor = anchors.row_anchor(led).astype(np.int64)
    seg = grp["segment"].to_numpy().astype(np.int64)
    fraud_phase = grp["kind"].to_numpy() == probes.FRAUD_PHASE
    return np.where(fraud_phase & (seg > 0), seg, anchor)


def cmd_ceiling(args, led, grp) -> dict:
    probe = grp["probe"].to_numpy()
    seg = grp["segment"].to_numpy().astype(np.int64)
    kind = grp["kind"].to_numpy()
    evid = grp["evidence"].to_numpy()
    y = led["y"].to_numpy()
    period = led["period"].to_numpy()
    train = ~led["is_test"].to_numpy()
    reach = _reachable_class(led, grp)
    ordinary_home = ~evid & np.isin(kind, (probes.OWN, probes.BORROWED, probes.LEGIT_BLOCK))
    probe_fraud = float((train & (probe == probes.ORDINARY) & (y > 0)).sum())
    other_fraud = float((train & ordinary_home & (y > 0)).sum())
    rate = ceil.new_device_rate(probe_fraud, other_fraud)
    rep: dict = {"new_device_rate": rate, "classes": list(CLASSES[1:]), "periods": {}}
    print(f"[ceiling] share of ordinary fraud rows on a new phone (train labels): {rate:.4f}")
    for p in sorted(np.unique(period)):
        m = period == p
        n = int(m.sum())
        probe_k = np.bincount(seg[m & (probe == probes.ORDINARY)], minlength=N_CLASSES)[1:]
        hidden = ceil.hidden_rows(probe_k, rate)
        reach_k = np.bincount(reach[m], minlength=N_CLASSES)[1:].astype(np.float64)
        c = ceil.ceiling(reach_k, hidden, n)
        row = {
            "rows": n,
            "probe_rows": probe_k.tolist(),
            "hidden": hidden.tolist(),
            "reachable_anchor": reach_k.tolist(),
            "ceiling_macro": c["macro"],
            "ceiling_f1": c["f1"].tolist(),
            "hidden_share": c["hidden_share"],
        }
        if p <= LAST_TRAIN_PERIOD:
            # check against labels: fraud-labelled rows without evidence on home/legit-block phones
            lab = np.bincount(y[m & ordinary_home & (y > 0)], minlength=N_CLASSES)[1:]
            row["labelled_ordinary_home"] = lab.tolist()
            row["hidden_over_labelled"] = float(hidden.sum() / max(lab.sum(), 1))
        rep["periods"][int(p)] = row
        print(
            f"[ceiling] period {p}: {n} rows, hidden ordinary {hidden.sum():.0f} "
            f"({c['hidden_share']:.3f} of fraud), ceiling macro {c['macro']:.4f}"
            + (
                f", hidden/labelled {row['hidden_over_labelled']:.3f}"
                if "hidden_over_labelled" in row
                else ""
            )
        )
    if args.submission:
        sub = pl.read_csv(args.submission, infer_schema=False)
        code = {c: i for i, c in enumerate(CLASSES)}
        pred = np.bincount(
            sub[sub.columns[1]].replace_strict(code, return_dtype=pl.Int64).to_numpy(),
            minlength=N_CLASSES,
        )[1:].astype(np.float64)
        test_p = int(period[~train].max())
        hidden = np.asarray(rep["periods"][test_p]["hidden"])
        n = rep["periods"][test_p]["rows"]
        tp = args.precision * pred
        true = tp + hidden
        now = ceil.expected_scores(pred, tp, true, n)
        best = ceil.ceiling(tp, hidden, n)
        sub_rep = {
            "submission": str(args.submission),
            "precision": args.precision,
            "calls": pred.tolist(),
            "expected_macro": now["macro"],
            "ceiling_macro_with_these_reachable": best["macro"],
        }
        print(
            f"[ceiling] submission: {pred.sum():.0f} fraud calls, "
            f"expected macro {now['macro']:.4f} at precision {args.precision}, "
            f"ceiling {best['macro']:.4f}"
        )
        if args.target:
            need = ceil.extra_rows_for(args.target, pred, tp, true, hidden, n)
            sub_rep["target"] = {"macro": args.target, **need}
            if need["feasible"]:
                print(
                    f"[ceiling] reaching {args.target} needs {need['extra_rows']:.0f} extra "
                    f"correct rows = {need['share_of_hidden']:.3f} of the hidden ordinary rows"
                )
            else:
                print(
                    f"[ceiling] {args.target} is out of reach even with every hidden row found "
                    f"(max {need['max_macro']:.4f})"
                )
        if args.front_shares:
            n_public = round(args.public_share * n)
            rows = [ceil.split_scores(pred, tp, true, n, n_public, f) for f in args.front_shares]
            sub_rep["position_split"] = {"public_rows": n_public, "rows": rows}
            for r in rows:
                print(
                    f"[ceiling] if public = first {n_public} rows of the file and "
                    f"{r['front_share']:.0%} of its fraud calls sit there: "
                    f"public {r['public']:.4f}, private {r['private']:.4f}"
                )
        rep["submission"] = sub_rep
    return rep


def cmd_lookalike(args, led, grp) -> dict:
    from ringtrace.evidence import anchors

    frame, names, _cat = features(led, args.families, raw=not args.no_raw)
    tmpl, event = anchors.templates(led)
    train = ~led["is_test"].to_numpy()
    y = led["y"].to_numpy()
    ts = led["ts"].to_numpy()
    fraud_phase = (grp["kind"].to_numpy() == probes.FRAUD_PHASE).astype(np.float64)
    rep: dict = {"templates": {}}
    for t in args.templates:
        idx = np.flatnonzero(train & (tmpl == t))
        pair = anchors.TEMPLATE_PAIR[t]
        res = lookalike.audit_template(
            event[idx],
            rows_of(frame, idx),
            ts[idx],
            names,
            y[idx],
            pair,
            fraud_phase[idx],
            seeds=args.seeds,
            folds=args.folds,
            threads=threads.budget(),
        )
        res["pair"] = [CLASSES[pair[0]], CLASSES[pair[1]]]
        res["rule_class"] = CLASSES[anchors.TEMPLATE_CLASS[t]]
        rep["templates"][anchors.TEMPLATE_NAMES[t]] = res
        a = res.get("all", {}).get("auc", float("nan"))
        o = res.get("off_fraud_phase", {}).get("auc", float("nan"))
        print(
            f"[lookalike] {anchors.TEMPLATE_NAMES[t]:8s} {res['pair'][0]} vs {res['pair'][1]}: "
            f"{res['events']} events, second-type share {res['second_type_share']:.3f}, "
            f"AUC all {a:.4f}, off fraud phase {o:.4f}"
        )
    return rep


def cmd_shuffle(args, led, grp) -> dict:
    src = Path(args.submission)
    dst = (
        Path(args.out)
        if args.out
        else paths.out_dir() / "audit" / f"{src.stem}-order-probe-{args.seed}.csv"
    )
    rep = shuffle.shuffled(src, dst, args.seed)
    print(
        f"[shuffle] {rep['rows']} rows, {rep['rows_moved']} moved, "
        f"same content {rep['same_content']}\n"
        f"          source sha256 {rep['sha256_source']}\n"
        f"          probe  sha256 {rep['sha256_probe']}"
    )
    return rep


COMMANDS = {
    "ordinary": cmd_ordinary,
    "links": cmd_links,
    "shift": cmd_shift,
    "ceiling": cmd_ceiling,
    "lookalike": cmd_lookalike,
    "shuffle": cmd_shuffle,
}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m ringtrace.audit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, families=True):
        p.add_argument("--seeds", type=_ints, default=(0, 1, 2))
        p.add_argument("--folds", type=int, default=5)
        if families:
            p.add_argument(
                "--families", type=lambda s: [f for f in s.split(",") if f], default=["context"]
            )
            p.add_argument("--no-raw", action="store_true", help="leave out the raw row columns")

    p = sub.add_parser("ordinary")
    common(p)
    p.add_argument("--top", type=int, default=15)
    p = sub.add_parser("links")
    p.add_argument("--windows", type=_ints, default=(3600, 86400, 7 * 86400))
    p = sub.add_parser("shift")
    p.add_argument("--a", type=int, default=LAST_TRAIN_PERIOD)
    p.add_argument("--b", type=int, default=10)
    p.add_argument("--kind", default="own", choices=probes.KIND_NAMES)
    p.add_argument("--seeds", type=_ints, default=(0,))
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=200_000)
    p = sub.add_parser("ceiling")
    p.add_argument("--submission", default=None)
    p.add_argument(
        "--precision",
        type=float,
        default=0.95,
        help="share of the submission's fraud calls taken to be right",
    )
    p.add_argument("--target", type=float, default=None)
    p.add_argument(
        "--front-shares",
        type=lambda s: [float(v) for v in s.split(",") if v != ""],
        default=None,
        help="model a position-based public part: shares of the fraud calls in its rows",
    )
    p.add_argument("--public-share", type=float, default=0.3)
    p = sub.add_parser("lookalike")
    common(p)
    p.add_argument("--templates", type=_ints, default=(1, 2, 3, 4, 5, 6))
    p = sub.add_parser("shuffle")
    p.add_argument("submission")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return ap


def main(argv=None) -> dict:
    args = parser().parse_args(argv)
    threads.pin()
    t0 = time.time()
    if args.cmd == "shuffle":
        rep = cmd_shuffle(args, None, None)
    else:
        led = store.ledger()
        grp = probes.groups(led)
        rep = COMMANDS[args.cmd](args, led, grp)
    rep["wall_s"] = round(time.time() - t0, 1)
    target = _write(args.cmd, rep)
    print(f"[audit] wrote {target} in {rep['wall_s']} s")
    return rep


if __name__ == "__main__":
    main()
