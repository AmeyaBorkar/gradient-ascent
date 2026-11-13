"""A wide look for an untried lever: where the headroom sits and what could reach it.

    python analysis/farlook.py pairs      # split curves and separability inside the six templates
    python analysis/farlook.py loyalty    # do shared accounts keep one class of a look-alike pair
    python analysis/farlook.py generator  # id structure, counter order in time, dead columns
    python analysis/farlook.py cells      # impossible-fact and odd-phone cells among legit calls
    python analysis/farlook.py thin       # the counter's segments at October's size
    python analysis/farlook.py all

Every command reads the saved blend (``runs/blend``), decodes it as the submission is decoded and
writes ``$RINGTRACE_OUT/farlook/<command>.json``. Train rows are scored against the cleaned
labels; October is only ever counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from farlook_core import (  # noqa: E402
    gap_stats,
    keep_units,
    oracle_typing_gain,
    pair_concordance,
    rank_correlation,
    shuffled_concordance,
    split_curve,
)

from ringtrace import decode, paths, rules, silver, store  # noqa: E402
from ringtrace.classes import CLASSES, N_CLASSES  # noqa: E402
from ringtrace.evidence import anchors, phones  # noqa: E402
from ringtrace.evidence.anchors import TEMPLATE_CLASS, TEMPLATE_NAMES, TEMPLATE_PAIR  # noqa: E402

MONTHS = (7, 8, 9)
PAIRS = ((4, 5), (9, 6), (3, 1), (1, 2), (7, 8))  # look-alike class pairs
OCTOBER_SHARE = 1 / 6  # October's fraud block is one sixth of train's


def load(run: str = "blend") -> dict:
    led = store.ledger()
    is_test = led["is_test"].to_numpy().astype(bool)
    period = led["period"].to_numpy()
    out = paths.out_dir() / "runs" / run
    proba = np.zeros((led.height, N_CLASSES))
    proba[~is_test] = np.load(out / "oof.npy")
    proba[is_test] = np.load(out / "test.npy")
    calls = decode.decode_by_period(proba, period, fraud_scale=0.875)
    calls = rules.apply(["anchored_templates"], calls, proba, led, log=lambda *a: None)
    tmpl, event = anchors.templates(led)
    return {
        "led": led,
        "is_test": is_test,
        "period": period,
        "proba": proba,
        "calls": calls,
        "tmpl": tmpl,
        "event": event,
        "cleaned": silver.load(led, "label"),
        "gen": silver.load(led, "structured"),
    }


def _totals(month: np.ndarray, calls: np.ndarray, y: np.ndarray, c: int) -> dict:
    return {
        "tp": float((month & (calls == c) & (y == c)).sum()),
        "called": float((month & (calls == c)).sum()),
        "true": float((month & (y == c)).sum()),
    }


def _partner(t: int) -> tuple[int, int]:
    a = TEMPLATE_CLASS[t]
    return a, next(c for c in TEMPLATE_PAIR[t] if c != a)


# ---------------------------------------------------------------------------------------------


def cmd_pairs(d: dict) -> dict:
    """Is the template's class the best split of its pool, what would perfect typing add, and
    can anything label-free tell the partner class apart?"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    led, calls, y = d["led"], d["calls"], d["cleaned"]
    train = ~d["is_test"]
    report: dict = {}
    for t in TEMPLATE_PAIR:
        a, b = _partner(t)
        months = {}
        for m in MONTHS:
            month = train & (d["period"] == m)
            pool = month & (d["tmpl"] == t) & (calls == a)
            sizes = {
                "n": float(pool.sum()),
                "a": float((pool & (y == a)).sum()),
                "b": float((pool & (y == b)).sum()),
            }
            ta, tb = _totals(month, calls, y, a), _totals(month, calls, y, b)
            qs, delta = split_curve(ta, tb, sizes)
            months[m] = {
                "pool": int(sizes["n"]),
                "share_a": sizes["a"] / sizes["n"],
                "share_b": sizes["b"] / sizes["n"],
                "slope_at_0": float(delta[1] / qs[1]),
                "best_q": float(qs[delta.argmax()]),
                "gain_at_best": float(delta.max()),
                "gain_full_flip": float(delta[-1]),
                "oracle_typing": oracle_typing_gain(ta, tb, sizes),
            }
        rows = np.flatnonzero(train & (d["tmpl"] == t) & (calls == a) & np.isin(y, (a, b)))
        target = (y[rows] == b).astype(int)
        ratio = d["proba"][rows, b] / np.maximum(d["proba"][rows, a] + d["proba"][rows, b], 1e-12)
        feats = _pool_features(led, rows, d["event"][rows])
        oof = np.zeros(len(rows))
        for fit, val in GroupKFold(5).split(feats, target, d["event"][rows]):
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=8,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                random_state=0,
                n_jobs=4,
                verbose=-1,
            )
            model.fit(feats[fit], target[fit])
            oof[val] = model.predict_proba(feats[val])[:, 1]
        top = max(len(rows) // 10, 1)
        report[TEMPLATE_NAMES[t]] = {
            "template_class": CLASSES[a],
            "partner": CLASSES[b],
            "months": months,
            "rows": int(len(rows)),
            "events": int(len(np.unique(d["event"][rows]))),
            "partner_share": float(target.mean()),
            "blend_ratio_auc": float(roc_auc_score(target, ratio)),
            "blend_ratio_top_decile_share": float(target[np.argsort(-ratio)[:top]].mean()),
            "row_event_gbdt_auc": float(roc_auc_score(target, oof)),
            "row_event_gbdt_top_decile_share": float(target[np.argsort(-oof)[:top]].mean()),
        }
    return report


def _pool_features(led: pl.DataFrame, rows: np.ndarray, event: np.ndarray) -> np.ndarray:
    sub = led[rows]
    ts = sub["timestamp"].dt.epoch("s").to_numpy()
    code = lambda c: sub[c].cast(pl.Utf8).cast(pl.Categorical).to_physical()  # noqa: E731
    base = pl.DataFrame(
        {
            "amount": sub["amount"],
            "hour": sub["timestamp"].dt.hour(),
            "minute": sub["timestamp"].dt.minute(),
            "weekday": sub["timestamp"].dt.weekday(),
            "day": sub["timestamp"].dt.day(),
            "sim_age": sub["sim_age_hours"],
            "device_age": sub["device_age_days"],
            "account_age": sub["sender_account_age_days"],
            "new_payee": sub["new_payee"],
            "pin_retries": sub["pin_retries"],
            "login": sub["login_to_txn_sec"],
            "balance": sub["balance_before"],
            "ratio": sub["amount"] / sub["balance_before"],
            "to_person": sub["to_person"].cast(pl.Int8),
            "away": (sub["device_city"] != sub["s_home_city"]).cast(pl.Int8),
            "kyc": code("s_kyc_level"),
            "bank": code("s_bank_code"),
            "os": code("s_os"),
            "app": code("app"),
            "type": code("txn_type"),
            "receiver_kyc": code("r_kyc"),
            "sender": sub["sender"],
            "receiver": sub["receiver"],
        }
    ).to_numpy()
    _, g = np.unique(event, return_inverse=True)
    first = np.full(g.max() + 1, np.inf)
    last = np.full(g.max() + 1, -np.inf)
    np.minimum.at(first, g, ts)
    np.maximum.at(last, g, ts)
    extra = np.column_stack(
        [
            np.bincount(g)[g],
            (last - first)[g],
            ts - first[g],
            np.bincount(g, weights=sub["amount"].to_numpy())[g],
        ]
    )
    return np.column_stack([base.astype(np.float64), extra])


def cmd_loyalty(d: dict) -> dict:
    """Events of a look-alike pair that share an account: do they carry the same class more often
    than shuffled labels would? Then: do bot-pool accounts mark bot-written mule-style bursts?"""
    led, y = d["led"], d["cleaned"]
    train = ~d["is_test"]
    event = rules.label_free_events(led)
    rows = np.flatnonzero(train & (event >= 0) & (y > 0))
    frame = pl.DataFrame({"ev": event[rows], "y": y[rows]})
    major = frame.group_by("ev").agg(pl.col("y").mode().first().alias("label"))
    label_of = dict(zip(major["ev"].to_list(), major["label"].to_list(), strict=True))
    ev_label = np.array([label_of[e] for e in event[rows]])
    report: dict = {"pairs": {}}
    for pair in PAIRS:
        inside = np.isin(ev_label, pair)
        for side in ("sender", "receiver"):
            links = pl.DataFrame(
                {
                    "entity": led[side].to_numpy()[rows][inside],
                    "ev": event[rows][inside],
                    "label": ev_label[inside],
                }
            ).unique()
            links = links.filter(pl.col("ev").n_unique().over("entity") >= 2)
            if links.height == 0:
                continue
            ent, ev, lab = (links[c].to_numpy() for c in ("entity", "ev", "label"))
            same, total = pair_concordance(ent, lab, pair)
            mean, sd = shuffled_concordance(ent, ev, lab, pair, draws=200)
            report["pairs"][f"{CLASSES[pair[0]]}/{CLASSES[pair[1]]} by {side}"] = {
                "event_pairs": total,
                "same_label": same,
                "shuffled_mean": mean,
                "shuffled_sd": sd,
            }
    block = anchors.family_columns(led, ["ctr_block"])["ctr_block"].fill_null(0).to_numpy()
    ring = np.unique(led["sender"].to_numpy()[block == 1])
    in_ring = np.isin(led["sender"].to_numpy(), ring)
    pool = d["tmpl"] == 3
    per_event = (
        pl.DataFrame(
            {
                "ev": d["event"][pool],
                "ring": in_ring[pool],
                "test": d["is_test"][pool],
                "y": y[pool],
            }
        )
        .group_by("ev")
        .agg(
            pl.col("test").first(),
            pl.col("ring").sum(),
            pl.col("y").filter(pl.col("y") > 0).mode().first().alias("label"),
        )
    )
    tr = per_event.filter(~pl.col("test"))
    report["bot_ring_in_mule_bursts"] = {
        "ring_accounts": int(len(ring)),
        "train_events_bot": tr.filter(pl.col("label") == 2).height,
        "train_events_bot_with_ring_sender": tr.filter(
            (pl.col("label") == 2) & (pl.col("ring") > 0)
        ).height,
        "train_events_mule": tr.filter(pl.col("label") == 1).height,
        "train_events_mule_with_ring_sender": tr.filter(
            (pl.col("label") == 1) & (pl.col("ring") > 0)
        ).height,
        "october_events": per_event.filter(pl.col("test")).height,
        "october_events_with_2_ring_senders": per_event.filter(
            pl.col("test") & (pl.col("ring") >= 2)
        ).height,
    }
    return report


def cmd_generator(d: dict) -> dict:
    """Three places a generator could leak its pass or its class turn."""
    led = d["led"]
    ids = led["txn_id"].str.slice(2).str.to_integer(base=16).to_numpy()
    report: dict = {"txn_id": gap_stats(ids)}
    ctr = anchors.family_columns(led, ["ctr_block", "ctr_seg"])
    block = ctr["ctr_block"].fill_null(0).to_numpy()
    seg = ctr["ctr_seg"].cast(pl.Float64).fill_null(-1).to_numpy().astype(np.int64)
    device = led["device"].to_numpy()
    ts = led["timestamp"].dt.epoch("s").to_numpy()
    train = ~d["is_test"]
    order = {}
    for k in range(phones.K):
        rows = np.flatnonzero(train & (block == 2) & (seg == k))
        dev, first = np.unique(device[rows], return_index=True)
        order[CLASSES[k + 1]] = {
            "phones": int(len(dev)),
            "rank_corr_number_time": rank_correlation(dev.astype(np.float64), ts[rows][first]),
        }
    report["counter_order_in_time"] = order
    dead = {}
    for name, split in (("train", train), ("october", d["is_test"])):
        for bname, code in (("home", 0), ("bot_pool", 1), ("fraud_block", 2), ("legit_block", 3)):
            rows = split & (block == code)
            ip = led["ip_risk_score"].to_numpy()[rows]
            app = led["app_version"].to_numpy()[rows]
            dead[f"{name} {bname}"] = {
                "rows": int(rows.sum()),
                "ip_mean": float(np.nanmean(ip)),
                "ip_above_0.3": float(np.nanmean(ip > 0.3)),
                "ip_above_0.5": float(np.nanmean(ip > 0.5)),
                "app_mean": float(np.nanmean(app)),
            }
    report["dead_columns_by_block"] = dead
    return report


def cmd_cells(d: dict) -> dict:
    """Facts that cannot happen in ordinary traffic, and phones that are neither the sender's
    own nor single-use: how often is a legit-called row in the cell fraud?"""
    led, calls = d["led"], d["calls"]
    train, test = ~d["is_test"], d["is_test"]
    day = led["timestamp"].dt.date()
    age = (day - led["s_opened_on"]).dt.total_days().to_numpy().astype(np.float64)
    block = anchors.family_columns(led, ["ctr_block"])["ctr_block"].fill_null(0).to_numpy()
    own = led["device_id"].str.slice(2) == led["sender_id"].str.slice(2).str.zfill(7)
    own = own.to_numpy()
    cells = {
        "merchant not yet onboarded": (day < led["m_onboarded_on"]).fill_null(False).to_numpy(),
        "receiver account not yet open": (day < led["r_opened_on"]).fill_null(False).to_numpy(),
        "sender account not yet open": (day < led["s_opened_on"]).fill_null(False).to_numpy(),
        "phone older than the account": led["device_age_days"].to_numpy() > age + 1,
        "sender pays itself": (led["sender_id"] == led["receiver_id"]).to_numpy(),
        "another account's home phone": (block == 0) & ~own,
        "own phone away from home": own
        & (led["device_city"] != led["s_home_city"]).fill_null(False).to_numpy(),
        "login within 1 s": led["login_to_txn_sec"].to_numpy() <= 1,
        "sim younger than 1 h": led["sim_age_hours"].to_numpy() < 1,
        "phone first used today": led["device_age_days"].to_numpy() == 0,
    }
    base = train & (calls == 0)
    report = {
        "legit calls, all train": {
            "rows": int(base.sum()),
            "cleaned_fraud": float((d["cleaned"][base] > 0).mean()),
            "generator_made": float((d["gen"][base] > 0).mean()),
        }
    }
    for name, cell in cells.items():
        cell = np.asarray(cell, dtype=bool)
        legit = cell & base
        report[name] = {
            "train_rows": int((cell & train).sum()),
            "train_called_legit": int(legit.sum()),
            "cleaned_fraud_among_legit_calls": float((d["cleaned"][legit] > 0).mean())
            if legit.any()
            else None,
            "generator_made_among_legit_calls": float((d["gen"][legit] > 0).mean())
            if legit.any()
            else None,
            "october_rows": int((cell & test).sum()),
            "october_called_legit": int((cell & test & (calls == 0)).sum()),
        }
    return report


def cmd_thin(d: dict, seeds: int = 30) -> dict:
    """The counter's segments are fitted per split from anchors. October's fraud block has one
    sixth of train's phones: thin train's block to that size, refit the segments without labels
    and compare with the full block's segments on the same unanchored fraud rows."""
    led, y = d["led"], d["cleaned"]
    device = led["device"].to_numpy().astype(np.int64)
    split = d["is_test"].astype(np.int64)
    anchor = anchors.row_anchor(led).astype(np.int64)
    blocks = phones.find_blocks(device, split)
    block = phones.block_codes(device, split, blocks)
    lo, edge, _hi = blocks.ranges[0]
    rows = np.flatnonzero((block == 2) & (split == 0))
    nums_all, inv_all = np.unique(device[rows], return_inverse=True)

    def segments(keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = rows[keep[inv_all]]
        nums, inv = np.unique(device[r], return_inverse=True)
        counts = np.zeros((len(nums), phones.K + 1))
        np.add.at(counts, (inv, anchor[r]), 1.0)
        used = np.zeros(edge - lo + 2, dtype=bool)
        used[nums - lo] = True
        own = phones.Blocks(f0=blocks.f0, used=used, base=lo)
        phase = phones.decode_phase(nums, counts, own, lo, edge)
        return r, phase.seg[inv] + 1, phase.conf[inv]

    r, seg, _ = segments(np.ones(len(nums_all), dtype=bool))
    full = np.zeros(led.height, dtype=np.int64)
    full[r] = seg
    open_rows = (anchor[r] == 0) & (y[r] > 0)
    tail = open_rows & (seg >= 8)
    report = {
        "full_block": {
            "rows": int(len(r)),
            "unanchored_fraud_rows": int(open_rows.sum()),
            "agreement_unanchored": float((seg[open_rows] == y[r][open_rows]).mean()),
            "agreement_unanchored_last_four_segments": float((seg[tail] == y[r][tail]).mean()),
        }
    }
    per_seed = []
    for seed in range(seeds):
        r, seg, conf = segments(keep_units(len(nums_all), OCTOBER_SHARE, seed))
        open_rows = (anchor[r] == 0) & (y[r] > 0)
        tail = open_rows & (full[r] >= 8)
        right_thin = seg == y[r]
        right_full = full[r] == y[r]
        per_seed.append(
            [
                open_rows.sum(),
                right_thin[open_rows].mean(),
                right_full[open_rows].mean(),
                (open_rows & right_full & ~right_thin).sum(),
                (open_rows & ~right_full & right_thin).sum(),
                tail.sum(),
                right_thin[tail].mean(),
                right_full[tail].mean(),
                (open_rows & (conf < 0.9)).sum(),
            ]
        )
    names = [
        "unanchored_fraud_rows",
        "agreement_thinned_fit",
        "agreement_full_fit_same_rows",
        "rows_lost_by_thinning",
        "rows_gained_by_thinning",
        "last_four_rows",
        "last_four_agreement_thinned_fit",
        "last_four_agreement_full_fit",
        "unanchored_rows_conf_below_0.9",
    ]
    a = np.array(per_seed, dtype=np.float64)
    report["thinned_to_october_size"] = {
        n: {"mean": float(a[:, i].mean()), "sd": float(a[:, i].std())} for i, n in enumerate(names)
    }
    # October itself, counted only: how many rows lean on the segment
    ctr = anchors.family_columns(led, ["ctr_block", "ctr_seg", "ctr_conf"])
    oct_rows = d["is_test"] & (ctr["ctr_block"].fill_null(0).to_numpy() == 2)
    open_oct = oct_rows & (anchor == 0)
    conf = ctr["ctr_conf"].cast(pl.Float64).fill_null(1.0).to_numpy()
    seg_oct = ctr["ctr_seg"].cast(pl.Float64).fill_null(-1).to_numpy()
    report["october"] = {
        "fraud_block_rows": int(oct_rows.sum()),
        "unanchored_rows": int(open_oct.sum()),
        "unanchored_conf_below_0.9": int((open_oct & (conf < 0.9)).sum()),
        "last_three_segments_rows": int((oct_rows & (seg_oct >= 8)).sum()),
        "last_three_segments_conf_below_0.9": int((oct_rows & (seg_oct >= 8) & (conf < 0.9)).sum()),
        "calls_equal_segment": int((oct_rows & (d["calls"] == seg_oct + 1)).sum()),
    }
    return report


COMMANDS = {
    "pairs": cmd_pairs,
    "loyalty": cmd_loyalty,
    "generator": cmd_generator,
    "cells": cmd_cells,
    "thin": cmd_thin,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--run", default="blend")
    args = parser.parse_args()
    d = load(args.run)
    target = paths.out_dir() / "farlook"
    target.mkdir(parents=True, exist_ok=True)
    for name in COMMANDS if args.command == "all" else [args.command]:
        report = COMMANDS[name](d)
        (target / f"{name}.json").write_text(json.dumps(report, indent=1))
        print(f"== {name} ==")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
