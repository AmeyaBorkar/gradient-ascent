"""velocity_bot and layering_chain in October: missed events, or a real change of mix?

    python analysis/velocity.py            # every section, report to $RINGTRACE_OUT/velocity/

Sections (October rows are only ever counted):

* ``bursts``: bot-pool bursts per month (size, span, accounts, category, hour, amount), whether
  October's bursts are whole or thinned (size test against train bursts thinned to 50 %), and
  the Poisson chance of October's burst count at the train rate per row.
* ``offpool``: sessions off the bot pool (per device and per sender; gaps 5, 15 and 60 minutes;
  four row filters), at full density and with train thinned to 50 % (seeds 0 to 2), plus the
  devices shared by several accounts.
* ``zones``: rows per 100k on single-use fraud-pass devices with no template, by segment class:
  a label-free count of each class's ordinary rows.
* ``layering``: the layering calls split by money-flow shape and hop delay.
* ``estimate``: the expected October F1 per class (``velocity_core.rescaled_f1``), with a
  backtest that estimates each train month from the other two.
* ``probe``: velocity_bot called on every row with P(bot) above a cut, on the train months.

Inputs: the ledger, the cached arrays of the best file (``$RINGTRACE_OUT/closelook/prep.npz``:
blended probabilities, out-of-fold and October calls, targets, counter segments, templates and
anchors) and the money-flow family.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from velocity_core import per_100k, rescaled_f1, sessions, size_test, thin_keep  # noqa: E402

from ringtrace import paths, store, tables  # noqa: E402
from ringtrace.classes import CLASSES, N_CLASSES  # noqa: E402

MONTHS = (7, 8, 9)
ALL = (7, 8, 9, 10)
BOT, LAYER, MULE = 2, 3, 1
POOL = (420_000, 422_999)
THIN_SEEDS = (0, 1, 2)
COLS = [
    "row",
    "ts",
    "device",
    "sender",
    "receiver",
    "amount",
    "txn_type",
    "merchant_category",
    "is_test",
    "month",
    "to_person",
    "login_to_txn_sec",
]


def load() -> dict:
    names = ("train", "test", "accounts", "merchants")
    key = "-".join(tables.file_digest(paths.data_dir() / f"{n}.csv", 8) for n in names)
    led = pl.read_parquet(paths.cache_dir() / "tables" / f"ledger-{key}.parquet", columns=COLS)
    z = np.load(paths.out_dir() / "closelook" / "prep.npz")
    is_test = z["is_test"]
    n = led.height
    if len(is_test) != n:
        raise ValueError(f"prep.npz has {len(is_test)} rows, the ledger {n}")
    d = {c: led[c].to_numpy() for c in COLS if c not in ("merchant_category",)}
    d["cat"] = led["merchant_category"].fill_null("").to_numpy()
    d["is_test"] = is_test
    d["m"] = np.where(is_test, 10, d["month"]).astype(np.int64)
    for k in ("calls", "proba", "tmpl", "anchor", "ctr_seg", "period"):
        d[k] = z[k]
    for t in ("given", "cleaned", "gen"):
        full = np.full(n, -1, dtype=np.int64)
        full[~is_test] = z[t]
        d[t] = full
    d["pool"] = (d["device"] >= POOL[0]) & (d["device"] <= POOL[1])
    d["zord"] = (d["ctr_seg"] > 0) & (d["tmpl"] == 0) & (d["anchor"] == 0)
    d["n"] = n
    return d


def _share(x: np.ndarray, value: int) -> float | None:
    return float((x == value).mean()) if len(x) else None


def sec_bursts(d: dict) -> dict:
    sid, size = sessions(d["device"], d["is_test"], d["ts"], d["row"], 300)
    rep = {"months": {}}
    sizes = {}
    for m in ALL:
        s = d["pool"] & (d["m"] == m)
        f = pl.DataFrame({"sid": sid[s], "ts": d["ts"][s], "snd": d["sender"][s]})
        g = f.group_by("sid").agg(
            pl.len().alias("n"),
            (pl.col("ts").max() - pl.col("ts").min()).alias("span"),
            pl.col("snd").n_unique().alias("acc"),
        )
        sizes[m] = g["n"].to_numpy()
        cats, cn = np.unique(d["cat"][s], return_counts=True)
        hour = (d["ts"][s] % 86400) // 3600
        r = {
            "bursts": g.height,
            "bursts_per_100k": g.height / (d["m"] == m).sum() * 1e5,
            "rows_per_100k": per_100k(s, d["m"], [m])[0],
            "rows_q": np.quantile(g["n"].to_numpy(), [0, 0.5, 1]).tolist(),
            "span_s_q": np.quantile(g["span"].to_numpy(), [0, 0.5, 1]).tolist(),
            "accounts_q": np.quantile(g["acc"].to_numpy(), [0, 0.5, 1]).tolist(),
            "categories": {str(c): float(k / s.sum()) for c, k in zip(cats, cn, strict=True)},
            "hour3_share": (np.bincount(hour // 3, minlength=8) / s.sum()).round(3).tolist(),
            "amount_q": np.quantile(d["amount"][s], [0.05, 0.5, 0.95]).tolist(),
            "called_bot": _share(d["calls"][s], BOT),
            "rows_in_session_lt10": int((size[s] < 10).sum()),
        }
        if m < 10:
            r["given_bot"] = _share(d["given"][s], BOT)
            r["gen_bot"] = _share(d["gen"][s], BOT)
        rep["months"][m] = r
    tr_sizes = np.concatenate([sizes[m] for m in MONTHS])
    rep["size_test"] = size_test(sizes[10], tr_sizes, keep=0.5)
    ntr = sum((d["m"] == m).sum() for m in MONTHS)
    expected = len(tr_sizes) / ntr * (d["m"] == 10).sum()
    rep["october_expected_bursts"] = float(expected)
    rep["october_poisson_p_le"] = float(poisson.cdf(len(sizes[10]), expected))
    return rep


def _scan(d: dict, keep: np.ndarray, key: np.ndarray, gap: int, filt: np.ndarray) -> dict:
    idx = np.flatnonzero(keep & filt & ~d["pool"])
    _, size = sessions(key[idx], d["is_test"][idx], d["ts"][idx], d["row"][idx], gap)
    out = {}
    for lo in (3, 4, 6):
        for m in ALL:
            ii = idx[(d["m"][idx] == m) & (size >= lo)]
            nrows = (keep & (d["m"] == m)).sum()
            r = {
                "per_100k": float(len(ii) / nrows * 1e5),
                "called_bot": _share(d["calls"][ii], BOT),
            }
            r["called_fraud"] = float((d["calls"][ii] > 0).mean()) if len(ii) else None
            if m < 10:
                r["given_bot"] = _share(d["given"][ii], BOT)
                r["given_fraud"] = float((d["given"][ii] > 0).mean()) if len(ii) else None
            out[f"k{lo}_m{m}"] = r
    return out


def sec_offpool(d: dict) -> dict:
    n = d["n"]
    merch = ~d["to_person"]
    pay = d["txn_type"] == "PAY"
    amt = d["amount"]
    small = (amt >= 10) & (amt <= 499) & (amt == np.round(amt))
    filters = {
        "small_integer_merchant_pay": pay & merch & small,
        "merchant_pay_any_amount": pay & merch,
        "festive_category_any_type": merch
        & np.isin(d["cat"], ["ecommerce", "gaming", "travel", "electronics"]),
        "any_row": np.ones(n, dtype=bool),
    }
    rep = {}
    for fname, filt in filters.items():
        for kname, key in (("device", d["device"]), ("sender", d["sender"])):
            for gap in (300, 900, 3600):
                full = _scan(d, np.ones(n, dtype=bool), key, gap, filt)
                thin = [
                    _scan(d, thin_keep(n, d["is_test"], 0.5, s), key, gap, filt) for s in THIN_SEEDS
                ]
                for k in full:
                    full[k]["thin50_per_100k"] = float(np.mean([t[k]["per_100k"] for t in thin]))
                rep[f"{fname}|{kname}|{gap}"] = full
    f = pl.DataFrame({"dev": d["device"], "snd": d["sender"], "m": d["m"], "pool": d["pool"]})
    g = (
        f.filter(~pl.col("pool"))
        .group_by("dev", "m")
        .agg(pl.col("snd").n_unique().alias("acc"), pl.len().alias("rows"))
    )
    shared = {}
    for lo in (3, 4, 6):
        for m in ALL:
            k = g.filter((pl.col("m") == m) & (pl.col("acc") >= lo))
            shared[f"acc{lo}_m{m}"] = {
                "devices": k.height,
                "rows_per_100k": float(k["rows"].sum() / (d["m"] == m).sum() * 1e5),
            }
    rep["shared_devices"] = shared
    return rep


def sec_zones(d: dict) -> dict:
    rep = {}
    tr = d["m"] < 10
    for c in range(1, N_CLASSES):
        s = (d["ctr_seg"] == c) & d["zord"]
        rates = per_100k(s, d["m"], ALL)
        te = s & (d["m"] == 10)
        rep[CLASSES[c]] = {
            "per_100k": rates,
            "ratio": rates[3] / max(np.mean(rates[:3]), 1e-12),
            "october_called_same": _share(d["calls"][te], c),
            "train_given_precision": _share(d["given"][s & tr], c),
        }
    return rep


def sec_layering(d: dict) -> dict:
    fl = store.family("flows", verbose=False).select(
        "mf_shape", "mf_ch_delay_in", "mf_ch_delay_out"
    )
    shape = fl["mf_shape"].fill_null(0).to_numpy()
    din = np.nan_to_num(fl["mf_ch_delay_in"].to_numpy().astype(float), nan=-1)
    dout = np.nan_to_num(fl["mf_ch_delay_out"].to_numpy().astype(float), nan=-1)
    dmax = np.fmax(din, dout)
    groups = {
        "chain_fast_le25min": (shape == 1) & (dmax <= 1500),
        "chain_25_60min": (shape == 1) & (dmax > 1500) & (dmax <= 3600),
        "chain_1_6h": (shape == 1) & (dmax > 3600),
        "mule_burst_inflow": shape == 2,
        "mule_burst_onward": shape == 3,
        "fan_in_3_7": shape == 4,
        "zone_layering_no_shape": (shape == 0) & (d["ctr_seg"] == LAYER),
    }
    rep = {}
    tr = d["m"] < 10
    for g, s in groups.items():
        te = s & (d["m"] == 10)
        rep[g] = {
            "rows_per_100k": per_100k(s, d["m"], ALL),
            "layering_calls_per_100k": per_100k(s & (d["calls"] == LAYER), d["m"], ALL),
            "train_called_layering": _share(d["calls"][s & tr], LAYER),
            "train_cleaned_layering": _share(d["cleaned"][s & tr], LAYER),
            "train_gen_layering": _share(d["gen"][s & tr], LAYER),
            "train_gen_mule": _share(d["gen"][s & tr], MULE),
            "october_called_layering": _share(d["calls"][te], LAYER),
            "october_called_mule": _share(d["calls"][te], MULE),
        }
    rep["all_layering_calls_per_100k"] = per_100k(d["calls"] == LAYER, d["m"], ALL)
    fast = groups["chain_fast_le25min"]
    rep["fast_chain_called_mule_cleaned_mule"] = _share(
        d["cleaned"][fast & tr & (d["calls"] == MULE)], MULE
    )
    return rep


def _estimate(d: dict, tr: np.ndarray, te: np.ndarray, target: str) -> dict:
    return rescaled_f1(
        d[target][tr],
        d["gen"][tr],
        d["calls"][tr],
        d["zord"][tr],
        d["ctr_seg"][tr],
        d["calls"][te],
        d["zord"][te],
        d["ctr_seg"][te],
    )


def _macro(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p, labels=list(range(N_CLASSES)), average="macro", zero_division=0))


def sec_estimate(d: dict) -> dict:
    rep = {}
    oct_ = _estimate(d, d["m"] < 10, d["m"] == 10, "cleaned")
    rep["october"] = {
        "per_class": {
            CLASSES[c]: {
                "a": oct_["a"][c],
                "b": oct_["b"][c],
                "f1_train": oct_["f1_train"][c],
                "f1_october": oct_["f1_test"][c],
                "ordinary_share": oct_["ordinary_share"][c],
            }
            for c in range(N_CLASSES)
        },
        "macro_train_pooled": oct_["macro_train"],
        "macro_october": oct_["macro_test"],
    }
    back = {}
    for h in MONTHS:
        tr = (d["m"] < 10) & (d["m"] != h)
        te = d["m"] == h
        for target in ("given", "cleaned"):
            e = _estimate(d, tr, te, target)
            back[f"{target}_m{h}"] = {
                "estimate": e["macro_test"],
                "actual": _macro(d[target][te], d["calls"][te]),
                "other_months_pooled": e["macro_train"],
            }
    rep["backtest"] = back
    return rep


def sec_probe(d: dict) -> dict:
    rep = {}
    tr = d["m"] < 10
    p = d["proba"][:, BOT]
    for cut in (0.1, 0.2, 0.35):
        new = d["calls"].copy()
        s = (p >= cut) & (d["calls"] != BOT)
        new[s] = BOT
        r = {"train_rows": int((s & tr).sum()), "october_rows": int((s & ~tr).sum())}
        for target in ("given", "cleaned", "gen"):
            r[target] = [
                _macro(d[target][d["m"] == m], new[d["m"] == m])
                - _macro(d[target][d["m"] == m], d["calls"][d["m"] == m])
                for m in MONTHS
            ]
        rep[f"p_bot_ge_{cut}"] = r
    return rep


SECTIONS = {
    "bursts": sec_bursts,
    "offpool": sec_offpool,
    "zones": sec_zones,
    "layering": sec_layering,
    "estimate": sec_estimate,
    "probe": sec_probe,
}


def main(argv=None) -> None:
    from ringtrace import threads

    threads.pin()
    names = (argv if argv is not None else sys.argv[1:]) or list(SECTIONS)
    t0 = time.time()
    d = load()
    rep = {}
    for name in names:
        rep[name] = SECTIONS[name](d)
        print(f"[velocity] {name} done ({time.time() - t0:.0f}s)", flush=True)
    out = paths.out_dir() / "velocity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[velocity] report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
