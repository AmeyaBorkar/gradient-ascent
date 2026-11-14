"""Gap scan: what could put a file at 0.63 public when a perfect typist of generator-made rows sits
near 0.57 to 0.60? Train labels are used row by row; October only through counts and rates.

    python analysis/gapscan.py ceiling   [--calls NPZ]
    python analysis/gapscan.py omnibus
    python analysis/gapscan.py boundary  [--calls NPZ]
    python analysis/gapscan.py lookalike [--calls NPZ]
    python analysis/gapscan.py festival  [--calls NPZ]
    python analysis/gapscan.py octlink   [--calls NPZ]
    python analysis/gapscan.py all       [--calls NPZ]

``--calls`` is an npz in ledger order with ``calls`` (out-of-fold calls on the train months and the
October calls of the file under study), ``proba`` (its blended probabilities) and the evidence
arrays ``tmpl``, ``anchor`` and ``ctr_block``; the default is
``$RINGTRACE_OUT/closelook/prep.npz``. Each command prints its tables and writes
``$RINGTRACE_OUT/gapscan/<command>.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_core import catch_needed, implied_r, shifted_f1, template_majority  # noqa: E402

from ringtrace import paths, silver, store  # noqa: E402
from ringtrace.classes import CLASSES  # noqa: E402
from ringtrace.evidence.anchors import TEMPLATE_PAIR  # noqa: E402

MONTHS = (7, 8, 9)
THREADS = int(os.environ.get("RINGTRACE_THREADS", "4"))


def _calls(path: str | None) -> dict:
    p = Path(path) if path else paths.out_dir() / "closelook" / "prep.npz"
    z = np.load(p)
    return {k: z[k] for k in z.files}


def _base(npz: dict | None = None) -> dict:
    led = store.ledger()
    d = {
        "led": led,
        "is_test": led["is_test"].to_numpy(),
        "period": led["period"].to_numpy(),
        "y": led["y"].to_numpy().astype(np.int64),
        "cleaned": silver.load(led, "label"),
        "gen": silver.load(led, "structured"),
    }
    if npz is not None:
        if len(npz["calls"]) != led.height:
            raise ValueError(f"calls have {len(npz['calls'])} rows, ledger {led.height}")
        d.update({k: npz[k] for k in ("calls", "proba", "tmpl", "anchor", "ctr_block") if k in npz})
    for k in ("cleaned", "gen"):
        if d[k] is None or len(d[k]) != led.height:
            raise ValueError(f"silver target {k} missing or not in ledger order")
    return d


def _save(name: str, obj: dict) -> None:
    out = paths.out_dir() / "gapscan"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


# --------------------------------------------------------------------------------------- ceiling
def ceiling(d: dict, public: float = 0.5499354, target: float = 0.6311) -> dict:
    tr = ~d["is_test"]
    calls, c, g, tm = d["calls"][tr], d["cleaned"][tr], d["gen"][tr], d["tmpl"][tr]
    typist = g.copy()
    maj = template_majority(typist, tm, TEMPLATE_PAIR)
    r_star = implied_r(c, calls, g, public)
    share = ((c > 0) & (g == 0)).sum() / tr.sum()
    print(
        f"r implied by our public {public}: {r_star:.3f} "
        f"(ordinary {100 * share * r_star:.2f} % of rows)"
    )
    res: dict = {
        "implied_r": r_star,
        "ordinary_share": share * r_star,
        "scores": {},
        "catch_needed": {},
    }
    files = [("ours", calls), ("typist", typist), ("template_majority", maj)]
    for r in (1.0, r_star, 1.14, 1.24):
        row = {nm: float(shifted_f1(c, p, g, r).mean()) for nm, p in files}
        res["scores"][f"{r:.3f}"] = row
        print(f"r={r:.3f}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
    for base_nm, base in files[:2]:
        for prec in (1.0, 0.6, 0.3):
            q = catch_needed(c, base, g, target, r=r_star, prec=prec)
            rows = q * share * r_star * 260_582
            res["catch_needed"][f"{base_nm}@{prec}"] = {"share": q, "october_rows": rows}
            print(
                f"{target} from {base_nm}, precision {prec}: "
                f"catch {q:.3f} of ordinary rows ({rows:,.0f})"
            )
    res["per_class"] = {nm: shifted_f1(c, p, g, r_star).round(4).tolist() for nm, p in files}
    return res


# --------------------------------------------------------------------------------------- omnibus
def artefact_features(led: pl.DataFrame) -> pl.DataFrame:
    """Generator-artefact and history features over the whole ledger, ledger order."""
    n = led.height
    idint = np.array([int(h, 16) for h in led["txn_id"].str.slice(2).to_list()], dtype=np.int64)
    day = pl.col("timestamp").dt.date()
    f = led.with_columns(
        pl.Series("idint", idint),
        (pl.col("ts") % 60).alias("sec"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.day().alias("dom"),
        ((pl.col("amount") * 100).round(0).cast(pl.Int64) % 100).alias("amt_cents"),
        (pl.col("amount").round(0).cast(pl.Int64) % 10).alias("amt_mod10"),
        ((pl.col("balance_before") * 100).round(0).cast(pl.Int64) % 100).alias("bal_cents"),
        (pl.col("amount") / (pl.col("balance_before") + 1)).alias("amt_bal"),
        ((pl.col("sim_age_hours") * 10).round(0).cast(pl.Int64) % 10).alias("sim_dec"),
        ((pl.col("device_age_days") * 10).round(0).cast(pl.Int64) % 10).alias("dev_dec"),
        ((day - pl.col("s_opened_on")).dt.total_days() - pl.col("sender_account_age_days")).alias(
            "acc_resid"
        ),
        (day - pl.col("r_opened_on")).dt.total_days().alias("r_age"),
        (day - pl.col("m_onboarded_on")).dt.total_days().alias("m_age"),
        (pl.col("device_age_days") - pl.col("sender_account_age_days")).alias("dev_minus_acc"),
        (pl.col("device_city") == pl.col("s_home_city")).alias("home"),
        (pl.col("device_city") == pl.col("r_home_city")).alias("r_city"),
        (pl.col("device_city") == pl.col("m_city")).alias("m_cityeq"),
        (pl.col("device_id").str.slice(3) == pl.col("sender_id").str.slice(2)).alias("own_dev"),
        pl.len().over("sender").alias("s_n"),
        pl.len().over("receiver").alias("r_n"),
        pl.col("sender").n_unique().over("device").alias("d_accts"),
        pl.len().over(["sender", "receiver"]).alias("pair_n"),
        pl.col("ts").rank("ordinal").over(["sender", "receiver"]).alias("pair_rank"),
        pl.col("sender").n_unique().over("receiver").alias("r_senders"),
    )
    f = f.with_columns(
        ((pl.col("pair_rank") == 1).cast(pl.Int8) - pl.col("new_payee")).alias("np_resid")
    )
    rev = (
        f.select(pl.col("receiver").alias("sender"), pl.col("sender").alias("receiver"))
        .unique()
        .with_columns(pl.lit(True).alias("rev_pair"))
    )
    f = f.join(rev, on=["sender", "receiver"], how="left").with_columns(
        pl.col("rev_pair").fill_null(False) & pl.col("to_person")
    )
    s_as_r = (
        f.filter(pl.col("to_person"))
        .group_by("receiver")
        .len()
        .rename({"receiver": "sender", "len": "s_recv_n"})
    )
    f = f.join(s_as_r, on="sender", how="left").with_columns(pl.col("s_recv_n").fill_null(0))
    out = f.select(pl.col("sender").alias("receiver"), pl.col("ts").alias("t_out")).sort("t_out")
    j = (
        f.select("row", "receiver", "ts", "to_person")
        .sort("ts")
        .join_asof(out, left_on="ts", right_on="t_out", by="receiver", strategy="forward")
    )
    j = j.select(
        "row", pl.when(pl.col("to_person")).then(pl.col("t_out") - pl.col("ts")).alias("onward_gap")
    )
    inn = (
        f.filter(pl.col("to_person"))
        .select(pl.col("receiver").alias("sender"), pl.col("ts").alias("t_in"))
        .sort("t_in")
    )
    j2 = (
        f.select("row", "sender", "ts")
        .sort("ts")
        .join_asof(inn, left_on="ts", right_on="t_in", by="sender", strategy="backward")
    )
    j2 = j2.select("row", (pl.col("ts") - pl.col("t_in")).alias("in_gap"))
    f = f.join(j, on="row", how="left").join(j2, on="row", how="left").sort("row")
    if f.height != n:
        raise AssertionError("artefact features changed the row count")
    keep = [
        "idint",
        "sec",
        "hour",
        "dom",
        "amt_cents",
        "amt_mod10",
        "bal_cents",
        "amt_bal",
        "sim_dec",
        "dev_dec",
        "acc_resid",
        "r_age",
        "m_age",
        "dev_minus_acc",
        "home",
        "r_city",
        "m_cityeq",
        "own_dev",
        "s_n",
        "r_n",
        "d_accts",
        "pair_n",
        "pair_rank",
        "r_senders",
        "np_resid",
        "rev_pair",
        "s_recv_n",
        "onward_gap",
        "in_gap",
    ]
    return f.select(keep)


RAW = [
    "amount",
    "balance_before",
    "sim_age_hours",
    "device_age_days",
    "sender_account_age_days",
    "login_to_txn_sec",
    "new_payee",
    "pin_retries",
    "app_version",
    "ip_risk_score",
    "qr_vpa_mismatch",
]
CAT = [
    "txn_type",
    "merchant_category",
    "app",
    "s_kyc_level",
    "s_bank_code",
    "s_os",
    "r_kyc",
    "receiver_type",
]


def omnibus(d: dict) -> dict:
    """Ordinary-looking fraud (given fraud, generator-only legit) against legit, held-out AUC."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    led = d["led"]
    X = pl.concat(
        [
            led.select(RAW),
            led.select([pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32) for c in CAT]),
            artefact_features(led),
        ],
        how="horizontal",
    )
    names = X.columns
    X = X.to_numpy().astype(np.float32)
    tr = ~d["is_test"]
    y, g, mo = d["y"], d["gen"], d["period"]
    keep = tr & ((y == 0) | (g == 0))
    ordn = (y > 0) & keep
    rng = np.random.default_rng(0)
    res: dict = {}
    for hold in (9, 7):
        fit = keep & (mo != hold) & (ordn | (rng.random(len(y)) < 0.25))
        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=100,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            num_threads=THREADS,
            verbose=-1,
            seed=0,
            deterministic=True,
        )
        m = lgb.train(
            params,
            lgb.Dataset(X[fit], ordn[fit].astype(int), feature_name=names),
            num_boost_round=300,
        )
        te = keep & (mo == hold)
        p = m.predict(X[te])
        auc = roc_auc_score(ordn[te], p)
        per = {}
        for k in range(1, 12):
            sel = te & ((y == 0) | (y == k))
            per[CLASSES[k]] = float(roc_auc_score(y[sel] == k, m.predict(X[sel])))
        o = np.argsort(-p)
        prec = {q: float(ordn[te][o[:q]].mean()) for q in (100, 1000, 10000)}
        res[str(hold)] = {
            "auc": float(auc),
            "per_class": per,
            "precision_at": prec,
            "base": float(ordn[te].mean()),
            "features": len(names),
        }
        print(
            f"held-out month {hold}: AUC {auc:.4f}; "
            f"per class {min(per.values()):.3f} to {max(per.values()):.3f}; "
            f"precision@1000 {prec[1000]:.4f} (base {ordn[te].mean():.4f}); {len(names)} features"
        )
    return res


# -------------------------------------------------------------------------------------- boundary
def boundary(d: dict) -> dict:
    led = (
        d["led"]
        .select("timestamp", "period")
        .with_columns(pl.col("timestamp").dt.month().alias("cal"))
    )
    cal = led["cal"].to_numpy()
    late = cal > d["period"]
    tr = ~d["is_test"]
    res: dict = {}
    for p in (9, 10):
        m = late & (d["period"] == p)
        row = {
            "rows": int(m.sum()),
            "called_fraud": int((d["calls"][m] > 0).sum()),
            "anchored": int((d["anchor"][m] > 0).sum()),
        }
        if p == 9:
            row["given_legit"] = int((d["y"][m] == 0).sum())
            row["generator_made"] = float((d["gen"][m] > 0).mean())
        res[f"dated_after_period_{p}"] = row
        print(f"rows filed in period {p} and dated later: {row}")
    ts = led["timestamp"]
    for m in (7, 8, 9, 10):
        start = dt.datetime(2026, m, 1)
        end = dt.datetime(2026, m + 1, 1)
        for nm, lo, hi in [
            ("first24h", start, start + dt.timedelta(hours=24)),
            ("days11_20", start + dt.timedelta(days=10), start + dt.timedelta(days=20)),
            ("last24h", end - dt.timedelta(hours=24), end),
        ]:
            sel = ((ts >= lo) & (ts < hi)).to_numpy() & (d["period"] == m)
            row = {
                "rows": int(sel.sum()),
                "fraud_calls_pct": float((d["calls"][sel] > 0).mean() * 100),
            }
            if m < 10:
                s2 = sel[tr]
                row["given_fraud_pct"] = float((d["y"][tr][s2] > 0).mean() * 100)
                row["generator_pct"] = float((d["gen"][tr][s2] > 0).mean() * 100)
                row["ordinary_pct"] = float(
                    ((d["y"][tr][s2] > 0) & (d["gen"][tr][s2] == 0)).mean() * 100
                )
            res[f"{m}_{nm}"] = row
            print(m, nm, {k: round(v, 2) for k, v in row.items()})
    return res


# ------------------------------------------------------------------------------------- lookalike
def lookalike(d: dict) -> dict:
    """Inside each look-alike shape, is the type predictable out of fold (AUC on the given type)?"""
    from sklearn.metrics import roc_auc_score

    tr = ~d["is_test"]
    proba, calls, g, s, per = (
        d["proba"][tr],
        d["calls"][tr],
        d["y"][tr],
        d["gen"][tr],
        d["period"][tr],
    )
    tm = d["tmpl"][tr]
    drn = store.family("drains")["drn_shape"].fill_null(0).to_numpy()[tr]
    groups = {f"template{t}": (tm == t, a, b) for t, (a, b) in TEMPLATE_PAIR.items()}
    groups["drain_shape3"] = (drn == 3, 4, 5)
    res: dict = {}
    for nm, (m, a, b) in groups.items():
        sel = m & np.isin(g, [a, b]) & (s > 0)
        yb = (g[sel] == b).astype(int)
        pb = proba[sel, b] / (proba[sel, a] + proba[sel, b] + 1e-12)
        aucs = [
            float(roc_auc_score(yb[per[sel] == mth], pb[per[sel] == mth]))
            for mth in MONTHS
            if len(np.unique(yb[per[sel] == mth])) == 2
        ]
        row = {
            "pair": [CLASSES[a], CLASSES[b]],
            "rows": int(sel.sum()),
            "share_b": float(yb.mean()),
            "call_acc": float(((calls[sel] == b) == (yb == 1)).mean()),
            "majority": float(max(yb.mean(), 1 - yb.mean())),
            "auc_by_month": aucs,
        }
        res[nm] = row
        print(
            f"{nm:13s} {CLASSES[a]}/{CLASSES[b]} rows {row['rows']} share_b {row['share_b']:.3f} "
            f"call acc {row['call_acc']:.3f} "
            f"(majority {row['majority']:.3f}) AUC {np.round(aucs, 3).tolist()}"
        )
    return res


# -------------------------------------------------------------------------------------- festival
def festival(d: dict) -> dict:
    """Do October's ordinary-fraud proxies (single-row fraud-block devices with no mark) carry the
    festival amount scale of October's legit rows? Geometric-mean ratio against legit-block rows,
    bootstrap 95 %."""
    led = d["led"].select("amount", "device")
    dn = led.select(pl.len().over("device")).to_series().to_numpy()
    amt = led["amount"].to_numpy()
    rng = np.random.default_rng(0)
    res: dict = {}
    for split, nm in [(False, "train"), (True, "october")]:
        base = d["is_test"] == split
        fb = base & (d["ctr_block"] == 2) & (dn == 1) & (d["anchor"] == 0) & (d["tmpl"] == 0)
        lb = base & (d["ctr_block"] == 3)
        la, lb_ = np.log(amt[fb]), np.log(amt[lb])
        bs = [
            rng.choice(la, la.size).mean() - rng.choice(lb_, lb_.size).mean() for _ in range(2000)
        ]
        row = {
            "proxies": int(fb.sum()),
            "legit_block": int(lb.sum()),
            "gm_ratio": float(np.exp(la.mean() - lb_.mean())),
            "ci95": np.exp(np.quantile(bs, [0.025, 0.975])).tolist(),
        }
        res[nm] = row
        print(nm, row)
    return res


# --------------------------------------------------------------------------------------- octlink
def octlink(d: dict) -> dict:
    """Share of legit-called rows whose sender, P2P receiver or device appears in a fraud-called row
    within 24 h, per month. A month written at half volume (half the events, half the legit rows per
    account) halves this share; thinning legit rows alone (September, p = 0.494, fraud-called rows
    whole) leaves it unchanged. If October's ordinary fraud were tied to its events, October's share
    would sit above half of the train share."""
    led = d["led"].select("row", "sender", "receiver", "device", "ts", "to_person")
    df = led.with_columns(
        pl.Series("call", d["calls"]),
        pl.Series("period", d["period"]),
        pl.Series("y", d["y"]),
        pl.Series("gen", d["gen"]),
    )

    def rates(f: pl.DataFrame) -> dict:
        fr = f.filter(pl.col("call") > 0)
        accts = pl.concat(
            [
                fr.select(pl.col("sender").alias("e"), pl.col("ts").alias("tf")),
                fr.filter(pl.col("to_person")).select(
                    pl.col("receiver").alias("e"), pl.col("ts").alias("tf")
                ),
            ]
        ).sort("tf")
        devs = fr.select(pl.col("device").alias("e"), pl.col("ts").alias("tf")).sort("tf")
        lg = f.filter(pl.col("call") == 0)
        out = {}
        for nm, col, tab in [
            ("sender", "sender", accts),
            ("p2p_receiver", "receiver", accts),
            ("device", "device", devs),
        ]:
            q = lg.select(pl.col(col).alias("e"), "ts", "y", "gen", "to_person")
            if nm == "p2p_receiver":
                q = q.filter(pl.col("to_person"))
            j = q.sort("ts").join_asof(tab, left_on="ts", right_on="tf", by="e", strategy="nearest")
            hit = ((pl.col("ts") - pl.col("tf")).abs() <= 86400).fill_null(False)
            r = j.select(
                hit.mean().alias("all"),
                hit.filter((pl.col("y") > 0) & (pl.col("gen") == 0)).mean().alias("ordinary"),
            ).row(0, named=True)
            out[nm] = {k: None if v is None else 100 * float(v) for k, v in r.items()}
        return out

    res: dict = {}
    for p in (*MONTHS, 10):
        res[str(p)] = rates(df.filter(pl.col("period") == p))
    sep = df.filter(pl.col("period") == 9)
    keep = (sep["call"].to_numpy() > 0) | (np.random.default_rng(0).random(sep.height) < 0.494)
    res["9_thinned"] = rates(sep.filter(pl.Series(keep)))
    for k, v in res.items():
        print(k, {nm: round(x["all"], 3) for nm, x in v.items()})
    return res


COMMANDS = {
    "ceiling": ceiling,
    "omnibus": omnibus,
    "boundary": boundary,
    "lookalike": lookalike,
    "festival": festival,
    "octlink": octlink,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--calls", default=None)
    a = ap.parse_args()
    d = _base(_calls(a.calls))
    for nm in COMMANDS if a.command == "all" else [a.command]:
        print(f"== {nm}")
        _save(nm, COMMANDS[nm](d))


if __name__ == "__main__":
    main()
