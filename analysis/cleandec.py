"""What changes when the October labels are clean: the train noise process, a clean-label proxy,
re-decoding under clean posteriors, and what the public scores say about the label model.

    python analysis/cleandec.py noise    # unreported / wrong-type rates by class and observables
    python analysis/cleandec.py decode   # clean-posterior decoders on the best file, five targets
    python analysis/cleandec.py public   # the fake_cashback -> phishing switch against label models
    python analysis/cleandec.py census   # label-free October census, ordinary-row consistency
    python analysis/cleandec.py build <variant>   # October file over the best file (candidates/)

The base is ``recipes/final-half.json`` (the best public file, ``blend-tabmF-bq.csv``). Its TabM
member is the run ``final-tabm``; where that run is missing, the saved ``tabm`` run (the same three
seeds) stands in. Train rows are scored per held-out month on the given, cleaned and gen-only
labels, the A-clean proxy (cleaned plus the expected unreported ordinary rows) and the counter
proxy (A-clean with the label-free counter imposed). October rows are only ever counted.
Outputs go to ``$RINGTRACE_OUT/cleandec/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cleandec_core import (  # noqa: E402
    counter_proxy,
    hidden_counts,
    macro,
    noise_rates,
    switch_likelihood,
    wrong_type_matrix,
)

from ringtrace import paths  # noqa: E402
from ringtrace.classes import CLASSES, N_CLASSES  # noqa: E402

MONTHS = (7, 8, 9)
RECIPE = "recipes/final-half.json"
BEST_FILE = "blend-tabmF-bq.csv"
# measured by ``noise`` on generator-only fraud rows (silver structured), index 0 unused
UNREPORTED_BY_CLASS = [
    0.0,
    0.117,
    0.128,
    0.120,
    0.097,
    0.120,
    0.128,
    0.124,
    0.124,
    0.112,
    0.130,
    0.126,
]

VARIANTS: dict[str, dict] = {
    "scale-0.80": {"fraud_scale": 0.80},
    "scale-0.85": {"fraud_scale": 0.85},
    "scale-0.879": {"fraud_scale": 0.879},
    "scale-0.95": {"fraud_scale": 0.95},
    "scale-1.0": {"fraud_scale": 1.0},
    "clean-u": {"fraud_scale": 1.0, "clean": {"unreported": 0.121}},
    "clean-uk": {"fraud_scale": 1.0, "clean": {"unreported": UNREPORTED_BY_CLASS}},
    "clean-u-flip": {"fraud_scale": 1.0, "clean": {"unreported": 0.121, "type_flip": 0.03}},
    "clean-uk-flip": {
        "fraud_scale": 1.0,
        "clean": {"unreported": UNREPORTED_BY_CLASS, "type_flip": 0.03},
    },
    "flip-only": {
        "fraud_scale": 0.875,
        "clean": {"unreported": 0.0, "legit_flip": 0.0, "type_flip": 0.03},
    },
    "hidden-1.1": {"fraud_scale": 0.875, "hidden_factor": 1.1},
    "hidden-0.9": {"fraud_scale": 0.875, "hidden_factor": 0.9},
}


def out_dir() -> Path:
    d = paths.out_dir() / "cleandec"
    d.mkdir(parents=True, exist_ok=True)
    return d


def best_recipe() -> dict:
    from ringtrace.final import load_recipe

    r = load_recipe(RECIPE)
    runs = paths.out_dir() / "runs"
    for m in r["members"]:
        if not (runs / m["run"]).exists() and m["run"] == "final-tabm":
            m["run"] = "tabm"
    return r


class Ctx:
    """The bench, the best recipe's mixed probabilities and calls, the targets and the counter."""

    def __init__(self):
        from ringtrace.final import TRAIN_MONTHS, Bench, mix
        from ringtrace.rules import _counter

        self.b = b = Bench.from_store(paths.out_dir() / "runs", None, log=lambda *_: None)
        self.recipe = best_recipe()
        arrays = [b.member(m) for m in self.recipe["members"]]
        roles = [m["role"] for m in self.recipe["members"]]
        w = b.weights(self.recipe)
        self.proba = np.zeros((b.led.height, N_CLASSES))
        for k in (*TRAIN_MONTHS, "october"):
            rows = b.te if k == "october" else b.tr[b.period[b.tr] == k]
            self.proba[rows] = mix([a[rows] for a in arrays], w[k], roles)
        self.calls = b.calls(self.recipe)["pred"]
        self.pt = b.period[b.tr]
        c = _counter(b.led)
        self.block, self.seg, self.conf = c["block"], c["seg_class"], c["conf"]
        t = b.targets
        self.cleaned, self.gen, self.given = t["cleaned"], t["gen"], t["given"]
        tr = b.tr
        self.proxy = counter_proxy(self.cleaned, self.block[tr], self.seg[tr], self.conf[tr])
        self.hid = hidden_counts(self.cleaned, self.gen, self.pt)
        self.hidp = hidden_counts(self.proxy, self.gen, self.pt)

    def scores(self, pred_tr: np.ndarray) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for m in MONTHS:
            s = self.pt == m
            p = pred_tr[s]
            for name, y, h in (
                ("given", self.given[s], None),
                ("cleaned", self.cleaned[s], None),
                ("gen", self.gen[s], None),
                ("aclean", self.cleaned[s], self.hid[m]),
                ("proxy", self.proxy[s], self.hidp[m]),
            ):
                out.setdefault(name, []).append(macro(y, p, h))
        return out

    def variant_calls(self, decoder: dict) -> np.ndarray:
        r = dict(self.recipe)
        r["decoder"] = decoder
        return self.b.calls(r)["pred"]


def pair_counts(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    ch = a != b
    keys, n = np.unique(a[ch] * N_CLASSES + b[ch], return_counts=True)
    return {
        f"{CLASSES[k // N_CLASSES]} -> {CLASSES[k % N_CLASSES]}": int(c)
        for k, c in zip(keys, n, strict=True)
    }


# --------------------------------------------------------------------------------------- noise


def cmd_noise(ctx: Ctx) -> dict:
    """Unreported and wrong-type rates over three fraud populations: generator-only fraud rows
    (silver structured), label-free fraud-block phones (class = counter segment) and label-free
    template rows (class = generator-only label)."""
    import polars as pl

    from ringtrace.evidence import anchors

    b = ctx.b
    tr = b.tr
    led = b.led
    tmpl = np.asarray(anchors.templates(led)[0], dtype=np.int64)[tr]
    cols = led.select(
        pl.col("s_bank_code").fill_null("?"),
        pl.col("s_kyc_level").fill_null("?"),
        pl.col("s_os").fill_null("?"),
        pl.col("app").fill_null("?"),
        pl.col("txn_type"),
        (pl.col("device_city") == pl.col("s_home_city")).fill_null(False).alias("home"),
        pl.col("amount").log10().floor().cast(pl.Int64).alias("amount_decade"),
        (pl.col("timestamp").dt.hour() // 4).alias("hour4"),
        pl.col("new_payee"),
    )[tr]
    groups = {c: cols[c].cast(pl.Utf8).to_numpy() for c in cols.columns}
    groups["month"] = ctx.pt.astype(str)
    groups["template"] = tmpl.astype(str)
    given, gen = ctx.given, ctx.gen
    seg = ctx.seg[tr]
    fb = (ctx.block[tr] == 2) & (seg >= 1) & (ctx.conf[tr] >= 0.8)
    pops = {
        "gen": (gen > 0, gen),
        "fraud_block": (fb, seg),
        "template": ((tmpl > 0) & (gen > 0), gen),
    }
    rep: dict = {}
    for pname, (mask, truth) in pops.items():
        rep[pname] = {"rows": int(mask.sum())}
        g = dict(groups, **{"class": truth.astype(str)})
        for gname, gv in g.items():
            r = noise_rates(given[mask], truth[mask], gv[mask])
            rep[pname][gname] = r
            print(
                f"{pname:12s} by {gname:14s} unreported {r['pooled_unreported']:.4f} wrong "
                f"{r['pooled_wrong_type']:.4f} chi2 {r['chi2']:7.1f} on {r['df']:3d} df",
                flush=True,
            )
        cm = wrong_type_matrix(given[mask], truth[mask])
        off = cm.sum(axis=1) - np.diag(cm)
        rep[pname]["wrong_type_by_class"] = {
            CLASSES[k]: float(off[k] / max(cm[k].sum(), 1)) for k in range(1, N_CLASSES)
        }
        pair = cm.copy()
        np.fill_diagonal(pair, 0)
        rep[pname]["wrong_type_top_pairs"] = [
            [CLASSES[i // N_CLASSES], CLASSES[i % N_CLASSES], int(pair.flat[i])]
            for i in np.argsort(-pair.ravel())[:8]
        ]
    # do our out-of-fold calls on generator rows follow the observables the way the labels do?
    calls = ctx.calls[tr]
    g = gen > 0
    rep["called_vs_unreported_by_bank"] = {}
    banks = groups["s_bank_code"]
    for bk in np.unique(banks[g]):
        s = g & (banks == bk)
        if s.sum() >= 150:
            rep["called_vs_unreported_by_bank"][str(bk)] = {
                "rows": int(s.sum()),
                "unreported": float((given[s] == 0).mean()),
                "called_fraud": float((calls[s] > 0).mean()),
            }
    v = rep["called_vs_unreported_by_bank"].values()
    u = np.array([x["unreported"] for x in v])
    cf = np.array([x["called_fraud"] for x in v])
    rep["bank_corr_called_vs_unreported"] = float(np.corrcoef(u, cf)[0, 1])
    print(
        "corr over banks, unreported rate vs our fraud-call rate on generator rows:",
        round(rep["bank_corr_called_vs_unreported"], 3),
        "| call-rate range",
        round(float(cf.min()), 4),
        round(float(cf.max()), 4),
    )
    (out_dir() / "noise.json").write_text(json.dumps(rep, indent=1))
    return rep


# --------------------------------------------------------------------------------------- decode


def cmd_decode(ctx: Ctx) -> dict:
    b = ctx.b
    base = ctx.scores(ctx.calls[b.tr])
    rep = {
        "base": base,
        "proxy_rows_changed": int((ctx.proxy != ctx.cleaned).sum()),
        "variants": {},
    }
    print("base", {t: [round(x, 5) for x in v] for t, v in base.items()}, flush=True)
    mass = {}
    for m in (*MONTHS, "october"):
        rows = b.te if m == "october" else b.tr[ctx.pt == m]
        mass[str(m)] = float((1 - ctx.proba[rows, 0]).sum())
    rep["fraud_mass"] = mass
    rep["truth_counts"] = {
        str(m): {
            "given": int((ctx.given[ctx.pt == m] > 0).sum()),
            "cleaned": int((ctx.cleaned[ctx.pt == m] > 0).sum()),
            "aclean": float((ctx.cleaned[ctx.pt == m] > 0).sum() + ctx.hid[m].sum()),
        }
        for m in MONTHS
    }
    for name, dec in VARIANTS.items():
        t0 = time.time()
        pred = ctx.variant_calls(dec)
        sc = ctx.scores(pred[b.tr])
        ch = {t: [x - y for x, y in zip(sc[t], base[t], strict=True)] for t in sc}
        rows = [
            int((pred[b.tr][ctx.pt == m] != ctx.calls[b.tr][ctx.pt == m]).sum()) for m in MONTHS
        ]
        octm = pair_counts(ctx.calls[b.te], pred[b.te])
        ok = all(c >= -1e-12 for t in ("given", "cleaned") for c in ch[t])
        pos = any(c > 1e-12 for t in ("given", "cleaned") for c in ch[t])
        rep["variants"][name] = {
            "decoder": dec,
            "change": ch,
            "train_rows": rows,
            "october": octm,
            "passes": bool(ok and pos),
        }
        print(
            f"{name:14s} "
            + " | ".join(
                f"m{m} " + " ".join(f"{t[0]}{ch[t][i] * 1e5:+6.1f}" for t in ch) + f" n{rows[i]}"
                for i, m in enumerate(MONTHS)
            )
            + f" | oct {sum(octm.values())} {octm} {'PASS' if ok and pos else ''}"
            + f" ({time.time() - t0:.0f}s)",
            flush=True,
        )
    (out_dir() / "decode.json").write_text(json.dumps(rep, indent=1))
    return rep


# --------------------------------------------------------------------------------------- public


def cmd_public(ctx: Ctx, observed: float = 0.5408 - 0.5486, moved: int = 94) -> dict:
    """The template-4 switch (bait answered within 4 h: fake_cashback -> phishing_collect). On
    the public slice it cost ``observed`` (94 October rows moved, counted from the two files). The
    train months give the same switch's effect on every target and the share of truly-cashback
    rows among the moved rows; the October public slice is sized from the train months' A-clean
    counts per row. The likelihood of the observed change is read for each cashback share."""
    from ringtrace.evidence import anchors
    from ringtrace.evidence.anchors import CASHB, PHISH

    b = ctx.b
    tmpl = np.asarray(anchors.templates(b.led)[0], dtype=np.int64)
    sw = (tmpl == 4) & (ctx.calls == CASHB)
    rep: dict = {"switched_rows": {str(m): int((sw[b.tr] & (ctx.pt == m)).sum()) for m in MONTHS}}
    rep["switched_rows"]["october"] = int(sw[b.te].sum())
    swt = sw[b.tr]
    effects, shares = {}, []
    a_tot = {"T": 0.0, "P": 0.0, "TP": 0.0}
    b_tot = {"T": 0.0, "P": 0.0, "TP": 0.0}
    n_tr = 0
    for m in MONTHS:
        s = ctx.pt == m
        base = ctx.calls[b.tr][s]
        alt = base.copy()
        alt[swt[s]] = PHISH
        y = ctx.cleaned[s]
        f = y[swt[s] & (y > 0)]
        shares.append(float((f == CASHB).sum() / max(((f == CASHB) | (f == PHISH)).sum(), 1)))
        for name, yy, h in (
            ("given", ctx.given[s], None),
            ("cleaned", y, None),
            ("gen", ctx.gen[s], None),
            ("aclean", y, ctx.hid[m]),
            ("proxy", ctx.proxy[s], ctx.hidp[m]),
        ):
            effects.setdefault(name, []).append(macro(yy, alt, h) - macro(yy, base, h))
        pure = y.copy()
        pure[swt[s] & (pure > 0)] = CASHB
        effects.setdefault("template-pure", []).append(
            macro(pure, alt, ctx.hid[m]) - macro(pure, base, ctx.hid[m])
        )
        from cleandec_core import expected_confusion

        cm = expected_confusion(y, base, ctx.hid[m])
        for cls, acc in ((CASHB, a_tot), (PHISH, b_tot)):
            acc["T"] += cm[cls].sum()
            acc["P"] += cm[:, cls].sum()
            acc["TP"] += cm[cls, cls]
        n_tr += int(s.sum())
    rep["train_effect"] = effects
    rep["train_cashback_share"] = shares
    # the October public slice, scaled from the train months per row, with October's own calls
    pub = 0.3 * len(b.te) / n_tr
    oc = ctx.calls[b.te]
    a = {"T": a_tot["T"] * pub, "P": 0.3 * (oc == CASHB).sum()}
    bb = {"T": b_tot["T"] * pub, "P": 0.3 * (oc == PHISH).sum()}
    for d_, tot in ((a, a_tot), (bb, b_tot)):
        f1 = 2 * tot["TP"] / (tot["T"] + tot["P"])
        d_["TP"] = f1 * (d_["T"] + d_["P"]) / 2
    rep["public_slice"] = {"cashback": a, "phishing": bb}
    grid = (0.3, 0.43, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    rep["likelihood"] = switch_likelihood(observed, moved, grid, a, bb)
    print("switched rows", rep["switched_rows"], "train cashback share", np.round(shares, 3))
    for t, v in effects.items():
        print(f"  train effect {t:14s} " + " / ".join(f"{x * 1e4:+.1f}" for x in v) + " x1e-4")
    print(
        "public slice",
        {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in rep["public_slice"].items()},
    )
    for c, r in rep["likelihood"].items():
        print(
            f"  cashback share {c}: mean {r['mean'] * 1e4:+6.1f} sd {r['sd'] * 1e4:5.1f} x1e-4, "
            f"P(<= observed) {r['p_at_or_below']:.3f}, density {r['density']:.3f}"
        )
    (out_dir() / "public.json").write_text(json.dumps(rep, indent=1))
    return rep


# --------------------------------------------------------------------------------------- census


def cmd_census(ctx: Ctx) -> dict:
    """Label-free October checks. Per month and device block: rows per 100k and the share we
    call fraud; fraud-block rows with no generator evidence (ordinary fraud on a new phone, so
    they scale with the month's ordinary-fraud rate); then, train only, whether ordinary fraud
    rows differ from legit rows in cross-month entity history, new_payee against the pair's own
    history, account and device age against the ledger, and own-phone use."""
    import polars as pl

    b = ctx.b
    led = b.led
    mon = np.where(b.is_test, 10, b.period)
    from ringtrace.evidence import anchors
    from ringtrace.rules import label_free_events

    tmpl = np.asarray(anchors.templates(led)[0], dtype=np.int64)
    evid = (label_free_events(led) > 0) | (tmpl > 0) | (anchors.row_anchor(led) > 0)
    rep: dict = {"blocks": {}, "fraud_block_no_evidence_per_100k": {}}
    for m in (*MONTHS, 10):
        n = int((mon == m).sum())
        for blk in np.unique(ctx.block):
            s = (ctx.block == blk) & (mon == m)
            rep["blocks"][f"{m}-{blk}"] = {
                "per_100k": float(s.sum() / n * 1e5),
                "called_fraud": float((ctx.calls[s] > 0).mean()) if s.any() else None,
            }
        s = (ctx.block == 2) & (mon == m) & ~evid
        rep["fraud_block_no_evidence_per_100k"][str(m)] = float(s.sum() / n * 1e5)
    print("blocks", json.dumps(rep["blocks"]))
    print("fraud-block rows with no evidence per 100k", rep["fraud_block_no_evidence_per_100k"])
    gen = np.full(led.height, -1)
    gen[b.tr] = ctx.gen
    f = (
        led.select(
            "sender",
            "receiver",
            "device",
            "ts",
            "is_test",
            "y",
            "new_payee",
            "device_age_days",
            "sender_account_age_days",
            "s_opened_on",
            "timestamp",
            "sender_id",
            "device_id",
        )
        .with_columns(pl.Series("gen", gen), pl.Series("m", mon))
        .with_columns(
            pl.when(pl.col("is_test"))
            .then(pl.lit("october"))
            .when(pl.col("y") == 0)
            .then(pl.lit("legit"))
            .when(pl.col("gen") > 0)
            .then(pl.lit("generator"))
            .otherwise(pl.lit("ordinary"))
            .alias("grp"),
            (pl.col("y") > 0).cast(pl.Int32).alias("fr"),
            (pl.col("sender") * 10_000_000 + pl.col("receiver")).alias("pair"),
        )
    )
    tr = f.filter(~pl.col("is_test"))
    hist = {}
    for key in ("sender", "receiver", "device", "pair"):
        per = tr.group_by(key, "m").agg(pl.len().alias("n"), pl.col("fr").sum().alias("f"))
        tot = per.group_by(key).agg(pl.col("n").sum().alias("N"), pl.col("f").sum().alias("F"))
        j = (
            tr.select(key, "m", "grp")
            .join(per, on=[key, "m"], how="left")
            .join(tot, on=key, how="left")
        )
        s = (
            j.group_by("grp")
            .agg(
                ((pl.col("N") - pl.col("n")) > 0).mean().alias("seen_other_month"),
                ((pl.col("F") - pl.col("f")) > 0).mean().alias("fraud_other_month"),
            )
            .sort("grp")
        )
        hist[key] = {
            r["grp"]: {k: v for k, v in r.items() if k != "grp"} for r in s.iter_rows(named=True)
        }
    rep["other_month_history"] = hist
    f = (
        f.sort("sender", "receiver", "ts")
        .with_columns(
            (pl.col("ts").rank("ordinal").over("sender", "receiver") > 1).alias("pair_seen_before")
        )
        .sort("sender", "device", "ts")
        .with_columns(
            ((pl.col("ts") - pl.col("ts").min().over("sender", "device")) / 86400.0).alias(
                "since_first"
            ),
            (pl.col("timestamp").dt.date() - pl.col("s_opened_on"))
            .dt.total_days()
            .alias("age_calc"),
        )
    )
    cons = (
        f.group_by("grp")
        .agg(
            pl.col("pair_seen_before")
            .filter(pl.col("new_payee") == 1)
            .mean()
            .alias("new_payee_1_seen_before"),
            pl.col("pair_seen_before")
            .filter(pl.col("new_payee") == 0)
            .mean()
            .alias("new_payee_0_seen_before"),
            (pl.col("sender_account_age_days") == pl.col("age_calc"))
            .mean()
            .alias("account_age_exact"),
            (pl.col("device_age_days") >= pl.col("since_first") - 1)
            .mean()
            .alias("device_age_covers_history"),
            (
                pl.col("sender_id").str.slice(2).cast(pl.Int64)
                == pl.col("device_id").str.slice(2).cast(pl.Int64)
            )
            .mean()
            .alias("own_phone"),
        )
        .sort("grp")
    )
    rep["consistency"] = {
        r["grp"]: {k: v for k, v in r.items() if k != "grp"} for r in cons.iter_rows(named=True)
    }
    for k, v in hist.items():
        print(
            f"history by {k}:",
            {g: {kk: round(vv, 4) for kk, vv in x.items()} for g, x in v.items()},
        )
    for g, v in rep["consistency"].items():
        print(f"consistency {g:10s}", {k: round(x, 4) for k, x in v.items()})
    (out_dir() / "census.json").write_text(json.dumps(rep, indent=1))
    return rep


# --------------------------------------------------------------------------------------- build


def cmd_build(ctx: Ctx, variant: str, name: str | None = None) -> dict:
    import hashlib

    import polars as pl

    from ringtrace.blend import write_submission

    b = ctx.b
    pred = ctx.variant_calls(VARIANTS[variant])
    name = name or f"cleandec-{variant}"
    path = paths.out_dir() / "candidates" / f"{name}.csv"
    sub = write_submission(
        path,
        b.led["txn_id"].to_numpy()[b.te],
        pred[b.te],
        paths.data_dir() / "sample_submission.csv",
    )
    ref = pl.read_csv(paths.out_dir() / "candidates" / BEST_FILE, infer_schema=False)
    mine = pl.read_csv(path, infer_schema=False)
    if ref["id"].to_list() != mine["id"].to_list():
        raise SystemExit("id order differs from the best file")
    idx = {c: i for i, c in enumerate(CLASSES)}
    ra = np.array([idx[x] for x in ref["label"].to_list()])
    rb = np.array([idx[x] for x in mine["label"].to_list()])
    rep = {
        "file": str(path.name),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "problems": sub["problems"],
        "rows_changed": int((ra != rb).sum()),
        "pairs": pair_counts(ra, rb),
        "decoder": VARIANTS[variant],
        "scores": ctx.scores(pred[b.tr]),
    }
    (out_dir() / f"build-{name}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "scores"}, indent=1))
    return rep


def main(argv=None) -> None:
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python analysis/cleandec.py")
    ap.add_argument("cmd", choices=("noise", "decode", "public", "census", "build"))
    ap.add_argument("variant", nargs="?", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    ctx = Ctx()
    leak = int((ctx.calls[ctx.b.te] != _best_file_calls(ctx)).sum())
    print(
        f"base rebuilt in {time.time() - t0:.0f}s; October rows differing from {BEST_FILE}: {leak}"
    )
    if args.cmd == "noise":
        cmd_noise(ctx)
    elif args.cmd == "decode":
        cmd_decode(ctx)
    elif args.cmd == "public":
        cmd_public(ctx)
    elif args.cmd == "census":
        cmd_census(ctx)
    else:
        if args.variant not in VARIANTS:
            raise SystemExit(f"variant must be one of {sorted(VARIANTS)}")
        cmd_build(ctx, args.variant)
    print(f"wall {time.time() - t0:.0f}s")


def _best_file_calls(ctx: Ctx) -> np.ndarray:
    import polars as pl

    f = pl.read_csv(paths.out_dir() / "candidates" / BEST_FILE, infer_schema=False)
    lab = dict(zip(f["id"].to_list(), f["label"].to_list(), strict=True))
    pos = {c: i for i, c in enumerate(CLASSES)}
    ids = ctx.b.led["txn_id"].to_numpy()[ctx.b.te]
    return np.array([pos[lab[str(i)]] for i in ids], dtype=np.int64)


if __name__ == "__main__":
    main()
