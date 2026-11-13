"""Error anatomy of the blend and October call-profile diagnostics (aggregates only).

    python analysis/fresh.py anatomy   [--run blend]
    python analysis/fresh.py october   [--run blend]

``anatomy`` splits each class's F1 loss on the held-out months into four pockets (see
``analysis/fresh_core.py``) on the given and cleaned targets. ``october`` compares the call
profile of every class across hour, day, amount band, app version, device age and receiver type
between train labels, train out-of-fold calls and October calls. October is only ever counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fresh_core import anatomy, band, profile_shift  # noqa: E402

from ringtrace import paths, silver, store  # noqa: E402
from ringtrace.blend import decode_all  # noqa: E402
from ringtrace.classes import CLASSES, N_CLASSES  # noqa: E402
from ringtrace.evidence import anchors  # noqa: E402

MONTHS = (7, 8, 9)


def load(run: str):
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    out = paths.out_dir() / "runs" / run
    proba = np.zeros((led.height, N_CLASSES))
    proba[~is_test] = np.load(out / "oof.npy")
    proba[is_test] = np.load(out / "test.npy")
    calls = decode_all(
        proba,
        period,
        is_test,
        {"fraud_scale": 0.875},
        ["anchored_templates"],
        led,
        log=lambda m: None,
    )["rules"]
    return led, proba, calls


def evidence(led) -> dict[str, np.ndarray]:
    anchor = anchors.row_anchor(led).astype(np.int64)
    tmpl = anchors.template_of(led).astype(np.int64)
    block = anchors.family_columns(led, ["ctr_block"])["ctr_block"].fill_null(0).to_numpy()
    return {"anchor": anchor, "tmpl": tmpl, "fraud_block": block == 2}


def cmd_anatomy(run: str) -> dict:
    led, proba, calls = load(run)
    tr = ~led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    gen = silver.load(led, "structured")
    targets = {"given": led["y"].to_numpy().astype(np.int64), "cleaned": silver.load(led, "label")}
    ev = evidence(led)
    report = {}
    for tname, y in targets.items():
        per_month = []
        for m in MONTHS:
            s = tr & (period == m)
            per_month.append(
                anatomy(
                    y[s], calls[s], gen[s], ev["anchor"][s], ev["tmpl"][s], ev["fraud_block"][s]
                )
            )
        report[tname] = per_month
        # mean over months of each pocket's stake
        print(f"== {tname}: macro-F1 per month", [round(p["macro"], 4) for p in per_month])
        keys = ["a_ordinary", "b_evidence_legit", "c_evidence_wrongtype", "d_legit_called"]
        print(f"{'class':<20}" + "".join(f"{k:>22}" for k in keys) + f"{'other_fp':>12}")
        for k in range(1, N_CLASSES):
            row = [
                np.mean([p["stake"][k][kk] for p in per_month])
                for kk in keys + ["e_otherfraud_called"]
            ]
            print(f"{CLASSES[k]:<20}" + "".join(f"{v:>22.5f}" for v in row))
        tot = {
            kk: float(
                np.mean([sum(p["stake"][k][kk] for k in range(1, N_CLASSES)) for p in per_month])
            )
            for kk in keys + ["e_otherfraud_called"]
        }
        print("total stake", json.dumps({k: round(v, 5) for k, v in tot.items()}))
        for k in range(1, N_CLASSES):
            c = {}
            for p in per_month:
                for pair, n in p["wrongtype_pairs"][k].items():
                    c[pair] = c.get(pair, 0) + n
            d = {}
            for p in per_month:
                for src, n in p["legit_called_by"][k].items():
                    d[src] = d.get(src, 0) + n
            b = {}
            for p in per_month:
                for src, n in p["evidence_legit_by"][k].items():
                    b[src] = b.get(src, 0) + n
            print(f"  {CLASSES[k]:<18} (b) by {b}  (c) to {c}  (d) by {d}")
    return report


def features(led) -> dict[str, np.ndarray]:
    ts = led["timestamp"]
    hour = (ts.dt.hour() + ts.dt.minute() / 60).to_numpy()
    return {
        "hour": band(hour, [0, 3, 5.5, 6, 9, 12, 15, 18, 21, 22, 24]),
        "day": band(ts.dt.day().to_numpy().astype(float), [1, 6, 11, 16, 21, 26, 32]),
        "amount": band(led["amount"].to_numpy(), [0, 100, 500, 2000, 10000, 50000, 1e12]),
        "app_version": band(
            np.nan_to_num(led["app_version"].to_numpy(), nan=-1), [-2, 0, 4, 5, 6, 100]
        ),
        "device_age": band(
            np.nan_to_num(led["device_age_days"].to_numpy(), nan=-1), [-2, 0, 1, 7, 30, 180, 1e9]
        ),
        "receiver_type": np.unique(
            led["receiver_type"].fill_null("?").to_numpy(), return_inverse=True
        )[1],
    }


def cmd_october(run: str) -> dict:
    led, proba, calls = load(run)
    te = led["is_test"].to_numpy()
    tr = ~te
    y = led["y"].to_numpy().astype(np.int64)
    feats = features(led)
    report = {}
    for fname, b in feats.items():
        nb = int(b.max()) + 1
        res = profile_shift(b, y, calls, tr, te, nb, N_CLASSES)
        report[fname] = res
        print(
            f"== {fname}: largest October departures from out-of-fold calls (share of class calls)"
        )
        for k, row in sorted(res.items(), key=lambda kv: -kv[1]["tv_oct_oof"])[:4]:
            print(
                f"  {CLASSES[int(k)]:<18} TV(oct,oof)={row['tv_oct_oof']:.3f}"
                f" TV(oof,label)={row['tv_oof_lab']:.3f}"
                f"  oct={row['oct']}  oof={row['oof']}  lab={row['lab']}"
            )
    # sim_swap night-window lead: sim-like evidence rows by hour, called-sim rate
    ev = evidence(led)
    ts = led["timestamp"]
    hour = (ts.dt.hour() + ts.dt.minute() / 60).to_numpy()
    step = anchors.family_columns(led, ["mrk_step025"])["mrk_step025"].fill_null(0).to_numpy() > 0
    simlike = (ev["anchor"] == anchors.SIM) | (ev["tmpl"] == 2) | step
    bins = {
        "00-05:30": (hour < 5.5),
        "05:30-06": (hour >= 5.5) & (hour < 6),
        "06-22": (hour >= 6) & (hour < 22),
        "22-24": hour >= 22,
    }
    sim = {}
    for part, mask in (("train", tr), ("october", te)):
        sim[part] = {}
        for bn, hb in bins.items():
            s = simlike & mask & hb
            n = int(s.sum())
            sim[part][bn] = {
                "rows": n,
                "called_sim": float((calls[s] == anchors.SIM).mean()) if n else None,
                "called_fraud": float((calls[s] > 0).mean()) if n else None,
            }
            if part == "train" and n:
                sim[part][bn]["label_sim"] = float((y[s] == anchors.SIM).mean())
        # every row's sim_swap call share by hour band (not only evidence)
    print("== sim-like evidence rows by hour:", json.dumps(sim, indent=1))
    allsim = {}
    for part, mask in (("train_label", tr), ("train_oof", tr), ("october", te)):
        v = y if part == "train_label" else calls
        s = mask & (v == anchors.SIM)
        allsim[part] = {bn: round(float((hb[s]).mean()), 4) for bn, hb in bins.items()}
    print("== hour mix of sim_swap rows:", json.dumps(allsim))
    report["sim_night"] = sim
    report["sim_hour_mix"] = allsim
    return report


def cmd_rule(run: str, rule: str, draws: int = 2000) -> dict:
    """The blend's calls with and without one extra rule: per-month scores on the three targets,
    a paired event bootstrap (95 % bounds) and the number of October calls changed."""
    import scipy.sparse as sp

    from ringtrace import rules as rulebook
    from ringtrace.blend import month_scores
    from ringtrace.compare import _counts, _macro
    from ringtrace.teacher import union_events

    led, proba, base = load(run)
    chal = rulebook.apply([rule], base, proba, led, log=print)
    te = led["is_test"].to_numpy()
    tr = np.flatnonzero(~te)
    period = led["period"].to_numpy()[tr]
    targets = {"given": led["y"].to_numpy().astype(np.int64)[tr]}
    for t, w in (("cleaned", "label"), ("gen", "structured")):
        targets[t] = silver.load(led, w).astype(np.int64)[tr]
    sa, sb = month_scores(base[tr], targets, period), month_scores(chal[tr], targets, period)
    ev = union_events(led.height)[tr]
    groups = np.unique(np.where(ev >= 0, ev, -1 - np.arange(len(tr))), return_inverse=True)[1]
    rng = np.random.default_rng(0)
    rep = {"rule": rule, "october_changed": int((base[te] != chal[te]).sum())}
    rep["train_changed"] = int((base[tr] != chal[tr]).sum())
    for t, y in targets.items():
        deltas, point = np.zeros(draws), 0.0
        for m in MONTHS:
            s = period == m
            gm = np.unique(groups[s], return_inverse=True)[1]
            ng = int(gm.max()) + 1
            ca, cb = _counts(gm, y[s], base[tr][s], ng), _counts(gm, y[s], chal[tr][s], ng)
            w = sp.csr_matrix(rng.poisson(1.0, size=(draws, ng)).astype(np.float64))
            deltas += (
                _macro(np.asarray((w @ cb).todense())) - _macro(np.asarray((w @ ca).todense()))
            ) / 3
        diff = [b - a for a, b in zip(sa[t]["months"], sb[t]["months"], strict=True)]
        point = float(np.mean(diff))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rep[t] = {
            "months": diff,
            "mean": point,
            "boot_mean": float(deltas.mean()),
            "lo": float(lo),
            "hi": float(hi),
        }
        print(
            f"[rule] {t:<8} per month "
            + " / ".join(f"{d:+.5f}" for d in diff)
            + f"  mean {point:+.5f}  95% [{lo:+.5f}, {hi:+.5f}]"
        )
    print(f"[rule] rows changed: train {rep['train_changed']}, October {rep['october_changed']}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["anatomy", "october", "rule"])
    ap.add_argument("--rule", default="blank_merchant_qr")
    ap.add_argument("--run", default="blend")
    a = ap.parse_args(argv)
    if a.cmd == "rule":
        rep = cmd_rule(a.run, a.rule)
    else:
        rep = cmd_anatomy(a.run) if a.cmd == "anatomy" else cmd_october(a.run)
    out = paths.out_dir() / "fresh"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.cmd}-{a.run}{'-' + a.rule if a.cmd == 'rule' else ''}.json").write_text(
        json.dumps(rep, indent=1, default=str)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
