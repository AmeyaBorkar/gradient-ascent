"""Label-free candidates on top of the five-member blend, scored leave-one-month-out.

    python -m ringtrace.squeeze lomo                 # every rule candidate vs the blend
    python -m ringtrace.squeeze select               # nested greedy ensemble selection
    python -m ringtrace.squeeze audit                # sim_swap / takeover edge, aggregates only
    python -m ringtrace.squeeze final --rules a,b    # the October file of an adopted combination

The base is the blend exactly as ``ringtrace.blend`` builds it: the equal-weight mean of the
members' probabilities, the fixed point per month (October on its own), then the anchored
template rule. A rule candidate is a different rule list on the same probabilities, so its
calls differ from the base's only where the rule acts. Scores are macro-F1 per held-out month on
the given, cleaned and gen-only targets, with a paired Poisson bootstrap over label-free events
(95 % interval, mean of the three months). Only aggregates of October are ever reported.

Nested selection (Caruana et al., 2004): greedy forward selection with replacement over the
members, scored on the two training months of each fold (given labels, decoded as the blend is)
and applied to the held-out month; the October weights are selected on all three months.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace.blend import decode_all, load_member, month_scores, weighted_mean
from ringtrace.classes import CLASSES, N_CLASSES

TRAIN_MONTHS = (7, 8, 9)
MEMBERS = {
    "champion": "blend-champion-m1",
    "lgbm": "lgbm",
    "catboost": "catboost",
    "clean-event": "clean-event",
    "clean-l2": "clean-l2",
}
DECODER = {"fraud_scale": 0.875}
BASE_RULES = ["anchored_templates"]
CANDIDATES = {
    "event_types": ["event_types", "anchored_templates"],
    "event_types_after": ["anchored_templates", "event_types"],
    "anchor_rescue": ["anchored_templates", "anchor_rescue"],
}
# every family column the rules read, loaded once so each decode skips the store
RULE_COLUMNS = [
    *(f"{p}_{k}" for p in ("bot", "drn", "mf", "col", "rng", "rep") for k in ("shape", "event")),
    "mrk_p2p_category",
    "mrk_qr_p2p_nocat",
    "mrk_bal102",
    "mrk_step025",
    "mrk_step035",
    "ctr_block",
    "ctr_seg",
    "ctr_conf",
    "ctr_dev_rows",
]


# ---------------------------------------------------------------------------------------------
# blending


def rank_mean(arrays: list[np.ndarray], weights: list[float], period: np.ndarray) -> np.ndarray:
    """Weighted mean of per-month, per-class ranks, mapped back onto the probability mean's own
    values in that month and class (so the decoder sees probabilities), rows renormalised."""
    from scipy.stats import rankdata

    pm = weighted_mean(arrays, weights)
    out = np.zeros_like(pm)
    for m in np.unique(period):
        s = np.flatnonzero(period == m)
        n = len(s)
        for k in range(pm.shape[1]):
            r = sum(float(w) * rankdata(a[s, k]) for a, w in zip(arrays, weights, strict=True))
            idx = np.clip(np.ceil(r - 1e-9).astype(np.int64) - 1, 0, n - 1)
            out[s, k] = np.sort(pm[s, k])[idx]
    out = np.maximum(out, 1e-12)
    return out / out.sum(axis=1, keepdims=True)


def combine(arrays, weights, period, mode: str) -> np.ndarray:
    if mode == "prob":
        return weighted_mean(arrays, weights)
    if mode == "rank":
        return rank_mean(arrays, weights, period)
    raise ValueError(mode)


def greedy(evaluate, n: int, steps: int = 10) -> tuple[np.ndarray, list[float]]:
    """Forward selection with replacement: add the member whose extra copy scores best, stop when
    no addition improves. Ties go to the earlier member. Returns copy counts and the path."""
    counts = np.zeros(n, dtype=np.int64)
    best, path = -np.inf, []
    for _ in range(steps):
        trials = []
        for j in range(n):
            c = counts.copy()
            c[j] += 1
            trials.append((evaluate(tuple(int(x) for x in c)), -j))
        score, neg_j = max(trials)
        if score <= best:
            break
        best = score
        counts[-neg_j] += 1
        path.append(float(score))
    return counts, path


# ---------------------------------------------------------------------------------------------
# data


class Data:
    """The ledger with the rules' columns attached, the targets and the members' arrays."""

    def __init__(self, runs_dir: Path, members: dict[str, str] = MEMBERS, log=print):
        from ringtrace import silver, store
        from ringtrace.evidence import anchors

        led = store.ledger()
        cols = anchors.family_columns(led, RULE_COLUMNS)
        self.led = led.with_columns(cols)
        self.is_test = led["is_test"].to_numpy()
        self.period = led["period"].to_numpy()
        self.tr = np.flatnonzero(~self.is_test)
        self.te = np.flatnonzero(self.is_test)
        self.targets = {"given": led["y"].to_numpy().astype(np.int64)[self.tr]}
        for tname, which in (("cleaned", "label"), ("gen", "structured")):
            arr = silver.load(led, which)
            if arr is not None:
                self.targets[tname] = arr.astype(np.int64)[self.tr]
        from ringtrace.rules import label_free_events

        ev = label_free_events(self.led)[self.tr]
        self.groups = np.where(ev >= 0, ev, -1 - np.arange(len(self.tr)))
        self.names = list(members)
        self.arrays = []
        for name, run in members.items():
            oof, test = load_member(runs_dir / run, len(self.tr), len(self.te))
            full = np.zeros((led.height, N_CLASSES))
            full[self.tr], full[self.te] = oof, test
            self.arrays.append(full)
            log(f"[squeeze] loaded {name} ({run})")

    def calls_on(self, rows: np.ndarray, proba_rows: np.ndarray, rules: list) -> np.ndarray:
        """Decode one month's (or October's) rows on their own, then the rules; other rows 0."""
        from ringtrace import rules as rulebook
        from ringtrace.decode import fixed_point

        pred = np.zeros(self.led.height, dtype=np.int64)
        pred[rows] = fixed_point(proba_rows, **DECODER)
        if rules:
            full = np.zeros((self.led.height, N_CLASSES))
            full[rows] = proba_rows
            pred = rulebook.apply(rules, pred, full, self.led, log=_quiet)
        return pred

    def calls(self, proba: np.ndarray, rules: list) -> np.ndarray:
        return decode_all(proba, self.period, self.is_test, DECODER, rules, self.led, log=_quiet)[
            "rules" if rules else "fixed_point"
        ]


def _quiet(*_):
    return None


def bootstrap95(d: Data, y: np.ndarray, a: np.ndarray, b: np.ndarray, draws=2000) -> dict:
    from ringtrace.compare import bootstrap

    r = bootstrap(d.groups, y, d.period[d.tr], a, b, draws=draws, q=(2.5, 97.5))
    return {"delta": r["delta"], "lo": r["p05"], "hi": r["p95"]}


def lomo_compare(d: Data, base: np.ndarray, chal: np.ndarray, draws=2000) -> dict:
    """Per-month scores of both on every target, and the event bootstrap of the change."""
    ptr = d.period[d.tr]
    sa = month_scores(base[d.tr], d.targets, ptr)
    sb = month_scores(chal[d.tr], d.targets, ptr)
    out = {}
    for t in d.targets:
        out[t] = {
            "base": sa[t]["months"],
            "challenger": sb[t]["months"],
            "change": [b - a for a, b in zip(sa[t]["months"], sb[t]["months"], strict=True)],
            **bootstrap95(d, d.targets[t], base[d.tr], chal[d.tr], draws),
        }
    out["rows_changed"] = {
        "train": int((base[d.tr] != chal[d.tr]).sum()),
        "october": int((base[d.te] != chal[d.te]).sum()),
    }
    out["never_loses"] = all(min(out[t]["change"]) >= 0 for t in d.targets)
    return out


def _line(name: str, r: dict) -> str:
    parts = [
        f"{t} "
        + " / ".join(f"{c:+.4f}" for c in r[t]["change"])
        + f" [{r[t]['lo']:+.4f}, {r[t]['hi']:+.4f}]"
        for t in r
        if isinstance(r[t], dict) and "change" in r[t]
    ]
    return f"[squeeze] {name}: " + "  ".join(parts) + f"  rows {r['rows_changed']}"


def pair_counts(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    """Number of rows per (from, to) class pair where the calls differ."""
    diff = a != b
    keys, n = np.unique(a[diff] * N_CLASSES + b[diff], return_counts=True)
    return {
        f"{CLASSES[k // N_CLASSES]} -> {CLASSES[k % N_CLASSES]}": int(c)
        for k, c in zip(keys, n, strict=True)
    }


# ---------------------------------------------------------------------------------------------
# commands


def cmd_lomo(d: Data, out: Path, log=print) -> dict:
    t0 = time.time()
    proba = weighted_mean(d.arrays, [1 / len(d.arrays)] * len(d.arrays))
    base = d.calls(proba, BASE_RULES)
    rep = {"base": month_scores(base[d.tr], d.targets, d.period[d.tr])}
    log("[squeeze] blend: " + json.dumps({t: v["mean"] for t, v in rep["base"].items()}))
    for name, rules in CANDIDATES.items():
        chal = d.calls(proba, rules)
        rep[name] = lomo_compare(d, base, chal)
        rep[name]["october_pairs"] = pair_counts(base[d.te], chal[d.te])
        log(_line(name, rep[name]))
    rep["wall_s"] = time.time() - t0
    (out / "lomo.json").write_text(json.dumps(rep, indent=1))
    return rep


def cmd_select(d: Data, out: Path, steps: int = 10, log=print) -> dict:
    t0 = time.time()
    n = len(d.arrays)
    ptr = d.period[d.tr]
    y = d.targets["given"]
    from ringtrace.scoring import macro_f1

    cache: dict = {}

    def month_score(mode: str, counts: tuple, m: int) -> float:
        key = (mode, counts, m)
        if key not in cache:
            s = np.flatnonzero(d.period == m)
            w = [c / sum(counts) for c in counts]
            used = [i for i in range(n) if counts[i]]
            p = combine([d.arrays[i][s] for i in used], [w[i] for i in used], d.period[s], mode)
            c = d.calls_on(s, p, BASE_RULES)
            cache[key] = macro_f1(y[ptr == m], c[d.tr][ptr == m])
        return cache[key]

    equal = tuple([1] * n)
    base_proba = weighted_mean(d.arrays, [1 / n] * n)
    base = d.calls(base_proba, BASE_RULES)
    rep: dict = {"members": d.names}
    for mode in ("prob", "rank"):
        nested = np.zeros(len(d.led), dtype=np.int64)
        folds = {}
        for m in TRAIN_MONTHS:
            fit = [k for k in TRAIN_MONTHS if k != m]
            counts, path = greedy(
                lambda c, fit=fit, mode=mode: float(
                    np.mean([month_score(mode, c, k) for k in fit])
                ),
                n,
                steps,
            )
            folds[m] = {"counts": counts.tolist(), "path": path}
            log(f"[squeeze] {mode} fold {m}: counts {counts.tolist()} path {path}")
            s = np.flatnonzero(d.period == m)
            used = [i for i in range(n) if counts[i]]
            w = [counts[i] / counts.sum() for i in used]
            p = combine([d.arrays[i][s] for i in used], w, d.period[s], mode)
            nested[s] = d.calls_on(s, p, BASE_RULES)[s]
        all_counts, all_path = greedy(
            lambda c, mode=mode: float(np.mean([month_score(mode, c, k) for k in TRAIN_MONTHS])),
            n,
            steps,
        )
        used = [i for i in range(n) if all_counts[i]]
        w = [all_counts[i] / all_counts.sum() for i in used]
        oct_p = combine([d.arrays[i][d.te] for i in used], w, d.period[d.te], mode)
        nested[d.te] = d.calls_on(d.te, oct_p, BASE_RULES)[d.te]
        r = lomo_compare(d, base, nested)
        r["folds"] = folds
        r["october_counts"] = all_counts.tolist()
        r["october_pairs"] = pair_counts(base[d.te], nested[d.te])
        rep[f"greedy_{mode}"] = r
        log(_line(f"greedy_{mode}", r) + f" october counts {all_counts.tolist()}")
    # the plain equal-weight rank blend of all five, no selection
    rank_eq = d.calls(combine(d.arrays, [1 / n] * n, d.period, "rank"), BASE_RULES)
    r = lomo_compare(d, base, rank_eq)
    r["october_pairs"] = pair_counts(base[d.te], rank_eq[d.te])
    rep["rank_equal"] = r
    log(_line("rank_equal", r))
    rep["equal_counts"] = list(equal)
    rep["wall_s"] = time.time() - t0
    (out / "select.json").write_text(json.dumps(rep, indent=1))
    return rep


def cmd_audit(d: Data, out: Path, log=print) -> dict:
    """Drain rows of the sim_swap / takeover pair against the counter's cut: aggregates only."""
    SIM, ATO = 4, 5
    proba = weighted_mean(d.arrays, [1 / len(d.arrays)] * len(d.arrays))
    base = d.calls(proba, BASE_RULES)
    led = d.led
    drn = led["drn_shape"].fill_null(0).to_numpy()
    block = led["ctr_block"].fill_null(0).to_numpy()
    seg = led["ctr_seg"].cast(pl.Float64).fill_null(-1).to_numpy() + 1
    conf = led["ctr_conf"].cast(pl.Float64).fill_null(-1).to_numpy()
    rep: dict = {}
    for split, rows in (("train", d.tr), ("october", d.te)):
        sp_rep = {}
        for shape in (1, 2, 3, 4):
            r = rows[drn[rows] == shape]
            in_pair = (block[r] == 2) & np.isin(seg[r], (SIM, ATO))
            c = conf[r][in_pair]
            sp_rep[f"drain{shape}"] = {
                "rows": int(len(r)),
                "calls_sim": int((base[r] == SIM).sum()),
                "calls_ato": int((base[r] == ATO).sum()),
                "calls_legit": int((base[r] == 0).sum()),
                "fraud_block_pair_segment": int(in_pair.sum()),
                "conf_bins": np.histogram(c, bins=[0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01])[
                    0
                ].tolist(),
                "member_spread_rows": int(
                    (
                        np.std([a[r, SIM] / (a[r, SIM] + a[r, ATO] + 1e-12) for a in d.arrays], 0)
                        > 0.1
                    ).sum()
                ),
            }
            if split == "train":
                y = d.targets["given"][np.searchsorted(d.tr, r)]
                sp_rep[f"drain{shape}"]["given_sim"] = int((y == SIM).sum())
                sp_rep[f"drain{shape}"]["given_ato"] = int((y == ATO).sum())
        rep[split] = sp_rep
        log(f"[squeeze] audit {split}: {json.dumps(sp_rep)}")
    # moving the anchored rule's cut: rows that change, and the LOMO effect
    rep["cut"] = {}
    for cut in (0.7, 0.75, 0.85, 0.9):
        rules = [{"name": "anchored_templates", "min_conf": cut}]
        c = d.calls(proba, rules)
        r = lomo_compare(d, base, c, draws=500)
        r["october_pairs"] = pair_counts(base[d.te], c[d.te])
        rep["cut"][str(cut)] = r
        log(_line(f"cut {cut}", r) + f" october {r['october_pairs']}")
    (out / "audit.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def cmd_final(d: Data, out: Path, rules: list, base_dir: Path, sample: Path, log=print) -> dict:
    from ringtrace.blend import write_submission

    proba = weighted_mean(d.arrays, [1 / len(d.arrays)] * len(d.arrays))
    base = d.calls(proba, BASE_RULES)
    chal = d.calls(proba, rules)
    out.mkdir(parents=True, exist_ok=True)
    sub = write_submission(
        out / "submission.csv", d.led["txn_id"].to_numpy()[d.te], chal[d.te], sample
    )
    rep = {
        "rules": rules,
        "submission_sha256": sub["sha256"],
        "submission_problems": sub["problems"],
        "test_calls": sub["calls"],
        "october_pairs_vs_blend": pair_counts(base[d.te], chal[d.te]),
        "lomo": lomo_compare(d, base, chal),
    }
    blend_sum = base_dir / "summary.json"
    if blend_sum.exists():
        rep["blend_sha256"] = json.loads(blend_sum.read_text()).get("submission_sha256")
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    log(f"[squeeze] final {rules}: sha256 {sub['sha256']} problems {sub['problems']}")
    log(f"[squeeze] October changes vs the blend: {rep['october_pairs_vs_blend']}")
    return rep


def main(argv=None) -> dict:
    from ringtrace import paths, threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.squeeze")
    ap.add_argument("cmd", choices=["lomo", "select", "audit", "final"])
    ap.add_argument("--rules", default="", help="final: comma-separated rule list")
    ap.add_argument("--name", default="squeeze-final")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args(argv)
    runs = paths.out_dir() / "runs"
    out = paths.out_dir() / "squeeze"
    out.mkdir(parents=True, exist_ok=True)
    d = Data(runs)
    if args.cmd == "lomo":
        return cmd_lomo(d, out)
    if args.cmd == "select":
        return cmd_select(d, out, args.steps)
    if args.cmd == "audit":
        return cmd_audit(d, out)
    return cmd_final(
        d,
        runs / args.name,
        args.rules.split(","),
        runs / "blend",
        paths.data_dir() / "sample_submission.csv",
    )


if __name__ == "__main__":
    main()
