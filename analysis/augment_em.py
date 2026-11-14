"""Prior correction of the best file's decoded probabilities, tested on months resampled to a mix.

The best file's out-of-fold probabilities (``closelook/prep.npz``) are decoded per month with the
fixed point and the two row-level rules. Each held-out month is also resampled, unit by unit (an
event, or a row on its own), towards a target class mix: a unit of fraud class k appears
``floor(r_k)`` times plus once more with probability ``r_k - floor(r_k)``; legit units appear
once. A unit's class is the majority of its cleaned labels.

For every month, plain and resampled, three decodes are compared:

* ``none``: the probabilities as they are;
* ``em``: ``ringtrace.mix.em_priors`` on the whole 12-class vector, referenced to the mean
  out-of-fold probability of the other held-out months;
* ``em_types``: the same correction restricted to the fraud-type split. Each row keeps its
  P(legit), and the 11 fraud classes are reweighted by an EM prior of the type mix in which every
  row counts with its fraud mass;
* ``known`` and ``known_types``: no estimate at all; the class odds are multiplied by the target
  mix itself (the ratios the month was resampled with, or October's estimate), on the whole
  vector or on the type split only. On a plain month the same ratios are applied, which is the
  cost of the correction when the month does not carry that mix.

The scored month uses the mean over all three held-out months as its reference; its October calls
are counted against the best file's (aggregates only) and, with ``--write``, saved as candidate
files. Only aggregates are printed.

    python analysis/augment_em.py [--draws 8] [--mix october] [--write]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ringtrace import paths
from ringtrace.classes import CLASSES, CODE, N_CLASSES
from ringtrace.decode import fixed_point
from ringtrace.mix import em_priors

MONTHS = (7, 8, 9)
TARGETS = ("given", "cleaned", "gen")
FRAUD_SCALE = 0.875
# October over train, class volume including ordinary rows, estimated from the counter segments.
OCTOBER_MIX = {
    "mule_account": 1.34,
    "refund_abuse": 1.26,
    "phishing_collect": 1.14,
    "merchant_collusion": 1.12,
    "layering_chain": 1.09,
    "qr_swap": 1.06,
    "velocity_bot": 1.04,
    "sim_swap": 1.01,
    "fake_cashback": 0.96,
    "account_takeover": 0.95,
    "loan_app_extortion": 0.77,
}
VARIANTS = ("none", "em", "em_types", "known", "known_types")
TAG = {"given": "gv", "cleaned": "cl", "gen": "gn"}


def ratio_vector(mix: dict[str, float]) -> np.ndarray:
    r = np.ones(N_CLASSES)
    for name, v in mix.items():
        r[CODE[name]] = float(v)
    return r


def macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    cm = np.bincount(truth * N_CLASSES + pred, minlength=N_CLASSES * N_CLASSES).reshape(
        N_CLASSES, N_CLASSES
    )
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def unit_classes(units: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inverse index of each row's unit, majority class of each unit; ties take the smaller)."""
    ids, inv = np.unique(units, return_inverse=True)
    counts = np.bincount(inv * N_CLASSES + truth, minlength=len(ids) * N_CLASSES).reshape(
        len(ids), N_CLASSES
    )
    return inv, counts.argmax(axis=1)


def resample_rows(inv: np.ndarray, ucls: np.ndarray, ratio: np.ndarray, seed: int) -> np.ndarray:
    """Row positions of one resampled month (rows repeat when their unit is drawn twice)."""
    r = ratio[ucls]
    base = np.floor(r)
    extra = np.random.default_rng(seed).random(len(ucls)) < (r - base)
    copies = (base + extra).astype(np.int64)
    return np.repeat(np.arange(len(inv)), copies[inv])


def em_types(post: np.ndarray, ref: np.ndarray, max_iter: int = 200, tol: float = 1e-8):
    """EM prior correction of the fraud-type split only; every row keeps its P(legit)."""
    post = np.asarray(post, dtype=np.float64)
    mass = post[:, 1:].sum(axis=1)
    cond = post[:, 1:] / np.maximum(mass, 1e-300)[:, None]
    ref_t = np.asarray(ref[1:], dtype=np.float64)
    ref_t = ref_t / ref_t.sum()
    q = ref_t.copy()
    w = mass / mass.sum()
    adj = cond
    for _ in range(max_iter):
        weighted = cond * (q / ref_t)[None, :]
        adj = weighted / weighted.sum(axis=1, keepdims=True)
        q_new = (adj * w[:, None]).sum(axis=0)
        shift = float(np.max(np.abs(q_new - q)))
        q = q_new
        if shift <= tol:
            break
    out = post.copy()
    out[:, 1:] = adj * mass[:, None]
    return q, out


def known_shift(p: np.ndarray, ratio: np.ndarray, types_only: bool) -> np.ndarray:
    """Multiply class odds by ``ratio``; with ``types_only`` each row keeps its P(legit)."""
    p = np.asarray(p, dtype=np.float64)
    if not types_only:
        w = p * ratio[None, :]
        return w / w.sum(axis=1, keepdims=True)
    mass = p[:, 1:].sum(axis=1, keepdims=True)
    w = p[:, 1:] * ratio[None, 1:]
    out = p.copy()
    out[:, 1:] = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300) * mass
    return out


def correct(
    variant: str, p: np.ndarray, ref: np.ndarray, ratio: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    if variant == "none":
        return p, None
    if variant == "em":
        q, adj = em_priors(p, ref)
        return adj, q
    if variant == "em_types":
        q, adj = em_types(p, ref)
        return adj, q
    if variant in ("known", "known_types"):
        return known_shift(p, ratio, variant == "known_types"), None
    raise ValueError(variant)


def apply_rules(pred: np.ndarray, p: np.ndarray, rt: dict, rows: np.ndarray) -> np.ndarray:
    """The anchored template rule, then blank_merchant_qr with this probability's fraud argmax.

    The anchored rule's target per row depends only on the counter and the templates, so it is
    read from the cached rule targets; the QR rule's class is recomputed from ``p``.
    """
    tmpl = rt["tmpl"][rows]
    target = rt["target"][rows]
    out = pred.copy()
    inpair = (tmpl > 0) & ((pred == rt["pair0"][tmpl]) | (pred == rt["pair1"][tmpl]))
    out[inpair] = target[inpair]
    q = rt["qr"][rows] & (out == 0)
    out[q] = p[q, 1:].argmax(axis=1) + 1
    return out


def decode(p: np.ndarray, rt: dict, rows: np.ndarray) -> np.ndarray:
    return apply_rules(fixed_point(p, fraud_scale=FRAUD_SCALE), p, rt, rows)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--write", action="store_true", help="write October candidate files")
    ap.add_argument("--prefix", default="augment")
    args = ap.parse_args(argv)
    t0 = time.time()
    src = paths.out_dir() / "closelook"
    z = np.load(src / "prep.npz")
    d = {k: z[k] for k in ("proba", "calls", "is_test", "period", "groups", *TARGETS)}
    zr = np.load(src / "rules.npz")
    rt = {k: zr[k] for k in zr.files}
    is_test = d["is_test"].astype(bool)
    tr = np.flatnonzero(~is_test)
    te = np.flatnonzero(is_test)
    per_tr = d["period"][tr]
    proba = d["proba"]
    ratio = ratio_vector(OCTOBER_MIX)
    month_rows = {m: tr[per_tr == m] for m in MONTHS}
    month_pos = {m: np.flatnonzero(per_tr == m) for m in MONTHS}
    mean_of = {m: proba[month_rows[m]].mean(axis=0) for m in MONTHS}
    rep: dict = {"mix": OCTOBER_MIX, "draws": args.draws, "plain": {}, "resampled": {}}

    # sanity: the plain decode reproduces the best file's calls on every train row
    for m in MONTHS:
        r = month_rows[m]
        same = decode(proba[r], rt, r)
        rep.setdefault("reproduced_rows_differing", {})[str(m)] = int((same != d["calls"][r]).sum())

    for m in MONTHS:
        r, pos = month_rows[m], month_pos[m]
        ref = np.mean([mean_of[o] for o in MONTHS if o != m], axis=0)
        truths = {t: d[t][pos] for t in TARGETS}
        p = proba[r]
        plain = {}
        for v in VARIANTS:
            # On a plain month the known shift applies the target mix to a month that does not
            # carry it: the cost of correcting when the estimate is wrong.
            adj, q = correct(v, p, ref, ratio)
            pred = decode(adj, rt, r)
            plain[v] = {t: macro_f1(truths[t], pred) for t in TARGETS}
            plain[v]["rows_changed"] = int((pred != d["calls"][r]).sum())
            if q is not None:
                plain[v]["prior_ratio"] = [float(x) for x in (q / ref if v == "em" else q)]
        rep["plain"][str(m)] = plain
        print(
            f"plain {m}: "
            + " | ".join(
                f"{v}: "
                + " ".join(f"{TAG[t]} {plain[v][t] - plain['none'][t]:+.5f}" for t in TARGETS)
                + f" n {plain[v]['rows_changed']}"
                for v in VARIANTS[1:]
            ),
            flush=True,
        )

        inv, ucls = unit_classes(d["groups"][pos], d["cleaned"][pos])
        draws = []
        for s in range(args.draws):
            idx = resample_rows(inv, ucls, ratio, seed=1000 * m + s)
            rr = r[idx]
            pp = p[idx]
            res = {}
            for v in VARIANTS:
                adj, q = correct(v, pp, ref, ratio)
                pred = decode(adj, rt, rr)
                res[v] = {t: macro_f1(truths[t][idx], pred) for t in TARGETS}
                if v == "none":
                    base_pred = pred
                res[v]["rows_changed"] = int((pred != base_pred).sum())
            share_true = np.bincount(d["cleaned"][pos][idx], minlength=N_CLASSES) / len(idx)
            share_plain = np.bincount(d["cleaned"][pos], minlength=N_CLASSES) / len(pos)
            res["true_ratio"] = [float(x) for x in share_true / np.maximum(share_plain, 1e-12)]
            draws.append(res)
        summ = {}
        for v in VARIANTS[1:]:
            summ[v] = {}
            for t in TARGETS:
                diffs = np.array([dr[v][t] - dr["none"][t] for dr in draws])
                summ[v][t] = {
                    "mean": float(diffs.mean()),
                    "min": float(diffs.min()),
                    "max": float(diffs.max()),
                }
        rep["resampled"][str(m)] = {"draws": draws, "change": summ}
        print(
            f"resampled {m}: "
            + " | ".join(
                f"{v}: "
                + " ".join(
                    f"{TAG[t]} {summ[v][t]['mean']:+.5f} [{summ[v][t]['min']:+.5f},"
                    f"{summ[v][t]['max']:+.5f}]"
                    for t in TARGETS
                )
                for v in VARIANTS[1:]
            ),
            flush=True,
        )

    # October: reference is the mean over all three held-out months
    ref_all = proba[tr].mean(axis=0)
    p_oct = proba[te]
    base_calls = d["calls"][te]
    rep["october"] = {}
    for v in VARIANTS:
        adj, q = correct(v, p_oct, ref_all, ratio)
        pred = decode(adj, rt, te)
        changed = pred != base_calls
        keys, n = np.unique(base_calls[changed] * N_CLASSES + pred[changed], return_counts=True)
        moves = {
            f"{CLASSES[k // N_CLASSES]} -> {CLASSES[k % N_CLASSES]}": int(c)
            for k, c in zip(keys, n, strict=True)
        }
        rep["october"][v] = {
            "rows_changed": int(changed.sum()),
            "moves": moves,
            "calls": {
                CLASSES[k]: int(c) for k, c in enumerate(np.bincount(pred, minlength=N_CLASSES))
            },
        }
        if q is not None:
            rep["october"][v]["prior_ratio"] = [float(x) for x in (q / ref_all if v == "em" else q)]
        print(f"october {v}: {int(changed.sum())} rows changed {moves}", flush=True)
        if args.write and v != "none":
            write_candidate(pred, te, f"{args.prefix}-{v.replace('_', '')}", rep["october"][v])
    rep["wall_s"] = time.time() - t0
    out = paths.out_dir() / "augment"
    out.mkdir(parents=True, exist_ok=True)
    (out / "em.json").write_text(json.dumps(rep, indent=1))
    print(f"done in {rep['wall_s']:.0f}s")
    return rep


def write_candidate(pred: np.ndarray, te: np.ndarray, name: str, info: dict) -> None:
    """October file in the best file's id order, checked against the sample submission."""
    import polars as pl

    best = pl.read_csv(paths.out_dir() / "candidates" / "blend-tabmF-bq.csv", infer_schema=False)
    led_path = sorted((paths.cache_dir() / "tables").glob("ledger-*.parquet"))
    if len(led_path) != 1:
        raise SystemExit("expected one ledger file")
    ids = pl.read_parquet(led_path[0], columns=["txn_id", "is_test"]).filter(pl.col("is_test"))
    if ids.height != len(te):
        raise SystemExit("ledger and probabilities disagree on the scored rows")
    calls = pl.DataFrame({"id": ids["txn_id"].cast(pl.Utf8), "label": np.asarray(CLASSES)[pred]})
    sub = best.select("id").join(calls, on="id", how="left")
    if sub["label"].null_count() or sub.height != best.height:
        raise SystemExit("candidate does not cover the best file's ids")
    sample = pl.read_csv(paths.data_dir() / "sample_submission.csv", infer_schema=False)
    if set(sample["id"].to_list()) != set(sub["id"].to_list()):
        raise SystemExit("candidate ids differ from the sample submission")
    path = paths.out_dir() / "candidates" / f"{name}.csv"
    sub.write_csv(path)
    info["file"] = str(path.name)
    info["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    print(f"wrote {path.name} sha256 {info['sha256']}")


if __name__ == "__main__":
    main()
