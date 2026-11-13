"""A pretrained in-context tabular model as a blend member, re-scoring the champion's doubtful rows.

    python -m ringtrace.icl lomo --name <run> [--model tabpfn-v2] [--k 100] [--context 50000]
    python -m ringtrace.icl time --fold 9 [--queries 5000]        (one small fold, run twice)
    python -m ringtrace.icl octlike --name <run> [--seeds 13,14,15] [--sets 0,1,2]

The member is the champion everywhere except on its candidate rows, where an in-context model
gives the fraud probability:

* **Candidates** are the rows whose champion fraud probability, 1 - p(legit), is at least ``tau``
  (train: the champion's out-of-fold arrays, October: its October array). ``tau`` is chosen once,
  as the quantile that lets ``share`` of the train rows in.
* **Features** are the champion recipe's matrix, cut to the ``k`` columns with the largest
  univariate separation (|AUC - 0.5|) on the context rows alone.
* **Target** is fraud vs legit on the given labels. When the context is larger than the model's
  limit, every fraud row is kept and legit rows are drawn at rate ``q`` (seeded); the predicted
  fraud odds are then multiplied by ``q`` to undo the shifted prior.
* **Folds** are whole months: a held-out month's candidates are scored with the other two months'
  candidates as context; October uses all three train months.
* **12 classes**: p(fraud type k) = p_icl(fraud) x p_champ(k) / p_champ(fraud), p(legit) =
  1 - p_icl(fraud).

``lomo`` writes the standard run layout under ``$RINGTRACE_OUT/runs/<name>/`` (``oof.npy``,
``test.npy``, ``summary.json`` with the scores decoded like the champion). ``octlike`` writes
``$RINGTRACE_OUT/octlike/<name>_s<seed>_r<r>.npz`` for every champion cell: the context is the
frame's July + August candidates (selected by the champion's out-of-fold arrays), the queries are
the frame's September candidates under that cell's champion probabilities.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata

from ringtrace.classes import N_CLASSES

TRAIN_MONTHS = (7, 8, 9)
SHARE = 0.05
TOP_K = 100
MAX_CONTEXT = 50_000
QUERY_BATCH = 4096
SEED = 0
CHAMPION_RECIPE = "recipes/champion.json"
CHAMPION_RUN = "blend-champion-m1"

# a predictor factory takes a seed and returns an object with fit(X, y) / predict_proba(X)
Factory = Callable[[int], object]


# ---------------------------------------------------------------------------------------------
# pieces (pure numpy, tested on synthetic data)


def fraud_prob(proba: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(proba, dtype=np.float64)[:, 0]


def choose_tau(p_fraud_train: np.ndarray, share: float = SHARE) -> float:
    """The threshold that lets ``share`` of the train rows in (ties may add a few more)."""
    return float(np.quantile(p_fraud_train, 1.0 - share, method="higher"))


def auc_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|AUC - 0.5| of every column for the binary target ``y``; missing values take the
    column median, a constant column scores 0."""
    X = np.asarray(X, dtype=np.float64).copy()
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(np.where(np.isnan(X).all(0), 0.0, X), axis=0)
    holes = np.isnan(X)
    X[holes] = np.take(med, np.nonzero(holes)[1])
    pos = np.asarray(y).astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.zeros(X.shape[1])
    ranks = rankdata(X, axis=0)
    auc = (ranks[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.abs(auc - 0.5)


def top_features(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` best columns by ``auc_scores``, ties broken by column order."""
    s = auc_scores(X, y)
    order = np.lexsort((np.arange(len(s)), -s))
    return np.sort(order[: min(k, len(s))])


def subsample_context(y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, float]:
    """Context row indices within ``max_rows``: all fraud rows, legit rows at a seeded rate.
    Returns the indices (sorted) and the legit keep rate ``q``."""
    y = np.asarray(y).astype(bool)
    if len(y) <= max_rows:
        return np.arange(len(y)), 1.0
    fraud, legit = np.flatnonzero(y), np.flatnonzero(~y)
    room = max_rows - len(fraud)
    if room <= 0:
        raise ValueError(f"{len(fraud)} fraud rows alone exceed the context limit {max_rows}")
    rng = np.random.default_rng(seed)
    keep = rng.choice(legit, size=room, replace=False)
    return np.sort(np.r_[fraud, keep]), room / len(legit)


def prior_correct(p: np.ndarray, q: float) -> np.ndarray:
    """Fraud probability from a model whose context kept legit rows at rate ``q``."""
    p = np.asarray(p, dtype=np.float64)
    if q == 1.0:
        return p
    return q * p / (q * p + (1.0 - p))


def compose(p_fraud: np.ndarray, champ: np.ndarray) -> np.ndarray:
    """12-class rows: legit 1 - p_fraud, each fraud type p_fraud x the champion's share."""
    champ = np.asarray(champ, dtype=np.float64)
    tot = champ[:, 1:].sum(1, keepdims=True)
    cond = np.where(tot > 0, champ[:, 1:] / np.maximum(tot, 1e-300), 1.0 / (N_CLASSES - 1))
    p = np.clip(np.asarray(p_fraud, dtype=np.float64), 0.0, 1.0)[:, None]
    out = np.concatenate([1.0 - p, p * cond], axis=1)
    return out / out.sum(1, keepdims=True)


def clean_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).copy()
    X[~np.isfinite(X)] = np.nan
    return X


def score_queries(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_qry: list[np.ndarray],
    factory: Factory,
    k: int,
    max_context: int,
    seed: int,
    log=print,
) -> tuple[list[np.ndarray], dict]:
    """Fit one in-context model on the context and return the prior-corrected fraud probability
    for each query block, plus what was done (features, context size, q, timings)."""
    t0 = time.time()
    feats = top_features(X_ctx, y_ctx, k)
    idx, q = subsample_context(y_ctx, max_context, seed)
    model = factory(seed)
    model.fit(clean_matrix(X_ctx[idx][:, feats]), np.asarray(y_ctx)[idx].astype(np.int64))
    t_fit = time.time() - t0
    outs = []
    for Xq in X_qry:
        Xq = clean_matrix(Xq[:, feats])
        p = np.zeros(len(Xq))
        for a in range(0, len(Xq), QUERY_BATCH):
            p[a : a + QUERY_BATCH] = model.predict_proba(Xq[a : a + QUERY_BATCH])[:, 1]
        outs.append(prior_correct(p, q))
    info = {
        "features": feats.tolist(),
        "context_rows": int(len(idx)),
        "context_fraud": int(np.asarray(y_ctx)[idx].sum()),
        "context_all": int(len(y_ctx)),
        "legit_rate": float(q),
        "queries": [int(len(x)) for x in X_qry],
        "fit_s": t_fit,
        "total_s": time.time() - t0,
    }
    log(
        f"  context {info['context_rows']} ({info['context_fraud']} fraud, q {q:.3f}), "
        f"queries {info['queries']}, {info['total_s']:.0f}s"
    )
    return outs, info


def lomo_member(
    X: np.ndarray,
    y_fraud: np.ndarray,
    period: np.ndarray,
    is_test: np.ndarray,
    champ: np.ndarray,
    factory: Factory,
    share: float = SHARE,
    k: int = TOP_K,
    max_context: int = MAX_CONTEXT,
    seed: int = SEED,
    log=print,
) -> tuple[np.ndarray, dict]:
    """The member's probabilities over every ledger row (``champ`` holds the champion's train
    out-of-fold and October rows in ledger order) and a report."""
    pf = fraud_prob(champ)
    tau = choose_tau(pf[~is_test], share)
    cand = pf >= tau
    train = ~is_test
    y_fraud = np.asarray(y_fraud).astype(np.int64)
    report: dict = {"tau": tau, "share": share, "k": k, "max_context": max_context, "seed": seed}
    report["candidates"] = {str(m): int((cand & (period == m)).sum()) for m in TRAIN_MONTHS}
    report["candidates"]["october"] = int((cand & is_test).sum())
    fraud_rows = train & (y_fraud == 1)
    report["fraud_covered"] = float((cand & fraud_rows).sum() / max(fraud_rows.sum(), 1))
    report["fraud_covered_by_month"] = {
        str(m): float(
            (cand & fraud_rows & (period == m)).sum() / max((fraud_rows & (period == m)).sum(), 1)
        )
        for m in TRAIN_MONTHS
    }
    log(
        f"[icl] tau {tau:.5f}: candidates {report['candidates']}, "
        f"fraud rows covered {report['fraud_covered']:.4f}"
    )
    out = np.asarray(champ, dtype=np.float64).copy()
    report["folds"] = {}
    folds = [(str(m), train & (period != m), train & (period == m)) for m in TRAIN_MONTHS]
    folds.append(("october", train, is_test))
    for name, ctx_rows, qry_rows in folds:
        ctx = np.flatnonzero(ctx_rows & cand)
        qry = np.flatnonzero(qry_rows & cand)
        log(f"[icl] fold {name}")
        (p,), info = score_queries(
            X[ctx], y_fraud[ctx], [X[qry]], factory, k, max_context, seed, log=log
        )
        out[qry] = compose(p, champ[qry])
        report["folds"][name] = info
    return out, report


# ---------------------------------------------------------------------------------------------
# models


class TabPFNv2:
    """TabPFN v2 classifier through the ``tabpfn`` package (v2 checkpoint, GPU, seeded)."""

    def __init__(self, seed: int, n_estimators: int | str = "auto"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        self.clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device="cuda",
            ignore_pretraining_limits=True,
            random_state=seed,
            n_estimators=n_estimators,
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


MODELS: dict[str, Callable[..., object]] = {"tabpfn-v2": TabPFNv2}


# ---------------------------------------------------------------------------------------------
# I/O and scoring


def _champion_full(led, run: str) -> np.ndarray:
    from ringtrace import paths

    d = paths.out_dir() / "runs" / run
    is_test = led["is_test"].to_numpy()
    full = np.zeros((led.height, N_CLASSES))
    full[~is_test] = np.load(d / "oof.npy")
    full[is_test] = np.load(d / "test.npy")
    return full


def _targets(led, tr_idx) -> dict[str, np.ndarray]:
    from ringtrace import silver

    targets = {"given": led["y"].to_numpy().astype(np.int64)[tr_idx]}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr[tr_idx]
    return targets


def scores_like_champion(led, proba: np.ndarray, recipe: dict, log=print) -> dict:
    """Fixed point per month, then the recipe's rules; macro-F1 per held-out month."""
    from ringtrace.blend import decode_all, month_scores

    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr_idx = np.flatnonzero(~is_test)
    targets = _targets(led, tr_idx)
    calls = decode_all(
        proba, period, is_test, recipe.get("decoder", {}), recipe.get("rules", []), led, log=log
    )
    return {st: month_scores(c[tr_idx], targets, period[tr_idx]) for st, c in calls.items()}


def cmd_lomo(args) -> dict:
    from ringtrace import paths, store
    from ringtrace import run as runner

    t0 = time.time()
    recipe = runner.load_recipe(args.recipe)
    led = store.ledger()
    X, cols = runner.feature_matrix(recipe)
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y_fraud = (led["y"].to_numpy() > 0).astype(np.int64)
    champ = _champion_full(led, args.champion)
    factory = MODELS[args.model]
    proba, report = lomo_member(
        X, y_fraud, period, is_test, champ, factory, args.share, args.k, args.context, args.seed
    )
    report["features"] = [cols[i] for i in report["folds"]["october"]["features"]]
    out = paths.out_dir() / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(proba).all() or np.abs(proba.sum(1) - 1).max() > 1e-6:
        raise ValueError("member probabilities are not finite rows summing to 1")
    np.save(out / "oof.npy", proba[~is_test].astype(np.float32))
    np.save(out / "test.npy", proba[is_test].astype(np.float32))
    member_recipe = {
        "name": args.name,
        "model": args.model,
        "base": args.champion,
        "decoder": recipe.get("decoder", {"fraud_scale": 0.875}),
        "rules": recipe.get("rules", []),
    }
    sc = scores_like_champion(led, proba, member_recipe)
    base = scores_like_champion(led, champ, member_recipe)
    final = "rules" if member_recipe["rules"] else "fixed_point"
    summary = {
        "recipe": member_recipe,
        "icl": report,
        **sc,
        "final": sc[final],
        "champion": base,
        "changed_october_argmax": int((proba[is_test].argmax(1) != champ[is_test].argmax(1)).sum()),
        "runtime_s": time.time() - t0,
    }
    for t in sc[final]:
        print(
            f"[icl] {t:<8} member "
            + " / ".join(f"{v:.4f}" for v in sc[final][t]["months"])
            + f" ({sc[final][t]['mean']:.4f})  champion "
            + " / ".join(f"{v:.4f}" for v in base[final][t]["months"])
            + f" ({base[final][t]['mean']:.4f})"
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"[icl] wrote {out} in {time.time() - t0:.0f}s")
    return summary


def cmd_time(args) -> dict:
    """One fold on a slice of queries, twice: wall time and run-to-run equality."""
    from ringtrace import run as runner
    from ringtrace import store

    recipe = runner.load_recipe(args.recipe)
    led = store.ledger()
    X, _ = runner.feature_matrix(recipe)
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y_fraud = (led["y"].to_numpy() > 0).astype(np.int64)
    pf = fraud_prob(_champion_full(led, args.champion))
    tau = choose_tau(pf[~is_test], args.share)
    cand = (pf >= tau) & ~is_test
    ctx = np.flatnonzero(cand & (period != args.fold))
    qry = np.flatnonzero(cand & (period == args.fold))[: args.queries]
    runs = []
    for _ in range(2):
        (p,), info = score_queries(
            X[ctx], y_fraud[ctx], [X[qry]], MODELS[args.model], args.k, args.context, args.seed
        )
        runs.append((p, info))
    diff = float(np.abs(runs[0][0] - runs[1][0]).max())
    res = {
        "fold": args.fold,
        "queries": int(len(qry)),
        "context_rows": runs[0][1]["context_rows"],
        "wall_s": [r[1]["total_s"] for r in runs],
        "fit_s": [r[1]["fit_s"] for r in runs],
        "max_abs_diff": diff,
        "identical": bool(np.array_equal(runs[0][0], runs[1][0])),
    }
    print(f"[icl] timing {json.dumps(res)}")
    return res


def cmd_octlike(args) -> dict:
    from ringtrace import octlike, paths, store
    from ringtrace import run as runner

    t0 = time.time()
    run_dir = paths.out_dir() / "runs" / args.name
    lomo = json.loads((run_dir / "summary.json").read_text())
    tau, k, max_context, seed = (lomo["icl"][x] for x in ("tau", "k", "max_context", "seed"))
    factory = MODELS[lomo["recipe"]["model"]]
    recipe = runner.load_recipe(args.recipe)
    store.use_frame(None)
    base = store.ledger()
    base_pf = fraud_prob(_champion_full(base, args.champion))
    base_y = (base["y"].to_numpy() > 0).astype(np.int64)
    report = {"tau": tau, "cells": {}}
    out_dir = paths.out_dir() / "octlike"
    for s in args.seeds:
        frame = octlike.Frame(s)
        X, _ = runner.feature_matrix(recipe)
        fit_rows = np.flatnonzero(frame.fit)
        ctx = fit_rows[base_pf[frame.stacked[fit_rows]] >= tau]
        hold = np.flatnonzero(frame.hold)
        cells = {r: octlike.cell_prob([("champion", 1.0)], s, r) for r in args.sets}
        qry = {}
        for r, (rows, prob) in cells.items():
            if not np.array_equal(rows, frame.stacked[hold]):
                raise ValueError(f"champion cell s{s} r{r} rows differ from the frame")
            qry[r] = np.flatnonzero(fraud_prob(prob) >= tau)
        print(
            f"[icl] s{s}: context {len(ctx)}, queries " + str({r: len(q) for r, q in qry.items()})
        )
        ps, info = score_queries(
            X[ctx],
            base_y[frame.stacked[ctx]],
            [X[hold[qry[r]]] for r in args.sets],
            factory,
            k,
            max_context,
            seed,
        )
        for r, p in zip(args.sets, ps, strict=True):
            rows, prob = cells[r]
            member = np.asarray(prob, dtype=np.float64).copy()
            member[qry[r]] = compose(p, prob[qry[r]])
            name = f"{args.name}_s{s}_r{r}"
            np.savez(out_dir / f"{name}.npz", rows=rows, prob=member.astype(np.float32))
            report["cells"][name] = {"queries": int(len(qry[r]))}
        report[f"s{s}"] = info
    store.use_frame(None)
    report["runtime_s"] = time.time() - t0
    (run_dir / "octlike.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"[icl] October-like cells written in {time.time() - t0:.0f}s")
    return report


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None):
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.icl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lomo", "time", "octlike"):
        p = sub.add_parser(name)
        p.add_argument("--recipe", default=CHAMPION_RECIPE)
        p.add_argument("--champion", default=CHAMPION_RUN)
        p.add_argument("--model", default="tabpfn-v2", choices=sorted(MODELS))
        p.add_argument("--share", type=float, default=SHARE)
        p.add_argument("--k", type=int, default=TOP_K)
        p.add_argument("--context", type=int, default=MAX_CONTEXT)
        p.add_argument("--seed", type=int, default=SEED)
        p.add_argument("--name", default="icl-tabpfn-v2")
    sub.choices["time"].add_argument("--fold", type=int, default=9)
    sub.choices["time"].add_argument("--queries", type=int, default=5000)
    sub.choices["octlike"].add_argument("--seeds", type=_ints, default=[13, 14, 15])
    sub.choices["octlike"].add_argument("--sets", type=_ints, default=[0, 1, 2])
    args = ap.parse_args(argv)
    return {"lomo": cmd_lomo, "time": cmd_time, "octlike": cmd_octlike}[args.cmd](args)


if __name__ == "__main__":
    main()
