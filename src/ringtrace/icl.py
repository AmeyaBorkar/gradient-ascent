"""A pretrained in-context tabular model as a blend member, on the champion's candidate rows.

    python -m ringtrace.icl run --name <run> [--k 100] [--max-context 50000] [--estimators 4]
                                [--folds 7,8,9,10] [--slice 2000] [--champion-run champion]

Design (the same for every in-context model):

* **Candidates.** Rows whose champion fraud probability ``1 - p(legit)`` (its out-of-fold array on
  train, its October array on test) is at least ``tau``; ``tau`` is chosen once so that ``SHARE``
  of the train rows qualify. Outside the candidates the member equals the champion.
* **Features.** The champion recipe's feature matrix, cut to the ``k`` columns with the largest
  univariate separation (``|AUC - 0.5|``) on the context rows of each fold; missing and infinite
  values are replaced by the context median.
* **Target.** Binary fraud vs legit on the given labels.
* **Folds.** Held-out month m: context = candidates of the other two months, queries = month m's
  candidates. October: context = candidates of all three months. If the context is larger than
  ``max_context``, every fraud row is kept, legit rows are subsampled at rate r, and the fraud
  probability is corrected for the prior: odds x r.
* **Member.** ``p(k) = p_icl(fraud) * p_champ(k) / p_champ(fraud)`` for fraud classes,
  ``p(legit) = 1 - p_icl(fraud)``.

Writes ``oof.npy`` (train rows, ledger order), ``test.npy`` and ``summary.json`` under
``$RINGTRACE_OUT/runs/<run>/``, plus ``icl.npz`` with the raw candidate probabilities.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ringtrace.classes import LEGIT, N_CLASSES

SHARE = 0.05
TRAIN_MONTHS = (7, 8, 9)
OCTOBER = 10

# fit(X_context, y_context, X_query, seed) -> p(fraud) per query row
Predictor = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def fraud_prob(proba: np.ndarray) -> np.ndarray:
    return 1.0 - proba[:, LEGIT]


def choose_tau(p_fraud_train: np.ndarray, share: float = SHARE) -> float:
    """The threshold above which ``share`` of the train rows lie."""
    return float(np.quantile(p_fraud_train, 1.0 - share, method="higher"))


def auc_score(x: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC of one column against a binary target (ties get mid-ranks)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    # mid-ranks over runs of equal values
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], len(xs)]
    mid = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(mid, ends - starts)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def medians(X: np.ndarray) -> np.ndarray:
    Xf = np.where(np.isfinite(X), X, np.nan)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(Xf, axis=0)
    return np.where(np.isfinite(med), med, 0.0).astype(np.float32)


def impute(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.broadcast_to(med, X.shape)[bad]
    return X


def rank_features(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` columns with the largest ``|AUC - 0.5|`` on these rows (constant
    columns last; ties broken by column order)."""
    Xi = impute(X, medians(X))
    score = np.array([abs(auc_score(Xi[:, j], y) - 0.5) for j in range(X.shape[1])])
    const = Xi.min(axis=0) == Xi.max(axis=0)
    score[const] = -1.0
    order = np.lexsort((np.arange(len(score)), -score))
    return np.sort(order[: min(k, X.shape[1])])


def subsample_context(y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, float]:
    """Every fraud row and a seeded share of the legit rows, at most ``max_rows`` in all.
    Returns the kept indices (sorted) and the legit keep rate."""
    n = len(y)
    if n <= max_rows:
        return np.arange(n), 1.0
    fraud = np.flatnonzero(y == 1)
    legit = np.flatnonzero(y == 0)
    room = max(max_rows - len(fraud), 1)
    rate = min(1.0, room / max(len(legit), 1))
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(legit, size=min(room, len(legit)), replace=False))
    return np.sort(np.concatenate([fraud, keep])), float(rate)


def prior_correct(p: np.ndarray, legit_rate: float) -> np.ndarray:
    """Undo legit subsampling at ``legit_rate``: fraud odds are multiplied by the rate."""
    if legit_rate >= 1.0:
        return p
    num = p * legit_rate
    return num / (num + (1.0 - p))


def compose(champ: np.ndarray, cand: np.ndarray, p_icl: np.ndarray, eps: float = 1e-12):
    """The member's 12-class rows: the champion's outside the candidates; inside them the
    model's fraud probability spread over the champion's fraud-type shares."""
    out = champ.astype(np.float64, copy=True)
    c = champ[cand].astype(np.float64)
    pf = np.clip(p_icl.astype(np.float64), 0.0, 1.0)
    fraud_mass = c[:, 1:].sum(axis=1, keepdims=True)
    shares = np.where(
        fraud_mass > eps, c[:, 1:] / np.maximum(fraud_mass, eps), 1.0 / (N_CLASSES - 1)
    )
    rows = np.empty_like(c)
    rows[:, LEGIT] = 1.0 - pf
    rows[:, 1:] = pf[:, None] * shares
    out[cand] = rows
    return out


def fold_predict(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_q: np.ndarray,
    predictor: Predictor,
    k: int,
    max_context: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """One fold: subsample the context, rank and cut the features, impute, predict, correct."""
    keep, rate = subsample_context(y_ctx, max_context, seed)
    Xc, yc = X_ctx[keep], y_ctx[keep]
    cols = rank_features(Xc, yc, k)
    med = medians(Xc[:, cols])
    p = predictor(impute(Xc[:, cols], med), yc, impute(X_q[:, cols], med), seed)
    p = prior_correct(np.asarray(p, dtype=np.float64), rate)
    info = {
        "context_rows": len(keep),
        "context_fraud": int(yc.sum()),
        "legit_rate": rate,
        "queries": len(X_q),
        "columns": cols.tolist(),
    }
    return p, info


def run_member(
    X: np.ndarray,
    y_fraud: np.ndarray,
    period: np.ndarray,
    is_test: np.ndarray,
    champ: np.ndarray,
    predictor: Predictor,
    k: int,
    max_context: int,
    seed: int = 0,
    folds: tuple[int, ...] = (*TRAIN_MONTHS, OCTOBER),
    share: float = SHARE,
    log=print,
) -> dict:
    """Everything except I/O; all arrays in ledger order. Returns the member's probabilities
    for every row, the candidate mask, tau and per-fold information."""
    pf = fraud_prob(champ)
    tau = choose_tau(pf[~is_test], share)
    cand = pf >= tau
    p_icl = np.full(len(y_fraud), np.nan)
    info: dict = {"tau": tau, "folds": {}}
    for m in folds:
        q = np.flatnonzero(cand & (period == m))
        ctx_months = [t for t in TRAIN_MONTHS if t != m]
        c = np.flatnonzero(cand & ~is_test & np.isin(period, ctx_months))
        t0 = time.time()
        p, fi = fold_predict(X[c], y_fraud[c], X[q], predictor, k, max_context, seed + m)
        fi["seconds"] = time.time() - t0
        p_icl[q] = p
        info["folds"][str(m)] = fi
        log(
            f"[icl] fold {m}: context {fi['context_rows']} ({fi['context_fraud']} fraud, legit "
            f"rate {fi['legit_rate']:.3f}), queries {len(q)}, {fi['seconds']:.0f}s"
        )
    done = cand & np.isfinite(p_icl)
    member = compose(champ, done, p_icl[done])
    return {"proba": member, "cand": cand, "p_icl": p_icl, "info": info}


def candidate_report(cand: np.ndarray, y: np.ndarray, period: np.ndarray, is_test) -> dict:
    out = {}
    for m in (*TRAIN_MONTHS, OCTOBER):
        s = period == m
        out[str(m)] = {"rows": int(s.sum()), "candidates": int((cand & s).sum())}
        if m != OCTOBER:
            f = s & (y != LEGIT)
            out[str(m)]["fraud_rows"] = int(f.sum())
            out[str(m)]["fraud_covered"] = float((cand & f).sum() / max(f.sum(), 1))
    tr = ~is_test
    f = tr & (y != LEGIT)
    out["train"] = {
        "candidates": int((cand & tr).sum()),
        "share": float((cand & tr).mean() / tr.mean()),
        "fraud_covered": float((cand & f).sum() / max(f.sum(), 1)),
        "candidate_fraud_rate": float((y[cand & tr] != LEGIT).mean()),
    }
    return out


def tabicl_predictor(estimators: int, gpu_fraction: float) -> Predictor:
    import torch
    from tabicl import TabICLClassifier

    if torch.cuda.is_available() and gpu_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)

    def predict(Xc, yc, Xq, seed):
        torch.manual_seed(seed)
        clf = TabICLClassifier(
            n_estimators=estimators,
            random_state=seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
            allow_auto_download=True,
        )
        clf.fit(Xc, yc)
        proba = clf.predict_proba(Xq)
        col = list(clf.classes_).index(1)
        out = proba[:, col].astype(np.float64)
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    return predict


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--recipe", default="recipes/champion.json")
    ap.add_argument("--champion-run", default="champion", help="dir under $RINGTRACE_OUT/runs")
    ap.add_argument("--champion-dir", default=None, help="explicit champion run dir")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-context", type=int, default=50_000)
    ap.add_argument("--estimators", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="7,8,9,10")
    ap.add_argument("--slice", type=int, default=0, help="timing: first N queries per fold")
    ap.add_argument("--gpu-fraction", type=float, default=0.5)
    args = ap.parse_args(argv)

    from ringtrace import paths, store, threads
    from ringtrace.run import feature_matrix, load_recipe

    threads.pin()
    t0 = time.time()
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y = led["y"].to_numpy().astype(np.int64)
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    cdir = Path(args.champion_dir) if args.champion_dir else paths.out_dir() / "runs"
    cdir = cdir if args.champion_dir else cdir / args.champion_run
    champ = np.zeros((led.height, N_CLASSES))
    champ[tr] = np.load(cdir / "oof.npy")
    champ[te] = np.load(cdir / "test.npy")
    X, cols = feature_matrix(load_recipe(args.recipe))
    print(f"[icl] {X.shape[1]} features, {X.shape[0]} rows ({time.time() - t0:.0f}s)", flush=True)

    base = tabicl_predictor(args.estimators, args.gpu_fraction)
    predictor = base
    if args.slice:

        def predictor(Xc, yc, Xq, seed):
            return np.concatenate(
                [base(Xc, yc, Xq[: args.slice], seed), np.full(len(Xq) - args.slice, np.nan)]
            )

    folds = tuple(int(f) for f in args.folds.split(","))
    res = run_member(
        X, (y != LEGIT).astype(np.int64), period, is_test, champ, predictor,
        args.k, args.max_context, args.seed, folds,
    )  # fmt: skip
    proba = res["proba"]
    out = paths.out_dir() / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "args": vars(args),
        "tau": res["info"]["tau"],
        "candidates": candidate_report(res["cand"], y, period, is_test),
        "folds": res["info"]["folds"],
        "feature_names": cols,
        "finite": bool(np.isfinite(proba).all()),
        "max_row_sum_error": float(np.abs(proba.sum(1) - 1).max()),
        "runtime_s": time.time() - t0,
    }
    for fi in summary["folds"].values():
        fi["columns"] = [cols[j] for j in fi["columns"]]
    np.savez_compressed(out / "icl.npz", p_icl=res["p_icl"], cand=res["cand"])
    if not args.slice and set(folds) == {*TRAIN_MONTHS, OCTOBER}:
        np.save(out / "oof.npy", proba[tr].astype(np.float32))
        np.save(out / "test.npy", proba[te].astype(np.float32))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("tau", "candidates")}, indent=1), flush=True)
    print(f"[icl] done in {summary['runtime_s']:.0f}s -> {out}", flush=True)
    return summary


if __name__ == "__main__":
    main()
