"""Statistics behind the audits (pure functions on arrays; no file access).

* ``cv_auc``: cross-validated AUC of a small gradient-boosted model, averaged over seeds; the
  standard "is there any signal" question for a two-group comparison.
* ``univariate``: one test per feature (rate difference for 0/1 features, Mann-Whitney AUC for
  the rest), so that a single strong column is not lost inside a weak model.
* ``link_share``: for every row, whether another member row shares its key (sender, receiver,
  phone ...) within a time window; compares how often two groups cluster.
* ``adversarial``: how well a model tells two periods apart, and which share of the most
  period-B-like rows really are from period B (a hidden sub-population shows up there).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BOOSTER = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "verbose": -1,
}


def _model(seed: int, threads: int | None, params: dict | None):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        random_state=seed, n_jobs=threads or 0, **{**BOOSTER, **(params or {})}
    )


def _fit(model, x: np.ndarray, y: np.ndarray, categorical: list[int] | None):
    if categorical:
        return model.fit(x, y, categorical_feature=categorical)
    return model.fit(x, y)


def oof_scores(
    x: np.ndarray,
    y: np.ndarray,
    *,
    categorical: list[int] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    folds: int = 5,
    threads: int | None = None,
    params: dict | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Seed-averaged out-of-fold scores and the AUC of each seed."""
    oof = np.zeros(len(y))
    per_seed = []
    for seed in seeds:
        s = np.zeros(len(y))
        for trn, val in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
            m = _fit(_model(seed, threads, params), x[trn], y[trn], categorical)
            s[val] = m.predict_proba(x[val])[:, 1]
        per_seed.append(float(roc_auc_score(y, s)))
        oof += s
    return oof / len(seeds), per_seed


def cv_auc(
    x: np.ndarray,
    y: np.ndarray,
    *,
    names: list[str] | None = None,
    categorical: list[int] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    folds: int = 5,
    threads: int | None = None,
    params: dict | None = None,
    top: int = 10,
    keep_oof: bool = False,
) -> dict:
    """Out-of-fold AUC of a booster separating ``y == 1`` from ``y == 0``.

    Returns the AUC of the seed-averaged out-of-fold scores, the AUC of each seed, the group
    sizes and the ``top`` features by gain of one model fitted on all rows."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).astype(np.int64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    out: dict = {"rows": len(y), "positives": n_pos, "negatives": n_neg}
    if min(n_pos, n_neg) < folds:
        out.update(auc=float("nan"), auc_seeds=[], top_features=[])
        return out
    oof, per_seed = oof_scores(
        x, y, categorical=categorical, seeds=seeds, folds=folds, threads=threads, params=params
    )
    out["auc"] = float(roc_auc_score(y, oof))
    out["auc_seeds"] = per_seed
    m = _fit(_model(seeds[0], threads, params), x, y, categorical)
    gain = m.booster_.feature_importance("gain")
    names = names or [f"f{i}" for i in range(x.shape[1])]
    order = np.argsort(-gain)[:top]
    out["top_features"] = [[names[i], float(gain[i])] for i in order if gain[i] > 0]
    if keep_oof:
        out["oof"] = oof
    return out


def univariate(columns: dict[str, np.ndarray], y: np.ndarray) -> list[dict]:
    """One test per column, sorted by p-value: two-proportion z for 0/1 columns, Mann-Whitney
    for the rest (``auc`` = P(group-1 value > group-0 value), ties counted half)."""
    y = np.asarray(y).astype(bool)
    res = []
    for name, col in columns.items():
        x = np.asarray(col, dtype=np.float64)
        ok = ~np.isnan(x)
        a, b = x[ok & y], x[ok & ~y]
        if len(a) < 2 or len(b) < 2:
            continue
        if np.isin(np.unique(x[ok]), (0.0, 1.0)).all():
            p1, p0, pool = float(a.mean()), float(b.mean()), float(x[ok].mean())
            se = np.sqrt(max(pool * (1 - pool), 1e-12) * (1 / len(a) + 1 / len(b)))
            z = (p1 - p0) / se
            res.append(
                {
                    "feature": name,
                    "kind": "rate",
                    "group1": p1,
                    "group0": p0,
                    "z": float(z),
                    "p": float(2 * st.norm.sf(abs(z))),
                }
            )
        else:
            u = st.mannwhitneyu(a, b, alternative="two-sided")
            res.append(
                {
                    "feature": name,
                    "kind": "rank",
                    "group1": float(np.median(a)),
                    "group0": float(np.median(b)),
                    "auc": float(u.statistic) / (len(a) * len(b)),
                    "p": float(u.pvalue),
                }
            )
    return sorted(res, key=lambda r: r["p"])


def link_share(key: np.ndarray, ts: np.ndarray, member: np.ndarray, window: int) -> np.ndarray:
    """For every row: does another ``member`` row with the same key lie within +-``window``
    seconds? (A member row does not count itself.)"""
    key = np.asarray(key, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.int64)
    member = np.asarray(member).astype(bool)
    t = ts - ts.min()
    span = int(t.max()) + 2 * int(window) + 2
    k = np.unique(key, return_inverse=True)[1].astype(np.int64)
    comp = k * span + t + int(window) + 1
    pool = np.sort(comp[member])
    lo = np.searchsorted(pool, comp - window, side="left")
    hi = np.searchsorted(pool, comp + window, side="right")
    return (hi - lo - member.astype(np.int64)) > 0


def adversarial(
    xa: np.ndarray,
    xb: np.ndarray,
    *,
    names: list[str] | None = None,
    categorical: list[int] | None = None,
    seeds: tuple[int, ...] = (0,),
    folds: int = 4,
    max_rows: int | None = None,
    quantiles: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05),
    threads: int | None = None,
    seed: int = 0,
) -> dict:
    """Out-of-fold AUC of a booster telling rows of B from rows of A (sizes balanced by
    subsampling), and the share of B among the most B-like rows at each quantile. A share well
    above 0.5 in the top quantiles while the AUC stays near 0.5 points to a small B-only group."""
    rng = np.random.default_rng(seed)
    n = min(len(xa), len(xb), max_rows or max(len(xa), len(xb)))
    ia = rng.choice(len(xa), n, replace=False)
    ib = rng.choice(len(xb), n, replace=False)
    x = np.vstack([np.asarray(xa)[ia], np.asarray(xb)[ib]]).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    res = cv_auc(
        x,
        y,
        names=names,
        categorical=categorical,
        seeds=seeds,
        folds=folds,
        threads=threads,
        keep_oof=True,
    )
    oof = res.pop("oof", None)
    res["rows_per_side"] = int(n)
    if oof is not None:
        order = np.argsort(-oof, kind="stable")
        res["tail_share_b"] = {
            str(q): float(y[order[: max(1, int(round(q * len(y))))]].mean()) for q in quantiles
        }
    return res
