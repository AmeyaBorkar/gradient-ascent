"""A pretrained in-context tabular model as a blend member, on the champion's doubtful rows.

    python -m ringtrace.icl prepare <job>                      # features + candidates (main venv)
    python -m ringtrace.icl fit <job> --model mitra [...]      # in-context fraud score (GPU)
    python -m ringtrace.icl assemble <job> [--tag <fit-tag>]   # 12-class member run
    python -m ringtrace.icl score <job>                        # alone and inside the blend

Design (identical for every model):

* **Candidates** are the rows whose champion fraud probability 1 - p(legit) is at least ``tau``;
  ``tau`` is the train quantile that leaves ``CAND_SHARE`` of train rows above it, chosen once
  on the champion's out-of-fold array. October uses the champion's October array and the same
  ``tau``. Outside the candidates the member equals the champion.
* **Features** are the champion's matrix (its recipe's families and drop list), restricted per
  fold to the top ``k`` columns by a univariate score (``|AUC - 1/2|`` against fraud) computed on
  the fold's context rows only.
* **Folds**: held-out month m uses the candidates of the other two months as context; October
  uses the candidates of all three. The target is binary fraud vs legit on the given labels.
* **Context limit**: every fraud row is kept, legit rows are drawn (seeded) at ``legit_ratio``
  per fraud row, and the probability is corrected for that prior shift.
* **12-class member**: p(k) = p_icl(fraud) x p_champ(k) / p_champ(fraud) for each fraud class,
  p(legit) = 1 - p_icl(fraud).

``prepare`` writes ``$RINGTRACE_OUT/icl/<job>/cand.npz`` (candidate features, labels, months,
champion probabilities and the row positions in the train / October arrays). ``fit`` reads only
that file, so it runs in a separate environment with the model's library; it writes
``fit-<tag>.npz`` (corrected p_icl(fraud) per candidate) and ``fit-<tag>.json`` (timings,
context sizes, features). ``assemble`` writes the standard run layout under
``$RINGTRACE_OUT/runs/<job>/``. ``score`` decodes as the champion is decoded (fixed point at
0.875 per month, then the anchored template rule) and reports per-month scores on the given,
cleaned and generator-only targets plus the paired event bootstrap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CAND_SHARE = 0.05
TRAIN_MONTHS = (7, 8, 9)
OCTOBER = 10
CHAMPION_RUN = "blend-champion-m1"
BLEND_RUNS = ("blend-champion-m1", "lgbm", "catboost", "clean-event", "clean-l2")
EPS = 1e-7
DECODER = {"fraud_scale": 0.875}
RULES = ["anchored_templates"]


# ---------------------------------------------------------------------------------------------
# pure pieces (tested on synthetic data)


def fraud_prob(proba: np.ndarray) -> np.ndarray:
    """The champion's fraud probability, 1 - p(legit)."""
    return 1.0 - np.asarray(proba, dtype=np.float64)[:, 0]


def choose_tau(p_fraud_train: np.ndarray, share: float = CAND_SHARE) -> float:
    """The threshold that leaves about ``share`` of train rows at or above it."""
    return float(np.quantile(p_fraud_train, 1.0 - share))


def auc_score(x: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC of one column against a binary target; missing values rank lowest, ties share."""
    from scipy.stats import rankdata

    x = np.where(np.isfinite(x), x, -np.inf)
    pos = y.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = rankdata(x)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rank_features(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the ``k`` most separating columns by ``|AUC - 1/2|`` (stable order:
    ties broken by column index), constant columns last."""
    score = np.array([abs(auc_score(X[:, j], y) - 0.5) for j in range(X.shape[1])])
    finite = np.where(np.isfinite(X), X, np.nan)
    with np.errstate(all="ignore"):
        const = ~(np.nanmax(finite, axis=0) > np.nanmin(finite, axis=0))
    score[const] = -1.0
    order = np.lexsort((np.arange(X.shape[1]), -score))
    return np.sort(order[: min(k, X.shape[1])])


def subsample_context(
    y_bin: np.ndarray, legit_ratio: float | None, seed: int
) -> tuple[np.ndarray, float]:
    """Indices of the context rows (every fraud row, ``legit_ratio`` legit rows per fraud row,
    drawn without replacement) and the legit keep rate q. ``None`` keeps everything."""
    fraud = np.flatnonzero(y_bin == 1)
    legit = np.flatnonzero(y_bin == 0)
    if legit_ratio is None:
        return np.arange(len(y_bin)), 1.0
    n_keep = min(len(legit), int(round(legit_ratio * len(fraud))))
    rng = np.random.default_rng(seed)
    kept = np.sort(rng.choice(legit, size=n_keep, replace=False))
    q = n_keep / len(legit) if len(legit) else 1.0
    return np.sort(np.r_[fraud, kept]), float(q)


def prior_correct(p: np.ndarray, q: float) -> np.ndarray:
    """Undo a legit keep rate q: the fitted odds are true odds / q, so true odds = odds x q."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    odds = p / (1 - p) * q
    return odds / (1 + odds)


def compose_member(proba_champ: np.ndarray, cand: np.ndarray, p_icl: np.ndarray) -> np.ndarray:
    """The 12-class member: the champion everywhere, and on ``cand`` rows legit = 1 - p_icl and
    each fraud class = p_icl x the champion's share of its fraud mass."""
    out = np.asarray(proba_champ, dtype=np.float64).copy()
    out /= out.sum(axis=1, keepdims=True)
    sub = out[cand]
    fraud_mass = sub[:, 1:].sum(axis=1, keepdims=True)
    n_fraud = sub.shape[1] - 1
    share = np.where(fraud_mass > 0, sub[:, 1:] / np.maximum(fraud_mass, 1e-300), 1.0 / n_fraud)
    p = np.clip(np.asarray(p_icl, dtype=np.float64), 0.0, 1.0)[:, None]
    out[cand] = np.hstack([1.0 - p, p * share])
    return out


def blend_weights(n_members: int, member_weight: float) -> list[float]:
    """Weights for the base members plus one extra member at ``member_weight``."""
    rest = (1.0 - member_weight) / n_members
    return [rest] * n_members + [member_weight]


# ---------------------------------------------------------------------------------------------
# paths


def job_dir(job: str) -> Path:
    from ringtrace import paths

    d = paths.out_dir() / "icl" / job
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------
# prepare (main environment)


def prepare(job: str, recipe_path: str = "recipes/champion.json", log=print) -> dict:
    from ringtrace import paths, run, store

    t0 = time.time()
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y = led["y"].to_numpy().astype(np.int64)
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    champ = paths.out_dir() / "runs" / CHAMPION_RUN
    oof, test = np.load(champ / "oof.npy"), np.load(champ / "test.npy")
    pf_tr, pf_te = fraud_prob(oof), fraud_prob(test)
    tau = choose_tau(pf_tr)
    c_tr, c_te = np.flatnonzero(pf_tr >= tau), np.flatnonzero(pf_te >= tau)
    X, cols = run.feature_matrix(run.load_recipe(recipe_path))
    rows = np.r_[tr[c_tr], te[c_te]]
    y_tr = y[tr]
    stats = {
        "tau": tau,
        "train_rows": int(len(tr)),
        "candidates_train": int(len(c_tr)),
        "candidate_share": float(len(c_tr) / len(tr)),
        "fraud_rows": int((y_tr > 0).sum()),
        "fraud_covered": float((y_tr[c_tr] > 0).sum() / (y_tr > 0).sum()),
        "candidate_fraud_share": float((y_tr[c_tr] > 0).mean()),
        "per_month": {
            int(m): {
                "rows": int((period[tr] == m).sum()),
                "candidates": int((period[tr][c_tr] == m).sum()),
                "fraud_in_candidates": int(((period[tr][c_tr] == m) & (y_tr[c_tr] > 0)).sum()),
                "fraud_rows": int(((period[tr] == m) & (y_tr > 0)).sum()),
            }
            for m in TRAIN_MONTHS
        },
        "october_rows": int(len(te)),
        "candidates_october": int(len(c_te)),
        "n_features": len(cols),
    }
    out = job_dir(job)
    np.savez(
        out / "cand.npz",
        X=X[rows],
        y=np.r_[y_tr[c_tr], np.full(len(c_te), -1)],
        month=np.r_[period[tr][c_tr], np.full(len(c_te), OCTOBER)],
        pos=np.r_[c_tr, c_te],
        champ=np.vstack([oof[c_tr], test[c_te]]).astype(np.float32),
        cols=np.asarray(cols),
    )
    stats["prepare_s"] = time.time() - t0
    (out / "candidates.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats, indent=1))
    return stats


# ---------------------------------------------------------------------------------------------
# fit (model environment)


def _mitra_predict(
    X_ctx, y_ctx, X_q, seed: int, model_id: str, revision: str | None, query_batch: int
) -> np.ndarray:
    """Pure in-context Mitra: the pretrained weights, the whole context as support, no tuning."""
    import torch
    from autogluon.tabular.models.mitra._internal.core.trainer_finetune import TrainerFinetune
    from autogluon.tabular.models.mitra.sklearn_interface import (
        MitraClassifier,
        mitra_deterministic_context,
    )

    torch.manual_seed(seed)
    hf = model_id
    if revision:
        from huggingface_hub import snapshot_download

        hf = snapshot_download(model_id, revision=revision)
    clf = MitraClassifier(fine_tune=False, hf_model=hf, seed=seed, verbose=False)
    with mitra_deterministic_context():
        cfg, tab2d = clf._create_config("classification", 10)
        cfg.hyperparams["max_samples_support"] = len(X_ctx)
        cfg.hyperparams["max_samples_query"] = query_batch
        model = tab2d.from_pretrained(hf, device=clf.device)
        trainer = TrainerFinetune(
            cfg,
            model,
            n_classes=10,
            device=clf.device,
            rng=np.random.RandomState(seed),
            verbose=False,
        )
        trainer.preprocessor.fit(X_ctx, y_ctx)
        logits = trainer.predict(X_ctx, y_ctx, X_q)[:, :2].astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e[:, 1] / e.sum(axis=1)


def _tabdpt_predict(X_ctx, y_ctx, X_q, seed: int, n_ensembles: int, context_size: int):
    import torch
    from tabdpt import TabDPTClassifier

    torch.manual_seed(seed)
    clf = TabDPTClassifier(compile=False)
    clf.fit(X_ctx, y_ctx)
    p = clf.ensemble_predict_proba(
        X_q, n_ensembles=n_ensembles, context_size=context_size, seed=seed
    )
    return np.asarray(p, dtype=np.float64)[:, 1]


def fit(args, log=print) -> dict:
    d = job_dir(args.job)
    z = np.load(d / "cand.npz")
    X, y, month = z["X"], z["y"], z["month"]
    cols = z["cols"]
    folds = [int(f) for f in args.folds.split(",")]
    tag = args.tag
    p_all = np.full(len(y), np.nan)
    info: dict = {"args": vars(args), "folds": {}}
    for f in folds:
        t0 = time.time()
        ctx_all = np.flatnonzero((month != f) & (month != OCTOBER))
        q_idx = np.flatnonzero(month == f)
        if args.limit:
            q_idx = q_idx[: args.limit]
        y_bin_all = (y[ctx_all] > 0).astype(np.int64)
        keep, q = subsample_context(y_bin_all, args.legit_ratio, seed=args.seed + f)
        ctx = ctx_all[keep]
        yb = y_bin_all[keep]
        feats = rank_features(X[ctx], yb, args.k)
        Xc = X[np.ix_(ctx, feats)].astype(np.float32)
        Xq = X[np.ix_(q_idx, feats)].astype(np.float32)
        t1 = time.time()
        if args.model == "mitra":
            p = _mitra_predict(
                Xc, yb, Xq, args.seed, args.model_id, args.revision, args.query_batch
            )
        elif args.model == "tabdpt":
            p = _tabdpt_predict(Xc, yb, Xq, args.seed, args.n_ensembles, args.context_size)
        else:
            raise ValueError(args.model)
        p_corr = prior_correct(p, q)
        p_all[q_idx] = p_corr
        info["folds"][f] = {
            "context": int(len(ctx)),
            "context_fraud": int(yb.sum()),
            "legit_keep_rate": q,
            "queries": int(len(q_idx)),
            "features": [str(cols[j]) for j in feats],
            "prep_s": t1 - t0,
            "model_s": time.time() - t1,
            "mean_p": float(p_corr.mean()),
        }
        if f != OCTOBER:
            from sklearn.metrics import roc_auc_score

            yq = (y[q_idx] > 0).astype(int)
            champ_pf = fraud_prob(z["champ"][q_idx])
            info["folds"][f]["auc_member"] = float(roc_auc_score(yq, p_corr))
            info["folds"][f]["auc_champion"] = float(roc_auc_score(yq, champ_pf))
        brief = {k: v for k, v in info["folds"][f].items() if k != "features"}
        log(f"[{tag}] fold {f}: " + json.dumps(brief))
    np.savez(d / f"fit-{tag}.npz", p=p_all)
    (d / f"fit-{tag}.json").write_text(json.dumps(info, indent=1, default=str))
    return info


# ---------------------------------------------------------------------------------------------
# assemble + score (main environment)


def assemble(job: str, tag: str, run_name: str | None = None, log=print) -> Path:
    from ringtrace import paths

    d = job_dir(job)
    z = np.load(d / "cand.npz")
    p = np.load(d / f"fit-{tag}.npz")["p"]
    month, pos = z["month"], z["pos"]
    if not np.isfinite(p).all():
        raise ValueError(f"fit-{tag} has {int((~np.isfinite(p)).sum())} rows without a score")
    champ = paths.out_dir() / "runs" / CHAMPION_RUN
    oof, test = np.load(champ / "oof.npy"), np.load(champ / "test.npy")
    tr_sel, te_sel = month != OCTOBER, month == OCTOBER
    cand_tr = np.zeros(len(oof), dtype=bool)
    cand_tr[pos[tr_sel]] = True
    cand_te = np.zeros(len(test), dtype=bool)
    cand_te[pos[te_sel]] = True
    p_tr = np.zeros(len(oof))
    p_tr[pos[tr_sel]] = p[tr_sel]
    p_te = np.zeros(len(test))
    p_te[pos[te_sel]] = p[te_sel]
    m_oof = compose_member(oof, cand_tr, p_tr[cand_tr])
    m_test = compose_member(test, cand_te, p_te[cand_te])
    for name, a in (("oof", m_oof), ("test", m_test)):
        if not np.isfinite(a).all() or np.abs(a.sum(1) - 1).max() > 1e-6 or (a < 0).any():
            raise ValueError(f"{name}: member rows are not probabilities")
    out = paths.out_dir() / "runs" / (run_name or job)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", m_oof.astype(np.float32))
    np.save(out / "test.npy", m_test.astype(np.float32))
    fit_info = json.loads((d / f"fit-{tag}.json").read_text())
    summary = {
        "recipe": {
            "name": out.name,
            "decoder": {"fraud_scale": 0.875},
            "rules": ["anchored_templates"],
            "icl": fit_info["args"],
        },
        "candidates": json.loads((d / "candidates.json").read_text()),
        "fit": {k: {kk: vv for kk, vv in v.items()} for k, v in fit_info["folds"].items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"[assemble] {out}: oof {m_oof.shape}, test {m_test.shape}")
    return out


def score(job: str, run_name: str | None = None, draws: int = 2000, log=print) -> dict:
    from ringtrace import paths, silver, store
    from ringtrace.blend import decode_all, month_scores
    from ringtrace.compare import bootstrap
    from ringtrace.teacher import union_events

    runs = paths.out_dir() / "runs"
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    targets = {"given": led["y"].to_numpy().astype(np.int64)[tr]}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr[tr]
    ev = union_events(led.height)[tr]
    groups = np.where(ev >= 0, ev, -1 - np.arange(len(tr)))
    per_tr = period[tr]

    def load(run):
        return np.load(runs / run / "oof.npy").astype(np.float64), np.load(
            runs / run / "test.npy"
        ).astype(np.float64)

    def calls(oof, test):
        p = np.zeros((led.height, oof.shape[1]))
        p[tr], p[te] = oof, test
        c = decode_all(p, period, is_test, DECODER, RULES, led, log=lambda m: None)["rules"]
        return c[tr], c[te]

    member = load(run_name or job)
    base = [load(r) for r in BLEND_RUNS]
    variants = {
        "champion": base[0],
        "member": member,
        "blend5": tuple(np.mean([b[i] for b in base], axis=0) for i in (0, 1)),
    }
    for w in (1 / 6, 1 / 3):
        ws = blend_weights(len(base), w)
        variants[f"blend6_w{w:.3f}"] = tuple(
            sum(wt * a[i] for wt, a in zip(ws, [*base, member], strict=True)) for i in (0, 1)
        )
    pred = {}
    report: dict = {"scores": {}, "bootstrap": {}, "october_changed": {}}
    for name, (o, t) in variants.items():
        pred[name] = calls(o, t)
        report["scores"][name] = month_scores(pred[name][0], targets, per_tr)
        line = "  ".join(
            f"{tn} " + " / ".join(f"{s:.4f}" for s in v["months"]) + f" ({v['mean']:.4f})"
            for tn, v in report["scores"][name].items()
        )
        log(f"[score] {name:<16} {line}")
    pairs = [("champion", "member"), ("blend5", "blend6_w0.167"), ("blend5", "blend6_w0.333")]
    for a, b in pairs:
        for tn, yt in targets.items():
            r = bootstrap(groups, yt, per_tr, pred[a][0], pred[b][0], draws=draws)
            report["bootstrap"][f"{b}_vs_{a}_{tn}"] = r
            log(
                f"[score] {b} vs {a} [{tn}]: {r['delta']:+.4f} "
                f"[{r['p05']:+.4f}, {r['p95']:+.4f}] -> {r['verdict']}"
            )
        report["october_changed"][f"{b}_vs_{a}"] = int((pred[a][1] != pred[b][1]).sum())
    log(f"[score] October calls changed: {report['october_changed']}")
    out = runs / (run_name or job) / "score.json"
    out.write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("job")
    p.add_argument("--recipe", default="recipes/champion.json")
    f = sub.add_parser("fit")
    f.add_argument("job")
    f.add_argument("--model", choices=["mitra", "tabdpt"], required=True)
    f.add_argument("--tag", required=True)
    f.add_argument("--folds", default="7,8,9,10")
    f.add_argument("--k", type=int, default=100)
    f.add_argument("--legit-ratio", type=float, default=None)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--limit", type=int, default=0, help="first N queries only (timing)")
    f.add_argument("--model-id", default="autogluon/mitra-classifier-2")
    f.add_argument("--revision", default=None)
    f.add_argument("--query-batch", type=int, default=1024)
    f.add_argument("--n-ensembles", type=int, default=1)
    f.add_argument("--context-size", type=int, default=2048)
    a = sub.add_parser("assemble")
    a.add_argument("job")
    a.add_argument("--tag", required=True)
    a.add_argument("--run", default=None)
    s = sub.add_parser("score")
    s.add_argument("job")
    s.add_argument("--run", default=None)
    s.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.job, args.recipe)
    elif args.cmd == "fit":
        fit(args)
    elif args.cmd == "assemble":
        assemble(args.job, args.tag, args.run)
    else:
        score(args.job, args.run, args.draws)


if __name__ == "__main__":
    main()
