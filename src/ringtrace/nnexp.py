"""Training-technique experiments for TabM: one fit on July + August per variant and cell.

    python -m ringtrace.nnexp teacher [--cells sep,s13,s14,s15]
    python -m ringtrace.nnexp fit <variant> [--cells sep,s13,s14,s15] [--seeds 0,1,2]
    python -m ringtrace.nnexp score <variant> [--base base] [--cells ...]
    python -m ringtrace.nnexp table <variant> [--base base]
    python -m ringtrace.nnexp mean <name> --of a,b,c [--cells ...]
    python -m ringtrace.nnexp member <name> --of <run> [--cells ...]

Cells: ``sep`` is the normal ledger with September held out (the leave-one-month-out fold 9);
``s13``..``s15`` are the October-like frames (``ringtrace.octlike``). Every variant is
``recipes/tabm.json`` plus the spec changes in ``VARIANTS``. September's probabilities go to
``$RINGTRACE_OUT/nntech/<variant>_<cell>.npz`` (``rows``, ``prob``); seeds other than 0-2 add
``-sd<seeds>`` to the name, which ``score`` then takes as the variant name.

``teacher`` cross-fits a booster between July and August (fit one, predict the other) on each
cell's fit rows: out-of-fold soft targets that never see September.

``score`` judges a variant against the base twice, with the same seeds: TabM alone, and
fraud-only TabM in the tree blend (legit column from TabM at 1/2 plus the trees at 1/2, fraud
types from the trees, see ``blend.fraud_split_from``). Trees: the five-way blend's fold-9
probabilities on ``sep``; on the October-like cells, the tree members whose r0 cells exist.
Calls: fixed point, then the template rule and the blank merchant QR rule. Aggregates only.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import paths

VARIANTS: dict[str, dict] = {
    "base": {},
    "kd30": {"distill_alpha": 0.3},
    "kd70": {"distill_alpha": 0.7},
    "pre": {"pretrain_epochs": 3},
    "qraug": {"qr_blank_p": 0.44},
    "mixup": {"mixup_alpha": 0.2},
    "noise": {"noise_std": 0.05},
    "ema": {"ema_decay": 0.999},
    "swa": {"swa_last": 3},
    "logitadj": {"loss": "logit_adjust", "logit_tau": 1.0},
    "focal": {"loss": "focal", "focal_gamma": 2.0},
    "smooth": {"smooth_fraud": 0.03},
}
# Teacher of the self-training variant: the plain seed-0 TabM's probabilities of the scored rows
# (fit on July + August, so it never saw their labels). Every pseudo row keeps its soft target.
# The cap is floor(sum of the teacher's mass) for each fraud class; legit is uncapped, because a
# legit cap drops the least confident legit-argmax rows, which hold the ambiguous fraud mass.
SELFTRAIN = {"teacher": "base-sd0", "w_pseudo": 0.3, "min_conf": 0.0}
TREES_SEP = ("blend-champion-m1", "lgbm", "catboost", "clean-event", "clean-l2")
TREES_OCT = ("champion", "lgbm", "catboost", "clean-event", "clean-l2")
RULES = ["anchored_templates", "blank_merchant_qr"]
FIT_PERIODS = (7, 8)


def variant_spec(name: str, recipe: dict):
    """The TabM spec of a variant: the recipe's, then ``VARIANTS[name]`` (``a+b`` combines)."""
    from ringtrace import run

    spec, _ = run.model(recipe)
    for part in name.split("+"):
        spec = replace(spec, **VARIANTS[part])
    return spec


def _dir() -> Path:
    d = paths.out_dir() / "nntech"
    d.mkdir(parents=True, exist_ok=True)
    return d


class SepCell:
    """The normal ledger, fit on July + August, September held out."""

    def __init__(self):
        from ringtrace import silver, store

        store.use_frame(None)
        led = store.ledger()
        self.led = led
        self.period = led["period"].to_numpy()
        is_test = led["is_test"].to_numpy()
        self.fit = ~is_test & np.isin(self.period, FIT_PERIODS)
        self.hold = ~is_test & (self.period == 9)
        self.stacked = np.arange(led.height)
        self.unit = np.arange(led.height)
        self.targets = {"given": led["y"].to_numpy().astype(np.int64)}
        for tname, which in (("cleaned", "label"), ("gen", "structured")):
            arr = silver.load(led, which)
            if arr is not None:
                self.targets[tname] = arr.astype(np.int64)


def open_cell(name: str):
    from ringtrace.octlike import Frame

    return SepCell() if name == "sep" else Frame(int(name.lstrip("s")))


def qr_person(led: pl.DataFrame) -> np.ndarray:
    """QR payments to a person whose check is filled (the rows October blanks)."""
    qr = (led["txn_type"] == "QR").to_numpy() & led["to_person"].to_numpy()
    return qr & led["qr_vpa_mismatch"].fill_nan(None).is_not_null().to_numpy()


def blanked_matrix(recipe: dict) -> np.ndarray:
    """The cell's feature matrix with the QR check blanked on every QR-to-person row (every
    family rebuilt on that ledger under its own cache key), then the cell's ledger restored."""
    from ringtrace import run, store

    led = store.ledger()
    saved = (led, store._KEY, dict(store._FIXED))
    blank = pl.Series(qr_person(led))
    led_b = led.with_columns(
        pl.when(blank).then(None).otherwise(pl.col("qr_vpa_mismatch")).alias("qr_vpa_mismatch")
    )
    store.use_frame(led_b, key=f"{store._ledger_key()}-qrblank", fixed=saved[2])
    try:
        X, _ = run.feature_matrix(recipe)
    finally:
        store.use_frame(*saved)
    return X


def teacher(cells: list[str], log=print) -> None:
    """Out-of-fold soft targets for each cell's fit rows: ``teacher.teacher_oof`` (one weighted
    multiclass booster per month, fitted on the other fit month) on the TabM recipe's features.
    The champion cascade cannot be used here: its screening stage is itself cross-fitted between
    months, so it needs two training months."""
    from ringtrace import run
    from ringtrace import teacher as tch

    recipe = run.load_recipe(Path("recipes") / "tabm.json")
    for c in cells:
        path = _dir() / f"teacher_{c}.npy"
        if path.exists():
            continue
        t0 = time.time()
        cell = open_cell(c)
        X, _ = run.feature_matrix(recipe)
        y = cell.targets["given"]
        p = tch.teacher_oof(X[cell.fit], y[cell.fit], cell.period[cell.fit])
        np.save(path, p.astype(np.float32))
        log(f"[nnexp] teacher {c}: {int(cell.fit.sum())} rows ({time.time() - t0:.0f}s)")


def _fit_selftrain(c: str, cell, X: np.ndarray, spec, log) -> np.ndarray:
    """Soft self-training on the scored rows' features (never their labels)."""
    from ringtrace import nn_selftrain

    z = np.load(_dir() / f"{SELFTRAIN['teacher']}_{c}.npz")
    if not np.array_equal(z["rows"], cell.stacked[cell.hold]):
        raise ValueError(f"cell {c}: teacher rows differ")
    P = z["prob"].astype(np.float64)
    cap = np.floor(P.sum(axis=0)).astype(np.int64)
    cap[0] = -1
    cap_map = {i: int(v) for i, v in enumerate(cap) if v >= 0}
    y = cell.targets["given"]
    (prob,) = nn_selftrain.fit_predict_selftrain(
        X[cell.fit],
        y[cell.fit],
        X[cell.hold],
        P,
        [X[cell.hold]],
        spec,
        w_pseudo=SELFTRAIN["w_pseudo"],
        min_conf=SELFTRAIN["min_conf"],
        cap=cap_map,
        log=log,
    )
    return prob


# Entity variant: hashed embeddings of these id columns, plus the strictly-past mean of up to
# ``ENTITY_POOL_N`` earlier rows of the same sender and of the same receiver over these raw
# columns (features only, from the cell's own ledger; a row never sees itself or a tie).
ENTITY_IDS = ("sender", "receiver", "device")
ENTITY_POOL_BY = ("sender", "receiver")
ENTITY_POOL_COLS = (
    "amount",
    "new_payee",
    "pin_retries",
    "login_to_txn_sec",
    "sim_age_hours",
    "device_age_days",
    "balance_before",
    "to_person",
)
ENTITY_POOL_N = 8


def entity_inputs(led: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(ids, pooled)``: the id matrix and the past-pooled columns (one block per entity)."""
    from ringtrace import nn_entity

    ids = np.stack([led[c].to_numpy() for c in ENTITY_IDS], axis=1).astype(np.int64)
    raw = led.select(
        pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in ENTITY_POOL_COLS
    ).to_numpy()
    ts = led["ts"].to_numpy()
    pooled = [
        nn_entity.past_pool(raw, led[e].to_numpy(), ts, n=ENTITY_POOL_N) for e in ENTITY_POOL_BY
    ]
    return ids, np.hstack(pooled)


def _fit_entity(c: str, cell, X: np.ndarray, spec, log, pool: bool = True) -> np.ndarray:
    """TabM with id embeddings (fit on the fit rows' labels only) and past-pooled history."""
    from dataclasses import asdict

    from ringtrace import nn_entity

    ids, pooled = entity_inputs(cell.led)
    Xe = np.hstack([X, pooled]) if pool else X
    seen = {
        e: float(np.isin(ids[cell.hold, j], ids[cell.fit, j]).mean())
        for j, e in enumerate(ENTITY_IDS)
    }
    log(f"[nnexp] {c}: share of scored rows whose id was seen in the fit rows {seen}")
    espec = nn_entity.EntitySpec(**asdict(spec))
    (prob,) = nn_entity.fit_predict_entity(
        Xe[cell.fit],
        ids[cell.fit],
        cell.targets["given"][cell.fit],
        None,
        [Xe[cell.hold]],
        [ids[cell.hold]],
        espec,
        log=log,
    )
    return prob


def _fit_entity_pool(c: str, cell, X: np.ndarray, spec, log) -> np.ndarray:
    """Ablation: the past-pooled history columns alone, in the plain TabM."""
    from ringtrace import tabm

    _, pooled = entity_inputs(cell.led)
    Xe = np.hstack([X, pooled])
    (prob,) = tabm.fit_predict(
        Xe[cell.fit],
        cell.targets["given"][cell.fit],
        cell.period[cell.fit],
        [Xe[cell.hold]],
        spec,
        log=log,
    )
    return prob


def _fit_entity_emb(c: str, cell, X: np.ndarray, spec, log) -> np.ndarray:
    """Ablation: the id embeddings alone, on the plain TabM inputs."""
    return _fit_entity(c, cell, X, spec, log, pool=False)


def _fit_tabr(c: str, cell, X: np.ndarray, spec, log) -> np.ndarray:
    """The TabR-style retrieval member; its memory is the fit rows only."""
    from ringtrace import tabr

    (prob,) = tabr.fit_predict(
        X[cell.fit],
        cell.targets["given"][cell.fit],
        cell.period[cell.fit],
        [X[cell.hold]],
        tabr.TabRSpec(seeds=spec.seeds),
        log=log,
    )
    return prob


# Shift variant: the fit rows weighted by a domain classifier toward the scored rows' features.
# On every feature the classifier tells September from July + August perfectly (AUC 1.0000),
# from calendar-like columns. Columns whose own AUC between July + August and the unthinned
# September is more than ``calendar_gap`` from 0.5 are left out of the classifier in every cell:
# between those months nothing but the calendar moves, so what they detect is the month.
SHIFT = {"clip": (0.2, 5.0), "sample": 300_000, "rounds": 200, "calendar_gap": 0.05}


def _shift_columns(c: str, cell, X: np.ndarray, log) -> np.ndarray:
    """Columns the domain classifier may read: fixed on ``sep``, then reused by every cell."""
    from ringtrace import nn_shift

    path = _dir() / "shift-columns.json"
    if c == "sep":
        gap = nn_shift.column_auc_gap(X[cell.fit], X[cell.hold])
        keep = np.flatnonzero(gap <= SHIFT["calendar_gap"])
        drop = np.flatnonzero(gap > SHIFT["calendar_gap"])
        path.write_text(
            json.dumps(
                {
                    "keep": keep.tolist(),
                    "drop": drop.tolist(),
                    "gap_of_dropped": [round(float(g), 4) for g in gap[drop]],
                },
                indent=1,
            )
        )
        log(f"[nnexp] shift: {drop.size} calendar-like columns left out, {keep.size} kept")
        return keep
    if not path.exists():
        raise FileNotFoundError("fit the shift variant on sep first: it fixes the columns")
    return np.asarray(json.loads(path.read_text())["keep"], dtype=np.intp)


def _fit_shift(c: str, cell, X: np.ndarray, spec, log) -> np.ndarray:
    """TabM on the fit rows reweighted toward the scored rows (features only, never labels)."""
    from ringtrace import nn_shift, threads

    cols = _shift_columns(c, cell, X, log)
    w, info = nn_shift.domain_weights(
        X[cell.fit][:, cols],
        X[cell.hold][:, cols],
        clip=SHIFT["clip"],
        sample=SHIFT["sample"],
        rounds=SHIFT["rounds"],
        seed=0,
        threads=threads.budget(),
        log=log,
    )
    (_dir() / f"shift-info_{c}.json").write_text(json.dumps(info, indent=1))
    (prob,) = nn_shift.fit_predict_weighted(
        X[cell.fit], cell.targets["given"][cell.fit], w, [X[cell.hold]], spec, log=log
    )
    return prob


# Metric variant: cross-entropy plus a REINFORCE estimate of batch macro-F1.
METRIC = {"metric_loss": "reinforce", "lam": 0.5}


def _fit_metric(c: str, cell, X: np.ndarray, spec, log) -> np.ndarray:
    from ringtrace import nn_metric

    (prob,) = nn_metric.fit_predict_metric(
        X[cell.fit],
        cell.targets["given"][cell.fit],
        cell.period[cell.fit],
        [X[cell.hold]],
        spec,
        log=log,
        **METRIC,
    )
    return prob


def _fit_gentarget(c: str, cell, X: np.ndarray, spec, log) -> np.ndarray:
    """The plain TabM trained on generator-only labels of the fit rows.

    The cached silver labels cannot be used: their teacher is out of fold by month, so the July
    and August labels come from a model that saw September. The target is rebuilt per cell from
    a booster cross-fitted between July and August, with every estimate on the fit rows only
    (``octlike._cleaned_target``)."""
    from ringtrace import tabm
    from ringtrace.octlike import _cleaned_target

    target, stats = _cleaned_target(cell, log, which="structured")
    (_dir() / f"gentarget-stats_{c}.json").write_text(json.dumps(stats, indent=1))
    t = target[cell.fit]
    if (t < 0).any():
        raise ValueError(f"cell {c}: {int((t < 0).sum())} fit rows have no generator-only label")
    y = cell.targets["given"][cell.fit]
    log(
        f"[nnexp] {c}: generator-only target keeps {int((t > 0).sum())} of {int((y > 0).sum())}"
        f" fraud labels ({int(((t > 0) & (y == 0)).sum())} legit-labelled rows become fraud)"
    )
    (prob,) = tabm.fit_predict(X[cell.fit], t, cell.period[cell.fit], [X[cell.hold]], spec, log=log)
    return prob


# Variants that are their own fit function, not a TabMSpec switch: name -> fit(c, cell, X, spec).
FAMILIES = {
    "selftrain": _fit_selftrain,
    "entity": _fit_entity,
    "entitypool": _fit_entity_pool,
    "entityemb": _fit_entity_emb,
    "tabr": _fit_tabr,
    "shift": _fit_shift,
    "metric": _fit_metric,
    "gentarget": _fit_gentarget,
}


def tag_of(variant: str, seeds: tuple[int, ...], job: str = "") -> str:
    """Output name of a fit: ``[<job>-]<variant>[-sd<seeds>]``."""
    tag = variant if seeds == (0, 1, 2) else f"{variant}-sd{''.join(map(str, seeds))}"
    return f"{job}-{tag}" if job else tag


def fit(variant: str, cells: list[str], seeds: tuple[int, ...], log=print, job: str = "") -> None:
    from ringtrace import run, tabm

    recipe = run.load_recipe(Path("recipes") / "tabm.json")
    fit_predict = tabm.fit_predict
    if variant == "realmlp":  # the second neural family, on the same features
        from ringtrace import realmlp

        spec, fit_predict = realmlp.RealMLPSpec(seeds=seeds), realmlp.fit_predict
    elif variant in FAMILIES:
        spec = replace(variant_spec("base", recipe), seeds=seeds)
    else:
        spec = replace(variant_spec(variant, recipe), seeds=seeds)
    tag = tag_of(variant, seeds, job)
    for c in cells:
        path = _dir() / f"{tag}_{c}.npz"
        if path.exists():
            log(f"[nnexp] {path.name} exists, skipped")
            continue
        t0 = time.time()
        cell = open_cell(c)
        X, _ = run.feature_matrix(recipe)
        y = cell.targets["given"]
        if variant in FAMILIES:
            prob = FAMILIES[variant](c, cell, X, spec, log)
            np.savez(path, rows=cell.stacked[cell.hold], prob=prob.astype(np.float32))
            log(f"[nnexp] {tag} {c} done ({time.time() - t0:.0f}s)")
            continue
        ex = tabm.Extras()
        if getattr(spec, "distill_alpha", 0) > 0:
            teacher([c], log)  # built once per cell, then read back
            cell = open_cell(c)
            X, _ = run.feature_matrix(recipe)
            ex.teacher = np.load(_dir() / f"teacher_{c}.npy")
        if getattr(spec, "qr_blank_p", 0) > 0:
            ex.X_blank = blanked_matrix(recipe)[cell.fit]
            ex.blank_rows = np.flatnonzero(qr_person(cell.led)[cell.fit])
            log(f"[nnexp] {c}: {len(ex.blank_rows)} QR-to-person fit rows can be blanked")
        if getattr(spec, "pretrain_epochs", 0) > 0:
            ex.X_unlabelled = X[~cell.fit]  # September and October: inputs only
        (prob,) = fit_predict(
            X[cell.fit],
            y[cell.fit],
            cell.period[cell.fit],
            [X[cell.hold]],
            spec,
            log=log,
            extras=ex,
        )
        np.savez(path, rows=cell.stacked[cell.hold], prob=prob.astype(np.float32))
        log(f"[nnexp] {tag} {c} done ({time.time() - t0:.0f}s)")


def tree_prob(cell_name: str, cell) -> np.ndarray:
    """Equal-weight mean of the tree members' September probabilities in this cell."""
    if cell_name == "sep":
        from ringtrace.blend import blend_arrays

        tr = np.flatnonzero(~cell.led["is_test"].to_numpy())
        oof = blend_arrays([(n, 1 / len(TREES_SEP)) for n in TREES_SEP])
        return oof[np.isin(tr, np.flatnonzero(cell.hold))]
    seed = int(cell_name.lstrip("s"))
    probs = []
    for n in TREES_OCT:
        f = paths.out_dir() / "octlike" / f"{n}_s{seed}_r0.npz"
        if f.exists():
            z = np.load(f)
            if not np.array_equal(z["rows"], cell.stacked[cell.hold]):
                raise ValueError(f"{n}: rows differ in cell {cell_name}")
            probs.append(z["prob"].astype(np.float64))
    if not probs:
        raise FileNotFoundError(f"no tree cells for {cell_name}")
    return np.mean(probs, axis=0)


def fraud_only(tabm_prob: np.ndarray, trees: np.ndarray) -> np.ndarray:
    from ringtrace.blend import fraud_split_from

    return fraud_split_from(trees, 0.5 * tabm_prob + 0.5 * trees)


def _decode(cell, prob: np.ndarray) -> np.ndarray:
    from ringtrace.octlike import decode

    return decode(cell, prob, None, RULES)


def score(variant: str, base: str, cells: list[str], draws: int = 1000, log=print) -> dict:
    from ringtrace.octlike import bootstrap
    from ringtrace.scoring import macro_f1

    report: dict = {"variant": variant, "base": base, "cells": {}}
    for c in cells:
        cell = open_cell(c)
        load = [np.load(_dir() / f"{v}_{c}.npz") for v in (base, variant)]
        for z in load:
            if not np.array_equal(z["rows"], cell.stacked[cell.hold]):
                raise ValueError(f"cell {c}: rows differ")
        pa, pb = (z["prob"].astype(np.float64) for z in load)
        trees = tree_prob(c, cell)
        res: dict = {}
        units = cell.unit[cell.hold]
        for role, (qa, qb) in {
            "alone": (pa, pb),
            "fraud_only": (fraud_only(pa, trees), fraud_only(pb, trees)),
        }.items():
            ya, yb = _decode(cell, qa), _decode(cell, qb)
            for t, truth in cell.targets.items():
                yt = truth[cell.hold]
                r = {"base": macro_f1(yt, ya), "variant": macro_f1(yt, yb)}
                if c != "sep":
                    r.update(bootstrap(units, yt, ya, yb, draws, seed=int(c[1:])))
                else:
                    r["delta"] = r["variant"] - r["base"]
                res.setdefault(role, {})[t] = r
            res[role]["changed_calls"] = int((ya != yb).sum())
        report["cells"][c] = res
        log(
            f"[nnexp] {variant} vs {base} {c}: "
            + "  ".join(
                f"{role} "
                + " ".join(f"{t} {res[role][t]['delta']:+.5f}" for t in cell.targets)
                + f" ({res[role]['changed_calls']} calls)"
                for role in res
            )
        )
    (_dir() / f"score_{variant}_vs_{base}.json").write_text(json.dumps(report, indent=1))
    return report


def mean_of(name: str, members: list[str], cells: list[str], log=print) -> None:
    """Save the equal-weight mean of fitted variants as variant ``name``, cell by cell."""
    for c in cells:
        zs = [np.load(_dir() / f"{m}_{c}.npz") for m in members]
        rows = zs[0]["rows"]
        for m, z in zip(members, zs, strict=True):
            if not np.array_equal(z["rows"], rows):
                raise ValueError(f"cell {c}: rows of {m} differ")
        prob = np.mean([z["prob"].astype(np.float64) for z in zs], axis=0)
        np.savez(_dir() / f"{name}_{c}.npz", rows=rows, prob=prob.astype(np.float32))
        log(f"[nnexp] {name} {c}: mean of {len(members)}")


def member_cells(run_name: str, name: str, cells: list[str], log=print) -> None:
    """A full member as a screen variant: its leave-one-month-out September rows on ``sep``
    (fit on July + August, like the screen's fits) and its r0 October-like cells."""
    for c in cells:
        if c == "sep":
            cell = SepCell()
            tr = np.flatnonzero(~cell.led["is_test"].to_numpy())
            oof = np.load(paths.out_dir() / "runs" / run_name / "oof.npy")
            rows = cell.stacked[cell.hold]
            prob = oof[np.isin(tr, np.flatnonzero(cell.hold))]
        else:
            z = np.load(paths.out_dir() / "octlike" / f"{run_name}_{c}_r0.npz")
            rows, prob = z["rows"], z["prob"]
        np.savez(_dir() / f"{name}_{c}.npz", rows=rows, prob=np.asarray(prob, dtype=np.float32))
        log(f"[nnexp] {name} {c}: {len(rows)} rows from {run_name}")


def table_rows(report: dict, name: str | None = None) -> list[str]:
    """Markdown rows of a ``score`` report: per role, each cell's change x 1e-4 on the targets
    in order (given / cleaned / gen-only), the calls changed, and whether no target loses."""
    name = name or report["variant"]
    cells = list(report["cells"])
    rows = []
    for role in ("fraud_only", "alone"):
        parts, worst = [], 0.0
        for c in cells:
            r = report["cells"][c][role]
            ds = [r[t]["delta"] for t in ("given", "cleaned", "gen") if t in r]
            worst = min([worst, *ds])
            parts.append(" / ".join(f"{1e4 * d:+.1f}" for d in ds) + f" ({r['changed_calls']})")
        label = role.replace("_", "-")
        verdict = "yes" if worst >= 0 else "no"
        rows.append(f"| {name} | {label} | " + " | ".join(parts) + f" | {verdict} |")
    return rows


def main(argv=None) -> None:
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.nnexp")
    ap.add_argument("cmd", choices=["teacher", "fit", "score", "table", "mean", "member"])
    ap.add_argument("variant", nargs="?", default="base")
    ap.add_argument("--cells", default="sep,s13,s14,s15")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--base", default="base")
    ap.add_argument("--job", default="", help="prefix of the fit's output name")
    ap.add_argument("--of", default="", help="mean: comma list of variants; member: the run")
    args = ap.parse_args(argv)
    cells = [c for c in args.cells.split(",") if c]
    if args.cmd == "teacher":
        teacher(cells)
    elif args.cmd == "fit":
        for v in args.variant.split(","):
            fit(v, cells, tuple(int(s) for s in args.seeds.split(",")), job=args.job)
    elif args.cmd == "mean":
        mean_of(args.variant, [m for m in args.of.split(",") if m], cells)
    elif args.cmd == "member":
        member_cells(args.of, args.variant, cells)
    elif args.cmd == "table":
        for v in args.variant.split(","):
            report = json.loads((_dir() / f"score_{v}_vs_{args.base}.json").read_text())
            print("\n".join(table_rows(report)))
    else:
        for v in args.variant.split(","):
            score(v, args.base, cells)


if __name__ == "__main__":
    main()
