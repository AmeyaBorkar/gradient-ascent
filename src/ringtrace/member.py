"""Seed-by-seed cascade fits: one set of fits serves several seed subsets of the same recipe.

    python -m ringtrace.member lomo <recipe> --seeds 0-4 --as member-cut04=0-2 --as member-x5=0-4
        [--folds 7]            # fit one held-out month only (one job per fold)
    python -m ringtrace.member cells <recipe> --frames 13,14,15 --seeds 0-14 \\
        --as member-cut04=0-2/3-5/6-8 --as member-x5=0-4/5-9/10-14 --cells <dir>

``lomo`` runs the recipe's leave-one-month-out folds as ``ringtrace.run`` does, but keeps every
seed's probabilities (``$RINGTRACE_OUT/member/<recipe>/lomo/``). Each ``--as name=seeds`` then
writes ``$RINGTRACE_OUT/runs/<name>/`` (``oof.npy``, ``test.npy``, ``summary.json``): the arrays
``python -m ringtrace.run`` gives for the recipe with those seeds, summed in the same order, so
they agree byte for byte (the screening score is seeded on its own and shared by every seed).

``cells`` does the same on the October-like frames, fitted on July + August as
``ringtrace.octlike.fit`` does: ``--as name=g0/g1/g2`` writes cell set r from the r-th seed group
as ``<cells>/<name>_s<frame>_r<r>.npz`` (``rows``, ``prob``), the format the gate reads.

Seeds already stored are not refitted, so a stopped job resumes where it stopped. Several jobs
can share one store with disjoint ``--seeds`` (``--shared``): an output may then name seeds
another job fits, and it is written by whichever job finds all of its seeds stored. Under
``RINGTRACE_DEVICE=cpu`` the seeds are stored apart (``member/<recipe>-cpu/``), so CPU and GPU
fits of a seed never mix in one output; the run summary records the device.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ringtrace import paths
from ringtrace.classes import CLASSES, N_CLASSES

TRAIN_MONTHS = (7, 8, 9)


# ---------------------------------------------------------------------------------------------
# seed lists


def parse_seeds(s: str) -> tuple[int, ...]:
    """``0,1,2`` / ``0-4`` / ``0-2,7`` -> the seeds in the order given; duplicates are refused."""
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, sep, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if sep else [int(a)])
    if not out or len(set(out)) != len(out) or min(out) < 0:
        raise ValueError(f"bad seed list {s!r}")
    return tuple(out)


def parse_outputs(items: list[str], groups: bool) -> dict[str, list[tuple[int, ...]]]:
    """``name=seeds`` -> {name: [seeds]}; with ``groups``, ``name=g0/g1/...`` -> one seed group
    per cell set."""
    out: dict[str, list[tuple[int, ...]]] = {}
    for item in items:
        name, _, rest = item.partition("=")
        if not name or not rest or name in out:
            raise ValueError(f"bad output {item!r}: expected name=seeds, names unique")
        parts = rest.split("/")
        if not groups and len(parts) != 1:
            raise ValueError(f"{item!r}: seed groups are for cells only")
        out[name] = [parse_seeds(p) for p in parts]
    return out


def mean_of(parts: list[np.ndarray]) -> np.ndarray:
    """The seed mean in ``cascade.fit_predict``'s order: zeros, plus each seed, divided."""
    o = np.zeros_like(parts[0], dtype=np.float64)
    for p in parts:
        o += p
    return o / len(parts)


class SeedStore:
    """Per-seed float64 probabilities, one ``<key>_seed<k>.npy`` file each."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str, seed: int) -> Path:
        return self.root / f"{key}_seed{seed}.npy"

    def has(self, key: str, seed: int) -> bool:
        return self.path(key, seed).exists()

    def save(self, key: str, seed: int, a: np.ndarray) -> None:
        p = self.path(key, seed)
        tmp = p.with_name(f".{p.stem}.tmp.npy")
        np.save(tmp, np.asarray(a, dtype=np.float64))
        tmp.replace(p)

    def load(self, key: str, seed: int) -> np.ndarray:
        return np.load(self.path(key, seed))

    def mean(self, key: str, seeds) -> np.ndarray:
        return mean_of([self.load(key, s) for s in seeds])


def fit_seeds(store: SeedStore, keys, X_tr, y_tr, period_tr, X_out, spec, seeds, log=print):
    """Fit the seeds whose outputs are not all stored; ``keys[i]`` names the outputs on
    ``X_out[i]``. Returns the seeds fitted."""
    from ringtrace.cascade import CascadeSpec, seed_outputs

    if not isinstance(spec, CascadeSpec):
        raise TypeError("seed-by-seed fits need a cascade recipe")
    if len(keys) != len(X_out):
        raise ValueError("one key per output matrix")
    todo = tuple(s for s in seeds if not all(store.has(k, s) for k in keys))
    if not todo:
        return ()
    t0 = time.time()
    for seed, res in seed_outputs(
        X_tr, y_tr, period_tr, X_out, dataclasses.replace(spec, seeds=todo), log=log
    ):
        for k, r in zip(keys, res, strict=True):
            store.save(k, seed, r)
        log(f"  seed {seed} stored ({time.time() - t0:.0f}s)")
    return todo


def store_root(recipe_name: str, part: str) -> Path:
    """Where a recipe's per-seed fits live; CPU-forced fits get their own directory."""
    from ringtrace.cascade import device_override

    tag = f"-{device_override()}" if device_override() else ""
    return paths.out_dir() / "member" / f"{recipe_name}{tag}" / part


def _given_only(recipe: dict) -> None:
    if recipe.get("train_on") not in (None, "given"):
        raise ValueError("seed-by-seed fits train on the given labels only")


# ---------------------------------------------------------------------------------------------
# leave-one-month-out runs


def write_run(name: str, recipe: dict, oof_tr, test, led, log=print) -> dict:
    """A run directory as ``ringtrace.run`` writes it (arrays, scores per month, October calls)."""
    from ringtrace import rules as rulebook
    from ringtrace import silver
    from ringtrace.decode import decode_by_period
    from ringtrace.run import check_proba, score_block
    from ringtrace.scoring import per_class_f1

    out = paths.out_dir() / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y = led["y"].to_numpy().astype(np.int64)
    tr_idx, te_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    check_proba(oof_tr, f"{name}: train rows")
    check_proba(test, f"{name}: October rows")
    targets = {"given": y[tr_idx]}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr[tr_idx]
    from ringtrace.cascade import device_override

    report: dict = {"recipe": recipe, "device": device_override() or "recipe"}
    per_tr = period[tr_idx]
    dec = recipe.get("decoder", {"fraud_scale": 0.875})
    pred = np.zeros(len(y), dtype=np.int64)
    pred[tr_idx] = decode_by_period(oof_tr, per_tr, **dec)
    pred[te_idx] = decode_by_period(test, period[te_idx], **dec)
    score_block("fixed_point", pred[tr_idx], targets, per_tr, report, log)
    rules = recipe.get("rules", [])
    if rules:
        proba = np.zeros((len(y), N_CLASSES))
        proba[tr_idx], proba[te_idx] = oof_tr, test
        pred = rulebook.apply(rules, pred, proba, led, log=log)
        score_block("rules", pred[tr_idx], targets, per_tr, report, log)
    report["final"] = report["rules" if rules else "fixed_point"]
    report["per_class"] = {
        t: dict(zip(CLASSES, map(float, per_class_f1(v, pred[tr_idx])), strict=True))
        for t, v in targets.items()
    }
    np.save(out / "oof.npy", oof_tr.astype(np.float32))
    np.save(out / "test.npy", test.astype(np.float32))
    calls = np.bincount(pred[te_idx], minlength=N_CLASSES)
    report["test_calls"] = {CLASSES[k]: int(calls[k]) for k in range(N_CLASSES) if calls[k]}
    report["arrays_sha256"] = {
        f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("oof.npy", "test.npy")
    }
    (out / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"[member] {name}: October calls {report['test_calls']}")
    return report


def lomo(recipe_path: str, seeds, outputs: dict, folds=TRAIN_MONTHS, log=print) -> dict:
    """Fit the held-out months in ``folds`` (the others are left as stored, so several jobs can
    fit one fold each); write each output once every fold of it is stored."""
    from ringtrace import run as runner
    from ringtrace import store

    t0 = time.time()
    recipe = runner.load_recipe(recipe_path)
    _given_only(recipe)
    store.use_frame(None)
    led = store.ledger()
    X, cols = runner.feature_matrix(recipe)
    spec, _ = runner.model(recipe)
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y = led["y"].to_numpy().astype(np.int64)
    tr_idx, te_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    st = SeedStore(store_root(recipe["name"], "lomo"))
    log(f"[member] {recipe['name']}: {len(cols)} features, seeds {list(seeds)}")
    for m in folds:
        fit = tr_idx[period[tr_idx] != m]
        hold = tr_idx[period[tr_idx] == m]
        log(f"fold {m}: train {len(fit)} rows, hold {len(hold)} ({time.time() - t0:.0f}s)")
        fit_seeds(
            st,
            [f"hold{m}", f"oct{m}"],
            X[fit],
            y[fit],
            period[fit],
            [X[hold], X[te_idx]],
            spec,
            seeds,
            log,
        )
    del X
    reports = {}
    for name, (sub,) in outputs.items():
        if not all(
            st.has(f"{k}{m}", s) for m in TRAIN_MONTHS for k in ("hold", "oct") for s in sub
        ):
            log(f"[member] {name}: not every fold is stored yet, not written")
            continue
        oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
        test_sum = np.zeros((len(te_idx), N_CLASSES), dtype=np.float64)
        for m in TRAIN_MONTHS:
            oof[tr_idx[period[tr_idx] == m]] = st.mean(f"hold{m}", sub)
            test_sum += st.mean(f"oct{m}", sub)
        rec = json.loads(json.dumps(recipe))
        rec["name"] = name
        rec.setdefault("cascade", {})["seeds"] = list(sub)
        reports[name] = write_run(name, rec, oof[tr_idx], test_sum / len(TRAIN_MONTHS), led, log)
    log(f"[member] done in {time.time() - t0:.0f}s")
    return reports


# ---------------------------------------------------------------------------------------------
# October-like cells


def cells(recipe_path: str, frames, seeds, outputs: dict, cells_dir: Path, log=print) -> dict:
    from ringtrace import octlike as ol
    from ringtrace import run as runner
    from ringtrace import store
    from ringtrace.scoring import macro_f1

    t0 = time.time()
    recipe = runner.load_recipe(recipe_path)
    _given_only(recipe)
    cells_dir = Path(cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    scores: dict = {}
    try:
        for fs in frames:
            frame = ol.Frame(fs)
            y = frame.led["y"].to_numpy().astype(np.int64)
            fit_rows = frame.fit & np.isin(frame.period, ol.FIT_PERIODS)
            X, _ = runner.feature_matrix(recipe)
            spec, _ = runner.model(recipe)
            st = SeedStore(store_root(recipe["name"], f"s{fs}"))
            log(
                f"[member] frame s{fs}: fit {int(fit_rows.sum())}, September "
                f"{int(frame.hold.sum())} ({time.time() - t0:.0f}s)"
            )
            fit_seeds(
                st,
                ["sept"],
                X[fit_rows],
                y[fit_rows],
                frame.period[fit_rows],
                [X[frame.hold]],
                spec,
                seeds,
                log,
            )
            del X
            rows = frame.stacked[frame.hold]
            for name, groups in outputs.items():
                for r, g in enumerate(groups):
                    if not all(st.has("sept", k) for k in g):
                        log(f"[member] {name} s{fs} r{r}: seeds not all stored yet, not written")
                        continue
                    prob = st.mean("sept", g)
                    path = cells_dir / f"{name}_s{fs}_r{r}.npz"
                    tmp = path.with_name(f".{path.stem}.tmp.npz")
                    np.savez(tmp, rows=rows, prob=prob.astype(np.float32))
                    tmp.replace(path)
                    pred = ol.decode(frame, prob, recipe.get("decoder"), recipe.get("rules", []))
                    sc = {t: macro_f1(v[frame.hold], pred) for t, v in frame.targets.items()}
                    scores[path.stem] = {**sc, "seeds": list(g)}
                    log(
                        f"[member] {path.stem}: September "
                        + ", ".join(f"{t} {v:.4f}" for t, v in sc.items())
                    )
    finally:
        store.use_frame(None)
    log(f"[member] cells done in {time.time() - t0:.0f}s")
    return scores


def main(argv=None):
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.member")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lo = sub.add_parser("lomo")
    lo.add_argument("recipe")
    lo.add_argument("--seeds", required=True)
    lo.add_argument("--folds", default="7,8,9", help="held-out months to fit in this job")
    lo.add_argument("--as", dest="outputs", action="append", default=[], required=True)
    ce = sub.add_parser("cells")
    ce.add_argument("recipe")
    ce.add_argument("--frames", default="13,14,15")
    ce.add_argument("--seeds", required=True)
    ce.add_argument("--as", dest="outputs", action="append", default=[], required=True)
    ce.add_argument("--cells", required=True, help="directory of the cell files")
    for p in (lo, ce):
        p.add_argument(
            "--shared",
            action="store_true",
            help="outputs may use seeds that another job fits into the same store",
        )
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    outputs = parse_outputs(a.outputs, groups=a.cmd == "cells")
    missing = {s for gs in outputs.values() for g in gs for s in g} - set(seeds)
    if missing and not a.shared:
        raise SystemExit(f"outputs use seeds that are not fitted: {sorted(missing)}")
    if a.cmd == "lomo":
        folds = parse_seeds(a.folds)
        if not set(folds) <= set(TRAIN_MONTHS):
            raise SystemExit(f"folds must be among {TRAIN_MONTHS}")
        return lomo(a.recipe, seeds, outputs, folds)
    return cells(a.recipe, parse_seeds(a.frames), seeds, outputs, Path(a.cells))


if __name__ == "__main__":
    main()
