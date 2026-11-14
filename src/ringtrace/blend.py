"""Blend finished runs into one submission:  python -m ringtrace.blend recipes/blend.json

A blend recipe names member runs (directories under ``$RINGTRACE_OUT/runs``) and their weights::

    {"name": "blend",
     "members": [{"name": "champion", "recipe": "recipes/champion.json",
                  "run": "blend-champion-m1", "weight": 0.2}, ...],
     "decoder": {"fraud_scale": 0.875},
     "rules": ["anchored_templates"]}

Each member run holds ``oof.npy`` (train rows, ledger order) and ``test.npy`` (October rows). The
blend is the weighted mean of the members' probabilities, decoded per month (October on its own)
with the recipe's decoder (the fixed point unless ``method`` is ``local_search``), then the
recipe's rules. Every member is decoded the same way on its own, so
the report holds the blend's and each member's scores on the given / cleaned / gen-only targets
per held-out month, plus the October rows on which the blend's call differs from each member's.

Writes under ``$RINGTRACE_OUT/runs/<name>/``: ``oof.npy``, ``test.npy``, ``submission.csv`` (in
the sample submission's id order, validated like the champion build) and ``summary.json``.
``--final`` also copies the validated file to ``$RINGTRACE_OUT/submission.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace.classes import CLASSES, N_CLASSES
from ringtrace.decode import decode_by_period
from ringtrace.scoring import macro_f1

TRAIN_MONTHS = (7, 8, 9)
WEIGHT_TOL = 1e-9
# runs save float32 probabilities, so re-decoding them can move a handful of rows; misaligned rows
# would cost far more than this
ALIGN_TOL = 2e-3


class BlendError(ValueError):
    """A member run is missing, misshapen or not aligned with the ledger."""


def load_recipe(path: str | Path) -> dict:
    recipe = json.loads(Path(path).read_text())
    recipe.setdefault("name", Path(path).stem)
    members = recipe.get("members", [])
    if not members:
        raise BlendError("a blend recipe needs at least one member")
    names = [m["name"] for m in members]
    if len(set(names)) != len(names):
        raise BlendError(f"duplicate member names: {names}")
    total = sum(float(m["weight"]) for m in members)
    if abs(total - 1.0) > WEIGHT_TOL or any(float(m["weight"]) < 0 for m in members):
        raise BlendError(f"member weights must be >= 0 and sum to 1 (sum is {total})")
    return recipe


def load_member(run_dir: Path, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """A member's train and October probabilities, checked for shape and sanity."""
    arrays = []
    for fname, n in (("oof.npy", n_train), ("test.npy", n_test)):
        path = run_dir / fname
        if not path.exists():
            raise BlendError(f"{path} is missing")
        a = np.load(path).astype(np.float64)
        if a.shape != (n, N_CLASSES):
            raise BlendError(f"{path} has shape {a.shape}, expected {(n, N_CLASSES)}")
        if not np.isfinite(a).all() or (a < 0).any():
            raise BlendError(f"{path} has negative or non-finite probabilities")
        sums = a.sum(axis=1)
        if np.abs(sums - 1.0).max() > 1e-3:
            raise BlendError(f"{path}: rows do not sum to 1 (worst {np.abs(sums - 1).max():.4g})")
        arrays.append(a)
    return arrays[0], arrays[1]


def weighted_mean(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for a, w in zip(arrays, weights, strict=True):
        out += float(w) * a
    return out


def decode_all(
    proba: np.ndarray,
    period: np.ndarray,
    is_test: np.ndarray,
    decoder: dict,
    rules: list,
    led: pl.DataFrame | None,
    log=print,
) -> dict[str, np.ndarray]:
    """Fixed point per month over every ledger row, then the rules; both stages returned."""
    fixed = np.zeros(len(proba), dtype=np.int64)
    for mask in (~is_test, is_test):
        if mask.any():
            fixed[mask] = decode_by_period(proba[mask], period[mask], **decoder)
    out = {"fixed_point": fixed}
    if rules:
        from ringtrace import rules as rulebook

        out["rules"] = rulebook.apply(rules, fixed, proba, led, log=log)
    return out


def month_scores(pred: np.ndarray, targets: dict[str, np.ndarray], period: np.ndarray) -> dict:
    """Macro-F1 per held-out month and their mean, for every target (train rows only)."""
    res = {}
    for tname, y in targets.items():
        per = [macro_f1(y[period == m], pred[period == m]) for m in TRAIN_MONTHS]
        res[tname] = {"months": per, "mean": float(np.mean(per))}
    return res


def check_alignment(name: str, summary: dict, scores: dict, log=print) -> None:
    """A member's own fixed-point score, recomputed here from its ``oof.npy`` and this ledger's
    labels, must match the score its run reported: the rows are then in the same order."""
    reported = summary.get("fixed_point", {}).get("given", {}).get("months")
    if reported is None:
        log(f"[blend] {name}: no reported fixed-point score, alignment not checked")
        return
    mine = scores["fixed_point"]["given"]["months"]
    if not np.allclose(reported, mine, atol=ALIGN_TOL):
        raise BlendError(
            f"{name}: recomputed given scores {mine} differ from the run's "
            f"{reported}; rows are not aligned with the ledger"
        )
    log(f"[blend] {name}: aligned (given fixed point {np.mean(mine):.4f} as reported)")


def write_submission(path: Path, ids: np.ndarray, calls: np.ndarray, sample_path: Path) -> dict:
    """The October calls in the sample submission's id order, validated; returns the file's
    sha256, the calls per class and any validation problems."""
    from ringtrace.build import validate

    sub = pl.DataFrame({"id": ids, "label": np.asarray(CLASSES)[calls]})
    sample = pl.read_csv(sample_path, infer_schema=False).select("id")
    sample.join(sub, on="id", how="left", maintain_order="left").write_csv(path)
    problems = validate(path, sample_path)
    counts = np.bincount(calls, minlength=N_CLASSES)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "problems": problems,
        "calls": dict(zip(CLASSES, map(int, counts), strict=True)),
    }


def blend(
    recipe: dict,
    led: pl.DataFrame,
    runs_dir: Path,
    targets: dict[str, np.ndarray],
    sample_path: Path,
    out: Path,
    log=print,
) -> dict:
    """Everything but the I/O of the ledger and labels; ``targets`` are train-row arrays."""
    t0 = time.time()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr_idx, te_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    per_tr = period[tr_idx]
    decoder = recipe.get("decoder", {"fraud_scale": 0.875})
    rules = recipe.get("rules", [])
    final = "rules" if rules else "fixed_point"

    def full(oof: np.ndarray, test: np.ndarray) -> np.ndarray:
        p = np.zeros((led.height, N_CLASSES))
        p[tr_idx], p[te_idx] = oof, test
        return p

    def evaluate(proba: np.ndarray) -> tuple[dict, dict]:
        calls = decode_all(proba, period, is_test, decoder, rules, led, log=lambda m: log("  " + m))
        scores = {stage: month_scores(c[tr_idx], targets, per_tr) for stage, c in calls.items()}
        scores["argmax"] = month_scores(proba[tr_idx].argmax(1), targets, per_tr)
        return calls, scores

    report: dict = {"recipe": recipe, "members": {}}
    oofs, tests, member_calls = [], [], {}
    for m in recipe["members"]:
        run_dir = runs_dir / m["run"]
        oof, test = load_member(run_dir, len(tr_idx), len(te_idx))
        calls, scores = evaluate(full(oof, test))
        summ_path = run_dir / "summary.json"
        summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
        check_alignment(m["name"], summary, scores, log=log)
        member_calls[m["name"]] = calls[final]
        report["members"][m["name"]] = {
            "run": m["run"],
            "weight": float(m["weight"]),
            "scores": scores,
            "submission_sha256": summary.get("submission_sha256"),
        }
        log(f"[blend] {m['name']}: " + _line(scores[final]))
        oofs.append(oof)
        tests.append(test)

    weights = [float(m["weight"]) for m in recipe["members"]]
    oof_b, test_b = weighted_mean(oofs, weights), weighted_mean(tests, weights)
    calls, scores = evaluate(full(oof_b, test_b))
    report["scores"] = scores
    report["final"] = scores[final]
    log(f"[blend] {recipe['name']}: " + _line(scores[final]))

    pred = calls[final]
    report["changed_vs_members"] = {
        name: {
            "train": int((c[tr_idx] != pred[tr_idx]).sum()),
            "october": int((c[te_idx] != pred[te_idx]).sum()),
        }
        for name, c in member_calls.items()
    }
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", oof_b.astype(np.float32))
    np.save(out / "test.npy", test_b.astype(np.float32))
    sub = write_submission(
        out / "submission.csv", led["txn_id"].to_numpy()[te_idx], pred[te_idx], sample_path
    )
    report["submission_sha256"] = sub["sha256"]
    report["submission_problems"] = sub["problems"]
    report["test_calls"] = sub["calls"]
    log(f"[blend] October calls: {sub['calls']}")
    log(f"[blend] rows changed vs each member's own decode: {report['changed_vs_members']}")
    log(
        ("VALID" if not sub["problems"] else "INVALID: " + "; ".join(sub["problems"]))
        + f" submission sha256 {sub['sha256']}"
    )
    report["runtime_s"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def _line(scores: dict) -> str:
    return "  ".join(
        f"{t} " + " / ".join(f"{s:.4f}" for s in v["months"]) + f" ({v['mean']:.4f})"
        for t, v in scores.items()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument(
        "--members",
        action="store_true",
        help="print 'recipe run' for each member and exit (for the build script)",
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="also copy the validated file to $RINGTRACE_OUT/submission.csv",
    )
    args = ap.parse_args(argv)
    recipe = load_recipe(args.recipe)
    if args.members:
        for m in recipe["members"]:
            print(m["recipe"], m["run"])
        return 0
    from ringtrace import paths, silver, store

    led = store.ledger()
    tr_idx = np.flatnonzero(~led["is_test"].to_numpy())
    targets = {"given": led["y"].to_numpy().astype(np.int64)[tr_idx]}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr[tr_idx]
    out = paths.out_dir() / "runs" / recipe["name"]
    report = blend(
        recipe,
        led,
        paths.out_dir() / "runs",
        targets,
        paths.data_dir() / "sample_submission.csv",
        out,
    )
    if report["submission_problems"]:
        return 1
    if args.final:
        shutil.copy(out / "submission.csv", paths.out_dir() / "submission.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
