"""The one command:  python -m ringtrace.build recipes/champion.json

1. builds (or reuses) every feature family the recipe names;
2. freezes the cleaned labels if they are not in the cache yet (only used for scoring);
3. runs the recipe on the month folds and predicts October;
4. validates the submission against ``sample_submission.csv`` and writes it to
   ``$RINGTRACE_OUT/submission.csv`` with its sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys

import polars as pl

from ringtrace import paths
from ringtrace.classes import CLASSES


def validate(path, sample_path) -> list[str]:
    """Problems with a submission file; empty when it is valid."""
    problems = []
    sub = pl.read_csv(path, infer_schema=False)
    sample = pl.read_csv(sample_path, infer_schema=False)
    if sub.columns != ["id", "label"]:
        problems.append(f"header is {sub.columns}, expected ['id', 'label']")
        return problems
    if sub.height != sample.height:
        problems.append(f"{sub.height} rows, sample has {sample.height}")
    elif not (sub["id"] == sample["id"]).all():
        problems.append("ids differ from the sample (order or values)")
    bad = set(sub["label"].unique().to_list()) - set(CLASSES)
    if bad:
        problems.append(f"unknown labels: {sorted(bad)}")
    if sub["label"].null_count():
        problems.append("missing labels")
    if path.stat().st_size > 15 * 1024 * 1024:
        problems.append("file larger than 15 MB")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    args = ap.parse_args(argv)
    from ringtrace import run, silver, store

    recipe = run.load_recipe(args.recipe)
    led = store.ledger()
    for fam in recipe["families"]:
        store.family(fam)
    if silver.load(led, "label") is None:
        from ringtrace import teacher

        teacher.main([])
    report = run.main([args.recipe])
    sub_path = paths.out_dir() / "runs" / recipe["name"] / "submission.csv"
    sample_path = paths.data_dir() / "sample_submission.csv"
    # the sample's id order is the contract: reorder ours to it
    sub = pl.read_csv(sub_path, infer_schema=False)
    sample = pl.read_csv(sample_path, infer_schema=False).select("id")
    ordered = sample.join(sub, on="id", how="left", maintain_order="left")
    final = paths.out_dir() / "submission.csv"
    ordered.write_csv(final)
    problems = validate(final, sample_path)
    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    if problems:
        print("INVALID: " + "; ".join(problems))
        return 1
    shutil.copy(final, sub_path.with_name("submission_validated.csv"))
    print(f"VALID submission {final} sha256 {digest}")
    print(f"final scores: {report['final']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
