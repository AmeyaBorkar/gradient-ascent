"""An order probe of a submission: the same id -> label rows, permuted.

Uploading the probe beside its source asks whether the leaderboard's public part depends on the
file's row order. Under an id-based public/private split both files score the same; a different
score means the public rows are picked by position. The content is identical, so the probe's
expected private score equals the source's.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shuffled(src: Path, dst: Path, seed: int = 0) -> dict:
    """Write ``src`` with its rows permuted to ``dst`` and check that every id keeps its label."""
    src, dst = Path(src), Path(dst)
    df = pl.read_csv(src, infer_schema=False)
    perm = np.random.default_rng(seed).permutation(df.height)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df[perm].write_csv(dst)
    back = pl.read_csv(dst, infer_schema=False)
    key = df.columns[0]
    joined = df.join(back, on=key, how="full", suffix="_probe", coalesce=True)
    value = df.columns[1]
    same = (
        back.height == df.height
        and back[key].n_unique() == df[key].n_unique()
        and joined[value].equals(joined[f"{value}_probe"])
    )
    moved = int((back[key] != df[key]).sum())
    return {
        "rows": df.height,
        "seed": seed,
        "rows_moved": moved,
        "same_content": bool(same),
        "sha256_source": sha256(src),
        "sha256_probe": sha256(dst),
        "probe": str(dst),
    }
