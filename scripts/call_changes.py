"""Aggregate comparison of two submission files (same ids): sha256, per-class calls, and the
(from, to) class pairs that change. No row is printed.

    python scripts/call_changes.py <base.csv> <candidate.csv>
"""

import hashlib
import sys
from collections import Counter
from pathlib import Path

import polars as pl

base_path, cand_path = map(Path, sys.argv[1:3])
base = pl.read_csv(base_path, infer_schema=False)
cand = pl.read_csv(cand_path, infer_schema=False)
assert base["id"].to_list() == cand["id"].to_list(), "ids or order differ"
for p in (base_path, cand_path):
    print(f"{p}: sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}, rows {cand.height}")
bc, cc = Counter(base["label"].to_list()), Counter(cand["label"].to_list())
print("\n| class | base | candidate | diff |\n|---|---|---|---|")
for k in sorted(set(bc) | set(cc)):
    print(f"| {k} | {bc[k]} | {cc[k]} | {cc[k] - bc[k]:+d} |")
pairs = Counter(
    (a, b) for a, b in zip(base["label"].to_list(), cand["label"].to_list(), strict=True) if a != b
)
print(f"\nchanged rows: {sum(pairs.values())}\n\n| from | to | rows |\n|---|---|---|")
for (a, b), n in pairs.most_common():
    print(f"| {a} | {b} | {n} |")
