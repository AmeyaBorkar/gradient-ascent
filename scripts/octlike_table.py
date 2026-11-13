"""Print the October-like summary as a Markdown table (run on the box, aggregates only).

    python scripts/octlike_table.py [~/outputs/octlike]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/outputs/octlike").expanduser()
rows = []
cleaning = {}
for f in sorted(out.glob("summary_s*.json")):
    s = json.loads(f.read_text())
    seed = f.stem.split("_s")[1]
    for k, v in s.items():
        if k.endswith("_cleaning"):
            cleaning[seed] = v
        elif isinstance(v, dict) and "given_fixed_point" in v:
            rows.append((k, v))
print("| cell | seeds | September rows | given macro-F1, fixed point | argmax |")
print("|---|---|---|---|---|")
for k, v in sorted(rows):
    print(f"| {k} | {v['seeds']} | {v['rows']} | {v['given_fixed_point']:.4f} | "
          f"{v['given_argmax']:.4f} |")
for prefix in ("champion", "cleantrain"):
    vals = [v["given_fixed_point"] for k, v in rows if k.startswith(prefix)]
    if vals:
        mean = sum(vals) / len(vals)
        print(f"\n{prefix}: {len(vals)} cells, mean {mean:.4f}, "
              f"min {min(vals):.4f}, max {max(vals):.4f}")
for seed, st in sorted(cleaning.items()):
    print(f"\ncleaning s{seed}: {json.dumps(st)}")
