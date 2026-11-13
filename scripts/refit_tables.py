"""Markdown tables from the refit runs (aggregates only).

python scripts/refit_tables.py [~/outputs/runs]
"""

import json
import sys
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "~/outputs/runs").expanduser()
TARGETS = ("given", "cleaned", "gen")

cur = json.loads((runs / "refit-curve" / "summary.json").read_text())
months = sorted(cur["months"])
print(f"curve: {cur['runtime_s'] / 60:.1f} min\n")
print("| arm | " + " | ".join(f"{t} " + " / ".join(months) + " (mean)" for t in TARGETS) + " |")
print("|---|" + "---|" * len(TARGETS))


def arm_scores(arm_of_month):
    cells = []
    for t in TARGETS:
        vals = []
        for m in months:
            arms = arm_of_month(m)
            vals.append(sum(cur["months"][m][a][t]["macro"] for a in arms) / len(arms))
        cells.append(" / ".join(f"{v:.4f}" for v in vals) + f" ({sum(vals) / len(vals):.4f})")
    return cells


def ones(m):
    return [a for a in cur["months"][m] if a.startswith("one_")]


rows = [
    ("one (mean of the two single-month fits)", ones),
    ("avg_one (probability mean)", lambda m: ["avg_one"]),
    ("two (base rounds)", lambda m: ["two"]),
    ("two_x1.5", lambda m: ["two_x1.5"]),
]
for label, fn in rows:
    print(f"| {label} | " + " | ".join(arm_scores(fn)) + " |")
for m in months:
    print(
        f"\nhold {m}: "
        + ", ".join(f"{a} {cur['months'][m][a]['given']['macro']:.4f}" for a in ones(m))
    )

classes = list(cur["months"][months[0]]["two"]["given"]["per_class"])
print("\n| class | " + " | ".join(f"{t}: one / two / gain" for t in TARGETS) + " |")
print("|---|" + "---|" * len(TARGETS))
for c in classes:
    cells = []
    for t in TARGETS:
        one = sum(cur["months"][m][a][t]["per_class"][c] for m in months for a in ones(m)) / (
            2 * len(months)
        )
        two = sum(cur["months"][m]["two"][t]["per_class"][c] for m in months) / len(months)
        cells.append(f"{one:.3f} / {two:.3f} / {two - one:+.3f}")
    print(f"| {c} | " + " | ".join(cells) + " |")

print(
    "\n| member | rounds | argmax changed (refit) | decoded changed (refit) | "
    "argmax changed (hedge) | decoded changed (hedge) | wall |"
)
print("|---|---|---|---|---|---|---|")
for name in ("champion", "clean-event", "clean-l2"):
    p = runs / f"{name}-refit" / "summary.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    print(
        f"| {name} | {r['rounds']} | {r['refit']['argmax_vs_fold_mean']['changed']} | "
        f"{r['refit']['decoded_changed']} | {r['hedge']['argmax_vs_fold_mean']['changed']} | "
        f"{r['hedge']['decoded_changed']} | {r['runtime_s'] / 60:.1f} min |"
    )
    print(
        f"|  fraud calls fold / refit / hedge | | {sum(r['fold_mean_decoded_counts'][1:])} / "
        f"{sum(r['refit']['decoded_counts'][1:])} / {sum(r['hedge']['decoded_counts'][1:])} "
        "| | | | |"
    )
