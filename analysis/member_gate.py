"""The 0.04-cut cascade member against the best file, on the full bar.

    python analysis/member_gate.py gate [<recipe> ...] [--cells DIR] [--out DIR] [--report NAME]
    python analysis/member_gate.py build <recipe> ... [--cells DIR] [--out DIR] [--best FILE]

``gate`` scores each group of ``GROUPS`` against its base with ``ringtrace.final.table``: the
blend candidates against ``recipes/final-half.json``, the swaps against the blend whose champion
is the CPU-built one (like with like: the members are CPU-built), the 0.04 cut alone against the
CPU-built champion alone, and the CPU-built champion against the GPU-built one (the size of the
device effect). Each table has the per-month change
on the given, cleaned and gen-only targets, the paired event bootstrap of 2,000 draws, the nine
October-like cells with 2,000 draws), then adds for each challenger:

* the bar of this round: leave-one-month-out change >= 0 on given and cleaned in every month and
  > 0 in at least one, and an October-like mean change > 0 with its pooled 95 % lower bound > 0
  on given and cleaned;
* on the train rows whose call changed: how many the challenger gets right and how many the base
  got right, per target and month (the precision of the moved calls);
* October: rows changed and class moves (counts only).

``build`` writes the October file of each recipe, validated against the sample submission, with
its sha256 and the class moves against the best file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ringtrace import paths
from ringtrace.classes import N_CLASSES

BASE = "recipes/final-half.json"
C = "recipes/cand/"
SWAPS = (C + "member-swap.json",)
GROUPS = (
    (
        "blend",
        BASE,
        (
            C + "member-type6.json",
            C + "member-mass6.json",
            C + "member-full6.json",
            *SWAPS,
            C + "member-champcpu.json",
        ),
    ),
    ("swap-cpu", C + "member-champcpu.json", SWAPS),
    (
        "solo",
        C + "member-solo-champion-cpu.json",
        (C + "member-solo-cut04.json",),
    ),
    ("device", C + "member-solo-champion.json", (C + "member-solo-champion-cpu.json",)),
)
MONTHS = (7, 8, 9)
HIDDEN_TOTAL = 3482  # train fraud rows the cleaned labels leave legit (earlier measurement)


def round_bar(lomo: dict, octlike: dict | None) -> dict:
    """This round's bar (given and cleaned): months non-negative, one positive; October-like mean
    change positive with the pooled lower bound above 0."""
    ch = {t: lomo[t]["change"] for t in ("given", "cleaned")}
    months_ok = all(min(c) >= 0 for c in ch.values()) and any(max(c) > 0 for c in ch.values())
    if octlike is None:
        oct_ok = False
        oct_mean = None
    else:
        oct_mean = {
            t: float(np.mean(list(octlike["per_seed"][t].values()))) for t in ("given", "cleaned")
        }
        oct_ok = all(oct_mean[t] > 0 and octlike["lower_bound"][t] > 0 for t in oct_mean)
    return {
        "months_ok": months_ok,
        "octlike_mean": oct_mean,
        "octlike_ok": oct_ok,
        "passes": bool(months_ok and oct_ok),
    }


def moved_calls(bench, base: dict, chal: dict) -> dict:
    """On train rows whose call changed: rows, right after, right before, per target and month."""
    a = bench.calls(base)["pred"][bench.tr]
    b = bench.calls(chal)["pred"][bench.tr]
    ptr = bench.period[bench.tr]
    out: dict = {}
    for t, truth in bench.targets.items():
        rows = {}
        for m in MONTHS:
            s = (ptr == m) & (a != b)
            rows[str(m)] = {
                "rows": int(s.sum()),
                "challenger_right": int((b[s] == truth[s]).sum()),
                "base_right": int((a[s] == truth[s]).sum()),
            }
        out[t] = rows
    return out


def aclean_macro(truth: np.ndarray, pred: np.ndarray, hidden: np.ndarray) -> float:
    """Macro-F1 of the A-clean proxy: each cleaned-legit row also stands for ``hidden[c]`` rows of
    class c spread evenly over the cleaned-legit rows, read off the expected confusion matrix."""
    k = N_CLASSES
    cm = np.bincount(truth * k + pred, minlength=k * k).reshape(k, k).astype(float)
    legit = cm[0].copy()
    h = np.asarray(hidden, dtype=float) / max(legit.sum(), 1.0)
    h[0] = 0.0
    cm[0] -= h.sum() * legit
    cm += h[:, None] * legit[None, :]
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0).mean())


def aclean_change(bench, base: dict, chal: dict) -> list[float]:
    """A-clean change per month; a month's hidden rows of class k are ``HIDDEN_TOTAL`` shared out
    by its ordinary-looking class-k labels (cleaned fraud, generator-only legit)."""
    cl, gen = bench.targets["cleaned"], bench.targets["gen"]
    ptr = bench.period[bench.tr]
    ordinary = (cl > 0) & (gen == 0)
    grand = max(int(ordinary.sum()), 1)
    a = bench.calls(base)["pred"][bench.tr]
    b = bench.calls(chal)["pred"][bench.tr]
    out = []
    for m in MONTHS:
        s = ptr == m
        hid = np.bincount(cl[ordinary & s], minlength=N_CLASSES) * HIDDEN_TOTAL / grand
        out.append(aclean_macro(cl[s], b[s], hid) - aclean_macro(cl[s], a[s], hid))
    return out


def _bench(cells: Path):
    from ringtrace.final import Bench

    return Bench.from_store(paths.out_dir() / "runs", cells)


def cmd_gate(cells: Path, out: Path, only: list[str] | None = None, report_name="gate") -> dict:
    from ringtrace.final import _json_default, load_recipe, table

    b = _bench(cells)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, base_path, chal_paths in GROUPS:
        base = load_recipe(base_path)
        chals = [load_recipe(p) for p in chal_paths if not only or p in only]
        if not chals:
            continue
        rep = table(b, base, chals, draws=2000, oct_draws=2000)
        for c in chals:
            r = rep["challengers"][c["name"]]
            if "error" in r:
                continue
            r["round_bar"] = round_bar(r["lomo"], r["octlike"])
            r["moved_calls"] = moved_calls(b, base, c)
            r["aclean_change"] = aclean_change(b, base, c)
            mc = r["moved_calls"]
            rb = r["round_bar"]
            print(
                f"[member] {c['name']}: bar {'PASS' if rb['passes'] else 'FAIL'} "
                f"(months {rb['months_ok']}, October-like {rb['octlike_ok']}"
                f" {rb['octlike_mean']}); moved train calls right after / before "
                + "; ".join(
                    f"{t} "
                    + " ".join(
                        f"{m}: {v['challenger_right']}/{v['base_right']} of {v['rows']}"
                        for m, v in mc[t].items()
                    )
                    for t in mc
                )
                + "; A-clean "
                + " / ".join(f"{x:+.5f}" for x in r["aclean_change"])
                + f"; October rows {r['lomo']['rows_changed']['october']} "
                f"{r['lomo']['october_pairs']}",
                flush=True,
            )
        report[label] = rep
        (out / f"{report_name}.json").write_text(
            json.dumps(report, indent=1, default=_json_default)
        )
    return report


def cmd_build(recipes: list[str], cells: Path, out: Path, best: Path) -> dict:
    from ringtrace.final import build, load_recipe

    b = _bench(cells)
    reps = {}
    for p in recipes:
        r = load_recipe(p)
        reps[r["name"]] = build(b, r, out, paths.data_dir() / "sample_submission.csv", best)
    return reps


def main(argv=None) -> None:
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python analysis/member_gate.py")
    ap.add_argument("cmd", choices=("gate", "build"))
    ap.add_argument("recipes", nargs="*")
    ap.add_argument("--cells", default=str(paths.out_dir() / "member" / "cells"))
    ap.add_argument("--out", default=str(paths.out_dir() / "member"))
    ap.add_argument("--best", default=str(paths.out_dir() / "member" / "blend-tabmF-bq.csv"))
    ap.add_argument("--report", default="gate", help="report name for gate")
    a = ap.parse_args(argv)
    if a.cmd == "gate":
        cmd_gate(Path(a.cells), Path(a.out), a.recipes or None, a.report)
    else:
        cmd_build(a.recipes, Path(a.cells), Path(a.out) / "build", Path(a.best))


if __name__ == "__main__":
    sys.exit(main())
