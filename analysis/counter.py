"""The counter's segments at October's size: a cut-position prior and a per-segment gap emission.

    python analysis/counter.py gate        # thinning simulation over the settings grid
    python analysis/counter.py october --spread S --gap-weight W   # October, counted only

``gate`` thins train's fraud phase to October's size (each phone kept with probability 1/6, seeds
0..29), decodes the kept phones without and with the prior, and scores both against the cleaned
labels on the kept unanchored fraud rows. The prior of each draw is learned from the phones outside
it. A stress variant shifts the prior's cuts by the offset between October's own fit and train's
(how far a train-built prior sits from October's cuts). The settings are chosen on seeds 0..14
and reported on seeds 15..29. ``october`` decodes the real frame with the test split taught by the
train split and counts how October's fraud-block rows move. Writes ``$RINGTRACE_OUT/counter/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from counter_core import choose, paired, passes  # noqa: E402
from farlook_core import keep_units  # noqa: E402

from ringtrace import paths, silver, store  # noqa: E402
from ringtrace.evidence import anchors, phones  # noqa: E402

OCTOBER_SHARE = 1 / 6
SPREADS = (None, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1)
GAP_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
CHOOSE_SEEDS = range(0, 15)
REPORT_SEEDS = range(15, 30)
METRICS = ("agreement", "last_four", "lost", "gained")


def setting_name(spread, gap_weight) -> str:
    return f"spread={spread}/gap={gap_weight}"


class Frame:
    """Train's fraud phase and October's, read once."""

    def __init__(self):
        led = store.ledger()
        self.y = silver.load(led, "label")
        self.device = led["device"].to_numpy().astype(np.int64)
        self.split = led["is_test"].to_numpy().astype(np.int64)
        self.anchor = anchors.row_anchor(led).astype(np.int64)
        self.blocks = phones.find_blocks(self.device, self.split)
        self.block = phones.block_codes(self.device, self.split, self.blocks)
        self.lo, self.edge, _ = self.blocks.ranges[0]
        self.rows = np.flatnonzero((self.block == 2) & (self.split == 0))
        self.nums_all, self.inv_all = np.unique(self.device[self.rows], return_inverse=True)

    def decode(self, keep: np.ndarray, prior: phones.Prior | None = None):
        """Rows, 1-based segment class and confidence of the kept phones' own fit."""
        r = self.rows[keep[self.inv_all]]
        nums, inv = np.unique(self.device[r], return_inverse=True)
        counts = np.zeros((len(nums), phones.K + 1))
        np.add.at(counts, (inv, self.anchor[r]), 1.0)
        used = np.zeros(self.edge - self.lo + 2, dtype=bool)
        used[nums - self.lo] = True
        own = phones.Blocks(f0=self.blocks.f0, used=used, base=self.lo)
        ph = phones.decode_phase(nums, counts, own, self.lo, self.edge, prior=prior, edge=self.edge)
        return r, ph.seg[inv] + 1, ph.conf[inv], ph

    def october_phase(self) -> tuple[phones.Phase, int, int]:
        lo, edge, hi = self.blocks.ranges[1]
        r = np.flatnonzero((self.block == 2) & (self.split == 1))
        nums, inv = np.unique(self.device[r], return_inverse=True)
        counts = np.zeros((len(nums), phones.K + 1))
        np.add.at(counts, (inv, self.anchor[r]), 1.0)
        return phones.decode_phase(nums, counts, self.blocks, lo, hi), lo, edge


def with_cuts(prior: phones.Prior, cuts: np.ndarray, spread, gap_weight) -> phones.Prior:
    return phones.Prior(cuts=cuts, occupancy=prior.occupancy, spread=spread, gap_weight=gap_weight)


def cmd_gate(f: Frame, seeds: int = 30) -> dict:
    y, anchor = f.y, f.anchor
    every = np.ones(len(f.nums_all), dtype=bool)
    r_full, seg_full, _, ph_full = f.decode(every)
    full = np.zeros(len(y), dtype=np.int64)
    full[r_full] = seg_full
    train_prior = phones.learn_prior(ph_full, f.lo, f.edge)
    oct_phase, oct_lo, oct_edge = f.october_phase()
    oct_prior = phones.learn_prior(oct_phase, oct_lo, oct_edge)
    shift = oct_prior.cuts - train_prior.cuts
    settings = [(s, w) for s in SPREADS for w in GAP_WEIGHTS if not (s is None and w == 0.0)]

    def score(r, seg, conf, base_seg=None):
        open_rows = (anchor[r] == 0) & (y[r] > 0)
        tail = open_rows & (full[r] >= 8)
        right = seg == y[r]
        out = {
            "rows": int(open_rows.sum()),
            "agreement": float(right[open_rows].mean()),
            "last_four": float(right[tail].mean()) if tail.any() else float("nan"),
            "conf_below_0.9": int((open_rows & (conf < 0.9)).sum()),
        }
        if base_seg is not None:
            base_right = base_seg == y[r]
            out["lost"] = int((open_rows & base_right & ~right).sum())
            out["gained"] = int((open_rows & ~base_right & right).sum())
            out["moved"] = int((open_rows & (seg != base_seg)).sum())
        return out

    # the full block: its own fit, then taught by itself and by October's fit
    base_full = score(r_full, seg_full, np.ones(len(r_full)))
    full_rows = {}
    for s, w in settings:
        row = {}
        for teacher_name, teacher in (("self", train_prior), ("october", oct_prior)):
            _, seg, conf, _ = f.decode(every, with_cuts(teacher, teacher.cuts, s, w))
            row[teacher_name] = score(r_full, seg, conf, seg_full)
        full_rows[setting_name(s, w)] = row

    per_seed: dict[str, list] = {setting_name(s, w): [] for s, w in settings}
    stress: dict[str, list] = {setting_name(s, w): [] for s, w in settings}
    base_seeds = []
    t0 = time.time()
    for seed in range(seeds):
        keep = keep_units(len(f.nums_all), OCTOBER_SHARE, seed)
        r, seg0, conf0, _ = f.decode(keep)
        base = score(r, seg0, conf0)
        base_seeds.append(base)
        _, _, _, ph_out = f.decode(~keep)
        taught = phones.learn_prior(ph_out, f.lo, f.edge)
        for s, w in settings:
            name = setting_name(s, w)
            for store_to, cuts in ((per_seed, taught.cuts), (stress, taught.cuts + shift)):
                _, seg, conf, _ = f.decode(keep, with_cuts(taught, cuts, s, w))
                sc = score(r, seg, conf, seg0)
                sc["d_agreement"] = sc["agreement"] - base["agreement"]
                sc["d_last_four"] = sc["last_four"] - base["last_four"]
                store_to[name].append(sc)
    wall = time.time() - t0

    def summarise(rows: list, idx) -> dict:
        pick = [rows[i] for i in idx]
        return {
            "agreement": paired([p["d_agreement"] for p in pick]),
            "last_four": paired([p["d_last_four"] for p in pick]),
            "lost": paired([p["lost"] for p in pick]),
            "gained": paired([p["gained"] for p in pick]),
            "moved": paired([p["moved"] for p in pick]),
            "conf_below_0.9": paired([p["conf_below_0.9"] for p in pick]),
        }

    choose_idx = [i for i in CHOOSE_SEEDS if i < seeds]
    report_idx = [i for i in REPORT_SEEDS if i < seeds]
    all_idx = list(range(seeds))
    grid = {}
    for s, w in settings:
        name = setting_name(s, w)
        grid[name] = {
            "choose_half": summarise(per_seed[name], choose_idx),
            "report_half": summarise(per_seed[name], report_idx),
            "all_seeds": summarise(per_seed[name], all_idx),
            "stress_choose_half": summarise(stress[name], choose_idx),
            "stress_report_half": summarise(stress[name], report_idx),
            "stress_all_seeds": summarise(stress[name], all_idx),
            "full_block": full_rows[name],
        }

    def flat(g: dict) -> dict:
        return {
            **g["choose_half"],
            **{f"stress_{k}": v for k, v in g["stress_choose_half"].items()},
        }

    rules = {
        # as specified: the plain simulation alone
        "plain": choose(
            {n: flat(g) for n, g in grid.items()}, keys=("agreement",), guards=("last_four",)
        ),
        # the worse of the plain and the October-shifted simulation (the rule adopted)
        "maximin": choose(
            {n: flat(g) for n, g in grid.items()},
            keys=("agreement", "stress_agreement"),
            guards=("last_four", "stress_last_four"),
        ),
    }

    def gate(chosen: str) -> dict:
        g = grid[chosen]
        fb = g["full_block"]
        v = {
            "chosen_on_seeds_0_14": chosen,
            "report_half_rise_and_2se": passes(g["report_half"]["agreement"]),
            "all_seeds_rise_and_2se": passes(g["all_seeds"]["agreement"]),
            "last_four_not_lower_report_half": g["report_half"]["last_four"]["mean"] >= 0,
            "last_four_not_lower_all_seeds": g["all_seeds"]["last_four"]["mean"] >= 0,
            "full_block_not_lower_self": fb["self"]["agreement"] >= base_full["agreement"],
            "full_block_not_lower_october_teacher": fb["october"]["agreement"]
            >= base_full["agreement"],
            "stress_report_half_not_lower": g["stress_report_half"]["agreement"]["mean"] >= 0,
            "stress_last_four_not_lower": g["stress_all_seeds"]["last_four"]["mean"] >= 0,
        }
        v["first_gate_passes"] = all(
            v[k]
            for k in (
                "report_half_rise_and_2se",
                "all_seeds_rise_and_2se",
                "last_four_not_lower_report_half",
                "last_four_not_lower_all_seeds",
                "full_block_not_lower_self",
            )
        )
        v["robust_checks_pass"] = all(
            v[k]
            for k in (
                "full_block_not_lower_october_teacher",
                "stress_report_half_not_lower",
                "stress_last_four_not_lower",
            )
        )
        return v

    verdict = {rule: gate(chosen) for rule, chosen in rules.items()}
    base_arr = {k: np.array([b[k] for b in base_seeds], dtype=np.float64) for k in base_seeds[0]}
    return {
        "verdict": verdict,
        "baseline_thinned": {
            k: {"mean": float(v.mean()), "sd": float(v.std())} for k, v in base_arr.items()
        },
        "baseline_full_block": base_full,
        "train_prior": {
            "cuts": train_prior.cuts.tolist(),
            "occupancy": train_prior.occupancy.tolist(),
        },
        "october_fit_cuts_minus_train": shift.tolist(),
        "october_fit_occupancy": oct_prior.occupancy.tolist(),
        "grid": grid,
        "decode_wall_s": wall,
    }


def cmd_october(f: Frame, spread, gap_weight: float) -> dict:
    """October's fraud block decoded with and without the train-taught prior, counted only."""
    base = phones.counter(f.device, f.split, f.anchor)
    new = phones.counter(
        f.device, f.split, f.anchor, teach={"spread": spread, "gap_weight": gap_weight}
    )
    fb = (f.block == 2) & (f.split == 1)
    same_train = bool(
        np.array_equal(base["seg"][f.split == 0], new["seg"][f.split == 0], equal_nan=True)
        and np.array_equal(base["conf"][f.split == 0], new["conf"][f.split == 0], equal_nan=True)
    )
    moved = fb & (base["seg"] != new["seg"])
    open_ = fb & (f.anchor == 0)
    out = {
        "setting": setting_name(spread, gap_weight),
        "train_split_unchanged": same_train,
        "fraud_block_rows": int(fb.sum()),
        "unanchored_rows": int(open_.sum()),
        "segment_moved_rows": int(moved.sum()),
        "segment_moved_unanchored": int((moved & open_).sum()),
        "segment_moved_unanchored_base_conf_below_0.9": int(
            (moved & open_ & (base["conf"] < 0.9)).sum()
        ),
        "segment_moved_unanchored_new_conf_0.9_up": int(
            (moved & open_ & (new["conf"] >= 0.9)).sum()
        ),
        "unanchored_base_conf_below_0.9": int((open_ & (base["conf"] < 0.9)).sum()),
        "unanchored_new_conf_below_0.9": int((open_ & (new["conf"] < 0.9)).sum()),
        "unanchored_base_below_new_up": int(
            (open_ & (base["conf"] < 0.9) & (new["conf"] >= 0.9)).sum()
        ),
        "unanchored_base_up_new_below": int(
            (open_ & (base["conf"] >= 0.9) & (new["conf"] < 0.9)).sum()
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["gate", "october"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--spread", type=float, default=None)
    ap.add_argument("--gap-weight", type=float, default=0.0)
    args = ap.parse_args()
    f = Frame()
    target = paths.out_dir() / "counter"
    target.mkdir(parents=True, exist_ok=True)
    if args.command == "gate":
        report = cmd_gate(f, args.seeds)
    else:
        report = cmd_october(f, args.spread, args.gap_weight)
    name = args.command if args.command == "gate" else f"october-{args.spread}-{args.gap_weight}"
    (target / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            report if args.command != "gate" else {k: v for k, v in report.items() if k != "grid"},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
