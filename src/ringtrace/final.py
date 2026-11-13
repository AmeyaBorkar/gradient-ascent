"""The final system: one recipe, one command, and a harness that scores recipes from saved arrays.

    python -m ringtrace.final recipes/final.json                      # build the October file
    python -m ringtrace.final score recipes/final.json recipes/x.json  # gate x against the base
    python -m ringtrace.final table recipes/a.json recipes/b.json ...  # every check, many recipes

A recipe::

    {"name": "final",
     "members": [{"name": "champion", "run": "blend-champion-m1", "variant": "fold",
                  "cells": "champion"}, ...],
     "weights": {"mode": "equal"},          # or {"mode": "fixed", "values": {name: w}}
                                            # or {"mode": "nested", "combine": "prob", "steps": 10}
     "decoder": {"fraud_scale": 0.875},
     "rules": ["anchored_templates"],
     "base_rate_rules": [{"name": "blank_merchant_qr", "max_october_rows": 40}]}

``variant`` picks the October probabilities of a member: ``fold`` (the mean of the three month
folds, ``runs/<run>/test.npy``), ``refit`` (``runs/<run>-refit/test.npy``, all train months) or
``hedge`` (``runs/<run>-hedge/test.npy``). The train rows always come from ``runs/<run>/oof.npy``,
so the variant cannot be measured leave-one-month-out; its evidence comes from the October-like
refit test. ``cells`` names the member's October-like cells (``<cells>_s<seed>_r<r>.npz``).

Nested weights are learned per held-out month on the other two months (given labels, decoded as
the recipe decodes) and applied to the held-out month; October uses all three months, and the
October-like cells (fitted on July + August) use the September fold's weights.

``rules`` run after the fixed point; ``base_rate_rules`` run last. They are components that the
training months cannot measure (their October case never occurs in train) and enter on a
documented base-rate argument; each carries a cap on the October rows it may change, and the
build stops if the cap is exceeded.

The harness (``score``, ``table``) never refits: it reads ``oof.npy`` / ``test.npy`` and the cells,
and reports per-month scores on the given, cleaned and gen-only targets, the paired event bootstrap
of the change (95 %), the nine October-like cells per thinning seed, the rows changed, and the gate
verdict. Only aggregates of October are ever reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace.classes import CLASSES, N_CLASSES

TRAIN_MONTHS = (7, 8, 9)
VARIANTS = {"fold": "", "refit": "-refit", "hedge": "-hedge"}
WEIGHT_MODES = ("equal", "fixed", "nested")
DEFAULT_DECODER = {"fraud_scale": 0.875}


class RecipeError(ValueError):
    """A final recipe is malformed."""


# ---------------------------------------------------------------------------------------------
# recipes


def load_recipe(path: str | Path) -> dict:
    recipe = json.loads(Path(path).read_text())
    recipe.setdefault("name", Path(path).stem)
    return check_recipe(recipe)


def check_recipe(recipe: dict) -> dict:
    members = recipe.get("members", [])
    if not members:
        raise RecipeError("a final recipe needs at least one member")
    names = [m["name"] for m in members]
    if len(set(names)) != len(names):
        raise RecipeError(f"duplicate member names: {names}")
    for m in members:
        m.setdefault("variant", "fold")
        m.setdefault("role", "full")
        if m["role"] not in ("full", "fraud"):
            raise RecipeError(f"{m['name']}: role must be 'full' or 'fraud'")
        if m["variant"] not in VARIANTS:
            raise RecipeError(f"{m['name']}: unknown variant {m['variant']!r}")
    if all(m["role"] == "fraud" for m in members):
        raise RecipeError("at least one member must have role 'full'")
    w = recipe.setdefault("weights", {"mode": "equal"})
    if w.get("mode") not in WEIGHT_MODES:
        raise RecipeError(f"unknown weight mode {w.get('mode')!r}; known {WEIGHT_MODES}")
    if w["mode"] == "fixed":
        vals = w.get("values", {})
        if set(vals) != set(names):
            raise RecipeError(f"fixed weights must name every member: {sorted(vals)} vs {names}")
        if any(float(v) < 0 for v in vals.values()) or sum(map(float, vals.values())) <= 0:
            raise RecipeError("fixed weights must be >= 0 with a positive sum")
    recipe.setdefault("decoder", dict(DEFAULT_DECODER))
    recipe.setdefault("rules", [])
    for r in recipe.setdefault("base_rate_rules", []):
        if "name" not in r or int(r.get("max_october_rows", -1)) < 0:
            raise RecipeError(f"a base-rate rule needs a name and max_october_rows: {r}")
    return recipe


def recipe_key(recipe: dict) -> str:
    """A digest of everything that changes the calls (not the name)."""
    keep = {k: recipe[k] for k in ("members", "weights", "decoder", "rules", "base_rate_rules")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def static_weights(recipe: dict) -> np.ndarray:
    names = [m["name"] for m in recipe["members"]]
    w = recipe["weights"]
    if w["mode"] == "equal":
        out = np.ones(len(names))
    elif w["mode"] == "fixed":
        out = np.array([float(w["values"][n]) for n in names])
    else:
        raise RecipeError("nested weights are learned, not static")
    return out / out.sum()


def mix(arrays: list[np.ndarray], weights, roles=None) -> np.ndarray:
    """Weighted mean of probabilities. A member whose role is ``fraud`` moves only the legit /
    fraud split: the legit probability is the weighted mean over every member, and the fraud mass
    is shared out by the ``full`` members' mean fraud-type distribution."""
    roles = roles or ["full"] * len(arrays)
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for a, w, r in zip(arrays, weights, roles, strict=True):
        if w and r == "full":
            out += float(w) * a
    if all(r == "full" for r in roles):
        return out
    legit = sum(float(w) * a[:, 0] for a, w in zip(arrays, weights, strict=True) if w)
    legit = legit / float(sum(weights))
    types = out[:, 1:]
    tot = types.sum(axis=1, keepdims=True)
    share = np.where(tot > 0, types / np.where(tot > 0, tot, 1), 1.0 / types.shape[1])
    out = np.empty_like(out)
    out[:, 0] = legit
    out[:, 1:] = (1.0 - legit)[:, None] * share
    return out


def greedy_counts(evaluate, n: int, steps: int = 10) -> np.ndarray:
    """Forward selection with replacement (copy counts); ties go to the earlier member, and the
    search stops when no extra copy improves the score."""
    counts = np.zeros(n, dtype=np.int64)
    best = -np.inf
    for _ in range(steps):
        trials = []
        for j in range(n):
            c = counts.copy()
            c[j] += 1
            trials.append((evaluate(tuple(int(x) for x in c)), -j))
        score, neg_j = max(trials)
        if score <= best:
            break
        best = score
        counts[-neg_j] += 1
    return counts


def capped(name: str, before: np.ndarray, after: np.ndarray, is_test: np.ndarray, cap: int) -> int:
    """October rows a base-rate rule changed; raises if above its cap."""
    n = int((before[is_test] != after[is_test]).sum())
    if n > cap:
        raise RecipeError(f"{name} changed {n} October rows, above its cap of {cap}")
    return n


def pair_counts(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    diff = a != b
    keys, n = np.unique(a[diff] * N_CLASSES + b[diff], return_counts=True)
    return {
        f"{CLASSES[k // N_CLASSES]} -> {CLASSES[k % N_CLASSES]}": int(c)
        for k, c in zip(keys, n, strict=True)
    }


def gate_verdict(lomo: dict, octlike: dict | None, targets=("given", "cleaned")) -> dict:
    """The protocol: never negative in any month and target, positive on every thinning seed for
    given and cleaned, and the event bootstrap's 95 % lower bound above 0."""
    tg = [t for t in lomo if isinstance(lomo[t], dict) and "change" in lomo[t]]
    months_ok = all(min(lomo[t]["change"]) >= 0 for t in tg)
    boot_ok = all(lomo[t]["lo"] > 0 for t in targets if t in tg)
    if octlike is None:
        oct_ok = None
    else:
        oct_ok = all(v > 0 for t in targets for v in octlike["per_seed"][t].values()) and all(
            octlike["lower_bound"][t] > 0 for t in targets
        )
    return {
        "months_non_negative": months_ok,
        "bootstrap_above_zero": boot_ok,
        "octlike_positive": oct_ok,
        "passes": bool(months_ok and boot_ok and oct_ok),
    }


# ---------------------------------------------------------------------------------------------
# the bench: ledger, targets and arrays loaded once, calls cached per recipe


class Bench:
    def __init__(
        self,
        led: pl.DataFrame,
        targets: dict[str, np.ndarray],
        groups: np.ndarray,
        runs_dir: Path,
        cells_dir: Path | None = None,
        log=print,
    ):
        self.led = led
        self.is_test = led["is_test"].to_numpy()
        self.period = led["period"].to_numpy()
        self.tr = np.flatnonzero(~self.is_test)
        self.te = np.flatnonzero(self.is_test)
        self.targets = targets
        self.groups = groups
        self.runs_dir = Path(runs_dir)
        self.cells_dir = cells_dir
        self.log = log
        self._arrays: dict = {}
        self._calls: dict = {}
        self._weights: dict = {}

    @classmethod
    def from_store(cls, runs_dir: Path, cells_dir: Path | None = None, log=print) -> Bench:
        from ringtrace import silver, store
        from ringtrace.evidence import anchors
        from ringtrace.rules import label_free_events
        from ringtrace.squeeze import RULE_COLUMNS

        store.use_frame(None)
        led = store.ledger()
        led = led.with_columns(anchors.family_columns(led, RULE_COLUMNS))
        tr = np.flatnonzero(~led["is_test"].to_numpy())
        targets = {"given": led["y"].to_numpy().astype(np.int64)[tr]}
        for tname, which in (("cleaned", "label"), ("gen", "structured")):
            arr = silver.load(led, which)
            if arr is not None:
                targets[tname] = arr.astype(np.int64)[tr]
        ev = label_free_events(led)[tr]
        groups = np.where(ev >= 0, ev, -1 - np.arange(len(tr)))
        return cls(led, targets, groups, runs_dir, cells_dir, log=log)

    # arrays -------------------------------------------------------------------------------

    def member(self, m: dict) -> np.ndarray:
        """Full-ledger probabilities: out-of-fold train rows, the variant's October rows."""
        key = (m["run"], m["variant"])
        if key not in self._arrays:
            from ringtrace.blend import load_member

            oof, test = load_member(self.runs_dir / m["run"], len(self.tr), len(self.te))
            if m["variant"] != "fold":
                _, test = load_member(
                    self.runs_dir / (m["run"] + VARIANTS[m["variant"]]), len(self.tr), len(self.te)
                )
            full = np.zeros((self.led.height, N_CLASSES))
            full[self.tr], full[self.te] = oof, test
            self._arrays[key] = full
        return self._arrays[key]

    # decoding -----------------------------------------------------------------------------

    def decode_rows(self, rows: np.ndarray, proba_rows: np.ndarray, recipe: dict) -> np.ndarray:
        """One month's (or October's) rows decoded on their own, then the model rules."""
        from ringtrace import rules as rulebook
        from ringtrace.decode import fixed_point

        pred = np.zeros(self.led.height, dtype=np.int64)
        pred[rows] = fixed_point(proba_rows, **recipe["decoder"])
        if recipe["rules"]:
            full = np.zeros((self.led.height, N_CLASSES))
            full[rows] = proba_rows
            pred = rulebook.apply(recipe["rules"], pred, full, self.led, log=lambda *_: None)
        return pred[rows]

    def weights(self, recipe: dict) -> dict:
        """Weights per held-out month and for October."""
        key = recipe_key(recipe)
        if key in self._weights:
            return self._weights[key]
        if recipe["weights"]["mode"] != "nested":
            w = static_weights(recipe)
            out = {m: w for m in (*TRAIN_MONTHS, "october")}
        else:
            out = self._nested(recipe)
        self._weights[key] = out
        return out

    def _nested(self, recipe: dict) -> dict:
        from ringtrace.scoring import macro_f1

        spec = recipe["weights"]
        if spec.get("combine", "prob") != "prob":
            raise RecipeError("nested selection supports combine 'prob' only")
        arrays = [self.member(m) for m in recipe["members"]]
        roles = [m["role"] for m in recipe["members"]]
        n = len(arrays)
        y = self.targets["given"]
        ptr = self.period[self.tr]
        cache: dict = {}

        def month_score(counts: tuple, k: int) -> float:
            if (counts, k) not in cache:
                s = np.flatnonzero(self.period == k)
                p = mix([a[s] for a in arrays], np.array(counts) / sum(counts), roles)
                cache[(counts, k)] = macro_f1(y[ptr == k], self.decode_rows(s, p, recipe))
            return cache[(counts, k)]

        out = {}
        for held in (*TRAIN_MONTHS, "october"):
            fit = [k for k in TRAIN_MONTHS if k != held]
            c = greedy_counts(
                lambda cc, fit=fit: float(np.mean([month_score(cc, k) for k in fit])),
                n,
                int(spec.get("steps", 10)),
            )
            out[held] = c / c.sum()
            self.log(f"[final] nested weights, held {held}: {c.tolist()}")
        return out

    def calls(self, recipe: dict) -> dict:
        """Calls on every ledger row, plus the rows each base-rate rule changed in October."""
        key = recipe_key(recipe)
        if key in self._calls:
            return self._calls[key]
        from ringtrace import rules as rulebook

        arrays = [self.member(m) for m in recipe["members"]]
        roles = [m["role"] for m in recipe["members"]]
        w = self.weights(recipe)
        pred = np.zeros(self.led.height, dtype=np.int64)
        proba = np.zeros((self.led.height, N_CLASSES))
        for k in (*TRAIN_MONTHS, "october"):
            rows = self.te if k == "october" else self.tr[self.period[self.tr] == k]
            proba[rows] = mix([a[rows] for a in arrays], w[k], roles)
        # as ringtrace.blend decodes: the fixed point per month, then the rules over every row
        from ringtrace.decode import fixed_point

        for k in (*TRAIN_MONTHS, "october"):
            rows = self.te if k == "october" else self.tr[self.period[self.tr] == k]
            pred[rows] = fixed_point(proba[rows], **recipe["decoder"])
        if recipe["rules"]:
            pred = rulebook.apply(recipe["rules"], pred, proba, self.led, log=_quiet)
        changed = {}
        for r in recipe["base_rate_rules"]:
            params = {k: v for k, v in r.items() if k not in ("name", "max_october_rows")}
            after = rulebook.apply(
                [{"name": r["name"], **params}], pred, proba, self.led, log=lambda *_: None
            )
            changed[r["name"]] = {
                "october": capped(r["name"], pred, after, self.is_test, int(r["max_october_rows"])),
                "train": int((pred[self.tr] != after[self.tr]).sum()),
            }
            pred = after
        self._calls[key] = {"pred": pred, "base_rate_changes": changed}
        return self._calls[key]

    # scoring ------------------------------------------------------------------------------

    def lomo(self, recipe: dict) -> dict:
        from ringtrace.blend import month_scores

        pred = self.calls(recipe)["pred"]
        return month_scores(pred[self.tr], self.targets, self.period[self.tr])

    def compare(self, base: dict, chal: dict, draws: int = 2000) -> dict:
        from ringtrace.squeeze import lomo_compare

        a, b = self.calls(base)["pred"], self.calls(chal)["pred"]
        r = lomo_compare(self, a, b, draws)
        r["october_pairs"] = pair_counts(a[self.te], b[self.te])
        return r

    # October-like -------------------------------------------------------------------------

    def has_cells(self, recipe: dict, seeds, sets) -> bool:
        if self.cells_dir is None:
            return False
        return all(
            "cells" in m and (self.cells_dir / f"{m['cells']}_s{s}_r{r}.npz").exists()
            for m in recipe["members"]
            for s in seeds
            for r in sets
        )

    def octlike(
        self, base: dict, chals: list[dict], seeds=(13, 14, 15), sets=(0, 1, 2), draws=1000
    ) -> list[dict | None]:
        """Each challenger against the base in the nine cells (thinned September, fitted on July
        + August); the September fold's weights; base-rate rules apply as in the recipe."""
        from ringtrace import octlike as ol
        from ringtrace import rules as rulebook
        from ringtrace.scoring import macro_f1

        todo = [i for i, c in enumerate(chals) if self.has_cells(c, seeds, sets)]
        if not self.has_cells(base, seeds, sets) or not todo:
            return [None] * len(chals)
        wsep = {recipe_key(r): self.weights(r)[9] for r in [base, *chals]}
        cells: dict = {i: {} for i in todo}
        try:
            for seed in seeds:
                frame = ol.Frame(seed)
                units = frame.unit[frame.hold]
                hold = np.flatnonzero(frame.hold)

                def calls(rec, r, frame=frame, hold=hold, seed=seed):
                    rows, probs = None, []
                    for m in rec["members"]:
                        f = np.load(self.cells_dir / f"{m['cells']}_s{seed}_r{r}.npz")
                        if rows is None:
                            rows = f["rows"]
                        elif not np.array_equal(rows, f["rows"]):
                            raise ValueError(f"{m['cells']}: rows differ in s{seed} r{r}")
                        probs.append(f["prob"].astype(np.float64))
                    if not np.array_equal(rows, frame.stacked[frame.hold]):
                        raise ValueError(f"cells s{seed} r{r} do not match the current frame")
                    p = mix(probs, wsep[recipe_key(rec)], [m["role"] for m in rec["members"]])
                    pred = ol.decode(frame, p, rec["decoder"], rec["rules"])
                    if rec["base_rate_rules"]:
                        full_pred = np.zeros(frame.led.height, dtype=np.int64)
                        full_pred[hold] = pred
                        full_p = np.zeros((frame.led.height, N_CLASSES))
                        full_p[hold] = p
                        names = [
                            {k: v for k, v in x.items() if k != "max_october_rows"}
                            for x in rec["base_rate_rules"]
                        ]
                        pred = rulebook.apply(names, full_pred, full_p, frame.led, log=_quiet)[hold]
                    return pred

                for r in sets:
                    ya = calls(base, r)
                    for i in todo:
                        yb = calls(chals[i], r)
                        cell = {"rows_changed": int((ya != yb).sum())}
                        for t, truth in frame.targets.items():
                            yt = truth[frame.hold]
                            bs = ol.bootstrap(units, yt, ya, yb, draws, seed=seed * 100 + r)
                            cell[t] = {"base": macro_f1(yt, ya), "chal": macro_f1(yt, yb), **bs}
                        cells[i][(seed, r)] = cell
        finally:
            from ringtrace import store

            store.use_frame(None)
        out: list[dict | None] = [None] * len(chals)
        for i in todo:
            c = cells[i]
            tg = [t for t in next(iter(c.values())) if t != "rows_changed"]
            out[i] = {
                "per_seed": {
                    t: {s: float(np.mean([c[(s, r)][t]["delta"] for r in sets])) for s in seeds}
                    for t in tg
                },
                "lower_bound": {t: float(np.mean([v[t]["lo"] for v in c.values()])) for t in tg},
                "base_mean": {t: float(np.mean([v[t]["base"] for v in c.values()])) for t in tg},
                "rows_changed": int(sum(v["rows_changed"] for v in c.values())),
                "cells": {f"s{s}_r{r}": v for (s, r), v in c.items()},
            }
        return out


def _quiet(*_):
    return None


# ---------------------------------------------------------------------------------------------
# reports


def fmt_line(name: str, lomo: dict, octlike: dict | None, verdict: dict) -> str:
    parts = [f"{name}:"]
    for t in ("given", "cleaned", "gen"):
        if t in lomo:
            ch = " / ".join(f"{c:+.5f}" for c in lomo[t]["change"])
            parts.append(f"{t} {ch} [{lomo[t]['lo']:+.5f}, {lomo[t]['hi']:+.5f}]")
    if octlike is not None:
        for t in ("given", "cleaned"):
            ps = " / ".join(f"{v:+.5f}" for v in octlike["per_seed"][t].values())
            parts.append(f"octlike {t} {ps} (lb {octlike['lower_bound'][t]:+.5f})")
    else:
        parts.append("octlike n/a")
    parts.append(f"rows {lomo['rows_changed']}")
    parts.append("PASS" if verdict["passes"] else "FAIL")
    return "  ".join(parts)


def table(
    bench: Bench, base: dict, chals: list[dict], draws=2000, oct_draws=1000, log=print
) -> dict:
    t0 = time.time()
    rep = {"base": base["name"], "base_scores": bench.lomo(base), "challengers": {}}
    octs = bench.octlike(base, chals, draws=oct_draws)
    for c, o in zip(chals, octs, strict=True):
        lo = bench.compare(base, c, draws)
        v = gate_verdict(lo, o)
        rep["challengers"][c["name"]] = {
            "scores": bench.lomo(c),
            "lomo": lo,
            "octlike": o,
            "verdict": v,
            "base_rate_changes": bench.calls(c)["base_rate_changes"],
        }
        log("[final] " + fmt_line(c["name"], lo, o, v))
    rep["wall_s"] = time.time() - t0
    return rep


# ---------------------------------------------------------------------------------------------
# build


def build(bench: Bench, recipe: dict, out_dir: Path, sample: Path, against: Path | None, log=print):
    from ringtrace.blend import write_submission

    t0 = time.time()
    res = bench.calls(recipe)
    calls = res["pred"][bench.te]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{recipe['name']}.csv"
    sub = write_submission(path, bench.led["txn_id"].to_numpy()[bench.te], calls, sample)
    (out_dir / f"{recipe['name']}.csv.sha256").write_text(f"{sub['sha256']}  {path.name}\n")
    summary = {
        "recipe": recipe,
        "sha256": sub["sha256"],
        "problems": sub["problems"],
        "calls": sub["calls"],
        "base_rate_changes": res["base_rate_changes"],
        "weights": {str(k): list(map(float, v)) for k, v in bench.weights(recipe).items()},
        "scores": bench.lomo(recipe),
    }
    if against is not None and against.exists():
        ref = pl.read_csv(against, infer_schema=False)
        mine = pl.read_csv(path, infer_schema=False)
        j = ref.join(mine, on="id", how="inner", suffix="_new")
        idx = {c: i for i, c in enumerate(CLASSES)}
        a = np.array([idx[x] for x in j["label"].to_list()])
        b = np.array([idx[x] for x in j["label_new"].to_list()])
        summary["against"] = {"file": against.name, "pairs": pair_counts(a, b)}
        summary["against"]["rows_changed"] = int((a != b).sum())
    summary["wall_s"] = time.time() - t0
    (out_dir / f"{recipe['name']}.json").write_text(json.dumps(summary, indent=1))
    log(f"[final] {path}: sha256 {sub['sha256']} problems {sub['problems']}")
    log(f"[final] calls {sub['calls']}")
    log(f"[final] base-rate changes {res['base_rate_changes']}")
    if "against" in summary:
        log(f"[final] against {against.name}: {summary['against']}")
    return summary


def main(argv=None) -> dict:
    from ringtrace import paths, threads

    threads.pin()
    argv = list(argv) if argv is not None else __import__("sys").argv[1:]
    cmd = argv[0] if argv and argv[0] in ("score", "table", "build") else "build"
    if argv and argv[0] == cmd:
        argv = argv[1:]
    ap = argparse.ArgumentParser(prog=f"python -m ringtrace.final {cmd}")
    ap.add_argument("recipes", nargs="+")
    ap.add_argument("--cells", default=None, help="October-like cell directory")
    ap.add_argument("--out", default=None, help="where the October file goes")
    ap.add_argument("--against", default=None, help="a submission to count changes against")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--oct-draws", type=int, default=1000)
    ap.add_argument("--report", default=None, help="json report path (score / table)")
    args = ap.parse_args(argv)
    cells = Path(args.cells) if args.cells else paths.out_dir() / "gate" / "octlike"
    bench = Bench.from_store(paths.out_dir() / "runs", cells)
    recipes = [load_recipe(p) for p in args.recipes]
    if cmd == "build":
        out = Path(args.out) if args.out else paths.out_dir() / "final"
        reps = {
            r["name"]: build(
                bench,
                r,
                out,
                paths.data_dir() / "sample_submission.csv",
                Path(args.against) if args.against else None,
            )
            for r in recipes
        }
        return reps
    rep = table(bench, recipes[0], recipes[1:], args.draws, args.oct_draws)
    path = Path(args.report) if args.report else paths.out_dir() / "final" / "table.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1, default=_json_default))
    return rep


def _json_default(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
