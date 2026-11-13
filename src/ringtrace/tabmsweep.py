"""One-fit sweep of TabM configs: September from July + August, normal and October-like.

    python -m ringtrace.tabmsweep <config> [--parts lomo,13,14,15] [--seed 0]

A config is the base recipe (``recipes/tabm.json``) with the overrides in ``CONFIGS``. For each
part it fits one TabM seed on July + August and predicts September:

* ``lomo``: the normal ledger (September as in leave-one-month-out); probabilities go to
  ``$RINGTRACE_OUT/runs/tabm2-sw-<config>/sep.npy``;
* ``13`` / ``14`` / ``15``: the October-like frame of that thinning seed; probabilities go to
  ``$RINGTRACE_OUT/octlike/tabm2-sw-<config>_s<seed>_r0.npz`` (seed set r0 here means the one
  sweep seed, not seeds 0-2).

Each part is decoded like a test month (fixed point, then the recipe's rules) and scored on the
given, cleaned and gen-only labels; the scores go to ``summary.json`` in the run directory.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np

from ringtrace import paths

BASE_RECIPE = "recipes/tabm.json"
PREFIX = "tabm2-sw-"

# overrides of the base recipe: "tabm" keys go to TabMSpec, the rest to the recipe
CONFIGS: dict[str, dict] = {
    "base": {},
    "k32": {"tabm": {"k": 32}},
    "k64": {"tabm": {"k": 64}},
    "small": {"tabm": {"width": 256, "blocks": 2}},
    "large": {"tabm": {"width": 1024, "blocks": 4}},
    "ple": {"tabm": {"embedding": "ple"}},
    "long": {"tabm": {"epochs": 20, "lr": 1e-3}},
    "short": {"tabm": {"epochs": 6, "lr": 3e-3}},
    "cw-sqrt": {"tabm": {"class_weight_power": 0.5}},
    "cw-full": {"tabm": {"class_weight_power": 1.0}},
    "l2": {"train_on": "event_l2"},
    "two-stage": {"tabm": {"two_stage": True}},
    "recent2": {"tabm": {"recent_weight": 2.0}},
}


def recipe_for(config: str, base_path: str = BASE_RECIPE) -> dict:
    from ringtrace import run as runner

    recipe = runner.load_recipe(base_path)
    over = CONFIGS[config]
    recipe["tabm"] = {**recipe.get("tabm", {}), **over.get("tabm", {})}
    recipe.update({k: v for k, v in over.items() if k != "tabm"})
    recipe["name"] = PREFIX + config
    return recipe


def _decode_sep(led, period, hold, prob, recipe) -> np.ndarray:
    """September decoded on its own, then the recipe's rules (other rows stay legit, empty)."""
    from ringtrace import rules as rulebook
    from ringtrace.decode import fixed_point

    pred = np.zeros(led.height, dtype=np.int64)
    pred[hold] = fixed_point(prob, **recipe.get("decoder", {"fraud_scale": 0.875}))
    if recipe.get("rules"):
        full = np.zeros((led.height, prob.shape[1]))
        full[hold] = prob
        pred = rulebook.apply(recipe["rules"], pred, full, led, log=lambda *_: None)
    return pred[hold]


def part_lomo(recipe: dict, seed: int, log=print) -> dict:
    from ringtrace import run as runner
    from ringtrace import silver, store
    from ringtrace.scoring import macro_f1

    store.use_frame(None)
    led = store.ledger()
    X, _ = runner.feature_matrix(recipe)
    y = led["y"].to_numpy().astype(np.int64)
    period = led["period"].to_numpy()
    is_test = led["is_test"].to_numpy()
    fit = ~is_test & np.isin(period, (7, 8))
    hold = ~is_test & (period == 9)
    cl = silver.load(led, "label") if recipe.get("train_on") == "cleaned" else None
    target = runner.fit_target(recipe, y, fit, cl)
    spec, fit_predict = runner.model(recipe)
    spec = replace(spec, seeds=(seed,))
    t0 = time.time()
    (prob,) = fit_predict(X[fit], target[fit], period[fit], [X[hold]], spec, log=log)
    runner.check_proba(prob, "September")
    out = paths.out_dir() / "runs" / recipe["name"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "sep.npy", prob.astype(np.float32))
    pred = _decode_sep(led, period, hold, prob, recipe)
    targets = {"given": y}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr
    return {
        **{t: macro_f1(v[hold], pred) for t, v in targets.items()},
        "wall_s": time.time() - t0,
    }


def part_octlike(recipe: dict, thin_seed: int, seed: int, log=print) -> dict:
    from ringtrace import octlike
    from ringtrace import run as runner
    from ringtrace.scoring import macro_f1

    frame = octlike.Frame(thin_seed)
    X, _ = runner.feature_matrix(recipe)
    y = frame.led["y"].to_numpy().astype(np.int64)
    target = y
    if recipe.get("train_on", "given") not in ("given", "cleaned"):
        target = runner.fit_target(recipe, y, frame.fit)
    elif recipe.get("train_on") == "cleaned":
        raise ValueError("silver training labels are not supported in the sweep")
    spec, fit_predict = runner.model(recipe)
    spec = replace(spec, seeds=(seed,))
    t0 = time.time()
    (prob,) = fit_predict(
        X[frame.fit], target[frame.fit], frame.period[frame.fit], [X[frame.hold]], spec, log=log
    )
    runner.check_proba(prob, "September")
    path = octlike._out_dir() / f"{recipe['name']}_s{thin_seed}_r0.npz"
    np.savez(path, rows=frame.stacked[frame.hold], prob=prob.astype(np.float32))
    pred = octlike.decode(frame, prob, recipe.get("decoder"), recipe.get("rules", []))
    scores = {t: macro_f1(v[frame.hold], pred) for t, v in frame.targets.items()}
    from ringtrace import store

    store.use_frame(None)
    return {**scores, "wall_s": time.time() - t0}


FIVE_WAY = ("blend-champion-m1", "lgbm", "catboost", "clean-event", "clean-l2")


def role_scores(config: str, log=print) -> dict:
    """September of the month folds with the config in two blend roles, against the five-way
    blend (members' fold-9 rows, all fitted on July + August like the sweep fit):

    * ``mix``: the config at 1/2, each member at 1/10;
    * ``fraud_only``: legit column from ``mix``, the 11 types from the five-way blend rescaled
      to ``mix``'s fraud mass."""
    from ringtrace import silver, store
    from ringtrace.blend import fraud_split_from
    from ringtrace.scoring import macro_f1

    store.use_frame(None)
    led = store.ledger()
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    tr = np.flatnonzero(~is_test)
    hold = ~is_test & (period == 9)
    sep_in_tr = period[tr] == 9
    runs = paths.out_dir() / "runs"
    five = sum(np.load(runs / m / "oof.npy")[sep_in_tr].astype(np.float64) for m in FIVE_WAY) / 5
    mine = np.load(runs / (PREFIX + config) / "sep.npy").astype(np.float64)
    mix = 0.5 * mine + 0.5 * five
    arms = {"five": five, "mix": mix, "fraud_only": fraud_split_from(five, mix)}
    recipe = recipe_for(config)
    targets = {"given": led["y"].to_numpy().astype(np.int64)}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        targets[tname] = silver.load(led, which)
    out = {}
    for arm, prob in arms.items():
        pred = _decode_sep(led, period, hold, prob, recipe)
        out[arm] = {t: macro_f1(v[hold], pred) for t, v in targets.items()}
    for arm in ("mix", "fraud_only"):
        out[arm + "_delta"] = {t: out[arm][t] - out["five"][t] for t in targets}
    log(f"[role] {config}: " + json.dumps(out))
    return out


def table(base: str = "base") -> list[str]:
    """Markdown rows: each finished config's scores minus the base config's, per part."""
    runs = paths.out_dir() / "runs"
    ref = json.loads((runs / (PREFIX + base) / "summary.json").read_text())
    lines = []
    for name in CONFIGS:
        path = runs / (PREFIX + name) / "summary.json"
        if not path.exists():
            continue
        got = json.loads(path.read_text())
        cells = []
        for t in ("given", "cleaned"):
            sep = got.get("lomo", {}).get(t)
            oct_ = [got[s][t] - ref[s][t] for s in ("13", "14", "15") if s in got]
            cells.append("-" if sep is None else f"{sep - ref['lomo'][t]:+.4f}")
            cells.append(
                " / ".join(f"{d:+.4f}" for d in oct_) + (f" ({np.mean(oct_):+.4f})" if oct_ else "")
                if oct_
                else "-"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def main(argv=None) -> dict:
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.tabmsweep")
    ap.add_argument("config", choices=sorted(CONFIGS) + ["table"])
    ap.add_argument("--parts", default="lomo,13,14,15")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--role", action="store_true", help="only the blend-role scores of September")
    args = ap.parse_args(argv)
    if args.config == "table":
        for line in table():
            print(line)
        return {}
    recipe = recipe_for(args.config)
    out = paths.out_dir() / "runs" / recipe["name"]
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["recipe"] = recipe
    summary["seed"] = args.seed

    def log(msg):
        print(msg, flush=True)

    if args.role:
        summary["role"] = role_scores(args.config, log)
        summary_path.write_text(json.dumps(summary, indent=1))
        return summary
    for part in args.parts.split(","):
        log(f"[sweep] {args.config} part {part}")
        if part == "lomo":
            res = part_lomo(recipe, args.seed, log)
        else:
            res = part_octlike(recipe, int(part), args.seed, log)
        summary[part] = res
        log(f"[sweep] {args.config} {part}: " + json.dumps(res))
        summary_path.write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
