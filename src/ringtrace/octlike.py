"""October-like mode: September made to look like October, scored by models fitted on July + August.

    python -m ringtrace.octlike build [--seeds 13,14,15]
    python -m ringtrace.octlike fit champion:0,1,2 [--seeds 13,14,15]
    python -m ringtrace.octlike score <base> <challenger> [--seeds 13,14,15] [--sets 0,1,2]

``build`` makes one frame per thinning seed s from the normal ledger:

* September rows are grouped into protected units, kept or dropped whole: generator events (rows
  sharing an event id in any evidence module, joined across modules where events overlap) and
  label units (connected fraud-labelled rows plus the legit-labelled rows next to them: same
  sender or device within 1 h, same receiving person within 72 h, same pair within 7 days, money
  passed on within 24 h), merged where they share a row; every other row (marked rows outside
  events included) is its own unit;
* each unit is kept with probability ``KEEP`` (October's legit rows per day against
  September's); July, August and October stay complete;
* among kept September QR payments to a person, ``qr_vpa_mismatch`` is blanked on ``QR_BLANK``
  of them, as in October.

The frame (kept stacked rows, blanked rows, unit of every kept September row) is saved under
``$RINGTRACE_CACHE/octlike/s<seed>.npz``. ``fit`` rebuilds every feature family on the frame's
ledger (its own cache key), except the device counter, whose columns are held at their values on
the unthinned build (thinning breaks its block finder). The recipe is fitted on July + August
only (the cascade's screening score is cross-fitted between those two months); seed set r uses
cascade seeds 3r, 3r + 1, 3r + 2. September's pre-decoder probabilities go to
``$RINGTRACE_OUT/octlike/<recipe>_s<seed>_r<r>.npz`` (``rows`` = stacked row, ``prob`` = n x 12).
September is decoded on its own, like a test month: fixed point, then the recipe's rules.

``score`` is the gate: the challenger (a recipe, or a blend ``a+b`` / ``a*0.6+b*0.4`` of recipes'
cells) against the base in every cell, on the given, cleaned and gen-only targets, with a paired
Poisson bootstrap over the protected units. Only aggregates are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ringtrace import paths

KEEP = 0.494
QR_BLANK = 0.44
SEEDS = (13, 14, 15)
SEED_SETS = (0, 1, 2)
FIT_PERIODS = (7, 8)
HOLD_PERIOD = 9
HOUR = 3600
EVENT_COLUMNS = ("bot_event", "drn_event", "mf_event", "col_event", "rng_event", "rep_event")
EVENT_FAMILIES = ("bursts", "drains", "flows", "requests", "merchant", "repeats")
HELD_FAMILIES = ("phones", "phones_prior")
DRAWS = 1000
DEFAULT_DECODER = {"fraud_scale": 0.875}
DEFAULT_RULES = ["anchored_templates"]


# ---------------------------------------------------------------------------------------------
# protected units


def _dense(*keys: np.ndarray) -> list[np.ndarray]:
    """Map several key arrays into one dense id space (negative keys stay -1)."""
    allk = np.concatenate(keys)
    ok = allk >= 0
    out = np.full(len(allk), -1, dtype=np.int64)
    out[ok] = np.unique(allk[ok], return_inverse=True)[1]
    return np.split(out, np.cumsum([len(k) for k in keys])[:-1])


def _range_edges(
    q_rows: np.ndarray,
    q_key: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    d_key: np.ndarray,
    d_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Edges joining each query row to every eligible row with the same key and a time in
    [q_lo, q_hi]. Rows of one range are chained in time order (same components, fewer edges)."""
    elig = np.flatnonzero(d_key >= 0)
    if len(elig) == 0 or len(q_rows) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    o = elig[np.lexsort((d_ts[elig], d_key[elig]))]
    span = int(max(d_ts.max(), q_hi.max()) - min(d_ts.min(), q_lo.min())) + 2
    base = int(min(d_ts.min(), q_lo.min()))
    comp = d_key[o] * span + (d_ts[o] - base)
    use = q_key >= 0
    q_rows, q_key, q_lo, q_hi = q_rows[use], q_key[use], q_lo[use], q_hi[use]
    lo = np.searchsorted(comp, q_key * span + (q_lo - base), side="left")
    hi = np.searchsorted(comp, q_key * span + (q_hi - base), side="right")
    hit = hi > lo
    a = [q_rows[hit]]
    b = [o[lo[hit]]]
    diff = np.zeros(len(o) + 1, dtype=np.int64)
    np.add.at(diff, lo[hit], 1)
    np.add.at(diff, hi[hit] - 1, -1)
    link = np.flatnonzero(np.cumsum(diff)[:-1] > 0)
    a.append(o[link])
    b.append(o[link + 1])
    return np.concatenate(a), np.concatenate(b)


def label_edges(
    sender: np.ndarray,
    receiver: np.ndarray,
    to_person: np.ndarray,
    device: np.ndarray,
    ts: np.ndarray,
    fraud: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Edges of the label units: every fraud-labelled row to the rows next to it."""
    ts = ts.astype(np.int64)
    f = np.flatnonzero(fraud)
    person_rcv = np.where(to_person, receiver, -1)
    snd_d, rcv_d = _dense(sender.astype(np.int64), person_rcv.astype(np.int64))
    (dev_d,) = _dense(device.astype(np.int64))
    rcv64 = receiver.astype(np.int64)
    (pair,) = _dense(sender.astype(np.int64) * (int(rcv64.max(initial=0)) + 1) + rcv64)
    parts = []
    for key, w in ((snd_d, HOUR), (dev_d, HOUR), (rcv_d, 72 * HOUR), (pair, 7 * 24 * HOUR)):
        parts.append(_range_edges(f, key[f], ts[f] - w, ts[f] + w, key, ts))
    day = 24 * HOUR
    # money passed on: a pays person R, R pays onward within 24 h (a fraud: look forward ...)
    parts.append(_range_edges(f, rcv_d[f], ts[f], ts[f] + day, snd_d, ts))
    # ... b fraud: look back for payments into b's sender
    parts.append(_range_edges(f, snd_d[f], ts[f] - day, ts[f], rcv_d, ts))
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))


def event_edges(events: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Edges chaining the rows of each generator event (one id array per evidence module)."""
    a, b = [], []
    for ev in events:
        rows = np.flatnonzero(ev >= 0)
        o = rows[np.argsort(ev[rows], kind="stable")]
        same = ev[o[1:]] == ev[o[:-1]]
        a.append(o[:-1][same])
        b.append(o[1:][same])
    if not a:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(a), np.concatenate(b)


def protected_units(n: int, edges: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Connected components of the rows under the given edges (dense ids, deterministic)."""
    a = np.concatenate([e[0] for e in edges] + [np.zeros(0, np.int64)])
    b = np.concatenate([e[1] for e in edges] + [np.zeros(0, np.int64)])
    g = sp.coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n)).tocsr()
    _, lab = connected_components(g, directed=False)
    return lab.astype(np.int64)


def september_units(
    led: pl.DataFrame,
    events: list[np.ndarray],
    sept: np.ndarray,
    with_labels: bool = True,
    with_events: bool = True,
) -> np.ndarray:
    """Unit id of every September row (indices ``sept`` into ``led``)."""
    s = led[sept]
    edges = []
    if with_events:
        edges.append(event_edges([e[sept] for e in events]))
    if with_labels:
        y = s["y"].to_numpy()
        edges.append(
            label_edges(
                s["sender"].to_numpy(),
                s["receiver"].to_numpy(),
                s["to_person"].to_numpy(),
                s["device"].fill_null(-1).to_numpy(),
                s["ts"].to_numpy(),
                y > 0,
            )
        )
    return protected_units(s.height, edges)


def thin(units: np.ndarray, seed: int, keep: float = KEEP) -> np.ndarray:
    """Keep each unit whole with probability ``keep`` (seeded); returns the row mask."""
    rng = np.random.default_rng([seed, 0])
    kept_unit = rng.random(units.max() + 1 if len(units) else 0) < keep
    return kept_unit[units]


def qr_blank(candidates: np.ndarray, seed: int, share: float = QR_BLANK) -> np.ndarray:
    """Which of the candidate rows lose the QR check (seeded)."""
    rng = np.random.default_rng([seed, 1])
    return candidates[rng.random(len(candidates)) < share]


# ---------------------------------------------------------------------------------------------
# frames


def _frame_dir() -> Path:
    d = paths.cache_dir() / "octlike"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out_dir() -> Path:
    d = paths.out_dir() / "octlike"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_frame(seed: int, log=print) -> dict:
    """Thin September of the normal ledger and save the frame definition."""
    from ringtrace import store

    store.use_frame(None)
    led = store.ledger()
    period = led["period"].to_numpy()
    is_test = led["is_test"].to_numpy()
    sept = np.flatnonzero(~is_test & (period == HOLD_PERIOD))
    events = [
        store.family(f)[c].fill_null(-1).to_numpy().astype(np.int64)
        for f, c in zip(EVENT_FAMILIES, EVENT_COLUMNS, strict=True)
    ]
    units = september_units(led, events, sept)
    kept = thin(units, seed)
    keep_rows = np.sort(
        np.concatenate([np.flatnonzero(~np.isin(np.arange(led.height), sept)), sept[kept]])
    )
    qr = (led["txn_type"] == "QR").to_numpy() & led["to_person"].to_numpy()
    cand = sept[kept & qr[sept]]
    blanked = qr_blank(cand, seed)
    unit_of = np.full(led.height, -1, dtype=np.int64)
    unit_of[sept] = units
    res = {
        "seed": seed,
        "rows": keep_rows.astype(np.int64),
        "blank": blanked.astype(np.int64),
        "unit": unit_of[keep_rows],
    }
    np.savez(_frame_dir() / f"s{seed}.npz", **res)
    sizes, counts = np.unique(np.bincount(units), return_counts=True)
    stats = {
        "september_rows": int(len(sept)),
        "units": int(units.max() + 1),
        "largest_unit": int(sizes.max()),
        "units_over_one_row": int(counts[sizes > 1].sum()),
        "kept_september_rows": int(kept.sum()),
        "kept_share": float(kept.mean()),
        "qr_person_kept": int(len(cand)),
        "qr_blanked": int(len(blanked)),
        "frame_rows": int(len(keep_rows)),
    }
    (_frame_dir() / f"s{seed}.json").write_text(json.dumps(stats, indent=1))
    log(f"[octlike] frame s{seed}: {json.dumps(stats)}")
    return stats


class Frame:
    """A thinned frame installed as the store's ledger, with the held counter columns."""

    def __init__(self, seed: int):
        from ringtrace import silver, store

        store.use_frame(None)
        base = store.ledger()
        self.base_height = base.height
        f = np.load(_frame_dir() / f"s{seed}.npz")
        self.seed = seed
        self.stacked = f["rows"]
        self.unit = f["unit"]
        digest = hashlib.sha256(self.stacked.tobytes() + f["blank"].tobytes()).hexdigest()[:12]
        self.targets = {"given": base["y"].to_numpy().astype(np.int64)[self.stacked]}
        for tname, which in (("cleaned", "label"), ("gen", "structured")):
            arr = silver.load(base, which)
            if arr is not None:
                self.targets[tname] = arr.astype(np.int64)[self.stacked]
        held = {name: store.family(name)[self.stacked] for name in HELD_FAMILIES}
        blank = np.isin(self.stacked, f["blank"])
        led = base[self.stacked].with_columns(
            pl.Series("stacked_row", self.stacked),
            pl.int_range(0, len(self.stacked), dtype=pl.Int32).alias("row"),
            pl.when(pl.Series(blank))
            .then(None)
            .otherwise(pl.col("qr_vpa_mismatch"))
            .alias("qr_vpa_mismatch"),
        )
        store.use_frame(led, key=f"octlike-s{seed}-{digest}", fixed=held)
        self.led = led
        self.period = led["period"].to_numpy()
        is_test = led["is_test"].to_numpy()
        self.fit = ~is_test & np.isin(self.period, FIT_PERIODS)
        self.hold = ~is_test & (self.period == HOLD_PERIOD)


def decode(
    frame: Frame, prob: np.ndarray, decoder: dict | None = None, rules: list | None = None
) -> np.ndarray:
    """September decoded on its own: the fixed point, then the rules (label-free)."""
    from ringtrace import rules as rulebook
    from ringtrace.decode import fixed_point

    hold = np.flatnonzero(frame.hold)
    pred = np.zeros(frame.led.height, dtype=np.int64)
    pred[hold] = fixed_point(prob, **(DEFAULT_DECODER if decoder is None else decoder))
    rules = DEFAULT_RULES if rules is None else rules
    if rules:
        full = np.zeros((frame.led.height, prob.shape[1]))
        full[hold] = prob
        pred = rulebook.apply(rules, pred, full, frame.led, log=lambda *_: None)
    return pred[hold]


def _cleaned_target(frame: Frame, log) -> tuple[np.ndarray, dict]:
    """Cleaned labels for July + August from a teacher that never sees September."""
    from ringtrace import silver, store, teacher

    feats = store.frame(["context", "marks"])
    cols = [c for c in feats.columns if not c.endswith("_event")]
    Xt = feats.select(cols).to_numpy().astype(np.float32)
    y = frame.led["y"].to_numpy().astype(np.int64)
    proba = teacher.teacher_oof(Xt[frame.fit], y[frame.fit], frame.period[frame.fit])
    events = teacher.union_events(frame.led.height)
    res = silver.build(frame.led, proba, events, log=log, mask=frame.fit)
    return res["label"], res["stats"]


def train_target(
    recipe: dict, y: np.ndarray, fit_rows: np.ndarray, cleaned=None, events=None
) -> np.ndarray:
    """The target a recipe trains on: given labels, or a cleaned copy where only ``fit_rows``
    vote and change (so September never shapes them)."""
    from ringtrace import eventclean
    from ringtrace.run import EVENT_MODES

    mode = recipe.get("train_on")
    target = y.copy()
    if mode in (None, "given"):
        return target
    if mode == "cleaned":
        target[fit_rows] = cleaned()[fit_rows]
        return target
    if mode in EVENT_MODES:
        return eventclean.clean_labels(y, events(), EVENT_MODES[mode], mask=fit_rows)
    raise ValueError(f"unknown train_on {mode!r}")


def fit(
    seeds: list[int],
    cells: list[tuple[str, list[int]]],
    recipes_dir: Path,
    months: tuple[int, ...] = FIT_PERIODS,
    use_screen: bool = True,
    tag: str = "",
    log=print,
) -> dict:
    """Fit cells ``<recipe><tag>_s<seed>_r<r>``. ``months`` (default July + August) are the fit
    rows; ``use_screen=False`` turns the cascade's screening score off (one month alone cannot
    be cross-fitted)."""
    from ringtrace import eventclean
    from ringtrace import run as runner
    from ringtrace.scoring import macro_f1

    t0 = time.time()
    summary_path = _out_dir() / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for seed in seeds:
        frame = Frame(seed)
        y = frame.led["y"].to_numpy().astype(np.int64)
        fit_rows = frame.fit & np.isin(frame.period, months)
        for recipe_name, sets in cells:
            recipe = runner.load_recipe(recipes_dir / f"{recipe_name}.json")
            X, _ = runner.feature_matrix(recipe)
            log(
                f"[octlike] s{seed} {recipe_name}{tag}: {X.shape[1]} features, fit "
                f"{int(fit_rows.sum())}, September {int(frame.hold.sum())} "
                f"({time.time() - t0:.0f}s)"
            )

            def cleaned(name=recipe_name, frame=frame, seed=seed):
                lab, stats = _cleaned_target(frame, log)
                summary[f"{name}_s{seed}_cleaning"] = stats
                return lab

            target = train_target(recipe, y, fit_rows, cleaned, eventclean.ledger_events)
            if recipe.get("train_on") not in (None, "given"):
                changed = int((target != y)[fit_rows].sum())
                summary[f"{recipe_name}{tag}_s{seed}_changed"] = changed
                log(f"[octlike] {recipe['train_on']}: {changed} fit rows relabelled")
            spec, fit_predict = runner.model(recipe)
            if hasattr(spec, "use_screen"):
                spec.use_screen = use_screen
            for r in sets:
                name = f"{recipe_name}{tag}_s{seed}_r{r}"
                path = _out_dir() / f"{name}.npz"
                if path.exists():
                    log(f"[octlike] {name} exists, skipped")
                    continue
                t1 = time.time()
                spec.seeds = (3 * r, 3 * r + 1, 3 * r + 2)
                (prob,) = fit_predict(
                    X[fit_rows],
                    target[fit_rows],
                    frame.period[fit_rows],
                    [X[frame.hold]],
                    spec,
                    log=lambda *_: None,
                )
                tmp = path.with_name(f".{name}.tmp.npz")
                np.savez(tmp, rows=frame.stacked[frame.hold], prob=prob.astype(np.float32))
                tmp.replace(path)
                pred = decode(frame, prob, recipe.get("decoder"), recipe.get("rules", []))
                scores = {t: macro_f1(v[frame.hold], pred) for t, v in frame.targets.items()}
                summary[name] = {
                    **scores,
                    "seeds": list(spec.seeds),
                    "months": list(months),
                    "screen": use_screen,
                    "rows": int(frame.hold.sum()),
                    "wall_s": time.time() - t1,
                }
                log(
                    f"[octlike] {name}: September "
                    + ", ".join(f"{t} {v:.4f}" for t, v in scores.items())
                    + f" ({time.time() - t1:.0f}s)"
                )
                summary_path.write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------------------------------------
# the gate


def parse_blend(spec: str) -> list[tuple[str, float]]:
    """``a`` -> [(a, 1)]; ``a+b`` -> equal weights; ``a*0.6+b*0.4`` -> fixed weights."""
    parts = []
    for p in spec.split("+"):
        name, _, w = p.partition("*")
        parts.append((name, float(w) if w else 1.0))
    total = sum(w for _, w in parts)
    return [(n, w / total) for n, w in parts]


def cell_prob(blend: list[tuple[str, float]], seed: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    rows, prob = None, 0.0
    for name, w in blend:
        f = np.load(_out_dir() / f"{name}_s{seed}_r{r}.npz")
        if rows is None:
            rows = f["rows"]
        elif not np.array_equal(rows, f["rows"]):
            raise ValueError(f"{name}: rows differ in cell s{seed} r{r}")
        prob = prob + w * f["prob"].astype(np.float64)
    return rows, prob


def bootstrap(
    units: np.ndarray,
    truth: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict:
    """Paired Poisson bootstrap of macro-F1(b) - macro-F1(a) over units: point, 95 % bounds."""
    from ringtrace.compare import _counts, _macro

    _, g = np.unique(units, return_inverse=True)
    ng = int(g.max()) + 1
    ca = _counts(g, truth, pred_a, ng)
    cb = _counts(g, truth, pred_b, ng)
    rng = np.random.default_rng(seed)
    w = sp.csr_matrix(rng.poisson(1.0, size=(draws, ng)).astype(np.float64))
    d = _macro(np.asarray((w @ cb).todense())) - _macro(np.asarray((w @ ca).todense()))
    one = np.ones((1, ng))
    point = float(_macro(np.asarray(one @ cb))[0] - _macro(np.asarray(one @ ca))[0])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": point, "lo": float(lo), "hi": float(hi)}


def _recipe_decoding(blend, recipes_dir: Path) -> tuple[dict | None, list | None]:
    if len(blend) == 1:
        p = recipes_dir / f"{blend[0][0]}.json"
        if p.exists():
            rec = json.loads(p.read_text())
            return rec.get("decoder"), rec.get("rules", [])
    return None, None


def score(
    base: str,
    challenger: str,
    seeds: list[int],
    sets: list[int],
    recipes_dir: Path,
    draws: int = DRAWS,
    log=print,
    rules_a: list | None = None,
    rules_b: list | None = None,
) -> dict:
    """``rules_a`` / ``rules_b`` replace the base's / challenger's rules (default decoder)."""
    from ringtrace.scoring import macro_f1

    ba, ch = parse_blend(base), parse_blend(challenger)
    dec_a, dec_b = _recipe_decoding(ba, recipes_dir), _recipe_decoding(ch, recipes_dir)
    if rules_a is not None:
        dec_a = (dec_a[0], rules_a)
    if rules_b is not None:
        dec_b = (dec_b[0], rules_b)
    cells = {}
    for seed in seeds:
        frame = Frame(seed)
        units = frame.unit[frame.hold]
        for r in sets:
            rows_a, pa = cell_prob(ba, seed, r)
            rows_b, pb = cell_prob(ch, seed, r)
            if not (
                np.array_equal(rows_a, rows_b) and np.array_equal(rows_a, frame.stacked[frame.hold])
            ):
                raise ValueError(f"cell s{seed} r{r}: rows differ from the frame")
            ya = decode(frame, pa, *dec_a)
            yb = decode(frame, pb, *dec_b)
            cell = {}
            for t, truth in frame.targets.items():
                yt = truth[frame.hold]
                bs = bootstrap(units, yt, ya, yb, draws, seed=seed * 100 + r)
                cell[t] = {"base": macro_f1(yt, ya), "challenger": macro_f1(yt, yb), **bs}
            cells[(seed, r)] = cell
    report = {"base": base, "challenger": challenger, "per_seed": {}, "lower_bound": {}}
    targets = list(next(iter(cells.values())))
    if rules_b is not None:
        report["challenger_rules"] = rules_b
    log(f"[gate] {challenger} vs {base}, {len(cells)} cells, {draws} draws, 95 %")
    for t in targets:
        per_seed = {s: float(np.mean([cells[(s, r)][t]["delta"] for r in sets])) for s in seeds}
        lb = float(np.mean([c[t]["lo"] for c in cells.values()]))
        base_mean = float(np.mean([c[t]["base"] for c in cells.values()]))
        report["per_seed"][t] = per_seed
        report["lower_bound"][t] = lb
        log(
            f"[gate] {t:<8} base {base_mean:.4f}  change per seed "
            + " / ".join(f"s{s} {v:+.4f}" for s, v in per_seed.items())
            + f"  mean {np.mean(list(per_seed.values())):+.4f}  lower bound {lb:+.4f}"
        )
    must = [t for t in ("given", "cleaned") if t in targets]
    report["passes"] = bool(
        all(v > 0 for t in must for v in report["per_seed"][t].values())
        and all(report["lower_bound"][t] > 0 for t in must)
    )
    report["cells"] = {f"s{s}_r{r}": c for (s, r), c in cells.items()}
    log(f"[gate] {'PASS' if report['passes'] else 'FAIL'}")
    return report


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None) -> dict | None:
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.octlike")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    f = sub.add_parser("fit")
    f.add_argument("cells", nargs="+", help="recipe:seedset,seedset ...")
    f.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    f.add_argument("--recipes", default="recipes")
    f.add_argument("--months", default=",".join(map(str, FIT_PERIODS)))
    f.add_argument("--no-screen", action="store_true")
    f.add_argument("--tag", default="", help="cell name suffix after the recipe name")
    s = sub.add_parser("score")
    s.add_argument("base")
    s.add_argument("challenger")
    s.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    s.add_argument("--sets", default=",".join(map(str, SEED_SETS)))
    s.add_argument("--draws", type=int, default=DRAWS)
    s.add_argument("--recipes", default="recipes")
    s.add_argument("--rules-a", help="comma-separated rules replacing the base's")
    s.add_argument("--rules-b", help="comma-separated rules replacing the challenger's")
    s.add_argument("--tag", default="", help="suffix of the report file name")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        for seed in _ints(args.seeds):
            build_frame(seed)
        return None
    if args.cmd == "fit":
        cells = []
        for c in args.cells:
            name, _, sets = c.partition(":")
            cells.append((name, _ints(sets) if sets else list(SEED_SETS)))
        return fit(
            _ints(args.seeds),
            cells,
            Path(args.recipes),
            tuple(_ints(args.months)),
            not args.no_screen,
            args.tag,
        )
    rep = score(
        args.base,
        args.challenger,
        _ints(args.seeds),
        _ints(args.sets),
        Path(args.recipes),
        args.draws,
        rules_a=args.rules_a.split(",") if args.rules_a else None,
        rules_b=args.rules_b.split(",") if args.rules_b else None,
    )
    name = f"gate_{args.challenger}_vs_{args.base}{args.tag}.json".replace("*", "x")
    (_out_dir() / name).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
