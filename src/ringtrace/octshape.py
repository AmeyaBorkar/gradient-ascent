"""October-shaped training copies of the train months, built in octlike's protected mode.

    python -m ringtrace.octshape mix                       # artefact check (aggregates only)
    python -m ringtrace.octshape octlike --copies 0,1,2 [--seeds 13,14,15] [--sets 0]
    python -m ringtrace.octshape lomo --copies 0,1,2

A copy of month m with copy seed c is a frame of the normal ledger in which only month m is
changed, the way ``ringtrace.octlike`` changes September:

* month m's rows are grouped into protected units (generator events and label units together,
  ``octlike.september_units``) and each unit is kept whole with probability ``octlike.KEEP``;
* the amount of every kept legit-labelled row of month m that sits outside every label-free
  generator structure (an event id of any evidence module, or a row mark) is scaled by
  ``AMOUNT_SCALE`` (rounded to the paisa); ``scale="off"`` skips it (control arm), and
  ``scale="all"`` scales every legit-labelled row (the first build, which split rings);
* ``qr_vpa_mismatch`` is blanked on ``octlike.QR_BLANK`` of the kept QR payments to a person;
* every feature family is rebuilt on the frame (its own cache key), except the device counter,
  held at its values on the unthinned build.

The copy's training rows are month m's kept rows with their given labels and period m (so the
cascade's cross-fitted screen keeps a copy with its source month). Draws use the generator key
``[c, m, k]``; copy seeds are distinct from the validator's thinning seeds 13-15.

``mix`` compares the label-free event-size mix of the copies with raw train and real October.
``octlike`` fits the champion recipe on July + August plus 0, 1 or 2 copies of each, on the
octlike frames, and writes cells ``octshape-c<k>_s<seed>_r<r>`` for ``octlike score``.
``lomo`` fits the same arms month by month (copies of the training months only, the held-out
month raw) and writes a run per arm under ``$RINGTRACE_OUT/runs/octshape-lomo-c<k>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import octlike, paths

AMOUNT_SCALE = 1.166
COPY_SEEDS = (101, 102)
TRAIN_MONTHS = (7, 8, 9)
RECIPE = "recipes/champion.json"
WATCH = ("qr_swap", "merchant_collusion", "refund_abuse")
RING_SHAPES = (1, 2, 3)
SCALE_MODES = ("outside", "off", "all")


# ---------------------------------------------------------------------------------------------
# pure pieces


def keep_units(units: np.ndarray, seed: int, month: int, keep: float = octlike.KEEP) -> np.ndarray:
    """Row mask keeping each unit whole with probability ``keep``."""
    rng = np.random.default_rng([seed, month, 0])
    kept = rng.random(int(units.max()) + 1 if len(units) else 0) < keep
    return kept[units]


def blank_rows(cand: np.ndarray, seed: int, month: int, share: float = octlike.QR_BLANK):
    rng = np.random.default_rng([seed, month, 1])
    return cand[rng.random(len(cand)) < share]


def scale_amounts(amount: np.ndarray, rows: np.ndarray, factor: float = AMOUNT_SCALE):
    """A copy of ``amount`` with ``rows`` scaled by ``factor`` and rounded to two decimals."""
    out = np.asarray(amount, dtype=np.float64).copy()
    out[rows] = np.round(out[rows] * factor, 2)
    return out


def scale_rows(idx: np.ndarray, kept: np.ndarray, y: np.ndarray, structured: np.ndarray, mode: str):
    """Rows (of ``idx``) whose amount is scaled: kept and legit-labelled; with ``outside`` also
    outside every generator structure; none with ``off``."""
    if mode not in SCALE_MODES:
        raise ValueError(f"unknown scale mode {mode!r}")
    sel = kept & (y[idx] == 0)
    if mode == "outside":
        sel &= ~structured[idx]
    elif mode == "off":
        sel &= False
    return idx[sel]


def event_mix(event: np.ndarray, mask: np.ndarray) -> dict:
    """Events touching the masked rows: count per 100k rows and mean rows per event."""
    ev = event[mask]
    ev = ev[ev >= 0]
    n = int(mask.sum())
    if len(ev) == 0:
        return {"per_100k": 0.0, "rows_per_event": 0.0}
    k = len(np.unique(ev))
    return {"per_100k": 1e5 * k / max(n, 1), "rows_per_event": len(ev) / k}


def ring_mix(event: np.ndarray, shape: np.ndarray, size: np.ndarray, mask: np.ndarray) -> dict:
    """Merchant ring events (shapes 1-3) in the masked rows by size, per 100k rows."""
    sel = mask & (event >= 0) & np.isin(shape, RING_SHAPES)
    n = max(int(mask.sum()), 1)
    _, first = np.unique(event[sel], return_index=True)
    sz = size[sel][first]
    bins = {"size1": sz == 1, "size2_3": (sz >= 2) & (sz <= 3), "size4_8": (sz >= 4) & (sz <= 8)}
    bins["size9p"] = sz >= 9
    out = {k: 1e5 * float(v.sum()) / n for k, v in bins.items()}
    out["mean_size"] = float(sz.mean()) if len(sz) else 0.0
    return out


# ---------------------------------------------------------------------------------------------
# copy frames


def _dir() -> Path:
    d = paths.cache_dir() / "octshape"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(month: int, seed: int, scale: str) -> Path:
    return _dir() / f"m{month}_s{seed}_{scale}.npz"


def build_copy(month: int, seed: int, scale: str = "outside", log=print) -> dict:
    """Choose the kept, blanked and scaled rows of a copy of ``month`` and save them."""
    from ringtrace import store

    store.use_frame(None)
    led = store.ledger()
    period = led["period"].to_numpy()
    is_test = led["is_test"].to_numpy()
    idx = np.flatnonzero(~is_test & (period == month))
    events = [
        store.family(f)[c].fill_null(-1).to_numpy().astype(np.int64)
        for f, c in zip(octlike.EVENT_FAMILIES, octlike.EVENT_COLUMNS, strict=True)
    ]
    units = octlike.september_units(led, events, idx)
    kept = keep_units(units, seed, month)
    others = np.flatnonzero(~np.isin(np.arange(led.height), idx))
    rows = np.sort(np.concatenate([others, idx[kept]]))
    qr = (led["txn_type"] == "QR").to_numpy() & led["to_person"].to_numpy()
    blank = blank_rows(idx[kept & qr[idx]], seed, month)
    y = led["y"].to_numpy()
    structured = store.family("marks")["mrk_any"].to_numpy() > 0
    for ev in events:
        structured |= ev >= 0
    scaled = scale_rows(idx, kept, y, structured, scale)
    np.savez(_path(month, seed, scale), rows=rows, blank=blank, scaled=scaled)
    stats = {
        "month_rows": int(len(idx)),
        "kept": int(kept.sum()),
        "kept_share": float(kept.mean()),
        "blanked": int(len(blank)),
        "scaled": int(len(scaled)),
    }
    stats["legit_kept"] = int((kept & (y[idx] == 0)).sum())
    log(f"[octshape] copy m{month} s{seed} {scale}: {json.dumps(stats)}")
    return stats


class Copy:
    """A copy frame installed as the store's ledger (device counter held)."""

    def __init__(self, month: int, seed: int, scale: str = "outside"):
        from ringtrace import store

        path = _path(month, seed, scale)
        if not path.exists():
            build_copy(month, seed, scale)
        store.use_frame(None)
        base = store.ledger()
        f = np.load(path)
        self.stacked = f["rows"]
        digest = hashlib.sha256(
            self.stacked.tobytes() + f["blank"].tobytes() + f["scaled"].tobytes()
        ).hexdigest()[:12]
        amount = scale_amounts(base["amount"].to_numpy(), f["scaled"])
        blank = np.isin(self.stacked, f["blank"])
        held = store.family(octlike.HELD_FAMILY)[self.stacked]
        led = base[self.stacked].with_columns(
            pl.Series("amount", amount[self.stacked]).cast(base["amount"].dtype),
            pl.Series("stacked_row", self.stacked),
            pl.int_range(0, len(self.stacked), dtype=pl.Int32).alias("row"),
            pl.when(pl.Series(blank))
            .then(None)
            .otherwise(pl.col("qr_vpa_mismatch"))
            .alias("qr_vpa_mismatch"),
        )
        store.use_frame(
            led, key=f"octshape-m{month}-s{seed}-{digest}", fixed={octlike.HELD_FAMILY: held}
        )
        self.led = led
        self.month = month
        p = led["period"].to_numpy()
        self.mine = ~led["is_test"].to_numpy() & (p == month)


def copy_matrix(recipe: dict, month: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Features and given labels of a copy's own rows."""
    from ringtrace import run as runner
    from ringtrace import store

    c = Copy(month, seed)
    X, _ = runner.feature_matrix(recipe)
    y = c.led["y"].to_numpy().astype(np.int64)
    store.use_frame(None)
    return X[c.mine], y[c.mine]


def copies(recipe: dict, months, k: int, log=print) -> dict:
    out = {}
    for m in months:
        for s in COPY_SEEDS[:k]:
            t = time.time()
            out[(m, s)] = copy_matrix(recipe, m, s)
            log(f"[octshape] copy m{m} s{s}: {len(out[(m, s)][1])} rows ({time.time() - t:.0f}s)")
    return out


# ---------------------------------------------------------------------------------------------
# artefact check


def _mix_of(led: pl.DataFrame, mask: np.ndarray) -> dict:
    from ringtrace import store

    res = {"rows": int(mask.sum())}
    for fam, col in zip(octlike.EVENT_FAMILIES, octlike.EVENT_COLUMNS, strict=True):
        ev = store.family(fam)[col].fill_null(-1).to_numpy().astype(np.int64)
        res[fam] = event_mix(ev, mask)
    m = store.family("merchant")
    res["rings"] = ring_mix(
        m["rng_event"].fill_null(-1).to_numpy().astype(np.int64),
        m["rng_shape"].fill_null(0).to_numpy(),
        m["rng_size"].fill_null(0).to_numpy(),
        mask,
    )
    legit = mask & (led["y"].to_numpy() == 0) & ~led["is_test"].to_numpy()
    if legit.any():
        res["legit_amount_mean"] = float(led["amount"].to_numpy()[legit].mean())
    qr = mask & (led["txn_type"] == "QR").to_numpy() & led["to_person"].to_numpy()
    res["qr_person_blank_share"] = float(led["qr_vpa_mismatch"].is_null().to_numpy()[qr].mean())
    return res


def mix(seeds=COPY_SEEDS, scales=("outside", "off"), log=print) -> dict:
    from ringtrace import store

    store.use_frame(None)
    led = store.ledger()
    period = led["period"].to_numpy()
    is_test = led["is_test"].to_numpy()
    out = {f"raw_m{m}": _mix_of(led, ~is_test & (period == m)) for m in TRAIN_MONTHS}
    out["october"] = _mix_of(led, is_test)
    for m in TRAIN_MONTHS:
        for s in seeds:
            for sc in scales:
                c = Copy(m, s, sc)
                out[f"copy_m{m}_s{s}_{sc}"] = _mix_of(c.led, c.mine)
    for s in octlike.SEEDS:
        if (paths.cache_dir() / "octlike" / f"s{s}.npz").exists():
            f = octlike.Frame(s)
            out[f"octlike_s{s}"] = _mix_of(f.led, f.hold)
    store.use_frame(None)
    for k, v in out.items():
        r = v["rings"]
        log(
            f"[mix] {k:<22} rows {v['rows']:>7}  rings/100k s1 {r['size1']:.2f} s2-3 "
            f"{r['size2_3']:.2f} s4-8 {r['size4_8']:.2f} s9+ {r['size9p']:.2f} "
            f"mean {r['mean_size']:.2f}  "
            + " ".join(
                f"{f[:4]} {v[f]['per_100k']:.0f}/{v[f]['rows_per_event']:.2f}"
                for f in octlike.EVENT_FAMILIES
            )
            + f"  legit amt {v.get('legit_amount_mean', float('nan')):.1f}"
            f"  qr blank {v['qr_person_blank_share']:.3f}"
        )
    return out


# ---------------------------------------------------------------------------------------------
# fits


def _out() -> Path:
    d = paths.out_dir() / "octshape"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fit_octlike(arms: list[int], seeds: list[int], sets: list[int], log=print) -> dict:
    """Champion recipe on July + August (+ copies of each), scored on the thinned September."""
    from ringtrace import run as runner
    from ringtrace import store
    from ringtrace.cascade import fit_predict
    from ringtrace.scoring import macro_f1

    recipe = runner.load_recipe(RECIPE)
    cps = copies(recipe, octlike.FIT_PERIODS, max(arms), log)
    summary_path = _out() / "octlike_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for seed in seeds:
        frame = octlike.Frame(seed)
        X, _ = runner.feature_matrix(recipe)
        y = frame.led["y"].to_numpy().astype(np.int64)
        spec = runner.cascade_spec(recipe)
        for k in arms:
            parts = [(X[frame.fit], y[frame.fit], frame.period[frame.fit])]
            for (m, s), (Xc, yc) in cps.items():
                if s in COPY_SEEDS[:k]:
                    parts.append((Xc, yc, np.full(len(yc), m, dtype=frame.period.dtype)))
            Xf = np.concatenate([p[0] for p in parts])
            yf = np.concatenate([p[1] for p in parts])
            pf = np.concatenate([p[2] for p in parts])
            for r in sets:
                name = f"octshape-c{k}_s{seed}_r{r}"
                path = octlike._out_dir() / f"{name}.npz"
                if path.exists():
                    log(f"[octshape] {name} exists, skipped")
                    continue
                t = time.time()
                spec.seeds = (3 * r, 3 * r + 1, 3 * r + 2)
                (prob,) = fit_predict(Xf, yf, pf, [X[frame.hold]], spec, log=lambda *_: None)
                np.savez(path, rows=frame.stacked[frame.hold], prob=prob.astype(np.float32))
                pred = octlike.decode(frame, prob, recipe.get("decoder"), recipe.get("rules"))
                scores = {t_: macro_f1(v[frame.hold], pred) for t_, v in frame.targets.items()}
                summary[name] = {**scores, "fit_rows": len(yf), "wall_s": time.time() - t}
                log(
                    f"[octshape] {name}: fit {len(yf)} "
                    + ", ".join(f"{a} {b:.4f}" for a, b in scores.items())
                    + f" ({time.time() - t:.0f}s)"
                )
                summary_path.write_text(json.dumps(summary, indent=1))
    store.use_frame(None)
    return summary


def watch_classes(base: str, challenger: str, seeds, sets, log=print) -> dict:
    """Per-cell F1 of the watched classes, base and challenger, on every target."""
    from ringtrace.classes import CLASSES
    from ringtrace.scoring import per_class_f1

    idx = [CLASSES.index(c) for c in WATCH]
    out = {}
    for seed in seeds:
        frame = octlike.Frame(seed)
        for r in sets:
            preds = []
            for name in (base, challenger):
                _, p = octlike.cell_prob([(name, 1.0)], seed, r)
                preds.append(octlike.decode(frame, p))
            cell = {}
            for t, truth in frame.targets.items():
                yt = truth[frame.hold]
                fa, fb = (per_class_f1(yt, p) for p in preds)
                cell[t] = {c: [float(fa[i]), float(fb[i])] for c, i in zip(WATCH, idx, strict=True)}
            out[f"s{seed}_r{r}"] = cell
            log(
                f"[watch] s{seed} r{r} "
                + "  ".join(
                    f"{c} {cell['given'][c][0]:.4f}->{cell['given'][c][1]:.4f}" for c in WATCH
                )
            )
    return out


def lomo(arms: list[int], log=print) -> dict:
    """Leave-one-month-out per arm; copies of the training months only."""
    from ringtrace import run as runner
    from ringtrace import silver, store
    from ringtrace.cascade import fit_predict
    from ringtrace.classes import CLASSES
    from ringtrace.decode import decode_by_period
    from ringtrace.rules import apply as apply_rules
    from ringtrace.scoring import macro_f1, per_class_f1

    recipe = runner.load_recipe(RECIPE)
    cps = copies(recipe, TRAIN_MONTHS, max(arms), log)
    store.use_frame(None)
    led = store.ledger()
    X, cols = runner.feature_matrix(recipe)
    is_test = led["is_test"].to_numpy()
    period = led["period"].to_numpy()
    y = led["y"].to_numpy().astype(np.int64)
    tr = np.flatnonzero(~is_test)
    targets = {"given": y}
    for tname, which in (("cleaned", "label"), ("gen", "structured")):
        arr = silver.load(led, which)
        if arr is not None:
            targets[tname] = arr.astype(np.int64)
    spec = runner.cascade_spec(recipe)
    report = {}
    for k in arms:
        t0 = time.time()
        oof = np.zeros((led.height, len(CLASSES)))
        for h in TRAIN_MONTHS:
            fit = tr[period[tr] != h]
            hold = tr[period[tr] == h]
            parts = [(X[fit], y[fit], period[fit])]
            for (m, s), (Xc, yc) in cps.items():
                if m != h and s in COPY_SEEDS[:k]:
                    parts.append((Xc, yc, np.full(len(yc), m, dtype=period.dtype)))
            Xf = np.concatenate([p[0] for p in parts])
            (oof[hold],) = fit_predict(
                Xf,
                np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]),
                [X[hold]],
                spec,
                log=lambda *_: None,
            )
            log(f"[octshape] lomo c{k} month {h}: fit {len(Xf)} ({time.time() - t0:.0f}s)")
        pred = np.zeros(led.height, dtype=np.int64)
        pred[tr] = decode_by_period(oof[tr], period[tr], **recipe["decoder"])
        pred = apply_rules(recipe["rules"], pred, oof, led, log=lambda *_: None)
        res = {}
        for t, truth in targets.items():
            months = [
                macro_f1(truth[tr][period[tr] == m], pred[tr][period[tr] == m])
                for m in TRAIN_MONTHS
            ]
            f1 = per_class_f1(truth[tr], pred[tr])
            res[t] = {
                "months": months,
                "mean": float(np.mean(months)),
                "watch": {c: float(f1[CLASSES.index(c)]) for c in WATCH},
            }
        name = f"octshape-lomo-c{k}"
        out = paths.out_dir() / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "oof.npy", oof[tr].astype(np.float32))
        rec = {**recipe, "name": name, "octshape_copies": k}
        summary = {"recipe": rec, "features": cols, "cascade": asdict(spec), "final": res}
        summary["wall_s"] = time.time() - t0
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        report[name] = res
        log(
            f"[octshape] {name}: "
            + "  ".join(
                f"{t} " + " / ".join(f"{v:.4f}" for v in r["months"]) + f" ({r['mean']:.4f})"
                for t, r in res.items()
            )
            + f" ({time.time() - t0:.0f}s)"
        )
    return report


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None):
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.octshape")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mix")
    o = sub.add_parser("octlike")
    o.add_argument("--copies", default="0,1,2")
    o.add_argument("--seeds", default="13,14,15")
    o.add_argument("--sets", default="0")
    w = sub.add_parser("watch")
    w.add_argument("base")
    w.add_argument("challenger")
    w.add_argument("--seeds", default="13,14,15")
    w.add_argument("--sets", default="0")
    lo = sub.add_parser("lomo")
    lo.add_argument("--copies", default="0,1,2")
    a = ap.parse_args(argv)
    if a.cmd == "mix":
        res = mix()
        (_out() / "mix.json").write_text(json.dumps(res, indent=1))
    elif a.cmd == "octlike":
        fit_octlike(_ints(a.copies), _ints(a.seeds), _ints(a.sets))
    elif a.cmd == "watch":
        res = watch_classes(a.base, a.challenger, _ints(a.seeds), _ints(a.sets))
        (_out() / f"watch_{a.challenger}.json").write_text(json.dumps(res, indent=1))
    else:
        res = lomo(_ints(a.copies))
        (_out() / "lomo.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
