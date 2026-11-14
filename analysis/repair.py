"""Device-order class repair (``ringtrace.repair``) on top of the best file, under the full bar.

    python analysis/repair.py lomo       # dense months, the stored counter, nested shape sets
    python analysis/repair.py alone      # each train month's fraud phones decoded on their own
    python analysis/repair.py octlike    # thinned September with the counter rebuilt on it
    python analysis/repair.py build      # October files repair-conservative / repair-full

Evidence the repair reads, per row:

* the counter: fraud-phase block (``ctr_block == 2``), the segment class and its posterior;
* anchors: a fraud-phase row whose own label-free anchor (``anchors.row_anchor``) names the
  class the counter gives it at posterior >= 0.80 (a counter anchor), or a template row whose
  segment is one of the template's pair at posterior >= 0.80, with its class handed to the other
  fraud-phase rows of its template event (template anchors);
* shapes: template, else first family shape, else first one-class mark.

``octlike`` rebuilds the counter on each thinned frame: the block of every phone is read from the
unthinned number line (a block is a range of numbers, which thinning other rows does not move;
October's own block finder works on its dense legit block), and the segments, posteriors and
anchors come from the kept September rows alone, decoded as their own fraud phase, as October is.
The base calls are the best recipe's, decoded as ``ringtrace.final.Bench.octlike`` does, with the
anchored template rule reading either the held counter (``held``) or the rebuilt one
(``rebuilt``). October rows are only ever counted.

Reports go to ``$RINGTRACE_OUT/repair/<command>.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import paths
from ringtrace import repair as rp
from ringtrace.classes import CLASSES, N_CLASSES

MONTHS = (7, 8, 9)
SEEDS = (13, 14, 15)
SETS = (0, 1, 2)
DRAWS = 2000
RECIPE = "recipes/final-half.json"
BEST_FILE = "blend-tabmF-bq.csv"
SHAPE_PREFIXES = ("bot", "drn", "mf", "col", "rng", "rep")
MARKS = ("mrk_p2p_category", "mrk_qr_p2p_nocat", "mrk_bal102", "mrk_step025", "mrk_step035")
TARGETS = ("given", "cleaned", "gen")
STRICT = False  # ``--strict``: anchors only on phones whose shapes allow one class


def out_dir() -> Path:
    d = paths.out_dir() / "repair"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(name: str, rep: dict) -> None:
    name = f"{name}-strict" if STRICT else name
    (out_dir() / f"{name}.json").write_text(json.dumps(rep, indent=1, default=_default))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def macro(y: np.ndarray, p: np.ndarray) -> float:
    from ringtrace.scoring import macro_f1

    return float(macro_f1(y, p))


# ---------------------------------------------------------------------------------------------
# evidence


def evidence(led: pl.DataFrame, anchor_row: np.ndarray | None = None) -> dict:
    """Label-free columns of ``led`` (normal ledger or a frame) that the repair reads."""
    from ringtrace.evidence import anchors
    from ringtrace.rules import label_free_events

    cols = anchors.family_columns(led, [f"{p}_shape" for p in SHAPE_PREFIXES] + list(MARKS))
    tmpl, tev = anchors.templates(led)
    shapes = [cols[f"{p}_shape"].fill_null(0).to_numpy() for p in SHAPE_PREFIXES]
    marks = [cols[m].fill_null(0).to_numpy() for m in MARKS]
    return {
        "tmpl": np.asarray(tmpl, dtype=np.int64),
        "tmpl_event": np.asarray(tev, dtype=np.int64),
        "anchor": (anchors.row_anchor(led) if anchor_row is None else anchor_row).astype(np.int64),
        "shape": rp.shape_key(tmpl, shapes, marks),
        "events": label_free_events(led),
        "device": led["device"].fill_null(-1).to_numpy().astype(np.int64),
        "split": led["is_test"].to_numpy().astype(np.int64),
    }


def anchor_rows(ev: dict, block: np.ndarray, seg: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Per row the anchor class (0 none): counter anchors, then template anchors."""
    from ringtrace.evidence.anchors import TEMPLATE_PAIR

    fb = block == 2
    sure = fb & (conf >= rp.MIN_CONF) & (seg > 0)
    out = np.where(sure & (ev["anchor"] == seg), seg, 0)
    tmpl, tev = ev["tmpl"], ev["tmpl_event"]
    direct = np.zeros(len(seg), dtype=bool)
    for t, pair in TEMPLATE_PAIR.items():
        direct |= sure & (tmpl == t) & np.isin(seg, pair)
    out = np.where(direct & (out == 0), seg, out)
    # hand an anchored template event's class to its other fraud-phase rows (majority class)
    key = tmpl * 10**9 + np.maximum(tev, 0)
    if direct.any():
        f = pl.DataFrame({"k": key[direct], "c": seg[direct]})
        maj = (
            f.group_by("k", "c")
            .agg(pl.len().alias("n"))
            .sort(["k", "n", "c"], descending=[False, True, False])
            .unique("k", keep="first", maintain_order=True)
        )
        lut = dict(zip(maj["k"].to_list(), maj["c"].to_list(), strict=True))
        hand = fb & (out == 0) & (tmpl > 0) & (tev >= 0)
        idx = np.flatnonzero(hand)
        cls = np.array([lut.get(int(k), 0) for k in key[idx]], dtype=np.int64)
        out[idx] = cls
    return out


def feasible(
    ev: dict, block: np.ndarray, seg: np.ndarray, conf: np.ndarray, table: dict, rows=None
) -> tuple[np.ndarray, dict]:
    """Per-row feasible mask (ALL off the fraud phase) for the fraud phases of ``rows``."""
    n = len(block)
    rows = np.ones(n, dtype=bool) if rows is None else rows
    anc = anchor_rows(ev, block, seg, conf)
    smask = rp.shape_masks(ev["shape"], table)
    if STRICT:
        # single-class anchors only: a phone whose own shapes allow two classes (a look-alike)
        # names a pair, not a class
        phone = ev["split"] * 10**7 + ev["device"]
        anc = np.where(rp.popcount(rp.group_and(phone, smask)) == 1, anc, 0)
    out = np.full(n, rp.ALL, dtype=np.int64)
    stats = {}
    for s in np.unique(ev["split"][rows]):
        ph = np.flatnonzero(rows & (block == 2) & (ev["split"] == s))
        if len(ph) == 0:
            continue
        f, st = rp.phase_masks(ev["device"][ph], anc[ph], smask[ph])
        out[ph] = f
        stats[int(s)] = st
    return out, stats


def groups(ev: dict) -> np.ndarray:
    phone = 10**12 + ev["split"] * 10**7 + ev["device"]
    return np.where(ev["events"] >= 0, ev["events"], phone)


def decision(proba: np.ndarray, calls: np.ndarray, period: np.ndarray, rows: np.ndarray):
    """Decoder values per row, each period on its own (``rows``: the rows to fill)."""
    val = np.zeros((len(calls), N_CLASSES))
    for p in np.unique(period[rows]):
        r = np.flatnonzero(rows & (period == p))
        val[r] = rp.decision_values(proba[r], calls[r])
    return val


def move_table(truth_by: dict, old: np.ndarray, new: np.ndarray, base_all: dict) -> dict:
    """Moves of one month or cell: pair counts, and per target the share of moved rows that are
    the new class, the old class, other fraud, legit; the new class's break-even (F1 / 2)."""
    ch = old != new
    rep = {"rows": int(ch.sum())}
    pairs = {}
    for a, b in zip(old[ch], new[ch], strict=True):
        k = f"{CLASSES[a]} -> {CLASSES[b]}"
        pairs[k] = pairs.get(k, 0) + 1
    rep["pairs"] = pairs
    if truth_by:
        for t, y in truth_by.items():
            yc = y[ch]
            r = {
                "new": int((yc == new[ch]).sum()),
                "old": int((yc == old[ch]).sum()),
                "other_fraud": int(((yc > 0) & (yc != new[ch]) & (yc != old[ch])).sum()),
                "legit": int((yc == 0).sum()),
            }
            be = {}
            for b in np.unique(new[ch]):
                be[CLASSES[b]] = rp.break_even(base_all[t][0], base_all[t][1], int(b))
            r["break_even_new"] = be
            rep[t] = r
    return rep


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


# ---------------------------------------------------------------------------------------------
# dense months


def _prep() -> dict:
    z = np.load(paths.out_dir() / "closelook" / "prep.npz")
    return {k: z[k] for k in z.files}


def _base_ev() -> tuple[dict, dict]:
    from ringtrace import store

    store.use_frame(None)
    led = store.ledger()
    d = _prep()
    if len(d["is_test"]) != led.height:
        raise ValueError("prep.npz does not match the ledger")
    ev = evidence(led, d["anchor"])
    if not np.array_equal(ev["tmpl"], d["tmpl"]):
        raise ValueError("templates differ from prep.npz")
    return d, ev


def _full(d: dict, name: str) -> np.ndarray:
    out = np.full(len(d["is_test"]), -1, dtype=np.int64)
    out[~d["is_test"]] = d[name]
    return out


def _lomo(d, ev, block, seg, conf, tag: str) -> dict:
    tr = ~d["is_test"]
    per = d["period"]
    y = {t: _full(d, t) for t in TARGETS}
    calls = d["calls"]
    grp = groups(ev)
    val = decision(d["proba"], calls, per, tr)
    rep = {"months": {}}
    full = {}
    for v in rp.VARIANTS:
        rep[v] = {"change": {t: [] for t in TARGETS}, "moves": {}}
        full[v] = calls.copy()
    for m in MONTHS:
        fit = tr & (per != m)
        table = rp.shape_table(ev["shape"], y["given"], fit)
        feas, st = feasible(ev, block, seg, conf, table, rows=tr)
        rows = np.flatnonzero(tr & (per == m))
        rep["months"][m] = {"phase": st}
        for v in rp.VARIANTS:
            new, moved = rp.retype(calls[rows], feas[rows], grp[rows], val[rows], v)
            full[v][rows] = new
            base_all = {t: (y[t][rows], calls[rows]) for t in TARGETS}
            for t in TARGETS:
                rep[v]["change"][t].append(macro(y[t][rows], new) - macro(y[t][rows], calls[rows]))
            rep[v]["moves"][m] = move_table(
                {t: y[t][rows] for t in TARGETS}, calls[rows], new, base_all
            )
    boot = None
    for v in rp.VARIANTS:
        if (full[v][tr] != calls[tr]).any():
            from ringtrace.pairboot import PairedBoot

            if boot is None:
                truth = {t: d[t] for t in TARGETS}
                boot = PairedBoot(d["groups"], truth, calls[tr], strata=per[tr], draws=DRAWS)
            rep[v]["bootstrap"] = boot.compare(full[v][tr])
        for t in TARGETS:
            k = sum(rep[v]["moves"][m][t]["new"] for m in MONTHS)
            o = sum(rep[v]["moves"][m][t]["old"] for m in MONTHS)
            n = sum(rep[v]["moves"][m]["rows"] for m in MONTHS)
            rep[v][f"{t}_precision"] = {"new": k, "old": o, "rows": n, "ci": wilson(k, n)}
        print(
            f"[repair] {tag} {v}: rows "
            + " / ".join(str(rep[v]["moves"][m]["rows"]) for m in MONTHS)
            + " | "
            + " | ".join(
                f"{t} " + " / ".join(f"{x:+.5f}" for x in rep[v]["change"][t]) for t in TARGETS
            )
            + f" | cleaned new/old {rep[v]['cleaned_precision']['new']}/"
            f"{rep[v]['cleaned_precision']['old']}"
            + (f" | bootstrap {rep[v]['bootstrap']}" if "bootstrap" in rep[v] else "")
        )
    return rep


def cmd_lomo() -> dict:
    d, ev = _base_ev()
    rep = _lomo(d, ev, d["ctr_block"], d["ctr_seg"], d["ctr_conf"], "lomo")
    save("lomo", rep)
    return rep


def phase_counter(
    device: np.ndarray, anchor: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Segment class (1..11) and posterior of ``rows`` (fraud-phase rows of one phase), decoded
    as a fraud phase of their own (``ringtrace.evidence.phones.decode_phase``)."""
    from ringtrace.evidence import phones

    n = len(device)
    seg = np.zeros(n, dtype=np.int64)
    conf = np.full(n, -1.0)
    if len(rows) == 0:
        return seg, conf
    nums, pinv = np.unique(device[rows], return_inverse=True)
    counts = np.zeros((len(nums), phones.K + 1))
    np.add.at(counts, (pinv, anchor[rows]), 1.0)
    lo, hi = int(nums.min()), int(nums.max())
    used = np.zeros(hi - lo + 1, dtype=bool)
    used[nums - lo] = True
    blocks = phones.Blocks(f0=lo, used=used, base=lo)
    ph = phones.decode_phase(nums, counts, blocks, lo, hi)
    seg[rows] = ph.seg[pinv] + 1
    conf[rows] = ph.conf[pinv]
    return seg, conf


def cmd_alone() -> dict:
    """Dense months, with each train month's fraud-phase rows decoded as their own phase."""
    d, ev = _base_ev()
    block = d["ctr_block"]
    seg = d["ctr_seg"].copy()
    conf = d["ctr_conf"].copy()
    for m in MONTHS:
        rows = np.flatnonzero(~d["is_test"] & (d["period"] == m) & (block == 2))
        s, c = phase_counter(ev["device"], ev["anchor"], rows)
        seg[rows], conf[rows] = s[rows], c[rows]
    tr = ~d["is_test"] & (block == 2)
    agree = float((seg[tr] == d["ctr_seg"][tr]).mean())
    print(f"[repair] month-alone segments agree with the joint phase on {agree:.4f} of rows")
    # the repair reads one phase per month: split ids per month
    ev2 = dict(ev)
    ev2["split"] = np.where(d["is_test"], 1, 10 + d["period"]).astype(np.int64)
    rep = _lomo(d, ev2, block, seg, conf, "alone")
    rep["segment_agreement"] = agree
    save("alone", rep)
    return rep


# ---------------------------------------------------------------------------------------------
# October-like


def _bench_recipe():
    from ringtrace.final import load_recipe, static_weights

    r = load_recipe(RECIPE)
    for m in r["members"]:
        if m["run"] == "final-tabm":
            m["run"] = "tabm"
    return r, static_weights(r)


def _decode_cell(frame, led, p: np.ndarray, recipe: dict) -> np.ndarray:
    """Fixed point on September, the recipe's rules and base-rate rules, reading ``led``."""
    from ringtrace import rules as rulebook
    from ringtrace.decode import fixed_point

    hold = np.flatnonzero(frame.hold)
    pred = np.zeros(led.height, dtype=np.int64)
    pred[hold] = fixed_point(p, **recipe["decoder"])
    full = np.zeros((led.height, N_CLASSES))
    full[hold] = p
    rules = list(recipe["rules"]) + [
        {k: v for k, v in x.items() if k != "max_october_rows"} for x in recipe["base_rate_rules"]
    ]
    pred = rulebook.apply(rules, pred, full, led, log=lambda *_: None)
    return pred[hold]


def cmd_octlike(seeds=SEEDS, sets=SETS) -> dict:
    from ringtrace import octlike as ol
    from ringtrace import store
    from ringtrace.final import mix
    from ringtrace.pairboot import PairedBoot

    recipe, w = _bench_recipe()
    roles = [m["role"] for m in recipe["members"]]
    cells_dir = paths.out_dir() / "lens-cells"
    rep = {"cells": {}, "phase": {}}
    pooled: dict = {}
    t0 = time.time()
    try:
        for seed in seeds:
            frame = ol.Frame(seed)
            led = frame.led
            hold = frame.hold
            ev = evidence(led)
            held = store.family("phones", verbose=False)
            block = held["ctr_block"].fill_null(0).to_numpy().astype(np.int64)
            hseg = held["ctr_seg"].cast(pl.Float64).fill_null(-1).to_numpy().astype(np.int64) + 1
            hconf = held["ctr_conf"].cast(pl.Float64).fill_null(-1).to_numpy()
            ph_rows = np.flatnonzero(hold & (block == 2))
            seg, conf = phase_counter(ev["device"], ev["anchor"], ph_rows)
            # off September the held values stay (never scored)
            seg = np.where(hold, seg, hseg)
            conf = np.where(hold, conf, hconf)
            ev["split"] = np.where(hold, 2, ev["split"])
            agree = float((seg[ph_rows] == hseg[ph_rows]).mean())
            led_rb = led.with_columns(
                pl.Series("ctr_block", block.astype(np.int8)),
                pl.Series("ctr_seg", np.where(seg > 0, seg - 1.0, np.nan)),
                pl.Series("ctr_conf", np.where(conf >= 0, conf, np.nan)),
                held["ctr_dev_rows"],
            )
            led_held = led.with_columns(
                held["ctr_block"], held["ctr_seg"], held["ctr_conf"], held["ctr_dev_rows"]
            )
            y = {t: frame.targets[t] for t in TARGETS}
            fit = frame.fit
            table = rp.shape_table(ev["shape"], y["given"], fit)
            feas, st = feasible(ev, block, seg, conf, table, rows=hold)
            rep["phase"][seed] = {"stats": st, "segment_agreement_with_held": agree}
            print(f"[repair] s{seed}: phase {st}, segments agree with held {agree:.3f}")
            grp = groups(ev)
            units = frame.unit[hold]
            truth = {t: y[t][hold] for t in TARGETS}
            hidx = np.flatnonzero(hold)
            for r in sets:
                probs, rows = [], None
                for m in recipe["members"]:
                    f = np.load(cells_dir / f"{m['cells']}_s{seed}_r{r}.npz")
                    rows = f["rows"] if rows is None else rows
                    if not np.array_equal(rows, f["rows"]):
                        raise ValueError(f"{m['cells']}: rows differ in s{seed} r{r}")
                    probs.append(f["prob"].astype(np.float64))
                if not np.array_equal(rows, frame.stacked[hold]):
                    raise ValueError(f"cells s{seed} r{r} do not match the frame")
                p = mix(probs, w, roles)
                for base_name, bled in (("held", led_held), ("rebuilt", led_rb)):
                    base = _decode_cell(frame, bled, p, recipe)
                    boot = PairedBoot(units, truth, base, draws=DRAWS, seed=seed * 100 + r)
                    val = rp.decision_values(p, base)
                    for v in rp.VARIANTS:
                        new, moved = rp.retype(base, feas[hidx], grp[hidx], val, v)
                        bs = boot.compare(new, keep_draws=True)
                        key = f"{base_name}|{v}"
                        cell = {
                            "moves": move_table(
                                truth, base, new, {t: (truth[t], base) for t in TARGETS}
                            )
                        }
                        for t in TARGETS:
                            pooled.setdefault(key, {}).setdefault(t, []).append(bs[t].pop("draws"))
                            cell[t] = bs[t]
                        rep["cells"].setdefault(key, {})[f"s{seed}_r{r}"] = cell
                print(
                    f"[repair] s{seed} r{r}: "
                    + " | ".join(
                        f"{k} rows {c[f's{seed}_r{r}']['moves']['rows']} cleaned "
                        f"{c[f's{seed}_r{r}']['cleaned']['delta']:+.5f}"
                        for k, c in rep["cells"].items()
                    )
                    + f" ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    finally:
        store.use_frame(None)
    rep["summary"] = {}
    for key, cells in rep["cells"].items():
        s = {"rows": int(sum(c["moves"]["rows"] for c in cells.values()))}
        for t in TARGETS:
            draws = np.mean(pooled[key][t], axis=0)
            s[t] = {
                "mean": float(np.mean([c[t]["delta"] for c in cells.values()])),
                "per_seed": {
                    sd: float(
                        np.mean([c[t]["delta"] for k, c in cells.items() if k[1:3] == str(sd)])
                    )
                    for sd in seeds
                },
                "lower_bound": float(np.percentile(draws, 2.5)),
                "upper_bound": float(np.percentile(draws, 97.5)),
            }
            k = sum(c["moves"][t]["new"] for c in cells.values())
            o = sum(c["moves"][t]["old"] for c in cells.values())
            s[t]["moved_new"] = k
            s[t]["moved_old"] = o
            s[t]["precision_ci"] = wilson(k, s["rows"])
        pairs: dict = {}
        for c in cells.values():
            for a, b in c["moves"]["pairs"].items():
                pairs[a] = pairs.get(a, 0) + b
        s["pairs"] = pairs
        rep["summary"][key] = s
        print(
            f"[repair] {key}: rows {s['rows']} | "
            + " | ".join(
                f"{t} mean {s[t]['mean']:+.5f} lb {s[t]['lower_bound']:+.5f} "
                f"new/old {s[t]['moved_new']}/{s[t]['moved_old']}"
                for t in TARGETS
            )
        )
    save("octlike", rep)
    return rep


# ---------------------------------------------------------------------------------------------
# October files


def cmd_build() -> dict:
    from ringtrace.blend import write_submission
    from ringtrace.store import ledger

    d, ev = _base_ev()
    te = d["is_test"]
    tr = ~te
    ids = ledger()["txn_id"].to_numpy()
    best = pl.read_csv(paths.out_dir() / "candidates" / BEST_FILE, infer_schema=False)
    pos = {t: i for i, t in enumerate(best["id"].to_list())}
    code = {c: i for i, c in enumerate(CLASSES)}
    best_calls = np.array([code[x] for x in best["label"].to_list()])[[pos[t] for t in ids[te]]]
    if not np.array_equal(best_calls, d["calls"][te]):
        raise ValueError("prep.npz October calls differ from the best file")
    table = rp.shape_table(ev["shape"], _full(d, "given"), tr)
    feas, st = feasible(ev, d["ctr_block"], d["ctr_seg"], d["ctr_conf"], table, rows=te)
    calls = d["calls"]
    val = decision(d["proba"], calls, d["period"], te)
    tix = np.flatnonzero(te)
    rep = {"phase": st}
    for v in rp.VARIANTS:
        new, moved = rp.retype(calls[tix], feas[tix], groups(ev)[tix], val[tix], v)
        stem = f"repair-strict-{v}" if STRICT else f"repair-{v}"
        path = paths.out_dir() / "candidates" / f"{stem}.csv"
        sub = write_submission(path, ids[te], new, paths.data_dir() / "sample_submission.csv")
        r = move_table({}, calls[tix], new, {})
        r.update({"file": str(path), "sha256": sub["sha256"], "problems": sub["problems"]})
        rep[v] = r
        print(f"[repair] October {v}: {json.dumps(r)}")
    save("build", rep)
    return rep


COMMANDS = {"lomo": cmd_lomo, "alone": cmd_alone, "octlike": cmd_octlike, "build": cmd_build}


def main(argv=None) -> None:
    from ringtrace import threads

    threads.pin()
    global STRICT
    args = list(argv if argv is not None else sys.argv[1:])
    STRICT = "--strict" in args
    names = [a for a in args if a != "--strict"] or ["lomo"]
    t0 = time.time()
    for name in names:
        COMMANDS[name]()
        print(f"[repair] {name} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
