"""Numbers behind the October border and event rules (``docs/findings/october-borders.md`` and
``october-events.md``). Aggregates only; test rows are counted, never listed.

    python analysis/october_rules.py replay [blocks]        # wrong-zone phones per thinned block
    python analysis/october_rules.py calibration [blocks]   # said against happened, per border
    python analysis/october_rules.py bursts                 # fresh-SIM bursts, train and October
    python analysis/october_rules.py rings                  # merchant ring sizes, train and October
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
import polars as pl

from ringtrace import silver, store
from ringtrace.bordershift import gap_mass
from ringtrace.evidence import anchors, borders, phones, refined
from ringtrace.evidence.phones import K

MIX_SD = (0.0, 0.25, 0.4)
BINS = np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98])
_PHASE: dict = {}


def phase_tables() -> dict:
    """The train phase on its number line (zone of every number from a monotone fit to the cleaned
    labels) and the size of the test phase."""
    if _PHASE:
        return _PHASE
    led = store.ledger()
    device = led["device"].to_numpy().astype(np.int64)
    split = led["is_test"].to_numpy().astype(np.int64)
    anchor = np.asarray(anchors.row_anchor(led), dtype=np.int64)
    blocks = phones.find_blocks(device, split)
    block = phones.block_codes(device, split, blocks)
    label = silver.load(led, "label").astype(np.int64)
    lo, edge, _ = blocks.ranges[0]
    rows = np.flatnonzero((block == 2) & (split == 0))
    nums, pinv = np.unique(device[rows], return_inverse=True)
    counts = np.zeros((len(nums), K + 1))
    np.add.at(counts, (pinv, anchor[rows]), 1.0)
    lab = np.full(len(nums), -1)
    for i in range(len(nums)):
        v = label[rows][pinv == i]
        v = v[v > 0]
        if len(v):
            lab[i] = np.bincount(v).argmax() - 1
    match = np.zeros((len(nums), K))
    match[np.flatnonzero(lab >= 0), lab[lab >= 0]] = 5.0
    path = phones.viterbi(match)
    n = int(edge - lo + 1)
    at = nums - lo
    zone = np.zeros(n, dtype=np.int64)
    for i in range(len(at)):
        zone[at[i] :] = path[i]
    used = np.zeros(n, dtype=bool)
    used[at] = True
    phone_of = np.full(n, -1)
    phone_of[at] = np.arange(len(at))
    t_lo, t_edge, _ = blocks.ranges[1]
    _PHASE.update(
        zone=zone, used=used, phone_of=phone_of, counts=counts, keep=(t_edge - t_lo + 1) / n
    )
    return _PHASE


def thinned(seed: int, mix_sd: float):
    """A test-sized copy of the train phase: numbers dropped and closed up, class mix shifted."""
    p = phase_tables()
    rng = np.random.default_rng(10_000 + seed)
    mix = np.exp(rng.normal(0, mix_sd, K)) if mix_sd > 0 else np.ones(K)
    keep = np.flatnonzero(rng.random(len(p["zone"])) < np.clip(p["keep"] * mix[p["zone"]], 0, 1))
    keep = keep[np.argmax(p["used"][keep]) :]
    u = p["used"][keep]
    return np.flatnonzero(u), p["counts"][p["phone_of"][keep[u]]], len(keep), p["zone"][keep[u]]


def _replay(args) -> dict:
    nums, counts, n, truth = thinned(*args)
    mode = {"refined": refined.decode_phase(nums, counts, None, 0, 0).post.argmax(axis=1)}
    for sd in (None, 0.3, 0.5, 0.8):
        share = borders.LENGTH_SHARE if sd else None
        post = borders.decode(nums, counts, 0, n - 1, share, sd)["post"]
        mode[f"border sd {sd}"] = post.argmax(axis=1)
    return {k: int((v != truth).sum()) for k, v in mode.items()}


def _calibration(args):
    nums, counts, n, truth = thinned(*args)
    cut = borders.decode(nums, counts, 0, n - 1)["cut"]
    said = [np.cumsum(gap_mass(cut[j], nums))[: len(nums)] for j in range(K - 1)]
    return np.stack(said, axis=1), truth[:, None] >= np.arange(1, K)[None, :]


def replay(blocks: int) -> None:
    phase_tables()  # built once here; the pool's workers inherit it
    for sd in MIX_SD:
        with Pool() as pool:
            got = pool.map(_replay, [(s, sd) for s in range(blocks)], chunksize=2)
        means = {k: np.mean([g[k] for g in got]) for k in got[0]}
        print(f"class-mix log-sd {sd}: " + ", ".join(f"{k} {v:.2f}" for k, v in means.items()))


def calibration(blocks: int) -> None:
    phase_tables()
    for sd in MIX_SD[:2]:
        with Pool() as pool:
            got = pool.map(_calibration, [(s, sd) for s in range(blocks)], chunksize=2)
        said = np.concatenate([g[0] for g in got])
        true = np.concatenate([g[1] for g in got])
        undecided = (said > BINS[0]) & (said < BINS[-1])
        b = np.digitize(said[undecided], BINS) - 1
        line = "  ".join(
            f"{said[undecided][b == i].mean():.2f}->{true[undecided][b == i].mean():.2f}"
            for i in range(len(BINS) - 1)
        )
        print(f"class-mix log-sd {sd}: undecided pairs per block {undecided.sum() / blocks:.1f}")
        print("  pooled said->happened: " + line)
        for j in range(K - 1):
            s, t = said[:, j], true[:, j]
            lo = (s > 0.02) & (s < 0.35)
            hi = (s >= 0.65) & (s < 0.98)
            print(
                f"  border {j}|{j + 1}: low {s[lo].mean():.2f}->{t[lo].mean():.2f} ({lo.sum()})"
                f"  high {s[hi].mean():.2f}->{t[hi].mean():.2f} ({hi.sum()})"
            )


def bursts() -> None:
    led = store.ledger()
    tmpl, event = anchors.templates(led)
    block = anchors.family_columns(led, ["ctr_block"])["ctr_block"].fill_null(0).to_numpy()
    label = silver.load(led, "label").astype(np.int64)
    frame = pl.DataFrame(
        {
            "ev": event,
            "burst": np.asarray(tmpl) == 3,
            "phase": block == 2,
            "sim": led["sim_age_hours"].to_numpy().astype(np.float64),
            "test": led["is_test"].to_numpy(),
            "bot": label == 2,
            "mule": label == 1,
        }
    ).filter(pl.col("burst"))
    ev = frame.group_by("ev").agg(
        pl.len().alias("rows"),
        pl.col("phase").any(),
        pl.col("sim").min(),
        pl.col("test").first(),
        (pl.col("bot").sum() > pl.col("mule").sum()).alias("is_bot"),
    )
    for name, part in (("train", ev.filter(~pl.col("test"))), ("October", ev.filter("test"))):
        free = part.filter(~pl.col("phase"))
        print(f"{name}: bursts {part.height}, without a phase phone {free.height}")
        for rows, sim in ((20, 50), (13, 50), (12, 50), (20, 30), (20, 100)):
            hit = free.filter((pl.col("rows") >= rows) & (pl.col("sim") < sim))
            bots = f", velocity_bot {int(hit['is_bot'].sum())}" if name == "train" else ""
            print(
                f"  rows >= {rows}, SIM < {sim} h: {hit.height} events, "
                f"{int(hit['rows'].sum())} rows{bots}"
            )


def rings() -> None:
    led = store.ledger()
    cols = anchors.family_columns(led, ["rng_shape", "rng_event"])
    label = silver.load(led, "label").astype(np.int64)
    frame = pl.DataFrame(
        {
            "ev": cols["rng_event"].fill_null(-1).to_numpy(),
            "test": led["is_test"].to_numpy(),
            "refund": label == 8,
            "collusion": label == 11,
        }
    ).filter(pl.col("ev") >= 0)
    ev = frame.group_by("ev").agg(
        pl.len().alias("rows"),
        pl.col("test").first(),
        pl.col("refund").sum(),
        pl.col("collusion").sum(),
    )
    for lo, hi in ((4, 7), (8, 8), (9, 9), (10, 19), (20, 99)):
        size = ev.filter((pl.col("rows") >= lo) & (pl.col("rows") <= hi))
        tr = size.filter(~pl.col("test"))
        print(
            f"size {lo}-{hi}: train {tr.height} events "
            f"(refund {int((tr['refund'] > tr['collusion']).sum())}, "
            f"collusion {int((tr['collusion'] > tr['refund']).sum())}), "
            f"October {size.filter('test').height} events"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "replay"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    if command == "replay":
        replay(count)
    elif command == "calibration":
        calibration(count)
    elif command == "bursts":
        bursts()
    elif command == "rings":
        rings()
    else:
        sys.exit(__doc__)
