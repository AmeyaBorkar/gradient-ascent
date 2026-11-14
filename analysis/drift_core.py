"""Pure helpers for the October drift scan (no file access; tested on synthetic data).

The scored month is compared with the three train months on rates per 100,000 rows, never on raw
counts: October has about half the rows of a train month. A Poisson z uses the mean of the three
train monthly rates as the baseline (the same baseline as the test/train ratio). A size histogram
instead uses the pooled train counts scaled by ``test_rows / train_rows``, so a pure volume shift
is a departure and the test total is not a constraint.

A test group of 1 to 4 rows is reported as ``<5`` and its rate and ratio are omitted. Zero stays
zero: an empty cell does not identify a test row. The z-score is still computed from the exact
count; it is the decision statistic and can be inverted to that count.

Near-miss bands are rows that meet a generator law except for one threshold, relaxed on the side
the law rejects. They are computed from ledger columns only. A band does not rebuild connected
components. A hop-ratio near-miss is a P2P of at least 2,000 whose receiver's first later send
within 24 h sits just outside the ratio band (the law's own rows, at least 5,000 within 6 h inside
the band, are excluded). A hop-delay near-miss is that first send when the ratio is in band and the
delay is past 6 h, on a P2P of at least 5,000. A drain near-miss is the sender's immediate next
row (the drain law also links the next-but-one row).

What a departure can show is a change in volume, in the size mix, or in traffic sitting just
outside a hard threshold. It does not by itself prove the generator changed: month length, the
half-density background, the festival amount scale and ordinary Poisson noise move rates too, and
the z-score treats rows as independent even though fraud arrives in events. A law can also move
and stay inside its acceptance window; this scan does not see that.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.stats import chi2

from ringtrace.evidence import bursts, drains, flows, merchant, repeats, requests

PER = 100_000
MIN_GROUP = 5
TRAIN = (7, 8, 9)
TEST = 10
MONTHS = (7, 8, 9, 10)
PILEUP_Z = 2.0

# Bands on the far side of a hard threshold. Widths are part of the scan, not of the laws.
HOP_RATIO_FLOOR = 0.90
HOP_RATIO_CEIL = 1.0
HOP_DELAY_BAND_S = 24 * 3600
# The law's own floor is 5,000 within 6 h. The pass-through edge is the receiver's first later
# send within 24 h after a P2P of at least 2,000, including amounts the 5,000 rule drops.
HOP_NEAR_MIN = 2000.0
SENDER_BAND = 2
BOT_GAP_BAND_S = 30 * 60
BOT_SHORT_BAND = 2
STEP_BAND = 0.02
NEAR_OFFSETS = (2, 5, 20, 25, 100)
NEAR_BAITS = (3, 20, 50)
BAIT_GAP_BAND_S = 30 * 60
ANSWER_BAND_S = 24 * 3600
REPEAT_GAP_BAND_S = 14 * repeats.DAY
EVENING_NEAR = (18, 19)
RING_CAP_RUPEES = 2 * max(c // 100 for c in merchant.RING_CENTS)

# Low side of a cut that splits two classes. Bins ``c`` and ``c + 1`` sit next to the cut.
CLASS_CUTS: dict[tuple[str, str], tuple[int, ...]] = {
    ("rng", "size"): (3, 8),  # 1-3 qr_swap, 4-8 refund_abuse, 9+ merchant_collusion
    ("mf", "fan_senders"): (7,),  # 7 layering fan-in, 8 mule burst
    ("mf", "bu_senders"): (7,),
}

TIE = -2  # event whose structured labels have no unique mode
UNKNOWN = -1  # test row, or a train row with no structured label


def per_100k(count: float, rows: int) -> float | None:
    """``count`` per 100,000 rows. None when the month has no rows."""
    if rows <= 0:
        return None
    return PER * float(count) / float(rows)


def shown_count(n: int, *, test: bool, floor: int = MIN_GROUP) -> int | str:
    """Test groups of 1 .. floor-1 are ``<5``. Zero and train groups stay numeric."""
    if test and 0 < int(n) < floor:
        return "<5"
    return int(n)


def month_cell(n: int, rows: int, *, test: bool) -> dict:
    """Count and rate for one month. A hidden test group has no rate."""
    hidden = test and 0 < int(n) < MIN_GROUP
    return {
        "n": shown_count(n, test=test),
        "rate": None if hidden else per_100k(n, rows),
    }


def poisson_z(test_count: int, test_rows: int, train_rate_per_row: float) -> float | None:
    """``(k - mu) / sqrt(mu)`` with ``mu = train_rate_per_row * test_rows``.

    None when the test month has no rows, or when the baseline is 0 (mu would be 0). A positive
    baseline of 0 with a positive test count is reported by the caller as ``no_baseline``.
    """
    if test_rows <= 0:
        return None
    mu = float(train_rate_per_row) * float(test_rows)
    if mu <= 0:
        return None
    return (float(test_count) - mu) / math.sqrt(mu)


def month_comparison(counts: dict[int, int], rows_by_month: dict[int, int]) -> dict:
    """Rates per 100k, test/mean-train ratio, and a Poisson z for one non-negative count series.

    The train baseline is the unweighted mean of the three monthly rates (count / rows), not the
    pooled rate. The ratio and the published test rate are omitted when the test count is a hidden
    group; the z-score still uses the exact count.
    """
    counts = {m: int(counts.get(m, 0)) for m in MONTHS}
    per_row: list[float] = []
    cells = []
    for m in MONTHS:
        n = counts[m]
        rows = int(rows_by_month[m])
        cell = month_cell(n, rows, test=(m == TEST))
        cell["month"] = m
        cells.append(cell)
        if m in TRAIN and rows > 0:
            per_row.append(n / rows)
    train_rate_row = float(np.mean(per_row)) if per_row else 0.0
    test_n = counts[TEST]
    test_rows = int(rows_by_month[TEST])
    hidden = 0 < test_n < MIN_GROUP
    z = poisson_z(test_n, test_rows, train_rate_row)
    no_baseline = train_rate_row <= 0 and test_n > 0
    ratio = None
    if not hidden and not no_baseline and test_rows > 0 and train_rate_row > 0:
        ratio = (test_n / test_rows) / train_rate_row
    test_rate = None if hidden else per_100k(test_n, test_rows)
    return {
        "months": cells,
        "train_rate": train_rate_row * PER,
        "test_rate": test_rate,
        "ratio": None if ratio is None else float(ratio),
        "z": None if z is None else float(z),
        "no_baseline": bool(no_baseline),
    }


def chi_square_scaled(
    train_counts: np.ndarray, test_counts: np.ndarray, train_rows: int, test_rows: int
) -> dict:
    """Pearson chi-square of test bin counts against pooled train counts scaled by rows.

    ``expected = train_counts * test_rows / train_rows``. Bins with a zero expectation are left
    out of the statistic (a zero denominator) and reported in ``unexpected``. Degrees of freedom
    equal the number of bins with a positive expectation: the test total is not fitted, so a
    volume shift with the same shape is a departure. ``z`` is ``(obs - exp) / sqrt(exp)`` and NaN
    where the expectation is 0.
    """
    train_counts = np.asarray(train_counts, dtype=np.float64)
    test_counts = np.asarray(test_counts, dtype=np.float64)
    if train_rows <= 0:
        raise ValueError("train_rows must be positive")
    expected = train_counts * (float(test_rows) / float(train_rows))
    keep = expected > 0
    z = np.full(train_counts.shape, np.nan)
    unexpected = (~keep) & (test_counts > 0)
    if not keep.any():
        return {
            "chi2": None,
            "df": 0,
            "p": None,
            "expected": expected,
            "z": z,
            "unexpected": unexpected,
        }
    obs = test_counts[keep]
    exp = expected[keep]
    stat = float(np.sum((obs - exp) ** 2 / exp))
    df = int(keep.sum())
    # Divide only on positive expectations. ``np.where`` would evaluate the quotient everywhere
    # and warn on the zero bins.
    np.divide(test_counts - expected, np.sqrt(expected), out=z, where=keep)
    return {
        "chi2": stat,
        "df": df,
        "p": float(chi2.sf(stat, df)),
        "expected": expected,
        "z": z,
        "unexpected": unexpected,
    }


def pileup_flags(
    bins: np.ndarray,
    z: np.ndarray,
    test_counts: np.ndarray,
    expected: np.ndarray,
    cuts: tuple[int, ...] | list[int],
    z_min: float = PILEUP_Z,
) -> np.ndarray:
    """True on a bin next to a class cut where the test month piles up.

    A cut ``c`` is the last size of the lower class, so the bins next to it are ``c`` and
    ``c + 1`` (ring size 8 and 9, sender count 7 and 8). Pile-up means the residual z is at least
    ``z_min``, or the train expectation is 0 and the test count is positive.
    """
    bins = np.asarray(bins)
    z = np.asarray(z, dtype=np.float64)
    test_counts = np.asarray(test_counts)
    expected = np.asarray(expected, dtype=np.float64)
    adjacent = np.zeros(len(bins), dtype=bool)
    for c in cuts:
        adjacent |= (bins == c) | (bins == c + 1)
    excess = np.isfinite(z) & (z >= z_min)
    novel = (expected <= 0) & (test_counts > 0)
    return adjacent & (excess | novel)


def mode_label(codes) -> int:
    """Unique mode of the non-negative codes.

    ``TIE`` when two classes share the mode, otherwise ``UNKNOWN`` when none are known.
    """
    known = [int(c) for c in codes if c is not None and int(c) >= 0]
    if not known:
        return UNKNOWN
    vals, cts = np.unique(np.asarray(known, dtype=np.int64), return_counts=True)
    if int((cts == cts.max()).sum()) > 1:
        return TIE
    return int(vals[int(np.argmax(cts))])


def collapse_events(
    rows: pl.DataFrame, measures: tuple[str, ...] = ()
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collapse event-member rows to one row per event and one row per event and shape.

    ``rows`` has ``event``, ``shape``, ``period``, ``gen`` and the optional ``measures`` (null
    where the measure does not apply). The event period is the minimum in its rows. The event
    class is :func:`mode_label` of ``gen``. ``n_rows`` on the event is its full length; on the
    shape frame it is the number of rows of that shape (a mule burst is one event in two shapes).
    """
    if rows.height == 0:
        events = rows.with_columns(pl.lit(0, dtype=pl.UInt32).alias("n_rows")).head(0)
        shapes = events.select("event", "shape", "period", "gen", "n_rows")
        return events, shapes
    aggs = [
        pl.col("period").min().alias("period"),
        pl.col("gen"),
        pl.len().alias("n_rows"),
    ]
    for m in measures:
        aggs.append(pl.col(m).drop_nulls().first().alias(m))
    keys = rows.group_by("event").agg(aggs)
    gens = [mode_label(g) for g in keys["gen"].to_list()]
    events = keys.drop("gen").with_columns(pl.Series("gen", gens, dtype=pl.Int64))
    shapes = (
        rows.group_by("event", "shape")
        .agg(pl.len().alias("n_rows"))
        .join(events.select("event", "period", "gen"), on="event", how="left")
    )
    return events, shapes


def _by_gen(
    frame: pl.DataFrame, rows_by_month: dict[int, int], names: tuple[str, ...]
) -> list[dict]:
    """Train-only event and row counts by month and generator-made class. Test rows are absent."""
    if frame.height == 0:
        return []
    use = frame.filter(pl.col("period").is_in(list(TRAIN)) & (pl.col("gen") != UNKNOWN))
    if use.height == 0:
        return []
    agg = (
        use.group_by("period", "gen")
        .agg(pl.len().alias("events"), pl.col("n_rows").sum().alias("rows"))
        .sort("period", "gen")
    )
    out = []
    for period, gen, events, nrows in agg.iter_rows():
        gen_i = int(gen)
        if gen_i == TIE:
            name = "tie"
        elif 0 <= gen_i < len(names):
            name = names[gen_i]
        else:
            name = str(gen_i)
        rows = int(rows_by_month[int(period)])
        out.append(
            {
                "month": int(period),
                "gen": gen_i,
                "name": name,
                "events": int(events),
                "event_rate": per_100k(events, rows),
                "rows": int(nrows),
                "row_rate": per_100k(nrows, rows),
            }
        )
    return out


def volume_table(
    frame: pl.DataFrame, rows_by_month: dict[int, int], names: tuple[str, ...]
) -> list[dict]:
    """Per-shape event counts and event-row counts, rated by the month's ledger rows.

    ``frame`` is one row per event (or per event and shape) with ``shape``, ``period``, ``n_rows``
    and ``gen``. A shape value of -1 is the whole family (one row per event).
    """
    if frame.height == 0:
        return []
    counted = frame.group_by("shape", "period").agg(
        pl.len().alias("events"), pl.col("n_rows").sum().alias("rows")
    )
    out = []
    for shape in sorted(frame["shape"].unique().to_list()):
        sub = counted.filter(pl.col("shape") == shape)
        ev = dict.fromkeys(MONTHS, 0)
        rw = dict.fromkeys(MONTHS, 0)
        for period, events, nrows in sub.select("period", "events", "rows").iter_rows():
            ev[int(period)] = int(events)
            rw[int(period)] = int(nrows)
        shaped = frame.filter(pl.col("shape") == shape)
        out.append(
            {
                "shape": int(shape),
                "events": month_comparison(ev, rows_by_month),
                "rows": month_comparison(rw, rows_by_month),
                "by_gen": _by_gen(shaped, rows_by_month, names),
            }
        )
    return out


def call_table(
    calls: np.ndarray, period: np.ndarray, rows_by_month: dict[int, int], names: tuple[str, ...]
) -> list[dict]:
    """One comparison per class code. Codes outside ``names`` are ignored."""
    calls = np.asarray(calls)
    period = np.asarray(period)
    out = []
    for code, name in enumerate(names):
        counts = {}
        for m in MONTHS:
            counts[m] = int(((calls == code) & (period == m)).sum())
        out.append({"code": code, "name": name, **month_comparison(counts, rows_by_month)})
    return out


def size_table(
    values: np.ndarray,
    period: np.ndarray,
    gen: np.ndarray,
    rows_by_month: dict[int, int],
    cuts: tuple[int, ...] | list[int],
    names: tuple[str, ...],
) -> dict:
    """Histogram of one event measure, a row-scaled chi-square, and pile-up flags.

    One observation per event. Train months are pooled for the expectation and also kept per
    month. ``by_gen`` is train only. A test bin of 1 to 4 events is hidden in that month's cell;
    its z and pile-up flag still use the exact count.
    """
    values = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)
    period = np.asarray(period, dtype=np.int64)
    gen = np.asarray(gen, dtype=np.int64)
    cuts_t = tuple(int(c) for c in cuts)
    if len(values) == 0:
        return {"cuts": list(cuts_t), "chi2": None, "df": 0, "p": None, "bins": [], "by_gen": []}
    bins = np.unique(values)
    train_m = np.isin(period, TRAIN)
    train_rows = int(sum(rows_by_month[m] for m in TRAIN))
    test_rows = int(rows_by_month[TEST])
    train_counts = np.array([int(((values == b) & train_m).sum()) for b in bins], dtype=np.float64)
    test_counts = np.array(
        [int(((values == b) & (period == TEST)).sum()) for b in bins], dtype=np.float64
    )
    stats = chi_square_scaled(train_counts, test_counts, train_rows, test_rows)
    flags = pileup_flags(bins, stats["z"], test_counts, stats["expected"], cuts_t)
    bin_rows = []
    for i, b in enumerate(bins):
        in_b = values == b
        counts = {m: int((in_b & (period == m)).sum()) for m in MONTHS}
        cell = month_comparison(counts, rows_by_month)
        z_i = stats["z"][i]
        exp_i = float(stats["expected"][i])
        bin_rows.append(
            {
                "bin": int(b),
                "months": cell["months"],
                "expected": exp_i,
                "z": None if not math.isfinite(float(z_i)) else float(z_i),
                "pileup": bool(flags[i]),
                "novel": bool(stats["unexpected"][i]),
            }
        )
    gen_frame = pl.DataFrame(
        {
            "shape": np.zeros(len(values), dtype=np.int64),
            "period": period,
            "gen": gen,
            "n_rows": np.ones(len(values), dtype=np.int64),
            "bin": values,
        }
    )
    by_gen = []
    train_obs = gen_frame.filter(pl.col("period").is_in(list(TRAIN)) & (pl.col("gen") != UNKNOWN))
    if train_obs.height:
        agg = (
            train_obs.group_by("period", "gen", "bin")
            .agg(pl.len().alias("n"))
            .sort("period", "gen", "bin")
        )
        for period_i, gen_i, bin_i, n in agg.iter_rows():
            gen_i = int(gen_i)
            if gen_i == TIE:
                name = "tie"
            elif 0 <= gen_i < len(names):
                name = names[gen_i]
            else:
                name = str(gen_i)
            rows = int(rows_by_month[int(period_i)])
            by_gen.append(
                {
                    "month": int(period_i),
                    "gen": gen_i,
                    "name": name,
                    "bin": int(bin_i),
                    "n": int(n),
                    "rate": per_100k(n, rows),
                }
            )
    return {
        "cuts": list(cuts_t),
        "chi2": stats["chi2"],
        "df": stats["df"],
        "p": stats["p"],
        "bins": bin_rows,
        "by_gen": by_gen,
    }


def band_table(
    mask: np.ndarray,
    period: np.ndarray,
    calls: np.ndarray,
    gen: np.ndarray | None,
    rows_by_month: dict[int, int],
) -> dict:
    """Per month: rate per 100k, share of rows the calls already mark fraud, and on train the
    share whose structured label is generator-made fraud (class > 0). Shares of a hidden test
    group are omitted."""
    mask = np.asarray(mask, dtype=bool)
    period = np.asarray(period)
    calls = np.asarray(calls)
    if len(mask) != len(period) or len(calls) != len(period):
        raise ValueError("mask, period and calls must share one length")
    if gen is not None and len(gen) != len(period):
        raise ValueError("gen must be one value per row")
    counts: dict[int, int] = {}
    shares: dict[int, tuple[float | None, float | None]] = {}
    gen_arr = None if gen is None else np.asarray(gen)
    for m in MONTHS:
        sel = mask & (period == m)
        n = int(sel.sum())
        counts[m] = n
        hidden = m == TEST and 0 < n < MIN_GROUP
        if n == 0 or hidden:
            shares[m] = (None, None)
            continue
        called = float(np.mean(calls[sel] > 0))
        gen_fraud = None
        if m != TEST and gen_arr is not None:
            gen_fraud = float(np.mean(gen_arr[sel] > 0))
        shares[m] = (called, gen_fraud)
    comp = month_comparison(counts, rows_by_month)
    for cell in comp["months"]:
        called, gen_fraud = shares[cell["month"]]
        cell["called"] = called
        cell["gen_fraud"] = gen_fraud
    return comp


def rank_departures(items: list[dict]) -> list[dict]:
    """Largest |z| first. A test count with no train baseline (z undefined) ranks ahead of them.
    Items with neither a z nor a missing baseline are dropped."""
    keep = [
        it
        for it in items
        if it.get("no_baseline") or (it.get("z") is not None and math.isfinite(it["z"]))
    ]

    def key(it: dict):
        missing = 0 if it.get("no_baseline") else 1
        mag = 0.0 if it.get("z") is None else -abs(float(it["z"]))
        return (missing, mag, str(it.get("name", "")))

    return sorted(keep, key=key)


def departure(kind: str, name: str, comp: dict) -> dict:
    """One ranked row: the comparison's z and its train baseline (mean train rate per 100k)."""
    test_cell = next(c for c in comp["months"] if c["month"] == TEST)
    return {
        "kind": kind,
        "name": name,
        "z": comp["z"],
        "no_baseline": comp["no_baseline"],
        "train_rate": comp["train_rate"],
        "test_rate": comp["test_rate"],
        "ratio": comp["ratio"],
        "test_n": test_cell["n"],
    }


# ---------------------------------------------------------------------------------------------
# near-miss bands


def _mask(n: int, idx) -> pl.Series:
    out = np.zeros(n, dtype=bool)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size:
        out[idx] = True
    return pl.Series("hit", out)


def _indexed(led: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return led.select(pl.int_range(pl.len(), dtype=pl.Int64).alias("i"), *cols)


def _hop_frames(led: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """P2P sources of at least ``HOP_NEAR_MIN``, and every row as a possible onward send.

    Keys are cast the way the hop family casts them, so the as-of join does not depend on the
    ledger's stored integer widths. ``split`` keeps a train row from matching a test row.
    """
    b = _indexed(led, ["is_test", "sender", "receiver", "to_person", "ts", "amount"]).with_columns(
        pl.col("is_test").cast(pl.Int8).alias("split"),
        pl.col("sender").cast(pl.Int64),
        pl.col("receiver").cast(pl.Int64),
        pl.col("ts").cast(pl.Int64),
        pl.col("amount").cast(pl.Float64),
        pl.col("to_person").fill_null(False),
    )
    src = b.filter(pl.col("to_person") & (pl.col("amount") >= HOP_NEAR_MIN)).select(
        pl.col("i").alias("a"),
        "split",
        pl.col("receiver").alias("acct"),
        pl.col("ts").alias("ta"),
        pl.col("amount").alias("xa"),
    )
    dst = b.select(
        pl.col("i").alias("b"),
        "split",
        pl.col("sender").alias("acct"),
        pl.col("ts").alias("tb"),
        pl.col("amount").alias("xb"),
    )
    return src, dst


def _first_forward(src: pl.DataFrame, dst: pl.DataFrame) -> pl.DataFrame:
    """Each source and its receiver's first later send within 24 h (any transaction type).

    Columns ``a``, ``b`` (ledger positions), ``dt`` seconds, ``ratio`` = onward / source, ``xa``.
    """
    schema = {
        "a": pl.Int64,
        "b": pl.Int64,
        "dt": pl.Int64,
        "ratio": pl.Float64,
        "xa": pl.Float64,
    }
    if src.height == 0 or dst.height == 0:
        return pl.DataFrame(schema=schema)
    hit = src.sort("split", "acct", "ta").join_asof(
        dst.sort("split", "acct", "tb", "b"),
        left_on="ta",
        right_on="tb",
        by=["split", "acct"],
        strategy="forward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    hit = hit.filter(pl.col("b").is_not_null()).with_columns(
        (pl.col("tb") - pl.col("ta")).alias("dt"),
        (pl.col("xb") / pl.col("xa")).alias("ratio"),
    )
    return hit.filter((pl.col("dt") > 0) & (pl.col("dt") <= HOP_DELAY_BAND_S)).select(
        "a", "b", "dt", "ratio", "xa", "split", "acct", "ta"
    )


def _later_inband(cands: pl.DataFrame, dst: pl.DataFrame) -> np.ndarray:
    """Source positions that have some onward row within 6 h at an in-band ratio.

    ``cands`` is the small set whose first send is not itself that row. This is what keeps a real
    hop (the law links any partner inside 6 h, not only the first send) out of the near-miss bands.
    """
    if cands.height == 0 or dst.height == 0:
        return np.array([], dtype=np.int64)
    lo, hi = flows.HOP_RATIO
    dt = pl.col("tb") - pl.col("ta")
    pairs = (
        cands.lazy()
        .join(dst.lazy(), on=["split", "acct"])
        .filter((dt > 0) & (dt <= flows.HOP_WINDOW))
        .with_columns((pl.col("xb") / pl.col("xa")).alias("ratio"))
        .filter((pl.col("ratio") >= lo) & (pl.col("ratio") <= hi))
        .select("a")
        .unique()
        .collect()
    )
    if pairs.height == 0:
        return np.array([], dtype=np.int64)
    return pairs["a"].to_numpy()


def _hop_band_masks(led: pl.DataFrame) -> dict[str, pl.Series]:
    """Near-miss pass-throughs from the receiver's first send within 24 h.

    ``hop_ratio_low`` / ``hop_ratio_high``: a P2P of at least 2,000 whose first onward row is at a
    ratio in [0.90, 0.9649) or (0.9951, 1]. ``hop_delay``: a P2P of at least 5,000 whose first
    onward row is inside the ratio band but after 6 h and by 24 h. A source the hop law already
    links (at least 5,000, some onward row within 6 h inside the ratio band) is not a near-miss.
    Both rows of the pair are marked.
    """
    n = led.height
    names = ("hop_ratio_low", "hop_ratio_high", "hop_delay")
    src, dst = _hop_frames(led)
    first = _first_forward(src, dst)
    empty = {name: _mask(n, []) for name in names}
    if first.height == 0:
        return empty
    lo, hi = flows.HOP_RATIO
    ratio = first["ratio"].to_numpy()
    dt = first["dt"].to_numpy()
    xa = first["xa"].to_numpy()
    a = first["a"].to_numpy()
    b = first["b"].to_numpy()
    law = (xa >= flows.HOP_MIN_AMOUNT) & (dt <= flows.HOP_WINDOW) & (ratio >= lo) & (ratio <= hi)
    suspect = first.filter(
        (pl.col("xa") >= flows.HOP_MIN_AMOUNT)
        & (pl.col("dt") <= flows.HOP_WINDOW)
        & ((pl.col("ratio") < lo) | (pl.col("ratio") > hi))
    )
    real = np.zeros(n, dtype=bool)
    if law.any():
        real[a[law]] = True
    later = _later_inband(suspect.select("a", "split", "acct", "ta", "xa"), dst)
    if later.size:
        real[later] = True
    keep = ~real[a]
    low = keep & (ratio >= HOP_RATIO_FLOOR) & (ratio < lo)
    high = keep & (ratio > hi) & (ratio <= HOP_RATIO_CEIL)
    slow = (
        keep
        & (xa >= flows.HOP_MIN_AMOUNT)
        & (dt > flows.HOP_WINDOW)
        & (ratio >= lo)
        & (ratio <= hi)
    )
    out = {}
    for name, hit in (("hop_ratio_low", low), ("hop_ratio_high", high), ("hop_delay", slow)):
        take = np.flatnonzero(hit)
        out[name] = _mask(n, np.concatenate([a[take], b[take]]) if take.size else [])
    return out


def hop_ratio_low(led: pl.DataFrame) -> pl.Series:
    """First send within 24 h at a ratio in [0.90, hop-ratio floor), source at least 2,000."""
    return _hop_band_masks(led)["hop_ratio_low"]


def hop_ratio_high(led: pl.DataFrame) -> pl.Series:
    """First send within 24 h at a ratio in (hop-ratio ceiling, 1], source at least 2,000."""
    return _hop_band_masks(led)["hop_ratio_high"]


def hop_delay(led: pl.DataFrame) -> pl.Series:
    """First send of an in-band ratio in (6 h, 24 h] after a P2P of at least 5,000."""
    return _hop_band_masks(led)["hop_delay"]


def _inflow_runs(led: pl.DataFrame, *, new_only: bool) -> pl.DataFrame:
    """Small whole P2P PAY runs (the mule / fan candidate), with a sender count per run."""
    df = _indexed(
        led, ["is_test", "sender", "receiver", "to_person", "txn_type", "ts", "amount", "new_payee"]
    )
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    cand = (
        pl.col("to_person").fill_null(False)
        & (pl.col("txn_type") == "PAY")
        & (cents % 1000 == 0)
        & (pl.col("amount") >= flows.INFLOW_RANGE[0])
        & (pl.col("amount") <= flows.INFLOW_RANGE[1])
    )
    if new_only:
        cand = cand & (pl.col("new_payee").fill_null(0).cast(pl.Int64) == 1)
    c = df.filter(cand).sort("is_test", "receiver", "ts", "i")
    if c.height == 0:
        return c.with_columns(pl.lit(0).alias("run"), pl.lit(0).alias("senders"))
    gap = pl.col("ts").diff().over("is_test", "receiver")
    c = c.with_columns((gap.is_null() | (gap > flows.RUN_GAP)).cum_sum().alias("run"))
    stats = c.group_by("run").agg(pl.col("sender").n_unique().alias("senders"))
    return c.join(stats, on="run", how="left")


def burst_senders(led: pl.DataFrame) -> pl.Series:
    """Inflow runs with sender count in [8 - band, 8): the burst minimum, relaxed."""
    runs = _inflow_runs(led, new_only=False)
    lo = flows.BURST_SENDERS - SENDER_BAND
    hit = runs.filter((pl.col("senders") >= lo) & (pl.col("senders") < flows.BURST_SENDERS))
    return _mask(led.height, hit["i"].to_numpy() if hit.height else [])


def fan_senders(led: pl.DataFrame) -> pl.Series:
    """New-payee inflow runs with sender count one below the fan minimum (2, not 3-7)."""
    runs = _inflow_runs(led, new_only=True)
    want = flows.FAN_SENDERS[0] - 1
    hit = runs.filter(pl.col("senders") == want)
    return _mask(led.height, hit["i"].to_numpy() if hit.height else [])


def _bot_sizes(led: pl.DataFrame) -> pl.DataFrame:
    """Strict (5 min) and relaxed (30 min) session sizes, one row per ledger row."""

    def sizes(gap_max: int, name: str) -> pl.DataFrame:
        df = _indexed(led, ["is_test", "device", "ts"]).sort("is_test", "device", "ts", "i")
        gap = pl.col("ts").diff().over("is_test", "device")
        df = df.with_columns((gap.is_null() | (gap > gap_max)).cum_sum().alias("sid"))
        df = df.with_columns(pl.len().over("sid").alias(name))
        return df.select("i", name)

    return sizes(bursts.SESSION_GAP_S, "strict_n").join(
        sizes(BOT_GAP_BAND_S, "relax_n"), on="i", how="left"
    )


def _bot_band_masks(led: pl.DataFrame) -> dict[str, pl.Series]:
    """Rows a 5-minute session misses that a 30-minute session of >= 10 rows would keep, and
    strict sessions of length 8 or 9 that a 30-minute gap does not stitch into a long one."""
    n = led.height
    both = _bot_sizes(led)
    strict = both["strict_n"].to_numpy()
    relax = both["relax_n"].to_numpy()
    idx = both["i"].to_numpy()
    gap = (relax >= bursts.MIN_ROWS) & (strict < bursts.MIN_ROWS)
    short_lo = bursts.MIN_ROWS - BOT_SHORT_BAND
    short = (strict >= short_lo) & (strict < bursts.MIN_ROWS) & (relax < bursts.MIN_ROWS)
    return {
        "bot_gap": _mask(n, idx[gap]),
        "bot_rows": _mask(n, idx[short]),
    }


def bot_gap(led: pl.DataFrame) -> pl.Series:
    """Device rows in a <= 30 min session of >= 10 that is not a <= 5 min session of >= 10."""
    return _bot_band_masks(led)["bot_gap"]


def bot_rows(led: pl.DataFrame) -> pl.Series:
    """Strict <= 5 min sessions of 8 or 9 rows, not already inside a 30 min session of >= 10."""
    return _bot_band_masks(led)["bot_rows"]


def _drain_pairs(led: pl.DataFrame) -> pl.DataFrame:
    """Each row and the sender's immediate next row inside the split."""
    df = _indexed(led, ["is_test", "sender", "ts", "balance_before"]).sort(
        "is_test", "sender", "ts", "i"
    )
    df = df.with_columns(
        pl.col("balance_before").shift(-1).alias("b1"),
        pl.col("ts").shift(-1).alias("t1"),
        pl.col("i").shift(-1).alias("j"),
        (
            (pl.col("sender").shift(-1) == pl.col("sender"))
            & (pl.col("is_test").shift(-1) == pl.col("is_test"))
        ).alias("same"),
    )
    return df.filter(pl.col("same").fill_null(False) & (pl.col("balance_before") > 0))


def _drain_band_masks(led: pl.DataFrame) -> dict[str, pl.Series]:
    """Step ratio within ±0.02 of 0.25 or 0.35 but outside TOL, inside 3 h; and an exact step
    whose delay is in (3 h, 6 h]."""
    n = led.height
    pairs = _drain_pairs(led)
    if pairs.height == 0:
        return {"drain_step": _mask(n, []), "drain_span": _mask(n, [])}
    b0 = pairs["balance_before"].to_numpy().astype(np.float64)
    b1 = pairs["b1"].to_numpy().astype(np.float64)
    dt = pairs["t1"].to_numpy().astype(np.float64) - pairs["ts"].to_numpy().astype(np.float64)
    real = np.zeros(len(b0), dtype=bool)
    near = np.zeros(len(b0), dtype=bool)
    for _code, c in drains.CONSTANTS:
        real |= np.abs(b1 - c * b0) <= drains.TOL
        near |= np.abs(b1 / b0 - c) <= STEP_BAND
    step = (~real) & near & (dt <= drains.MAX_SPAN_S)
    span = real & (dt > drains.MAX_SPAN_S) & (dt <= 2 * drains.MAX_SPAN_S)
    src = pairs["i"].to_numpy()
    dst = pairs["j"].to_numpy().astype(np.int64)

    def take(hit: np.ndarray) -> pl.Series:
        w = np.flatnonzero(hit)
        return _mask(n, np.concatenate([src[w], dst[w]]) if w.size else [])

    return {"drain_step": take(step), "drain_span": take(span)}


def drain_step(led: pl.DataFrame) -> pl.Series:
    """Next balance within ±0.02 of 0.25 or 0.35, absolute error outside TOL, delay <= 3 h."""
    return _drain_band_masks(led)["drain_step"]


def drain_span(led: pl.DataFrame) -> pl.Series:
    """Exact 0.25 or 0.35 step (within TOL) delayed by more than 3 h and at most 6 h."""
    return _drain_band_masks(led)["drain_span"]


def _near_amounts() -> list[int]:
    """Rupee amounts L - d for the real limits and the near offsets, excluding real offsets."""
    real = set(requests.JUST_UNDER)
    out = []
    for lim in requests.LIMITS:
        for d in NEAR_OFFSETS:
            amt = lim - d
            if amt not in real and amt > 0:
                out.append(amt)
    return out


def collect_offset(led: pl.DataFrame) -> pl.Series:
    """New-payee P2P COLLECT for L - d with d in {2, 5, 20, 25, 100}, not 1/10/50."""
    amounts = _near_amounts()
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    hit = (
        pl.col("to_person").fill_null(False)
        & (pl.col("txn_type") == "COLLECT")
        & (pl.col("new_payee").fill_null(0).cast(pl.Int64) == 1)
        & cents.is_in([100 * a for a in amounts])
    )
    return led.select(hit.fill_null(False).alias("hit"))["hit"]


def bait_amount(led: pl.DataFrame) -> pl.Series:
    """P2P PAY of 3, 20 or 50 rupees (the bait amounts beside 1/2/5/10)."""
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    hit = (
        pl.col("to_person").fill_null(False)
        & (pl.col("txn_type") == "PAY")
        & cents.is_in([100 * a for a in NEAR_BAITS])
    )
    return led.select(hit.fill_null(False).alias("hit"))["hit"]


def bait_gap(led: pl.DataFrame) -> pl.Series:
    """Consecutive real baits of one pair whose gap is in (10 min, 30 min]."""
    df = _indexed(led, ["is_test", "sender", "receiver", "to_person", "txn_type", "ts", "amount"])
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    bait = (
        pl.col("to_person").fill_null(False)
        & (pl.col("txn_type") == "PAY")
        & cents.is_in([100 * a for a in requests.BAIT_AMOUNTS])
    )
    c = df.filter(bait).sort("is_test", "sender", "receiver", "ts", "i")
    if c.height == 0:
        return _mask(led.height, [])
    c = c.with_columns(pl.col("i").shift(1).alias("prev"), pl.col("ts").shift(1).alias("prev_ts"))
    same = (
        (pl.col("sender") == pl.col("sender").shift(1))
        & (pl.col("receiver") == pl.col("receiver").shift(1))
        & (pl.col("is_test") == pl.col("is_test").shift(1))
    )
    gap = pl.col("ts") - pl.col("prev_ts")
    hit = c.filter(same.fill_null(False) & (gap > requests.SESSION_GAP) & (gap <= BAIT_GAP_BAND_S))
    if hit.height == 0:
        return _mask(led.height, [])
    return _mask(led.height, np.concatenate([hit["i"].to_numpy(), hit["prev"].to_numpy()]))


def bait_answer(led: pl.DataFrame) -> pl.Series:
    """A P2P row whose closest earlier bait the other way is in (72 h, 96 h].

    A bait inside the real 72 h window makes the row an answer of the law, not a near-miss.
    """
    df = _indexed(led, ["is_test", "sender", "receiver", "to_person", "txn_type", "ts", "amount"])
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    bait = (
        pl.col("to_person").fill_null(False)
        & (pl.col("txn_type") == "PAY")
        & cents.is_in([100 * a for a in requests.BAIT_AMOUNTS])
    )
    baits = df.filter(bait).select(
        pl.col("is_test").alias("split"),
        pl.col("sender").alias("rcv"),
        pl.col("receiver").alias("snd"),
        pl.col("ts").alias("tb"),
    )
    backs = df.filter(pl.col("to_person").fill_null(False)).select(
        "i",
        pl.col("is_test").alias("split"),
        "sender",
        "receiver",
        "ts",
    )
    if baits.height == 0 or backs.height == 0:
        return _mask(led.height, [])
    dt = pl.col("ts") - pl.col("tb")
    late = ANSWER_BAND_S + requests.ANSWER_WINDOW
    joined = backs.join(
        baits, left_on=["split", "sender", "receiver"], right_on=["split", "snd", "rcv"]
    )
    joined = joined.filter((dt > 0) & (dt <= late))
    if joined.height == 0:
        return _mask(led.height, [])
    best = joined.group_by("i").agg(dt.min().alias("delay"))
    hit = best.filter(pl.col("delay") > requests.ANSWER_WINDOW)
    return _mask(led.height, hit["i"].to_numpy() if hit.height else [])


def _evening_expr() -> pl.Expr:
    """The lending candidate laws with the hour set to 18 and 19 instead of 20-23."""
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    whole = cents % 100 == 0
    rupees = cents // 100
    hour = pl.col("timestamp").dt.hour().is_in(EVENING_NEAR)
    to_person = pl.col("to_person").fill_null(False)
    ttype = pl.col("txn_type")
    cat = pl.col("merchant_category")
    ratio_gap = (pl.col("amount") / pl.col("balance_before")) - repeats.RATIO_110
    ratio = ratio_gap.abs() < repeats.RATIO_TOL
    lending = (ttype == "PAY") & ~to_person & (cat == repeats.LENDING) & whole & hour
    lending_p2p = (
        (ttype == "PAY")
        & to_person
        & hour
        & (pl.col("new_payee").fill_null(0).cast(pl.Int64) == 1)
        & whole
        & (pl.col("amount") >= repeats.P2P_MIN)
        & (pl.col("amount") <= repeats.P2P_MAX)
        & ((rupees % 10 != 0) | ratio.fill_null(False))
    )
    return (lending.fill_null(False) | lending_p2p.fill_null(False)).alias("hit")


def evening(led: pl.DataFrame) -> pl.Series:
    """Lending PAY, merchant or person, that meets the law except the hour is 18 or 19."""
    return led.select(_evening_expr())["hit"]


def repeat_gap(led: pl.DataFrame) -> pl.Series:
    """Consecutive repeat candidates of one sender and kind whose gap is in (7 d, 14 d]."""
    df = _indexed(
        led,
        [
            "is_test",
            "sender",
            "ts",
            "timestamp",
            "to_person",
            "txn_type",
            "merchant_category",
            "amount",
            "balance_before",
            "new_payee",
        ],
    )
    cand = df.with_columns(repeats._kind(df).alias("kind")).filter(pl.col("kind") > 0)
    if cand.height == 0:
        return _mask(led.height, [])
    c = cand.sort("is_test", "sender", "kind", "ts", "i")
    c = c.with_columns(pl.col("i").shift(1).alias("prev"), pl.col("ts").shift(1).alias("prev_ts"))
    same = (
        (pl.col("sender") == pl.col("sender").shift(1))
        & (pl.col("kind") == pl.col("kind").shift(1))
        & (pl.col("is_test") == pl.col("is_test").shift(1))
    )
    gap = pl.col("ts") - pl.col("prev_ts")
    hit = c.filter(same.fill_null(False) & (gap > repeats.CHAIN_GAP) & (gap <= REPEAT_GAP_BAND_S))
    if hit.height == 0:
        return _mask(led.height, [])
    return _mask(led.height, np.concatenate([hit["i"].to_numpy(), hit["prev"].to_numpy()]))


def ring_amount(led: pl.DataFrame) -> pl.Series:
    """PAY or QR to a merchant for another round hundred of rupees, up to twice the largest ring.

    A round hundred is a positive multiple of 100 rupees. The five ring amounts are excluded.
    """
    cents = (pl.col("amount") * 100).round().cast(pl.Int64)
    rupees = cents // 100
    hit = (
        ~pl.col("to_person").fill_null(False)
        & pl.col("txn_type").is_in(["PAY", "QR"])
        & (cents % 100 == 0)
        & (rupees % 100 == 0)
        & (rupees >= 100)
        & (rupees <= RING_CAP_RUPEES)
        & ~cents.is_in(list(merchant.RING_CENTS))
    )
    return led.select(hit.fill_null(False).alias("hit"))["hit"]


def _blurb(name: str) -> str:
    lo, hi = flows.HOP_RATIO
    text = {
        "hop_ratio_low": (
            f"P2P >= {HOP_NEAR_MIN:g} whose first send within 24 h "
            f"is at ratio [{HOP_RATIO_FLOOR}, {lo})"
        ),
        "hop_ratio_high": (
            f"P2P >= {HOP_NEAR_MIN:g} whose first send within 24 h "
            f"is at ratio ({hi}, {HOP_RATIO_CEIL}]"
        ),
        "hop_delay": (
            f"P2P >= {flows.HOP_MIN_AMOUNT:g} whose first send within 24 h "
            "is an in-band ratio in (6 h, 24 h]"
        ),
        "burst_senders": (
            f"small whole P2P runs with {flows.BURST_SENDERS - SENDER_BAND}"
            f"-{flows.BURST_SENDERS - 1} senders"
        ),
        "fan_senders": f"new-payee runs with {flows.FAN_SENDERS[0] - 1} senders",
        "bot_gap": f"device session of >= {bursts.MIN_ROWS} rows only if gaps up to 30 min count",
        "bot_rows": (
            f"device session of {bursts.MIN_ROWS - BOT_SHORT_BAND}-{bursts.MIN_ROWS - 1} "
            "rows at <= 5 min"
        ),
        "drain_step": (
            f"balance step within ±{STEP_BAND} of 0.25/0.35, outside ±{drains.TOL} rupees, <= 3 h"
        ),
        "drain_span": "exact 0.25/0.35 step delayed by (3 h, 6 h]",
        "collect_offset": "new-payee P2P COLLECT for L - d, d in (2, 5, 20, 25, 100)",
        "bait_amount": "P2P PAY of 3, 20 or 50",
        "bait_gap": "consecutive 1/2/5/10 baits of one pair with gap (10 min, 30 min]",
        "bait_answer": "reverse P2P whose closest prior bait is in (72 h, 96 h]",
        "repeat_gap": "repeat candidates of one sender and kind with gap (7 d, 14 d]",
        "evening": "lending PAY (merchant or person) at 18:00-19:59",
        "ring_amount": (
            f"PAY/QR to a merchant for another multiple of 100 rupees, 100-{RING_CAP_RUPEES}"
        ),
    }
    return text[name]


BANDS: tuple[tuple[str, object], ...] = (
    ("hop_ratio_low", hop_ratio_low),
    ("hop_ratio_high", hop_ratio_high),
    ("hop_delay", hop_delay),
    ("burst_senders", burst_senders),
    ("fan_senders", fan_senders),
    ("bot_gap", bot_gap),
    ("bot_rows", bot_rows),
    ("drain_step", drain_step),
    ("drain_span", drain_span),
    ("collect_offset", collect_offset),
    ("bait_amount", bait_amount),
    ("bait_gap", bait_gap),
    ("bait_answer", bait_answer),
    ("repeat_gap", repeat_gap),
    ("evening", evening),
    ("ring_amount", ring_amount),
)


def band_masks(led: pl.DataFrame) -> dict[str, pl.Series]:
    """Every near-miss band, hop and bot and drain joins computed once each."""
    parts: dict[str, pl.Series] = {}
    parts.update(_hop_band_masks(led))
    parts.update(_bot_band_masks(led))
    parts.update(_drain_band_masks(led))
    shared = set(parts)
    for name, fn in BANDS:
        if name not in shared:
            parts[name] = fn(led)
    return {name: parts[name] for name, _fn in BANDS}
