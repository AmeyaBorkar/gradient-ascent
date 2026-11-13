"""Device counter: blocks on the extra-phone number line and the fraud phase's type segments.

Phones numbered above 420,000 are minted by a counter. A reserved bot range comes first, then each
split's *fraud phase* (fraud labels in label-list order, about 43 % of the integers used) and its
dense *legit block* (every number used once, one row each). Block edges are found per split from
occupancy of the stacked frame's number line:

* F0 = the smallest extra number whose device carries exactly one row; below it is the bot block;
* a split's range = the numbers >= F0 its rows use; its fraud/legit edge = the largest unused
  number of that range whose preceding window is sparse (an isolated hole in a dense block is
  ignored).

Inside a fraud phase the phones, sorted by number, run through the 11 fraud types in order. An
11-state left-to-right HMM with flat transitions (uniform prior over monotone paths that start in
the first segment) reads each row's label-free anchor (``anchors.row_anchor``) through a hard-anchor
emission table with zone-specific unmarked shares. sim_swap and account_takeover phones look the
same to every family, so the cut between them is placed by the occupancy drop (sim phones are
dense, takeover phones sparse) and snapped onto the longest run of unused numbers near it; a
second pass of the HMM then respects that cut. Fitted separately on each split's fraud phase.

A smaller phase can borrow what a full one teaches (off by default; ``counter(..., teach=...)``).
The train split's own fit gives a ``Prior``: where each cut falls on the number line, as a fraction
of the phase, and each segment's occupancy over the phase's mean. The test split is then decoded
with two soft terms: each cut is drawn from a normal around the train fraction (a path that cuts
between two phones takes the normal's mass between them), and the run of unused numbers before
each phone is geometric at its segment's occupancy, rescaled to the test phase's own mean.

Columns (prefix ``ctr``): block (0 none, 1 bot, 2 fraud phase, 3 legit block), pos (rank of the
phone in its fraud phase, 0..1), seg (Viterbi segment 0..10), conf (posterior of the Viterbi
segment), p00..p10 (segment posteriors), ato_vs_sim (1 takeover side, 0 sim side of the cut, for
phones of the sim/takeover zone), dev_rows (rows per device in the whole frame), dev_single (device
has one row in its split). Raw device numbers never leave this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.special import log_ndtr

from ringtrace.evidence import anchors
from ringtrace.tables import HOME_DEVICES

PREFIX = "ctr"
DEPENDS_ON = ("ringtrace.evidence.anchors", *anchors.DEPENDS_ON)

K = 11
SIM, ATO = 3, 4  # segment indices
OCC_WINDOW = 50  # numbers looked back from a candidate fraud/legit edge
OCC_DENSE = 0.75  # a candidate edge whose window is this full sits inside a dense block
CUT_MIN_SIDE = 5  # phones on each side of the sim/ato change point
SNAP_RADIUS = 10  # numbers around the change point searched for a run of unused numbers
SIDE_PENALTY = 6.0  # log-weight against the wrong side of the snapped sim/ato cut
TEMPER = 1.0  # scale of the log-emission for the posteriors (Viterbi unchanged)

# P(row carries no anchor | zone): the learned shape-tag table (spec-counter 2c)
UNMARKED = np.array([0.641, 0.948, 0.557, 0.150, 0.067, 0.551, 0.671, 0.597, 0.641, 0.730, 0.704])
# anchor class mix of the marked rows of each zone; the rest of the mass is spread evenly. A zone
# is mostly marked with its own class, except where the type is written in a partner's shape:
# bot-phase fan-ins look like mule bursts, sim-phase phones carry 0.35 drains, cashback events
# are half bait-answered-within-4 h (phishing majority).
MARKED: dict[int, dict[int, float]] = {
    0: {1: 0.85, 3: 0.05},
    1: {1: 0.85, 2: 0.05},
    2: {3: 0.90, 1: 0.05},
    3: {5: 0.90, 4: 0.08},
    4: {5: 0.93, 4: 0.04},
    5: {6: 0.88, 9: 0.05},
    6: {7: 0.88, 8: 0.05},
    7: {8: 0.80, 7: 0.10},
    8: {9: 0.50, 6: 0.40},
    9: {10: 0.80, 6: 0.10},
    10: {11: 0.80, 8: 0.10},
}


def emission_table() -> np.ndarray:
    """log P(anchor | segment), shape (K, K + 1); column 0 is "no anchor", column c class c."""
    e = np.zeros((K, K + 1))
    for k in range(K):
        mix = np.zeros(K)
        for c, w in MARKED[k].items():
            mix[c - 1] = w
        free = mix == 0
        mix[free] = (1.0 - mix.sum()) / free.sum()
        e[k, 0] = UNMARKED[k]
        e[k, 1:] = (1.0 - UNMARKED[k]) * mix
    return np.log(e)


# ---------------------------------------------------------------------------------------------
# blocks


@dataclass
class Blocks:
    f0: int
    # per split (0 train, 1 test): (lo, edge, hi); fraud phase lo..edge, legit block edge+1..hi
    ranges: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    used: np.ndarray | None = None  # used[n - base] for n in base..top
    base: int = 0

    def is_used(self, n: np.ndarray) -> np.ndarray:
        n = np.asarray(n, dtype=np.int64)
        i = n - self.base
        ok = (i >= 0) & (i < len(self.used))
        out = np.zeros(n.shape, dtype=bool)
        out[ok] = self.used[i[ok]]
        return out


def find_blocks(device: np.ndarray, split: np.ndarray) -> Blocks:
    """Bot block and each split's fraud-phase / legit-block edges from number-line occupancy."""
    device = np.asarray(device, dtype=np.int64)
    split = np.asarray(split, dtype=np.int64)
    extra = device > HOME_DEVICES
    nums, inv, counts = np.unique(device[extra], return_inverse=True, return_counts=True)
    if len(nums) == 0:
        return Blocks(f0=HOME_DEVICES + 1, used=np.zeros(0, dtype=bool), base=HOME_DEVICES + 1)
    single = counts == 1
    f0 = int(nums[single].min()) if single.any() else int(nums.max()) + 1
    base, top = int(nums.min()), int(nums.max())
    used = np.zeros(top - base + 1, dtype=bool)
    used[nums - base] = True
    blocks = Blocks(f0=f0, used=used, base=base)
    cum = np.concatenate([[0], np.cumsum(used)])
    sp = split[extra]
    for s in np.unique(sp):
        mine = np.zeros(len(nums), dtype=bool)
        mine[np.unique(inv[sp == s])] = True
        mine &= nums >= f0
        if not mine.any():
            continue
        lo, hi = int(nums[mine].min()), int(nums[mine].max())
        holes = np.flatnonzero(~used[lo - base : hi - base + 1]) + lo
        edge = lo - 1  # no fraud phase unless a sparse hole is found
        for b in holes[::-1]:
            start = max(lo, int(b) - OCC_WINDOW + 1)
            occ = (cum[b - base + 1] - cum[start - base]) / (b - start + 1)
            if occ < OCC_DENSE:
                edge = int(b)
                break
        blocks.ranges[int(s)] = (lo, edge, hi)
    return blocks


def block_codes(device: np.ndarray, split: np.ndarray, blocks: Blocks) -> np.ndarray:
    device = np.asarray(device, dtype=np.int64)
    split = np.asarray(split, dtype=np.int64)
    code = np.zeros(len(device), dtype=np.int8)
    extra = device > HOME_DEVICES
    code[extra & (device < blocks.f0)] = 1
    for s, (lo, edge, hi) in blocks.ranges.items():
        mine = extra & (split == s)
        code[mine & (device >= lo) & (device <= edge)] = 2
        code[mine & (device > edge) & (device <= hi)] = 3
    return code


# ---------------------------------------------------------------------------------------------
# HMM over the phones of one fraud phase


def _hmm_terms(
    ll: np.ndarray, enter: np.ndarray | None, end: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    n, k = ll.shape
    return (np.zeros((n, k)) if enter is None else enter, np.zeros(k) if end is None else end)


def forward_backward(
    ll: np.ndarray, enter: np.ndarray | None = None, end: np.ndarray | None = None
) -> np.ndarray:
    """Segment posteriors over monotone paths starting in segment 0. ``enter[t, k]`` is the
    log-weight of entering segment k at phone t and ``end[k]`` that of a path ending in segment k;
    both default to 0, a uniform prior over paths."""
    n, k = ll.shape
    enter, end = _hmm_terms(ll, enter, end)
    neg = np.full(1, -np.inf)
    la = np.full((n, k), -np.inf)
    la[0, 0] = ll[0, 0]
    for t in range(1, n):
        prev = la[t - 1]
        la[t] = np.logaddexp(prev, np.concatenate([neg, prev[:-1] + enter[t, 1:]])) + ll[t]
    lb = np.zeros((n, k))
    lb[n - 1] = end
    for t in range(n - 2, -1, -1):
        nxt = lb[t + 1] + ll[t + 1]
        lb[t] = np.logaddexp(nxt, np.concatenate([nxt[1:] + enter[t + 1, 1:], neg]))
    z = np.logaddexp.reduce(la[-1] + end)
    post = np.exp(la + lb - z)
    return post / post.sum(axis=1, keepdims=True)


def viterbi(
    ll: np.ndarray, enter: np.ndarray | None = None, end: np.ndarray | None = None
) -> np.ndarray:
    n, k = ll.shape
    enter, end = _hmm_terms(ll, enter, end)
    neg = np.full(1, -np.inf)
    d = np.full(k, -np.inf)
    d[0] = ll[0, 0]
    move = np.zeros((n, k), dtype=bool)
    for t in range(1, n):
        from_left = np.concatenate([neg, d[:-1] + enter[t, 1:]])
        move[t] = from_left > d
        d = np.where(move[t], from_left, d) + ll[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(d + end))
    for t in range(n - 1, 0, -1):
        path[t - 1] = path[t] - 1 if move[t, path[t]] else path[t]
    return path


def _geom_ll(g: np.ndarray) -> float:
    if len(g) == 0:
        return 0.0
    mu = g.mean()
    if mu <= 0:
        return 0.0
    return float(g.sum() * np.log(mu / (1 + mu)) - len(g) * np.log1p(mu))


def change_point(nums: np.ndarray) -> int | None:
    """Index of the first phone after the occupancy drop (two-segment geometric gap model)."""
    m = len(nums)
    if m < 2 * CUT_MIN_SIDE:
        return None
    gaps = (np.diff(nums) - 1).astype(np.float64)  # gaps[i] sits between phones i and i + 1
    best, best_j = -np.inf, None
    for j in range(CUT_MIN_SIDE, m - CUT_MIN_SIDE + 1):
        left, right = gaps[: j - 1], gaps[j:]
        score = _geom_ll(left) + _geom_ll(right)
        # a drop: the takeover side must be the sparser one
        if right.mean() > left.mean() and score > best:
            best, best_j = score, j
    return best_j


def unused_runs(blocks: Blocks, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Start and length of every maximal run of unused numbers inside ``lo..hi``."""
    free = ~blocks.is_used(np.arange(lo, hi + 1))
    edges = np.diff(np.concatenate([[0], free.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return starts + lo, ends - starts


def snap_cut(peak: int, blocks: Blocks, lo: int, hi: int) -> int:
    """Start of the longest run of unused numbers starting within SNAP_RADIUS of ``peak``."""
    starts, lengths = unused_runs(blocks, lo, hi)
    near = np.abs(starts - peak) <= SNAP_RADIUS
    if not near.any():
        return peak
    starts, lengths = starts[near], lengths[near]
    best = np.lexsort((np.abs(starts - peak), -lengths))[0]
    return int(starts[best])


@dataclass
class Phase:
    nums: np.ndarray  # phone numbers, sorted
    post: np.ndarray  # (n, K)
    seg: np.ndarray
    conf: np.ndarray
    side: np.ndarray  # 1 takeover side, 0 sim side, NaN outside the sim/takeover zone
    cut: int | None


@dataclass(frozen=True)
class Prior:
    """What one fraud phase's fit teaches another. ``cuts[k - 1]`` is where segment k starts, as a
    fraction of the teaching phase's number line (NaN if its fit never reached k); ``occupancy[k]``
    is segment k's share of used numbers over the phase's mean share."""

    cuts: np.ndarray
    occupancy: np.ndarray
    spread: float | None = 0.03  # sd of each cut, in fractions of the line; None drops the term
    gap_weight: float = 1.0  # weight on the gap emission; 0 drops the term


def learn_prior(
    phase: Phase, lo: int, edge: int, spread: float | None = 0.03, gap_weight: float = 1.0
) -> Prior:
    """Cut fractions and relative occupancies of a fitted phase spanning ``lo..edge``."""
    length = edge - lo + 1
    nums, seg = phase.nums, phase.seg
    cuts = np.full(K - 1, np.nan)
    for k in range(1, K):
        before, after = nums[seg < k], nums[seg >= k]
        if len(before) and len(after):
            cuts[k - 1] = ((before.max() + after.min()) / 2 - lo) / length
    starts = np.concatenate([[0.0], cuts, [1.0]])
    occ = np.ones(K)
    for k in range(K):
        if not np.isfinite(starts[k]):
            continue
        stop = next(v for v in starts[k + 1 :] if np.isfinite(v))
        n_k = int((seg == k).sum())
        if n_k and stop > starts[k]:
            occ[k] = n_k / ((stop - starts[k]) * len(nums))
    return Prior(cuts=cuts, occupancy=occ, spread=spread, gap_weight=gap_weight)


def interval_logmass(a: np.ndarray, b: np.ndarray, centre: float, spread: float) -> np.ndarray:
    """log of a normal's mass between ``a`` and ``b`` (elementwise, ``a < b``), tail-stable."""
    za = (np.asarray(a, dtype=np.float64) - centre) / spread
    zb = (np.asarray(b, dtype=np.float64) - centre) / spread
    upper = za > 0  # both above the centre: take the mass from the upper tails
    lo_z = np.where(upper, -zb, za)
    hi_z = np.where(upper, -za, zb)
    top = log_ndtr(hi_z)
    return top + np.log1p(-np.exp(log_ndtr(lo_z) - top))


def prior_terms(
    nums: np.ndarray, lo: int, edge: int, prior: Prior
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the phones ``nums`` of a phase ``lo..edge``: the gap log-emission (n, K), the
    log-weight of entering segment k at phone t (n, K) and that of a path ending in k (K,)."""
    n = len(nums)
    length = edge - lo + 1
    emit, enter, end = np.zeros((n, K)), np.zeros((n, K)), np.zeros(K)
    if n < 2:
        return emit, enter, end
    if prior.gap_weight > 0:
        p = np.clip(prior.occupancy * n / length, 1e-6, 1 - 1e-6)
        gaps = (np.diff(nums) - 1).astype(np.float64)
        emit[1:] = prior.gap_weight * (gaps[:, None] * np.log1p(-p) + np.log(p))
    if prior.spread is not None:
        x = (np.asarray(nums, dtype=np.float64) - lo) / length
        for k in range(1, K):
            c = prior.cuts[k - 1]
            if not np.isfinite(c):
                continue
            # entering k at phone t puts the cut between phones t - 1 and t
            enter[1:, k] = interval_logmass(x[:-1], x[1:], c, prior.spread)
            # a path that ends before k leaves the cut beyond the last phone
            end[:k] += log_ndtr((c - x[-1]) / prior.spread)
    return emit, enter, end


def decode_phase(
    nums: np.ndarray,
    counts: np.ndarray,
    blocks: Blocks,
    lo: int,
    hi: int,
    prior: Prior | None = None,
    edge: int | None = None,
) -> Phase:
    """Segment one fraud phase. ``counts[i, c]`` = rows of phone i with anchor c (0 = none).

    ``prior`` (off by default) adds a teaching phase's cut positions and occupancies, read over the
    phase ``lo..edge`` (``edge`` defaults to ``hi``)."""
    loge = emission_table()
    ll = counts @ loge.T
    enter = end = None
    if prior is not None:
        emit, enter, end = prior_terms(nums, lo, hi if edge is None else edge, prior)
        ll = ll + emit
    path = viterbi(ll, enter, end)
    zone = np.isin(path, (SIM, ATO))
    cut = None
    if zone.sum() >= 2 * CUT_MIN_SIDE:
        idx = np.flatnonzero(zone)
        j = change_point(nums[idx])
        if j is not None:
            peak = int(nums[idx[j - 1]]) + 1
            cut = snap_cut(peak, blocks, lo, hi)
            ll = ll.copy()
            ll[nums < cut, ATO] -= SIDE_PENALTY
            ll[nums >= cut, SIM] -= SIDE_PENALTY
            path = viterbi(ll, enter, end)
    post = forward_backward(ll * TEMPER, enter, end)
    conf = post[np.arange(len(path)), path]
    side = np.full(len(nums), np.nan)
    if cut is not None:
        zone |= np.isin(path, (SIM, ATO))
        side[zone] = (nums[zone] >= cut).astype(np.float64)
    return Phase(nums=nums, post=post, seg=path, conf=conf, side=side, cut=cut)


# ---------------------------------------------------------------------------------------------
# assembly


def counter(
    device: np.ndarray, split: np.ndarray, anchor: np.ndarray, teach: dict | None = None
) -> dict[str, np.ndarray]:
    """All counter columns as arrays, from device numbers, split ids (0/1) and row anchors.

    ``teach = {"spread": s, "gap_weight": w}`` (off by default) decodes every split after the
    first with the ``Prior`` learned from the first split's own fit."""
    device = np.asarray(device, dtype=np.int64)
    split = np.asarray(split, dtype=np.int64)
    anchor = np.asarray(anchor, dtype=np.int64)
    n_rows = len(device)
    blocks = find_blocks(device, split)
    block = block_codes(device, split, blocks)

    _, dev_inv, dev_n = np.unique(device, return_inverse=True, return_counts=True)
    _, ds_inv, ds_n = np.unique(device * 2 + split, return_inverse=True, return_counts=True)

    out: dict[str, np.ndarray] = {
        "block": block,
        "pos": np.full(n_rows, np.nan),
        "seg": np.full(n_rows, np.nan),
        "conf": np.full(n_rows, np.nan),
        **{f"p{k:02d}": np.full(n_rows, np.nan) for k in range(K)},
        "ato_vs_sim": np.full(n_rows, np.nan),
        "dev_rows": dev_n[dev_inv].astype(np.int32),
        "dev_single": (ds_n[ds_inv] == 1).astype(np.int8),
    }
    prior = None
    for s, (lo, edge, hi) in sorted(blocks.ranges.items()):
        rows = np.flatnonzero((block == 2) & (split == s))
        if len(rows) == 0:
            continue
        nums, pinv = np.unique(device[rows], return_inverse=True)
        counts = np.zeros((len(nums), K + 1))
        np.add.at(counts, (pinv, anchor[rows]), 1.0)
        if prior is None:
            ph = decode_phase(nums, counts, blocks, lo, hi)
            if teach is not None:
                prior = learn_prior(ph, lo, edge, **teach)
        else:
            ph = decode_phase(nums, counts, blocks, lo, hi, prior=prior, edge=edge)
        out["pos"][rows] = (pinv / max(len(nums) - 1, 1)).astype(np.float64)
        out["seg"][rows] = ph.seg[pinv]
        out["conf"][rows] = ph.conf[pinv]
        for k in range(K):
            out[f"p{k:02d}"][rows] = ph.post[pinv, k]
        out["ato_vs_sim"][rows] = ph.side[pinv]
    return out


def build(led: pl.DataFrame, anchor: np.ndarray | None = None) -> pl.DataFrame:
    if anchor is None:
        anchor = anchors.row_anchor(led)
    cols = counter(led["device"].to_numpy(), led["is_test"].to_numpy().astype(np.int64), anchor)
    dtypes = {"block": pl.Int8, "dev_rows": pl.Int32, "dev_single": pl.Int8}
    return pl.DataFrame(
        {f"{PREFIX}_{k}": pl.Series(v, dtype=dtypes.get(k, pl.Float64)) for k, v in cols.items()}
    )
