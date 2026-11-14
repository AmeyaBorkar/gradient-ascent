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
emission table with zone-specific unmarked shares. Each phone also carries its allocation
geometry: the run of unused numbers just below it (geometric per segment) and its row count.
sim_swap and account_takeover phones look the same to every family, so this is what separates
them (sim phones dense, takeover phones sparse). Fitted separately on each split's fraud phase.

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

# Allocation evidence per segment, fitted once on the train months' fraud phase (phones placed by
# a monotone fit to the cleaned labels) and fixed here, so the counter stays label-free when it
# runs: the mean run of unused numbers just below a phone (a geometric law per segment: sim_swap
# phones are dense, account_takeover phones sparse) and the share of phones with 1, 2, 3, 4 and
# 5+ rows (only the drain segments carry phones with more than one row).
GAP_MU = np.array([0.8771, 1.8352, 1.215, 0.4138, 2.8066, 0.7797, 1.9609, 1.0455, 0.8, 1.0, 1.169])
ROW_PMF = np.array(
    [
        [0.9987, 0.0005, 0.0004, 0.0004, 0.0],
        [0.9974, 0.001, 0.0009, 0.0007, 0.0],
        [0.9978, 0.0008, 0.0007, 0.0006, 0.0],
        [0.4417, 0.2377, 0.2308, 0.0898, 0.0],
        [0.2073, 0.2921, 0.2328, 0.2678, 0.0],
        [0.998, 0.0008, 0.0007, 0.0006, 0.0],
        [0.998, 0.0008, 0.0007, 0.0006, 0.0],
        [0.9958, 0.0016, 0.0014, 0.0012, 0.0],
        [0.9939, 0.0024, 0.002, 0.0017, 0.0],
        [0.9912, 0.0034, 0.0029, 0.0025, 0.0],
        [0.9935, 0.0025, 0.0021, 0.0018, 0.0],
    ]
)
ROW_FLOOR = 1e-4  # an unseen row count (5+ rows) costs every segment the same, never -inf

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


def forward_backward(ll: np.ndarray) -> np.ndarray:
    """Segment posteriors under a uniform prior over monotone paths starting in segment 0."""
    n, k = ll.shape
    neg = np.full(1, -np.inf)
    la = np.full((n, k), -np.inf)
    la[0, 0] = ll[0, 0]
    for t in range(1, n):
        prev = la[t - 1]
        la[t] = np.logaddexp(prev, np.concatenate([neg, prev[:-1]])) + ll[t]
    lb = np.zeros((n, k))
    for t in range(n - 2, -1, -1):
        nxt = lb[t + 1] + ll[t + 1]
        lb[t] = np.logaddexp(nxt, np.concatenate([nxt[1:], neg]))
    z = np.logaddexp.reduce(la[-1])
    post = np.exp(la + lb - z)
    return post / post.sum(axis=1, keepdims=True)


def viterbi(ll: np.ndarray) -> np.ndarray:
    n, k = ll.shape
    neg = np.full(1, -np.inf)
    d = np.full(k, -np.inf)
    d[0] = ll[0, 0]
    move = np.zeros((n, k), dtype=bool)
    for t in range(1, n):
        from_left = np.concatenate([neg, d[:-1]])
        move[t] = from_left > d
        d = np.where(move[t], from_left, d) + ll[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(d))
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


def allocation_ll(nums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-phone log-likelihood of each segment from the allocation geometry: the unused numbers
    just below the phone (none for the first phone) and the phone's row count."""
    n = len(nums)
    ll = np.zeros((n, K))
    gap = np.full(n, -1, dtype=np.int64)
    gap[1:] = np.diff(np.asarray(nums, dtype=np.int64)) - 1
    m = gap >= 0
    if m.any():
        g = gap[m][:, None].astype(np.float64)
        ll[m] = g * np.log(GAP_MU / (1.0 + GAP_MU)) - np.log1p(GAP_MU)
    pmf = np.maximum(ROW_PMF, ROW_FLOOR)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    rows = np.asarray(counts, dtype=np.float64).sum(axis=1).round().astype(np.int64)
    rb = np.clip(rows, 1, pmf.shape[1]) - 1
    return ll + np.log(pmf[:, rb].T)


@dataclass
class Phase:
    nums: np.ndarray  # phone numbers, sorted
    post: np.ndarray  # (n, K)
    seg: np.ndarray
    conf: np.ndarray
    side: np.ndarray  # 1 takeover side, 0 sim side, NaN outside the sim/takeover zone
    cut: int | None


def decode_phase(nums: np.ndarray, counts: np.ndarray, blocks: Blocks, lo: int, hi: int) -> Phase:
    """Segment one fraud phase. ``counts[i, c]`` = rows of phone i with anchor c (0 = none). The
    anchors and the allocation geometry (gaps and rows per phone) are read together, so the
    sim_swap / account_takeover cut falls where the dense sim phones give way to sparse takeover
    phones without a separate change-point step."""
    loge = emission_table()
    ll = counts @ loge.T + allocation_ll(nums, counts)
    path = viterbi(ll)
    post = forward_backward(ll * TEMPER)
    conf = post[np.arange(len(path)), path]
    zone = np.isin(path, (SIM, ATO))
    side = np.full(len(nums), np.nan)
    side[zone] = (path[zone] == ATO).astype(np.float64)
    takeover = np.flatnonzero(path == ATO)
    cut = int(nums[takeover[0]]) if len(takeover) else None
    return Phase(nums=nums, post=post, seg=path, conf=conf, side=side, cut=cut)


# ---------------------------------------------------------------------------------------------
# assembly


def counter(device: np.ndarray, split: np.ndarray, anchor: np.ndarray) -> dict[str, np.ndarray]:
    """All counter columns as arrays, from device numbers, split ids (0/1) and row anchors."""
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
    for s, (lo, _edge, hi) in sorted(blocks.ranges.items()):
        rows = np.flatnonzero((block == 2) & (split == s))
        if len(rows) == 0:
            continue
        nums, pinv = np.unique(device[rows], return_inverse=True)
        counts = np.zeros((len(nums), K + 1))
        np.add.at(counts, (pinv, anchor[rows]), 1.0)
        ph = decode_phase(nums, counts, blocks, lo, hi)
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
