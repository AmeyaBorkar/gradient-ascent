"""Refined device counter: champion blocks, allocation geometry inside each fraud phase.

Not a feature family, and not hashed into the feature cache. ``phones`` stays the champion
counter the members trained on. A recipe with ``rule_counter`` ``refined`` has its rules read
these columns instead. Block edges, the emission table and the HMM are the champion's
(``find_blocks``, ``emission_table``, ``viterbi``, ``forward_backward``); the phase decoder
reads the unused-number run just below a phone and the phone's row count together with the
anchor, and it does not snap a sim_swap / account_takeover change point.

``counter`` and ``build`` return the same column names and dtypes as ``phones.counter`` and
``phones.build`` (prefix ``ctr``).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import anchors
from ringtrace.evidence.phones import (
    ATO,
    PREFIX,
    SIM,
    TEMPER,
    Blocks,
    K,
    Phase,
    block_codes,
    emission_table,
    find_blocks,
    forward_backward,
    viterbi,
)

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
    """All counter columns as arrays, from device numbers, split ids (0/1) and row anchors.

    Same names as ``phones.counter``. Each fraud phase is segmented by ``decode_phase``.
    """
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
    """``phones.build``'s columns and dtypes, each fraud phase read by ``decode_phase``."""
    if anchor is None:
        anchor = anchors.row_anchor(led)
    cols = counter(led["device"].to_numpy(), led["is_test"].to_numpy().astype(np.int64), anchor)
    dtypes = {"block": pl.Int8, "dev_rows": pl.Int32, "dev_single": pl.Int8}
    return pl.DataFrame(
        {f"{PREFIX}_{k}": pl.Series(v, dtype=dtypes.get(k, pl.Float64)) for k, v in cols.items()}
    )
