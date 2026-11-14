"""Rows the refined device counter moves by one class relative to the previous counter.

The phone feature module's source is hashed into its cache key, so the previous fraud-phase
segmentation lives here and ``phones.py`` stays as it is. ``legacy_decode`` is that segmentation:
anchors-only Viterbi, then the sim_swap / account_takeover change point, ``snap_cut`` and
``SIDE_PENALTY``, then Viterbi again. ``news`` runs it beside the current ``phones.decode_phase``
on the same per-split tables the counter builds.

``counter_news`` moves a fraud-phase row whose call still equals the previous class when the
refined posterior mode is exactly one class away and at least ``min_post``. A call that already
differs from the previous segment is left alone.
"""

from __future__ import annotations

import numpy as np

from ringtrace.evidence import anchors, phones


def _split_counts(mask: np.ndarray, is_test: np.ndarray) -> str:
    return f"train {int((mask & ~is_test).sum())} / test {int((mask & is_test).sum())}"


def legacy_decode(
    nums: np.ndarray, counts: np.ndarray, blocks: phones.Blocks, lo: int, hi: int
) -> np.ndarray:
    """Segment one fraud phase the way ``decode_phase`` did before allocation geometry.

    ``counts[i, c]`` is the rows of phone ``i`` with anchor ``c`` (0 = none). Returns the segment
    index of each phone (0..10), the path the previous counter published. The old decoder also
    returned posteriors and the snapped cut; this returns the path only.
    """
    loge = phones.emission_table()
    ll = counts @ loge.T
    path = phones.viterbi(ll)
    zone = np.isin(path, (phones.SIM, phones.ATO))
    if zone.sum() >= 2 * phones.CUT_MIN_SIDE:
        idx = np.flatnonzero(zone)
        j = phones.change_point(nums[idx])
        if j is not None:
            peak = int(nums[idx[j - 1]]) + 1
            cut = phones.snap_cut(peak, blocks, lo, hi)
            ll = ll.copy()
            ll[nums < cut, phones.ATO] -= phones.SIDE_PENALTY
            ll[nums >= cut, phones.SIM] -= phones.SIDE_PENALTY
            path = phones.viterbi(ll)
    return path


def news(led) -> dict[str, np.ndarray]:
    """Row-level previous class, refined class and refined posterior.

    Off the fraud phase every value is 0. On it, ``old_class`` is the legacy segment plus one,
    ``new_class`` is the mode of the current phase posterior plus one (the earliest segment when
    the mode ties), and ``new_post`` is that probability. The per-split phone tables are built as
    ``phones.counter`` builds them.
    """
    device = np.asarray(led["device"].to_numpy(), dtype=np.int64)
    split = np.asarray(led["is_test"].to_numpy().astype(np.int64), dtype=np.int64)
    anchor = np.asarray(anchors.row_anchor(led), dtype=np.int64)
    n_rows = len(device)
    old_class = np.zeros(n_rows, dtype=np.int64)
    new_class = np.zeros(n_rows, dtype=np.int64)
    new_post = np.zeros(n_rows, dtype=np.float64)
    blocks = phones.find_blocks(device, split)
    block = phones.block_codes(device, split, blocks)
    for s, (lo, _edge, hi) in sorted(blocks.ranges.items()):
        rows = np.flatnonzero((block == 2) & (split == s))
        if len(rows) == 0:
            continue
        nums, pinv = np.unique(device[rows], return_inverse=True)
        counts = np.zeros((len(nums), phones.K + 1))
        np.add.at(counts, (pinv, anchor[rows]), 1.0)
        old_seg = legacy_decode(nums, counts, blocks, lo, hi)
        post = phones.decode_phase(nums, counts, blocks, lo, hi).post
        choice = post.argmax(axis=1)
        old_class[rows] = old_seg[pinv] + 1
        new_class[rows] = choice[pinv] + 1
        new_post[rows] = post[np.arange(len(choice)), choice][pinv]
    return {"old_class": old_class, "new_class": new_class, "new_post": new_post}


def counter_news(pred, proba, led, log=print, min_post: float = 0.7) -> np.ndarray:
    """A fraud-phase row whose call equals the previous counter class takes the refined class
    when that class is one away and its posterior is at least ``min_post``."""
    got = news(led)
    pred = np.asarray(pred, dtype=np.int64)
    old = got["old_class"]
    new = got["new_class"]
    post = got["new_post"]
    hit = (old > 0) & (np.abs(new - old) == 1) & (post >= min_post) & (pred == old)
    out = pred.copy()
    out[hit] = new[hit]
    log(f"  counter news: {_split_counts(hit, led['is_test'].to_numpy())}")
    return out
