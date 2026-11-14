"""Rows the refined device counter moves by one class relative to the champion counter.

``phones.py`` is the champion counter: its source is hashed into the feature cache, so the
members keep the segmentation they trained on. ``legacy_decode`` is that segmentation
(``phones.decode_phase(...).seg``). ``news`` runs it beside ``refined.decode_phase`` on the same
per-split tables the counter builds.

``counter_news`` moves a fraud-phase row whose call still equals the champion class when the
refined posterior mode is exactly one class away and at least ``min_post``. A call that already
differs from the champion segment is left alone.
"""

from __future__ import annotations

import numpy as np

from ringtrace.evidence import anchors, phones, refined


def _split_counts(mask: np.ndarray, is_test: np.ndarray) -> str:
    return f"train {int((mask & ~is_test).sum())} / test {int((mask & is_test).sum())}"


def legacy_decode(
    nums: np.ndarray, counts: np.ndarray, blocks: phones.Blocks, lo: int, hi: int
) -> np.ndarray:
    """Segment one fraud phase the way the champion counter does.

    ``counts[i, c]`` is the rows of phone ``i`` with anchor ``c`` (0 = none). Returns the segment
    index of each phone (0..10), the path ``phones.decode_phase`` publishes.
    """
    return phones.decode_phase(nums, counts, blocks, lo, hi).seg


def news(led) -> dict[str, np.ndarray]:
    """Row-level champion class, refined class and refined posterior.

    Off the fraud phase every value is 0. On it, ``old_class`` is the champion segment plus one,
    ``new_class`` is the mode of the refined phase posterior plus one (the earliest segment when
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
        post = refined.decode_phase(nums, counts, blocks, lo, hi).post
        choice = post.argmax(axis=1)
        old_class[rows] = old_seg[pinv] + 1
        new_class[rows] = choice[pinv] + 1
        new_post[rows] = post[np.arange(len(choice)), choice][pinv]
    return {"old_class": old_class, "new_class": new_class, "new_post": new_post}


def counter_news(pred, proba, led, log=print, min_post: float = 0.7) -> np.ndarray:
    """A fraud-phase row whose call equals the champion counter class takes the refined class
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
