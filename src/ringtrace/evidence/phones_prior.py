"""The counter's segments with the test split taught by the train split's fraud phase.

The same counter as ``ringtrace.evidence.phones``, except that the test split's fraud phase is
decoded with the ``phones.Prior`` learned from the train split's own fit: a normal prior on each
cut's position on the number line and a geometric emission for the unused numbers before each
phone (see ``phones``). The train split's columns equal the plain counter's. October's fraud phase
holds one sixth of train's phones; thinning train's phase to that size and teaching it from the
phones outside each draw raised agreement on its unanchored fraud rows by 0.030 (30 seeds; the
spread and the gap weight below were chosen on half the seeds, ``analysis/counter.py gate``).

Columns (prefix ``cpr``): seg (Viterbi segment 0..10) and conf (its posterior), NaN outside the
fraud phases. Not a feature family: the ``counter`` rule reads it with ``source="cpr"``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import anchors, phones

PREFIX = "cpr"
DEPENDS_ON = ("ringtrace.evidence.phones", *phones.DEPENDS_ON)

SPREAD = 0.01  # sd of each cut, in fractions of the fraud phase's number line
GAP_WEIGHT = 1.0  # weight on the gap emission


def build(led: pl.DataFrame, anchor: np.ndarray | None = None) -> pl.DataFrame:
    if anchor is None:
        anchor = anchors.row_anchor(led)
    cols = phones.counter(
        led["device"].to_numpy(),
        led["is_test"].to_numpy().astype(np.int64),
        anchor,
        teach={"spread": SPREAD, "gap_weight": GAP_WEIGHT},
    )
    return pl.DataFrame(
        {
            f"{PREFIX}_seg": pl.Series(cols["seg"], dtype=pl.Float64),
            f"{PREFIX}_conf": pl.Series(cols["conf"], dtype=pl.Float64),
        }
    )
