"""Refund-shaped merchant rings that sit exactly at the size cap, called merchant_collusion in test.

Round-amount merchant rings are typed by size: in train a ring of 4 to 8 rows is refund_abuse and
one of 10 or more is merchant_collusion, with nothing at 9. October departs from that on both
sides at once: five rings at exactly 8 rows where the train rate gives about 1.7, and none at 10
to 19 rows where it gives about 4.3. ``capped_rings`` reads the excess at the cap as the missing
mid-size collusion rings and calls every row of a test ring of exactly ``size`` rows
merchant_collusion when it is called refund_abuse.

Train cannot measure it (every 8-row train ring is refund_abuse), so it is a base-rate rule with a
row cap; one ring has one class, so its rows share one answer.
"""

from __future__ import annotations

import numpy as np

from ringtrace.evidence import anchors
from ringtrace.evidence.anchors import COLLU, REFUND

REFUND_RING = 2  # rng_shape of a refund-sized round-amount ring


def capped_rings(pred, proba, led, log=print, size: int = 8, only_test: bool = True) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int64)
    cols = anchors.family_columns(led, ["rng_shape", "rng_event"])
    shape = cols["rng_shape"].fill_null(0).to_numpy().astype(np.int64)
    event = cols["rng_event"].fill_null(-1).to_numpy().astype(np.int64)
    is_test = led["is_test"].to_numpy()
    ids, inv, n = np.unique(event, return_inverse=True, return_counts=True)
    at_cap = (event >= 0) & (n[inv] == int(size)) & (shape == REFUND_RING)
    hit = at_cap & (pred == REFUND) & (is_test | (not only_test))
    out = pred.copy()
    out[hit] = COLLU
    log(
        f"  capped rings (size {size}): events {len(np.unique(event[hit]))}, "
        f"train {int((hit & ~is_test).sum())} / test {int((hit & is_test).sum())} rows"
    )
    return out
