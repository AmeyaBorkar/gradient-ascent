"""Fast per-entity time-window aggregates on the stacked ledger.

Rows are sorted once by (entity, time). A composite integer ``entity * SPAN + seconds`` keeps
every entity's rows in one contiguous, time-ordered run, so a single ``searchsorted`` finds the
window bounds of every row at once and cumulative sums give window sums.
"""

from __future__ import annotations

import numpy as np

SPAN = np.int64(1) << np.int64(33)  # wider than any timestamp offset we use


class Timeline:
    """Rows of one entity key, ordered by time, with window queries in both directions."""

    def __init__(self, key: np.ndarray, ts: np.ndarray):
        key = np.asarray(key)
        ts = np.asarray(ts, dtype=np.int64)
        ok = key >= 0
        self.t0 = int(ts.min())
        _, dense = np.unique(np.where(ok, key, -1), return_inverse=True)
        comp = dense.astype(np.int64) * SPAN + (ts - self.t0)
        self.order = np.lexsort((np.arange(len(ts)), comp))
        self.comp = comp[self.order]
        self.valid = ok[self.order]
        self.n = len(ts)
        self.rank = np.empty(self.n, dtype=np.int64)
        self.rank[self.order] = np.arange(self.n)

    def bounds(self, back: int, ahead: int) -> tuple[np.ndarray, np.ndarray]:
        """Sorted positions [lo, hi) of rows within [t - back, t + ahead] of each sorted row."""
        lo = np.searchsorted(self.comp, self.comp - back, side="left")
        hi = np.searchsorted(self.comp, self.comp + ahead, side="right")
        return lo, hi

    def count(self, back: int, ahead: int) -> np.ndarray:
        lo, hi = self.bounds(back, ahead)
        out = (hi - lo).astype(np.float32)
        out[~self.valid] = np.nan
        return self._unsort(out)

    def total(self, values: np.ndarray, back: int, ahead: int) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)[self.order]
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])
        lo, hi = self.bounds(back, ahead)
        out = (cs[hi] - cs[lo]).astype(np.float32)
        out[~self.valid] = np.nan
        return self._unsort(out)

    def gap(self, direction: int) -> np.ndarray:
        """Seconds to the previous (direction=-1) or next (+1) row of the same entity."""
        c = self.comp
        same_prev = np.zeros(self.n, dtype=bool)
        same_prev[1:] = (c[1:] // SPAN) == (c[:-1] // SPAN)
        out = np.full(self.n, np.nan, dtype=np.float32)
        if direction < 0:
            d = (c[1:] - c[:-1]).astype(np.float32)
            out[1:] = np.where(same_prev[1:], d, np.nan)
        else:
            d = (c[1:] - c[:-1]).astype(np.float32)
            out[:-1] = np.where(same_prev[1:], d, np.nan)
        out[~self.valid] = np.nan
        return self._unsort(out)

    def position(self) -> tuple[np.ndarray, np.ndarray]:
        """Index of each row inside its entity's run, and the run length."""
        grp = self.comp // SPAN
        starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
        lengths = np.diff(np.r_[starts, self.n])
        run_start = np.repeat(starts, lengths)
        run_len = np.repeat(lengths, lengths)
        idx = np.arange(self.n) - run_start
        return self._unsort(idx), self._unsort(run_len)

    def _unsort(self, sorted_values: np.ndarray) -> np.ndarray:
        return sorted_values[self.rank]


def distinct_in_window(key: np.ndarray, other: np.ndarray, ts: np.ndarray,
                       back: int, ahead: int = 0) -> np.ndarray:
    """Distinct ``other`` values among rows of the same ``key`` in [t - back, t] (or
    [t, t + ahead] when ``back`` is 0). Polars evaluates the rolling windows natively."""
    import polars as pl

    if back and ahead:
        raise ValueError("use one direction per call")
    ts = np.asarray(ts, dtype=np.int64)
    sign = 1 if back else -1
    width = back or ahead
    df = pl.DataFrame({
        "i": np.arange(len(ts), dtype=np.int64),
        "k": np.asarray(key, dtype=np.int64),
        "o": np.asarray(other, dtype=np.int64),
        "t": sign * ts,
    }).sort("k", "t")
    agg = df.rolling(index_column="t", period=f"{width + 1}i", group_by="k").agg(
        pl.col("o").n_unique().alias("d"), pl.col("i").last().alias("i_last")
    )
    # rolling keeps one output per input row, in the frame's order
    out = np.empty(len(ts), dtype=np.float32)
    out[df["i"].to_numpy()] = agg["d"].to_numpy().astype(np.float32)
    bad = np.asarray(key) < 0
    out[bad] = np.nan
    return out
