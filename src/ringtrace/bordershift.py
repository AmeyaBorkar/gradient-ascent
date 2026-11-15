"""One class border of a fraud phase moved to a chosen position of its posterior.

``evidence.borders`` gives every border a posterior over positions on the number line. Phones
order the phase, so a border's position is a gap between two neighbouring phones, and the
posterior mass of the positions inside one gap is the probability that the border sits there.
``border_shift`` places border ``border`` (0..9, between class codes ``border + 1`` and
``border + 2``) at its ``rank``-th most likely gap (0 = the most likely) and gives every row of
the phones inside the border's band the class of its side, when the row's call is one of the two
classes. The band is the phones the border can lie on either side of with probability ``band``
or more. Rows of the same label-free event follow the phone's new class.

Rows near one border share one answer: only the border's position is unknown. ``only_test`` limits
the rule to the test phase, whose borders are the uncertain ones (it is a sixth of train's).
"""

from __future__ import annotations

import numpy as np

from ringtrace.evidence import anchors, borders, phones
from ringtrace.evidence.phones import K


def gap_mass(cut: np.ndarray, at: np.ndarray) -> np.ndarray:
    """``mass[g]``: probability that exactly ``g`` phones lie below the border, from the border's
    posterior over numbers ``cut[c]`` and the phones' positions ``at`` (sorted)."""
    g = np.searchsorted(at, np.arange(len(cut)), side="left")
    return np.bincount(g, weights=cut, minlength=len(at) + 1)


def border_shift(
    pred,
    proba,
    led,
    log=print,
    border: int = 0,
    rank: int = 0,
    band: float = 0.01,
    only_test: bool = True,
    sigma: float | None = borders.SIGMA,
) -> np.ndarray:
    from ringtrace import rules

    if not 0 <= int(border) < K - 1:
        raise ValueError(f"border {border} is not one of 0..{K - 2}")
    a, b = int(border) + 1, int(border) + 2
    pred = np.asarray(pred, dtype=np.int64)
    out = pred.copy()
    device = led["device"].to_numpy().astype(np.int64)
    split = led["is_test"].to_numpy().astype(np.int64)
    anchor = np.asarray(anchors.row_anchor(led), dtype=np.int64)
    blocks = phones.find_blocks(device, split)
    block = phones.block_codes(device, split, blocks)
    moved = np.zeros(len(pred), dtype=bool)
    for s, (lo, edge, _hi) in sorted(blocks.ranges.items()):
        if only_test and s != 1:
            continue
        rows = np.flatnonzero((block == 2) & (split == s))
        if len(rows) == 0:
            continue
        nums, pinv = np.unique(device[rows], return_inverse=True)
        counts = np.zeros((len(nums), K + 1))
        np.add.at(counts, (pinv, anchor[rows]), 1.0)
        got = borders.decode(nums, counts, lo, edge, borders.LENGTH_SHARE if sigma else None, sigma)
        mass = gap_mass(got["cut"][int(border)], nums - lo)
        order = np.argsort(-mass, kind="stable")
        if int(rank) >= len(order) or mass[order[int(rank)]] <= 0:
            continue
        g = int(order[int(rank)])
        below = np.cumsum(mass)  # below[i] = P(border has at most i phones below it)
        first = int(np.searchsorted(below, band, side="left"))
        last = int(np.searchsorted(below, 1.0 - band, side="left"))
        idx = pinv  # phone index of each phase row
        in_band = (idx >= first) & (idx < max(last, first))
        call = pred[rows]
        down = in_band & (idx < g) & (call == b)
        up = in_band & (idx >= g) & (call == a)
        out[rows[down]] = a
        out[rows[up]] = b
        moved[rows[down | up]] = True
        log(
            f"  border {a}|{b} split {s}: gap rank {rank} holds {mass[g]:.3f} "
            f"(top {mass[order[0]]:.3f}), band {max(last - first, 0)} phones, "
            f"rows to {a}: {int(down.sum())}, rows to {b}: {int(up.sum())}"
        )
    if moved.any():
        event = rules.label_free_events(led)
        ev = np.unique(event[moved & (event >= 0)])
        if len(ev):
            for e in ev:
                inside = event == e
                new = out[moved & inside][0]
                follow = inside & ~moved & np.isin(pred, (a, b)) & (pred != new)
                out[follow] = new
    is_test = led["is_test"].to_numpy()
    ch = out != pred
    log(
        f"  border shift {a}|{b}: train {int((ch & ~is_test).sum())} / "
        f"test {int((ch & is_test).sum())}"
    )
    return out
