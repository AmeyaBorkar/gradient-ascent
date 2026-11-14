"""Device-order class repair over the fraud phase of the device counter.

The phones of a split's fraud phase are minted class by class in label-list order (mule first,
collusion last), so along the number line the fraud types never go back. A call can look right
on its own rows and still sit between two neighbouring phones whose classes rule it out. The
repair retypes such calls, and nothing else:

1. **Shape sets.** Every row gets one event shape: its look-alike template, else the shape of
   the first evidence family that places it in an event, else its first one-class mark, else
   none. From the fraud-labelled fit rows, a shape allows every class with at least
   ``min_share`` of its rows, and always its most common class. A row with no shape (or a shape
   never seen on a fit row) allows every fraud class.
2. **Device sets.** A phone allows the classes that all of its rows' shapes allow.
3. **Anchors.** Phones of one fraud phase are ordered by number. A phone is anchored when its
   own rows carry a class the counter confirms (``anchor_class``); among the anchored phones the
   longest run whose classes never decrease is kept (``longest_nondecreasing``) and the others
   are dropped.
4. **Intervals.** A kept anchor allows its own class; any other phone allows the classes between
   the kept anchors on either side of it (the first and the last class at the ends), intersected
   with its device set.
5. **Retype.** A fraud call moves only when its class is outside a non-empty feasible set; fraud
   never becomes legit or the reverse. Rows of one event (or, outside events, of one phone) with
   the same original class move together, under the intersection of their feasible sets.
   ``conservative`` moves a group only when exactly one class is feasible; ``full`` also moves it
   when several are, to the feasible class with the highest summed decoder value.

Classes are held as bit masks: bit ``c - 1`` is fraud class ``c`` (1..11).
"""

from __future__ import annotations

import bisect

import numpy as np

K = 12
NF = K - 1
ALL = (1 << NF) - 1
MIN_SHARE = 0.015
MIN_CONF = 0.80
VARIANTS = ("conservative", "full")


def bit(c) -> np.ndarray:
    """Mask of fraud class ``c`` (scalar or array); 0 for class 0."""
    c = np.asarray(c, dtype=np.int64)
    return np.where(c > 0, np.left_shift(1, np.maximum(c, 1) - 1), 0)


def span(lo, hi) -> np.ndarray:
    """Mask of the fraud classes ``lo..hi`` (inclusive)."""
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    return np.left_shift(1, hi) - np.left_shift(1, lo - 1)


def popcount(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.int64)
    out = np.zeros(m.shape, dtype=np.int64)
    for c in range(NF):
        out += (m >> c) & 1
    return out


def lowest_class(m) -> np.ndarray:
    """The smallest class in each mask (0 for an empty mask)."""
    m = np.asarray(m, dtype=np.int64)
    out = np.zeros(m.shape, dtype=np.int64)
    for c in range(NF, 0, -1):
        out = np.where((m >> (c - 1)) & 1, c, out)
    return out


# ---------------------------------------------------------------------------------------------
# 1-2: shape sets and device sets


def shape_key(tmpl: np.ndarray, shapes: list[np.ndarray], marks: list[np.ndarray]) -> np.ndarray:
    """One event-shape code per row: ``100 + t`` for template t, else ``1000 * (f + 1) + s`` for
    shape s of the first family f (in the order given) that has one, else ``100000 + j`` for the
    first one-class mark j that is set, else 0."""
    tmpl = np.asarray(tmpl, dtype=np.int64)
    key = np.where(tmpl > 0, 100 + tmpl, 0)
    for f, s in enumerate(shapes):
        s = np.asarray(s, dtype=np.int64)
        free = (key == 0) & (s > 0)
        key[free] = 1000 * (f + 1) + s[free]
    for j, m in enumerate(marks):
        free = (key == 0) & (np.asarray(m) > 0)
        key[free] = 100_000 + j
    return key


def shape_table(
    key: np.ndarray, labels: np.ndarray, fit: np.ndarray, min_share: float = MIN_SHARE
) -> dict[int, int]:
    """Feasible-class mask of every shape seen on a fraud-labelled fit row."""
    rows = np.flatnonzero(fit & (labels > 0) & (key > 0))
    if len(rows) == 0:
        return {}
    uk, inv = np.unique(key[rows], return_inverse=True)
    cnt = np.zeros((len(uk), NF))
    np.add.at(cnt, (inv, labels[rows] - 1), 1.0)
    share = cnt / cnt.sum(axis=1, keepdims=True)
    keep = share >= min_share
    keep[np.arange(len(uk)), share.argmax(axis=1)] = True
    masks = (keep * (1 << np.arange(NF))).sum(axis=1)
    return {int(k): int(m) for k, m in zip(uk, masks, strict=True)}


def shape_masks(key: np.ndarray, table: dict[int, int]) -> np.ndarray:
    """Per-row mask from ``shape_table``; rows with no shape or an unseen shape allow all."""
    out = np.full(len(key), ALL, dtype=np.int64)
    for k, m in table.items():
        out[key == k] = m
    return out


def group_and(group: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """AND of ``mask`` over the rows of each group, returned per row."""
    if len(group) == 0:
        return np.zeros(0, dtype=np.int64)
    _, inv = np.unique(group, return_inverse=True)
    o = np.argsort(inv, kind="stable")
    starts = np.flatnonzero(np.r_[True, inv[o][1:] != inv[o][:-1]])
    red = np.bitwise_and.reduceat(np.asarray(mask, dtype=np.int64)[o], starts)
    return red[inv]


# ---------------------------------------------------------------------------------------------
# 3-4: anchors along the number line and the intervals between them


def longest_nondecreasing(c: np.ndarray) -> np.ndarray:
    """Mask of one longest non-decreasing subsequence of ``c`` (deterministic: among equally
    long ones, each element is taken as early as the patience sort allows)."""
    c = np.asarray(c, dtype=np.int64)
    n = len(c)
    tails: list[int] = []
    tail_idx: list[int] = []
    prev = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        j = bisect.bisect_right(tails, int(c[i]))
        if j > 0:
            prev[i] = tail_idx[j - 1]
        if j == len(tails):
            tails.append(int(c[i]))
            tail_idx.append(i)
        else:
            tails[j] = int(c[i])
            tail_idx[j] = i
    keep = np.zeros(n, dtype=bool)
    i = tail_idx[-1] if tail_idx else -1
    while i >= 0:
        keep[i] = True
        i = prev[i]
    return keep


def interval_masks(anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phones in number order with their anchor class (0 none): the feasible mask of each phone
    from the kept anchors, and which anchors were kept."""
    anchor = np.asarray(anchor, dtype=np.int64)
    n = len(anchor)
    idx = np.flatnonzero(anchor > 0)
    kept = np.zeros(n, dtype=bool)
    kept[idx[longest_nondecreasing(anchor[idx])]] = True
    cls = np.where(kept, anchor, 0)
    # previous kept class (1 before the first), next kept class (NF after the last)
    pos = np.arange(n)
    last = np.maximum.accumulate(np.where(kept, pos, -1))
    prev = np.where(last >= 0, cls[np.maximum(last, 0)], 1)
    nxt_pos = np.minimum.accumulate(np.where(kept, pos, n)[::-1])[::-1]
    nxt = np.where(nxt_pos < n, cls[np.minimum(nxt_pos, n - 1)], NF)
    mask = np.where(kept, bit(cls), span(prev, nxt))
    return mask, kept


def anchor_class(phone: np.ndarray, row_class: np.ndarray) -> np.ndarray:
    """Per row: the phone's anchor class when its anchored rows all name one class, else 0."""
    row_class = np.asarray(row_class, dtype=np.int64)
    out = np.zeros(len(phone), dtype=np.int64)
    a = row_class > 0
    if not a.any():
        return out
    _, inv = np.unique(phone, return_inverse=True)
    ng = int(inv.max()) + 1
    lo = np.full(ng, K, dtype=np.int64)
    hi = np.zeros(ng, dtype=np.int64)
    np.minimum.at(lo, inv[a], row_class[a])
    np.maximum.at(hi, inv[a], row_class[a])
    one = (lo == hi) & (hi > 0)
    out = np.where(one[inv], hi[inv], 0)
    return out


def phase_masks(
    phone: np.ndarray, anchor_row: np.ndarray, shape_mask: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Feasible mask per row of one fraud phase (all rows given belong to it)."""
    phone = np.asarray(phone, dtype=np.int64)
    dev_shape = group_and(phone, shape_mask)
    a = anchor_class(phone, anchor_row)
    nums, first, inv = np.unique(phone, return_index=True, return_inverse=True)
    mask_p, kept_p = interval_masks(a[first])
    feas = mask_p[inv] & dev_shape
    stats = {
        "phones": len(nums),
        "anchored_phones": int((a[first] > 0).sum()),
        "dropped_anchors": int(((a[first] > 0) & ~kept_p).sum()),
        "singleton_phones": int((popcount(mask_p & dev_shape[first]) == 1).sum()),
        "empty_phones": int(((mask_p & dev_shape[first]) == 0).sum()),
    }
    return feas, stats


# ---------------------------------------------------------------------------------------------
# 5: the retype


def decision_values(proba: np.ndarray, calls: np.ndarray, fraud_scale: float = 0.875) -> np.ndarray:
    """The fixed-point decoder's marginal gain of every class for every row, at the calls given
    (one month's rows): (2 p_ik - F_k) / (N_k + P_k)."""
    p = np.asarray(proba, dtype=np.float64).copy()
    p[:, 1:] /= fraud_scale
    p /= p.sum(axis=1, keepdims=True)
    k = p.shape[1]
    expected = p.sum(axis=0)
    counts = np.bincount(calls, minlength=k).astype(np.float64)
    tp = np.bincount(calls, weights=p[np.arange(len(p)), calls], minlength=k)
    denom = np.maximum(expected + counts, 1e-12)
    f1 = 2 * tp / denom
    return (2 * p - f1[None, :]) / denom[None, :]


def retype(
    calls: np.ndarray,
    feasible: np.ndarray,
    group: np.ndarray,
    value: np.ndarray | None = None,
    variant: str = "conservative",
) -> tuple[np.ndarray, np.ndarray]:
    """New calls and the moved-row mask. ``feasible`` is the per-row mask (``ALL`` where the row
    carries no device-order constraint); ``group`` joins rows that move together (with the same
    original class); ``value`` (n x 12) ranks the feasible classes for ``full``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    calls = np.asarray(calls, dtype=np.int64)
    out = calls.copy()
    moved = np.zeros(len(calls), dtype=bool)
    fr = np.flatnonzero(calls > 0)
    if len(fr) == 0:
        return out, moved
    key = np.asarray(group, dtype=np.int64)[fr] * K + calls[fr]
    _, inv = np.unique(key, return_inverse=True)
    g_mask = group_and(inv, np.asarray(feasible, dtype=np.int64)[fr])
    c = calls[fr]
    bad = (g_mask != 0) & ((g_mask & bit(c)) == 0)
    if variant == "conservative":
        bad &= popcount(g_mask) == 1
        new = lowest_class(g_mask)
    else:
        if value is None:
            raise ValueError("variant 'full' needs decoder values")
        new = lowest_class(g_mask)
        many = bad & (popcount(g_mask) > 1)
        if many.any():
            ng = int(inv.max()) + 1
            tot = np.zeros((ng, K))
            np.add.at(tot, inv[many], np.asarray(value, dtype=np.float64)[fr[many]])
            allowed = ((g_mask[many, None] >> np.arange(NF)[None, :]) & 1).astype(bool)
            v = np.where(allowed, tot[inv[many]][:, 1:], -np.inf)
            new[many] = v.argmax(axis=1) + 1
    rows = fr[bad]
    out[rows] = new[bad]
    moved[rows] = True
    return out, moved


def break_even(truth: np.ndarray, pred: np.ndarray, c: int) -> float:
    """F1 of class ``c`` divided by 2: the precision a new call of ``c`` needs to raise its F1."""
    tp = np.sum((truth == c) & (pred == c))
    den = np.sum(truth == c) + np.sum(pred == c)
    return float(tp / den) if den else 0.0
