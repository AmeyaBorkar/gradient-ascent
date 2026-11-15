"""Template events typed by the counter posterior restricted to the template's pair.

``anchored_templates`` lets a fraud-phase phone anchor its template event only when the counter's
most likely segment is one of the template's two classes with confidence ``min_conf``. A template
event is written by one of its two classes and no other, so the event is evidence about its
phones: ``pair_templates`` conditions each phone's posterior on that pair,
``share_b = p_b / (p_a + p_b)``, and adds the log odds of the event's fraud-phase phones (one
vote per phone, not per row). The event takes the pair's second class when the joint share is at
least ``min_share``, its first class when it is at most ``1 - min_share``, and the template's own
class otherwise. A phone counts only when the pair holds at least ``min_mass`` of its posterior.

``templates`` limits the pair reading to the listed templates; the others are read exactly as
``anchored_templates`` reads them. The rule needs the posterior columns ``ctr_p00 .. ctr_p10``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import anchors
from ringtrace.evidence.anchors import TEMPLATE_CLASS, TEMPLATE_NAMES, TEMPLATE_PAIR

N_SEG = 11
ODDS_CLIP = 1e-6


def pair_share(post: np.ndarray, pair: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``(mass, share_b)`` of a posterior matrix for a pair of class codes (1..11)."""
    a, b = pair
    pa, pb = post[:, a - 1], post[:, b - 1]
    mass = pa + pb
    share = np.divide(pb, mass, out=np.full(len(mass), 0.5), where=mass > 0)
    return mass, share


def event_share(event: np.ndarray, phone: np.ndarray, share: np.ndarray) -> dict[int, float]:
    """Joint share of the second class per event: the log odds of its distinct phones, summed."""
    if len(event) == 0:
        return {}
    s = np.clip(share, ODDS_CLIP, 1.0 - ODDS_CLIP)
    per_phone = (
        pl.DataFrame({"ev": event, "ph": phone, "lo": np.log(s) - np.log1p(-s)})
        .group_by("ev", "ph")
        .agg(pl.col("lo").mean())
        .group_by("ev")
        .agg(pl.col("lo").sum())
    )
    lo = per_phone["lo"].to_numpy()
    return dict(zip(per_phone["ev"].to_list(), (1.0 / (1.0 + np.exp(-lo))).tolist(), strict=True))


def pair_templates(
    pred,
    proba,
    led,
    log=print,
    min_share: float = 0.8,
    min_mass: float = 0.5,
    min_conf: float = 0.8,
    templates: list | None = None,
) -> np.ndarray:
    from ringtrace import rules

    pred = np.asarray(pred, dtype=np.int64)
    out = rules.anchored_templates(pred, proba, led, log=lambda *_: None, min_conf=min_conf)
    chosen = set(TEMPLATE_PAIR) if templates is None else {int(t) for t in templates}
    if not chosen <= set(TEMPLATE_PAIR):
        raise ValueError(f"unknown templates {sorted(chosen - set(TEMPLATE_PAIR))}")
    tmpl, event = anchors.templates(led)
    cols = anchors.family_columns(led, ["ctr_block"] + [f"ctr_p{k:02d}" for k in range(N_SEG)])
    block = cols["ctr_block"].fill_null(0).to_numpy().astype(np.int64)
    post = cols.drop("ctr_block").select(pl.all().cast(pl.Float64)).to_numpy()
    known = (block == 2) & np.isfinite(post).all(axis=1)
    post = np.where(known[:, None], post, 0.0)
    phone = led["device"].to_numpy().astype(np.int64)
    is_test = led["is_test"].to_numpy()
    for t in sorted(chosen):
        pair = TEMPLATE_PAIR[t]
        inside = tmpl == t
        mass, share = pair_share(post, pair)
        voter = inside & known & (mass >= min_mass)
        joint = event_share(event[voter], phone[voter], share[voter])
        target = np.full(len(pred), TEMPLATE_CLASS[t], dtype=np.int64)
        if joint:
            ev_ids = np.fromiter(joint.keys(), dtype=np.int64)
            ev_share = np.fromiter(joint.values(), dtype=np.float64)
            order = np.argsort(ev_ids)
            ev_ids, ev_share = ev_ids[order], ev_share[order]
            pos = np.clip(np.searchsorted(ev_ids, event), 0, len(ev_ids) - 1)
            held = inside & (ev_ids[pos] == event)
            s = ev_share[pos]
            target[held & (s >= min_share)] = pair[1]
            target[held & (s <= 1.0 - min_share)] = pair[0]
        hit = inside & np.isin(pred, pair)
        before = out[hit].copy()
        out[hit] = target[hit]
        moved = np.zeros(len(pred), dtype=bool)
        moved[np.flatnonzero(hit)] = out[hit] != before
        log(
            f"  {TEMPLATE_NAMES[t]:<8} voting phones rows "
            f"train {int((voter & ~is_test).sum())} / test {int((voter & is_test).sum())}, "
            f"events {len(joint)}, against anchored "
            f"train {int((moved & ~is_test).sum())} / test {int((moved & is_test).sum())}"
        )
    return out
