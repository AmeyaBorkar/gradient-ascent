"""Post-decode rules over the counter and the look-alike templates (spec-counter s4).

``apply(rules, pred, proba, led)`` runs the named rules in order on the decoded calls of every row
(train and test) and returns the new calls. A rule is a name or a dict ``{"name": ..., **params}``.

* ``counter``: every fraud-block row is fraud; in the first seven segments (mule .. qr) a confident
  segment (``ctr_conf >= min_conf``) sets the type, elsewhere the model keeps its fraud type.
* ``legit_block``: a one-row phone in a legit block is legit.
* ``plain_templates``: inside each of the six look-alike templates, every call of the template's
  pair takes the template's class.
* ``blank_merchant_qr``: a QR payment to a merchant with a blank QR check, called legit, takes its
  most likely fraud class.
* ``anchored_templates``: as ``plain_templates``, except that a template row on a fraud-block phone
  whose counter segment is one of the pair (``ctr_conf >= min_conf``) keeps its segment class and
  hands it to the rest of its family event.

Rules only read label-free columns; NaN never triggers one. Each logs the rows it changed.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace.evidence import anchors
from ringtrace.evidence.anchors import TEMPLATE_CLASS, TEMPLATE_NAMES, TEMPLATE_PAIR

TRUSTED_SEGMENTS = 7  # mule .. qr; the last four segments are unreliable in test
COUNTER_COLS = ["ctr_block", "ctr_seg", "ctr_conf", "ctr_dev_rows"]


def _counter(led: pl.DataFrame) -> dict[str, np.ndarray]:
    c = anchors.family_columns(led, COUNTER_COLS)
    block = c["ctr_block"].fill_null(0).to_numpy().astype(np.int64)
    seg = c["ctr_seg"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    conf = c["ctr_conf"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    rows = c["ctr_dev_rows"].fill_null(0).to_numpy().astype(np.int64)
    seg_class = np.where(np.isfinite(seg), np.nan_to_num(seg, nan=-1.0) + 1, 0).astype(np.int64)
    return {
        "block": block,
        "seg_class": seg_class,
        "conf": np.nan_to_num(conf, nan=-1.0),
        "dev_rows": rows,
    }


def _split_counts(mask: np.ndarray, is_test: np.ndarray) -> str:
    return f"train {int((mask & ~is_test).sum())} / test {int((mask & is_test).sum())}"


def counter_rule(
    pred, proba, led, log=print, min_conf: float = 0.9, trusted: int = TRUSTED_SEGMENTS
) -> np.ndarray:
    c = _counter(led)
    out = pred.copy()
    fb = c["block"] == 2
    called_legit = fb & (out == 0)
    if called_legit.any():
        out[called_legit] = np.asarray(proba)[called_legit, 1:].argmax(axis=1) + 1
    sure = fb & (c["seg_class"] >= 1) & (c["seg_class"] <= trusted) & (c["conf"] >= min_conf)
    out[sure] = c["seg_class"][sure]
    return out


def legit_block_rule(pred, proba, led, log=print) -> np.ndarray:
    c = _counter(led)
    out = pred.copy()
    out[(c["block"] == 3) & (c["dev_rows"] == 1)] = 0
    return out


def plain_templates(pred, proba, led, log=print) -> np.ndarray:
    tmpl = anchors.template_of(led)
    is_test = led["is_test"].to_numpy()
    out = pred.copy()
    for t, pair in TEMPLATE_PAIR.items():
        hit = (tmpl == t) & np.isin(pred, pair)
        out[hit] = TEMPLATE_CLASS[t]
        log(
            f"  {TEMPLATE_NAMES[t]:<8} pair calls {_split_counts(hit, is_test)}, "
            f"changed {_split_counts(hit & (out != pred), is_test)}"
        )
    return out


def _event_class(event: np.ndarray, cls: np.ndarray, conf: np.ndarray) -> dict[int, int]:
    """Majority anchor class of each event (ties: larger summed confidence)."""
    if len(event) == 0:
        return {}
    votes = (
        pl.DataFrame({"ev": event, "c": cls, "w": conf})
        .group_by("ev", "c")
        .agg(pl.len().alias("n"), pl.col("w").sum())
        .sort(["ev", "n", "w", "c"], descending=[False, True, True, False])
        .unique("ev", keep="first", maintain_order=True)
    )
    return dict(zip(votes["ev"].to_list(), votes["c"].to_list(), strict=True))


def anchored_templates(
    pred, proba, led, log=print, min_conf: float = 0.8, classes: dict | None = None
) -> np.ndarray:
    """``classes`` overrides the class a template gives its unanchored rows, e.g. ``{"4": 6}``;
    rows held by a counter anchor keep their segment class."""
    cls_of = {**TEMPLATE_CLASS, **{int(k): int(v) for k, v in (classes or {}).items()}}
    for t, c in cls_of.items():
        if c not in TEMPLATE_PAIR[t]:
            raise ValueError(f"template {t}: class {c} is not one of its pair {TEMPLATE_PAIR[t]}")
    tmpl, event = anchors.templates(led)
    c = _counter(led)
    is_test = led["is_test"].to_numpy()
    out = pred.copy()
    for t, pair in TEMPLATE_PAIR.items():
        inside = tmpl == t
        anchor = (
            inside & (c["block"] == 2) & np.isin(c["seg_class"], pair) & (c["conf"] >= min_conf)
        )
        by_event = _event_class(event[anchor], c["seg_class"][anchor], c["conf"][anchor])
        target = np.full(len(pred), cls_of[t], dtype=np.int64)
        if by_event:
            ev_ids = np.fromiter(by_event.keys(), dtype=np.int64)
            ev_cls = np.fromiter(by_event.values(), dtype=np.int64)
            order = np.argsort(ev_ids)
            ev_ids, ev_cls = ev_ids[order], ev_cls[order]
            pos = np.clip(np.searchsorted(ev_ids, event), 0, len(ev_ids) - 1)
            held = inside & (ev_ids[pos] == event)
            target[held] = ev_cls[pos[held]]
        target[anchor] = c["seg_class"][anchor]
        hit = inside & np.isin(pred, pair)
        out[hit] = target[hit]
        log(
            f"  {TEMPLATE_NAMES[t]:<8} anchors {_split_counts(anchor, is_test)}, "
            f"anchored events {len(by_event)}, "
            f"changed {_split_counts(hit & (out != pred), is_test)}"
        )
    return out


def blank_merchant_qr(pred, proba, led, log=print) -> np.ndarray:
    """A QR payment to a merchant whose QR check is blank is generator-made (every such train row
    is fraud on the cleaned labels); one called legit takes its most likely fraud class."""
    hit = (
        (led["txn_type"] == "QR").to_numpy()
        & (led["receiver_type"] == "MERCHANT").to_numpy()
        & led["qr_vpa_mismatch"].is_null().to_numpy()
        & (pred == 0)
    )
    out = pred.copy()
    if hit.any():
        out[hit] = np.asarray(proba)[hit, 1:].argmax(axis=1) + 1
    log(f"  blank merchant QR called legit {_split_counts(hit, led['is_test'].to_numpy())}")
    return out


RULES = {
    "blank_merchant_qr": blank_merchant_qr,
    "counter": counter_rule,
    "legit_block": legit_block_rule,
    "plain_templates": plain_templates,
    "anchored_templates": anchored_templates,
}


def apply(
    rules: list, pred: np.ndarray, proba: np.ndarray, led: pl.DataFrame, log=print
) -> np.ndarray:
    """Run ``rules`` in order on the calls ``pred`` (class codes, one per ledger row)."""
    pred = np.asarray(pred, dtype=np.int64).copy()
    is_test = led["is_test"].to_numpy()
    for rule in rules:
        if isinstance(rule, str):
            name, params = rule, {}
        else:
            params = dict(rule)
            name = params.pop("name")
        if name not in RULES:
            raise KeyError(f"unknown rule {name!r}; known: {sorted(RULES)}")
        log(f"[rules] {name} {params or ''}".rstrip())
        new = RULES[name](pred, proba, led, log=log, **params)
        log(f"[rules] {name}: changed {_split_counts(new != pred, is_test)} rows")
        pred = new
    return pred
