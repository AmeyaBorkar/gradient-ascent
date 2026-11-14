"""The mule/bot edge rule moves only zone rows in the mule segment that are called velocity_bot."""

from __future__ import annotations

import numpy as np
import polars as pl

from ringtrace import rules


def test_mule_edge_moves_only_zone_rows_called_bot_in_the_mule_segment(monkeypatch):
    seg = np.array([1, 1, 1, 1, 2, 0])  # counter segment class per row (0: none)
    tmpl = np.array([0, 0, 3, 0, 0, 0])  # row 2 sits in a template
    anchor = np.array([0, 0, 0, 1, 0, 0])  # row 3 carries a row anchor
    monkeypatch.setattr(
        rules,
        "_counter",
        lambda led: {
            "block": np.full(6, 2),
            "seg_class": seg,
            "conf": np.ones(6),
            "dev_rows": np.ones(6),
        },
    )
    monkeypatch.setattr(rules.anchors, "templates", lambda led: (tmpl, np.zeros(6)))
    monkeypatch.setattr(rules.anchors, "row_anchor", lambda led: anchor)
    led = pl.DataFrame({"is_test": [False, True, True, True, True, True]})
    pred = np.array([2, 2, 2, 2, 2, 2])  # every row called velocity_bot
    out = rules.mule_edge(pred, None, led, log=lambda *_: None)
    assert out.tolist() == [1, 1, 2, 2, 2, 2]


def test_mule_edge_is_registered():
    assert "mule_edge" in rules.RULES
