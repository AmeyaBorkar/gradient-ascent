"""The submission validator accepts a well-formed file and names every problem in a bad one."""

from __future__ import annotations

import polars as pl

from ringtrace.build import validate


def test_validate(tmp_path):
    sample = tmp_path / "sample.csv"
    pl.DataFrame({"id": ["TXa", "TXb", "TXc"], "label": ["legit"] * 3}).write_csv(sample)
    good = tmp_path / "good.csv"
    pl.DataFrame({"id": ["TXa", "TXb", "TXc"],
                  "label": ["legit", "qr_swap", "sim_swap"]}).write_csv(good)
    assert validate(good, sample) == []
    bad = tmp_path / "bad.csv"
    pl.DataFrame({"id": ["TXb", "TXa", "TXc"],
                  "label": ["legit", "qr-swap", "legit"]}).write_csv(bad)
    problems = validate(bad, sample)
    assert any("order" in p for p in problems)
    assert any("unknown labels" in p for p in problems)
