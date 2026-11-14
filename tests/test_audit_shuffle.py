"""The order probe keeps every id's label and only moves rows."""

from __future__ import annotations

import polars as pl

from ringtrace.audit import shuffle


def test_shuffled_keeps_content(tmp_path):
    src = tmp_path / "sub.csv"
    pl.DataFrame(
        {"id": [f"T{i:03d}" for i in range(200)], "label": ["legit"] * 190 + ["qr_swap"] * 10}
    ).write_csv(src)
    rep = shuffle.shuffled(src, tmp_path / "probe.csv", seed=4)
    assert rep["same_content"] and rep["rows"] == 200 and rep["rows_moved"] > 150
    again = shuffle.shuffled(src, tmp_path / "probe2.csv", seed=4)
    assert again["sha256_probe"] == rep["sha256_probe"] != rep["sha256_source"]
    a = pl.read_csv(src).sort("id")
    b = pl.read_csv(tmp_path / "probe.csv").sort("id")
    assert a.equals(b)
