"""Event-level sibling separability on synthetic templates."""

from __future__ import annotations

import numpy as np
import pytest

from ringtrace.audit import lookalike


def test_event_table_aggregates_per_event():
    event = np.array([5, 5, 9, -1, 9, 9])
    x = np.array([[1.0], [3.0], [2.0], [100.0], [2.0], [5.0]])
    ts = np.array([0, 60, 10, 0, 20, 40])
    ids, table, cols = lookalike.event_table(event, x, ts, ["v"])
    assert ids.tolist() == [5, 9]
    assert cols == ["size", "span", "mean_v", "std_v"]
    assert table[:, 0].tolist() == [2, 3] and table[:, 1].tolist() == [60, 30]
    assert table[:, 2] == pytest.approx([2.0, 3.0])
    assert table[0, 3] == pytest.approx(1.0)


def test_event_labels_majority_and_missing():
    event = np.array([1, 1, 1, 2, 2, 3])
    y = np.array([4, 5, 5, 4, 0, 0])
    ids, lab = lookalike.event_labels(event, y, (4, 5))
    assert ids.tolist() == [1, 2, 3] and lab.tolist() == [1, 0, -1]


def _template(rng, n_events, separable):
    rows_event, rows_x, rows_y, rows_ts = [], [], [], []
    for e in range(n_events):
        typ = rng.random() < 0.3
        size = rng.integers(2, 5)
        for i in range(size):
            shift = 2.0 if (separable and typ) else 0.0
            rows_event.append(e)
            rows_x.append([rng.normal() + shift, rng.normal()])
            rows_y.append(5 if typ else 4)
            rows_ts.append(e * 1000 + i * 60)
    return (np.array(rows_event), np.array(rows_x), np.array(rows_ts), np.array(rows_y))


@pytest.mark.parametrize("separable", [True, False])
def test_audit_template(separable):
    rng = np.random.default_rng(3)
    event, x, ts, y = _template(rng, 1500, separable)
    res = lookalike.audit_template(
        event, x, ts, ["a", "b"], y, (4, 5), np.zeros(len(y)), seeds=(0,), folds=3
    )
    assert res["events"] == 1500
    if separable:
        assert res["all"]["auc"] > 0.9
    else:
        assert abs(res["all"]["auc"] - 0.5) < 0.1
