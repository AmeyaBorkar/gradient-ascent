import numpy as np
import polars as pl
import pytest

from ringtrace import density
from ringtrace.tables import HOME_DEVICES


def _pass(rng, n_fraud=120, n_legit=300, months=(7, 8, 9)):
    """A synthetic shared pass: a sparse fraud phase, a dense legit block, months interleaved."""
    start = HOME_DEVICES + 3_000
    gaps = rng.integers(0, 4, n_fraud)
    fraud = start + np.cumsum(gaps + 1)
    legit = fraud[-1] + 2 + np.arange(n_legit)
    nums = np.concatenate([fraud, legit])
    block = np.concatenate([np.full(n_fraud, 2), np.full(n_legit, 3)])
    month = rng.choice(months, len(nums))
    return nums, block, month


def test_shared_line_keeps_own_gaps_and_the_transition():
    rng = np.random.default_rng(0)
    nums, block, month = _pass(rng)
    line = density.shared_line(nums, block, np.ones(len(nums), bool))
    assert np.array_equal(line.nums, np.sort(nums))
    assert line.gaps[0] == 0
    assert np.array_equal(line.gaps[1:], np.diff(np.sort(nums)) - 1)
    assert line.transition == 1
    assert (line.gaps[line.blocks == 3][1:] == 0).all()


def test_renumber_drops_other_months_and_keeps_order_and_gaps():
    rng = np.random.default_rng(1)
    nums, block, month = _pass(rng)
    line = density.shared_line(nums, block, np.ones(len(nums), bool))
    chosen = nums[month == 9]
    old, new = density.renumber(line, chosen, base=10_000_000)
    assert np.array_equal(old, np.sort(chosen))
    assert (np.diff(new) > 0).all()
    assert new[0] == 10_000_000
    steps = np.diff(new) - 1
    own = line.gaps[np.isin(line.nums, chosen)][1:]
    b = line.blocks[np.isin(line.nums, chosen)]
    first_legit = np.flatnonzero(b == 3)[0]
    # every phone keeps its own gap; the first legit phone at least the shared transition
    same = np.arange(1, len(old)) != first_legit
    assert np.array_equal(steps[same], own[same])
    assert steps[first_legit - 1] >= line.transition
    # the legit block stays dense
    legit_new = new[b == 3]
    assert (np.diff(legit_new) == 1).all()


def test_renumber_of_nothing_is_empty():
    line = density.shared_line(np.array([HOME_DEVICES + 10]), np.array([2]), np.array([True]))
    old, new = density.renumber(line, np.array([], dtype=np.int64))
    assert len(old) == 0 and len(new) == 0


def _frame(rng, month_keep=1.0):
    """Rows: one per pass phone (two on a few fraud phones), in a shared train pass."""
    nums, block, month = _pass(rng, n_fraud=200, n_legit=600)
    two = (block == 2) & (rng.random(len(nums)) < 0.3)
    device = np.concatenate([nums, nums[two]])
    period = np.concatenate([month, month[two]])
    prod_block = np.concatenate([block, block[two]])
    # anchors: fraud-phase phones carry their zone class (11 zones in number order)
    zone = np.zeros(len(nums), np.int64)
    fp = np.flatnonzero(block == 2)
    zone[fp] = 1 + (np.arange(len(fp)) * 11) // len(fp)
    anchor = np.concatenate([zone, zone[two]])
    anchor = np.where(rng.random(len(anchor)) < 0.5, anchor, 0)
    keep = rng.random(len(device)) < month_keep
    return device, period, prod_block, anchor, keep


def test_alone_rebuilds_blocks_and_ranks_inside_the_month():
    rng = np.random.default_rng(2)
    device, period, prod_block, anchor, _ = _frame(rng)
    train = np.ones(len(device), bool)
    line = density.shared_line(device, prod_block, train)
    cols = density.alone(line, device, period, train, anchor, 9)
    mine = period == 9
    assert len(cols["block"]) == int(mine.sum())
    # the month alone finds the same fraud phase and legit block as the shared pass
    assert (cols["block"] == prod_block[mine]).mean() > 0.97
    pos = cols["pos"][cols["block"] == 2]
    assert np.nanmin(pos) == 0.0 and np.nanmax(pos) == 1.0
    # ranks are taken among the month's own phones
    fp = cols["block"] == 2
    n_phones = len(np.unique(device[mine][fp]))
    assert len(np.unique(pos)) == n_phones


def test_alone_on_a_thinned_month_keeps_a_dense_legit_block():
    rng = np.random.default_rng(3)
    device, period, prod_block, anchor, keep = _frame(rng, month_keep=0.5)
    line = density.shared_line(device, prod_block, np.ones(len(device), bool))
    # the frame: every other month whole, the held month thinned
    in_frame = keep | (period != 9)
    d, p, a = device[in_frame], period[in_frame], anchor[in_frame]
    cols = density.alone(line, d, p, np.ones(len(d), bool), a, 9)
    pb = prod_block[in_frame][p == 9]
    # without the renumbering the thinned legit block is half empty and reads as a fraud phase
    assert (cols["block"] == pb).mean() > 0.97
    assert ((cols["block"] == 3) == (pb == 3)).mean() > 0.97


def test_replace_rows_changes_only_those_rows():
    base = pl.DataFrame(
        {
            "ctr_block": pl.Series([1, 2, 3], dtype=pl.Int8),
            "ctr_pos": [0.1, 0.2, 0.3],
            "ctr_dev_rows": pl.Series([1, 1, 1], dtype=pl.Int32),
        }
    )
    out = density.replace_rows(
        base,
        np.array([1]),
        {"block": np.array([3]), "pos": np.array([0.9]), "dev_rows": np.array([5])},
    )
    assert out["ctr_block"].to_list() == [1, 3, 3]
    assert out["ctr_pos"].to_list() == pytest.approx([0.1, 0.9, 0.3])
    assert out["ctr_dev_rows"].to_list() == [1, 5, 1]
    assert out.schema == base.schema


def test_swap_counter_takes_only_counter_columns():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    cols = ["a", "ctr_pos", "b"]
    ph = pl.DataFrame({"ctr_pos": [10.0, 11.0, 12.0, 13.0]})
    out = density.swap_counter(X, cols, np.array([1, 3]), ph)
    assert out.shape == (2, 3)
    assert out[:, 1].tolist() == [11.0, 13.0]
    assert out[:, 0].tolist() == [3.0, 9.0]
    assert X[1, 1] == 4.0  # the input is untouched
