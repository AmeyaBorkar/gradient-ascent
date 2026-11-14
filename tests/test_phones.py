"""Device counter on a synthetic number line: bot block, two fraud phases, dense legit blocks."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ringtrace.evidence import phones
from ringtrace.tables import HOME_DEVICES

K = phones.K
# skipped numbers per phone by segment (spec-counter 2a), and the multi-row share
GAP_MEAN = (0.87, 1.84, 1.22, 0.41, 2.81, 0.78, 1.97, 1.04, 0.80, 1.00, 1.17)
MULTI = {phones.SIM: 0.55, phones.ATO: 0.79}


def _phase(rng, start: int, per_seg: int, split: int) -> tuple[list, int, dict]:
    """Rows (device, split, anchor, true segment) of one fraud phase; returns rows, end, truth."""
    probs = np.exp(phones.emission_table())
    rows, n = [], start
    truth: dict[int, int] = {}
    for k in range(K):
        for i in range(per_seg):
            if i == 0 and k == phones.ATO:
                n += 6  # the sim/ato cut sits on a run of unused numbers
            elif not (k == 0 and i == 0):
                n += rng.geometric(1 / (1 + GAP_MEAN[k])) - 1
            # the first phone of a phase is a one-row phone; sim/ato hold the multi-row phones
            size = rng.integers(2, 5) if (i > 0 and rng.random() < MULTI.get(k, 0)) else 1
            for _ in range(size):
                rows.append((n, split, rng.choice(K + 1, p=probs[k]), k))
            truth[n] = k
            n += 1
    # the phase ends on an unused number
    return rows, n, truth


def synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    # home phones: several rows each, both splits
    for d in range(1, 301):
        for _ in range(rng.integers(1, 4)):
            rows.append((d, int(rng.random() < 0.3), 0, -1))
    # bot block: sparse multi-row devices shared by both splits
    bots = np.sort(rng.choice(np.arange(HOME_DEVICES + 1, HOME_DEVICES + 3000), 30, replace=False))
    for d in bots:
        for _ in range(rng.integers(10, 30)):
            rows.append((int(d), int(rng.random() < 0.2), rng.choice([0, 2]), -1))
    f0 = HOME_DEVICES + 3000
    train, end, truth_tr = _phase(rng, f0, 40, 0)
    rows += train
    legit_tr = (end + 1, end + 400)
    rows += [(d, 0, 0, -1) for d in range(legit_tr[0], legit_tr[1] + 1)]
    test, end2, truth_te = _phase(rng, legit_tr[1] + 1, 30, 1)
    rows += test
    legit_te = (end2 + 1, end2 + 150)
    rows += [(d, 1, 0, -1) for d in range(legit_te[0], legit_te[1] + 1)]
    order = rng.permutation(len(rows))  # row order must not matter
    arr = np.array(rows, dtype=np.int64)[order]
    # a fraud phase ends on its last unused number (spec-counter 1: B1, B2)
    edges = {
        "f0": f0,
        "train": (f0, end, legit_tr[1]),
        "test": (legit_tr[1] + 1, end2, legit_te[1]),
    }
    return arr, edges


@pytest.fixture(scope="module")
def case():
    arr, edges = synthetic()
    led = pl.DataFrame({"device": arr[:, 0], "is_test": arr[:, 1].astype(bool)})
    out = phones.build(led, anchor=arr[:, 2])
    return arr, edges, out


def test_blocks(case):
    arr, edges, out = case
    dev, split = arr[:, 0], arr[:, 1]
    blocks = phones.find_blocks(dev, split)
    assert blocks.f0 == edges["f0"]
    assert blocks.ranges[0] == edges["train"]
    assert blocks.ranges[1] == edges["test"]
    want = np.zeros(len(dev), dtype=np.int8)
    want[(dev > HOME_DEVICES) & (dev < edges["f0"])] = 1
    for s, name in ((0, "train"), (1, "test")):
        lo, edge, hi = edges[name]
        want[(split == s) & (dev >= lo) & (dev <= edge)] = 2
        want[(split == s) & (dev > edge) & (dev <= hi)] = 3
    assert np.array_equal(out["ctr_block"].to_numpy(), want)


def test_isolated_hole_in_legit_block_is_ignored():
    arr, edges = synthetic()
    lo, edge, hi = edges["train"]
    hole = edge + 200  # drop one legit phone deep inside the dense block
    keep = arr[:, 0] != hole
    blocks = phones.find_blocks(arr[keep, 0], arr[keep, 1])
    assert blocks.ranges[0] == edges["train"]


def test_segments(case):
    arr, edges, out = case
    truth, split = arr[:, 3], arr[:, 1]
    block = out["ctr_block"].to_numpy()
    seg = out["ctr_seg"].to_numpy()
    fb = block == 2
    assert np.isnan(seg[~fb]).all()
    assert np.isnan(out["ctr_pos"].to_numpy()[~fb]).all()
    for s in (0, 1):
        m = fb & (split == s)
        assert (truth[m] >= 0).all()
        agree = (seg[m] == truth[m]).mean()
        assert agree >= 0.9, (s, agree)
        # segments are monotone in the device number
        dev = arr[m, 0]
        order = np.argsort(dev, kind="stable")
        assert (np.diff(seg[m][order]) >= 0).all()
        pos = out["ctr_pos"].to_numpy()[m]
        assert pos.min() == 0.0 and pos.max() == 1.0


def test_sim_ato_cut(case):
    arr, edges, out = case
    truth = arr[:, 3]
    side = out["ctr_ato_vs_sim"].to_numpy()
    multi = out["ctr_dev_rows"].to_numpy() > 1
    zone = np.isin(truth, (phones.SIM, phones.ATO)) & (out["ctr_block"].to_numpy() == 2)
    m = zone & multi
    assert m.sum() > 50
    assert np.isfinite(side[m]).all()
    assert (side[m] == (truth[m] == phones.ATO)).mean() >= 0.98
    # outside the sim/ato zone the cut is not defined
    assert np.isnan(side[out["ctr_block"].to_numpy() != 2]).all()


def test_posteriors(case):
    _, _, out = case
    fb = out["ctr_block"].to_numpy() == 2
    post = out.select([f"ctr_p{k:02d}" for k in range(K)]).to_numpy()[fb]
    assert np.allclose(post.sum(axis=1), 1.0)
    seg = out["ctr_seg"].to_numpy()[fb].astype(int)
    assert np.allclose(out["ctr_conf"].to_numpy()[fb], post[np.arange(len(seg)), seg])


def test_device_counts(case):
    arr, _, out = case
    dev, split = arr[:, 0], arr[:, 1]
    rows = out["ctr_dev_rows"].to_numpy()
    single = out["ctr_dev_single"].to_numpy()
    for i in range(0, len(dev), 97):
        assert rows[i] == (dev == dev[i]).sum()
        assert single[i] == int(((dev == dev[i]) & (split == split[i])).sum() == 1)


def test_order_independent():
    arr, _ = synthetic(seed=11)
    led = pl.DataFrame({"device": arr[:, 0], "is_test": arr[:, 1].astype(bool)})
    a = phones.build(led, anchor=arr[:, 2])
    perm = np.random.default_rng(0).permutation(len(arr))
    b = phones.build(led[perm], anchor=arr[perm, 2])
    inv = np.argsort(perm)
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()[inv]
        assert np.allclose(x, y, equal_nan=True), c
