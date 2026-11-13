"""TabR on synthetic CPU problems: retrieval math, self-exclusion, determinism, vs an MLP."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ringtrace.classes import N_CLASSES

pytest.importorskip("torch")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from ringtrace import tabm, tabr  # noqa: E402
from ringtrace.scoring import macro_f1  # noqa: E402


def _spec(**kw):
    base = dict(
        seeds=(0,),
        d=16,
        k=4,
        bank_legit=10_000,
        epochs=3,
        batch=64,
        lr=2e-3,
        weight_decay=3e-4,
        dropout=0.0,
        device="cpu",
        hidden=32,
        quantiles=32,
        query_chunk=32,
        bank_chunk=16,
    )
    base.update(kw)
    return tabr.TabRSpec(**base)


def _labeled(n: int, p: int, seed: int, rare: bool):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    if rare:
        y = np.zeros(n, dtype=np.int64)
        y[: max(n // 5, 12)] = rng.integers(1, N_CLASSES, size=max(n // 5, 12))
    else:
        y = (np.arange(n) % (N_CLASSES - 1)) + 1  # no class 0: every row is in the bank
    X[0, 0] = np.nan
    return X, y.astype(np.int64)


def test_bank_keeps_every_rare_row_and_samples_legit_by_seed():
    y = np.array([0] * 50 + [1] * 4 + [2] * 3 + [0] * 10)
    a = tabr.build_bank(y, bank_legit=12, seed=0)
    b = tabr.build_bank(y, bank_legit=12, seed=0)
    c = tabr.build_bank(y, bank_legit=12, seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    rare = set(np.flatnonzero(y != 0).tolist())
    assert rare.issubset(set(a.tolist()))
    assert int((y[a] == 0).sum()) == 12
    assert int((y[a] != 0).sum()) == len(rare)
    # fewer legit rows than the cap: take them all
    assert set(tabr.build_bank(np.array([0, 1, 0, 4]), bank_legit=100, seed=0).tolist()) == {
        0,
        1,
        2,
        3,
    }
    # a zero cap keeps rares only
    assert set(tabr.build_bank(np.array([1, 0, 2, 0, 3]), bank_legit=0, seed=0).tolist()) == {
        0,
        2,
        4,
    }
    with pytest.raises(ValueError):
        tabr.build_bank(np.array([0]), bank_legit=0, seed=0)


def test_chunked_topk_matches_full_cdist_and_drops_self():
    # Hand-built points: query 0 sits on bank row 0, and the other distances are unique.
    bank = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
    )
    query = bank[:2].clone()
    exclude = torch.tensor([0, 1])
    idx, dist = tabr.neighbor_indices(query, bank, k=10, exclude=exclude, bank_chunk=2)
    assert idx.shape == (2, 4)  # k clipped to bank size minus the masked self
    assert idx[0].tolist() == [1, 2, 3, 4]
    assert set(idx[1].tolist()) == {0, 2, 3, 4}
    torch.testing.assert_close(dist[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.isfinite(dist).all()

    torch.manual_seed(0)
    rows, n_bank, d, k = 5, 17, 4, 3
    bank = torch.randn(n_bank, d)
    query = bank[:rows].clone()
    exclude = torch.arange(rows)
    full = torch.cdist(query, bank)
    full[torch.arange(rows), exclude] = float("inf")
    expect_d, expect_i = torch.topk(full, k, dim=1, largest=False)
    for chunk in (1, 2, 4, 7, 17, 100):
        idx, dist = tabr.neighbor_indices(query, bank, k, exclude, chunk)
        torch.testing.assert_close(dist, expect_d, atol=1e-5, rtol=1e-5)
        got = full[torch.arange(rows)[:, None], idx]
        torch.testing.assert_close(got, expect_d, atol=1e-5, rtol=1e-5)
        assert (idx != exclude[:, None]).all()
        assert torch.equal(idx, expect_i)
    none_i, none_d = tabr.neighbor_indices(query, bank, k, None, bank_chunk=5)
    neg_i, neg_d = tabr.neighbor_indices(query, bank, k, torch.full((rows,), -1), bank_chunk=5)
    assert torch.equal(none_i, neg_i)
    torch.testing.assert_close(none_d, neg_d)
    # without the mask the unique nearest neighbour of an exact copy is itself
    self_i, _ = tabr.neighbor_indices(query, bank, k, None, bank_chunk=3)
    assert torch.equal(self_i[:, 0], torch.arange(rows))


def test_attention_is_negative_squared_l2_and_labels_start_the_value():
    spec = _spec(d=8, k=3, hidden=8, dropout=0.0, bank_chunk=4, query_chunk=2)
    tabr._seed_all(0)
    net = tabr.TabRNet(4, spec).eval()
    assert torch.count_nonzero(net.cand_proj.weight) == 0
    assert torch.count_nonzero(net.diff_proj.weight) == 0
    x = torch.randn(5, 4)
    with torch.no_grad():
        query = net.encoder(x)
        assert torch.allclose(query.norm(dim=-1), torch.ones(5), atol=1e-4)
        bank = torch.cat([query, query + 0.4], dim=0)
        labels = torch.arange(10) % N_CLASSES
        exclude = torch.arange(5)
        logits = net(x, bank, labels, exclude)
        idx, _ = tabr.neighbor_indices(query, bank, spec.k, exclude, bank_chunk=4)
        diff = query[:, None, :] - bank[idx]
        attn = torch.softmax(-diff.square().sum(-1), dim=-1)
        mixed = (attn[:, :, None] * net.label(labels[idx])).sum(1)
        expected = net.predictor(torch.cat([query, mixed], dim=-1))
    torch.testing.assert_close(logits, expected)
    for i in range(5):
        assert int(exclude[i]) not in idx[i].tolist()


def test_training_never_retrieves_its_own_row(monkeypatch):
    real = tabr.neighbor_indices
    seen: list[np.ndarray] = []

    def wrapped(query, bank, k, exclude, bank_chunk):
        idx, dist = real(query, bank, k, exclude, bank_chunk)
        if exclude is not None:
            ex = exclude.detach().cpu().numpy().copy()
            got = idx.detach().cpu().numpy()
            for row_ex, row_idx in zip(ex, got, strict=True):
                if row_ex >= 0:
                    assert row_ex not in row_idx
            seen.append(ex)
        return idx, dist

    monkeypatch.setattr(tabr, "neighbor_indices", wrapped)
    n = 48
    X, y = _labeled(n, 5, seed=0, rare=False)
    spec = _spec(epochs=1, batch=16, d=8, hidden=8, k=4, bank_chunk=5, query_chunk=10)
    period = np.zeros(n, dtype=np.int64)
    (probs,) = tabr.fit_predict(X, y, period, [X[:8]], spec, log=lambda *_: None)
    assert probs.shape == (8, N_CLASSES)
    # every training row is in the bank, so each epoch excludes each bank slot once
    assert sorted(np.concatenate(seen).tolist()) == list(range(n)) * spec.epochs


def test_fit_predict_shapes_sum_to_one_and_matches_across_runs():
    X, y = _labeled(180, 6, seed=1, rare=True)
    spec = _spec(epochs=3, batch=64, dropout=0.1, bank_legit=15, bank_chunk=7, query_chunk=20)
    period = np.zeros(len(y), dtype=np.int64)
    kwargs = dict(log=lambda *_: None)
    first = tabr.fit_predict(X, y, period, [X[:20], X[20:55]], spec, **kwargs)
    second = tabr.fit_predict(X, y, period, [X[:20], X[20:55]], spec, **kwargs)
    assert [p.shape for p in first] == [(20, N_CLASSES), (35, N_CLASSES)]
    for one, two in zip(first, second, strict=True):
        assert np.isfinite(one).all()
        np.testing.assert_allclose(one.sum(1), 1.0, atol=1e-5)
        np.testing.assert_array_equal(one, two)


class _PlainMLP(nn.Module):
    """TabR's encoder plus a head of the same depth, with no memory."""

    def __init__(self, n_in: int, spec: tabr.TabRSpec):
        super().__init__()
        self.encoder = tabr.Encoder(n_in, spec)
        self.head = nn.Sequential(
            nn.Linear(spec.d, spec.hidden),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def _fit_mlp(Z: np.ndarray, y: np.ndarray, Z_te: np.ndarray, spec: tabr.TabRSpec) -> np.ndarray:
    tabr._seed_all(0)
    dev = torch.device("cpu")
    model = _PlainMLP(Z.shape[1], spec).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    features = torch.from_numpy(Z).to(dev)
    targets = torch.from_numpy(y.astype(np.int64)).to(dev)
    steps = math.ceil(len(Z) / spec.batch)
    total = steps * spec.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: tabr._lr_at(s, total))
    generator = torch.Generator(device=dev).manual_seed(0)
    for _epoch in range(spec.epochs):
        model.train()
        perm = torch.randperm(len(Z), device=dev, generator=generator)
        for i in range(steps):
            batch = perm[i * spec.batch : (i + 1) * spec.batch]
            logits = model(features[batch])
            loss = nn.functional.cross_entropy(logits, targets[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Z_te).to(dev)).argmax(1).cpu().numpy()
    return pred


def _neighbor_problem(seed: int = 0):
    """Tight clusters with a random label. The label is the neighbour's, not a smooth field."""
    rng = np.random.default_rng(seed)
    per_class, train_per, test_per, dim = 12, 6, 3, 8
    centers = rng.normal(size=(N_CLASSES * per_class, dim)) * 3.0
    labels = np.repeat(np.arange(N_CLASSES, dtype=np.int64), per_class)

    def draw(n_per: int):
        noise = rng.normal(scale=0.03, size=(len(centers), n_per, dim))
        X = (centers[:, None, :] + noise).reshape(-1, dim).astype(np.float32)
        y = np.repeat(labels, n_per)
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    return (*draw(train_per), *draw(test_per))


def test_tabr_beats_mlp_when_the_label_is_the_neighbours():
    Xtr, ytr, Xte, yte = _neighbor_problem(0)
    spec = _spec(
        d=32,
        k=4,
        hidden=64,
        epochs=12,
        batch=128,
        lr=2e-3,
        weight_decay=0.0,
        dropout=0.0,
        quantiles=64,
        bank_legit=1_000_000,
        query_chunk=256,
        bank_chunk=64,
    )
    # same encoder initialisation: TabRNet builds the encoder before its other layers
    tabr._seed_all(0)
    mlp = _PlainMLP(Xtr.shape[1], spec)
    tabr._seed_all(0)
    net = tabr.TabRNet(Xtr.shape[1], spec)
    assert torch.equal(mlp.encoder.fc1.weight, net.encoder.fc1.weight)
    assert torch.equal(mlp.encoder.skip.weight, net.encoder.skip.weight)
    del mlp, net

    prep = tabm.fit_prep(Xtr, spec.quantiles)
    Ztr, Zte = tabm.transform(prep, Xtr), tabm.transform(prep, Xte)
    d2 = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
    oracle = ytr[d2.argmin(1)]
    purity = float((oracle == yte).mean())
    oracle_f1 = macro_f1(yte, oracle)
    assert purity > 0.95

    mlp_pred = _fit_mlp(Ztr, ytr, Zte, spec)
    (probs,) = tabr.fit_predict(
        Xtr, ytr, np.zeros(len(ytr), dtype=np.int64), [Xte], spec, log=lambda *_: None
    )
    mlp_f1 = macro_f1(yte, mlp_pred)
    tabr_f1 = macro_f1(yte, probs.argmax(1))
    print(
        f"TABR_COMPARE purity {purity:.3f} oracle {oracle_f1:.3f} "
        f"tabr {tabr_f1:.3f} mlp {mlp_f1:.3f}"
    )
    assert tabr_f1 > mlp_f1 + 0.15, (
        f"tabr {tabr_f1:.3f} mlp {mlp_f1:.3f} oracle {oracle_f1:.3f} purity {purity:.3f}"
    )
    assert tabr_f1 > 0.7
