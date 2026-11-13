"""TabR: tabular classification with an MLP encoder and a nearest-neighbour memory.

Gorishniy et al. (2023) mix a learned embedding with labels retrieved from a context
bank. This member has the same ``fit_predict`` contract as ``ringtrace.tabm``:

* inputs are the training-fitted quantile map and gap flags (``tabm.fit_prep`` /
  ``tabm.transform``);
* the encoder ``E`` is an MLP, ``E(x)`` in ``R^d``, L2-normalised so squared distances
  lie in ``[0, 4]`` and ``softmax(-||.||^2)`` does not saturate;
* each query takes the ``k`` nearest bank rows under L2 (exact ``torch.cdist``, tiled);
* a candidate value is ``W E(x_c) + embed(y_c) + T(E(x) - E(x_c))``;
* attention is ``softmax(-||E(x) - E(x_c)||^2)``;
* an MLP on ``[E(x), weighted values]`` predicts 12 logits.

The bank is every rare row (label other than 0) plus a seeded sample of ``bank_legit``
class-0 rows, redrawn per seed. Embeddings of the bank are recomputed once per epoch
under ``no_grad`` and held fixed for that epoch, which is the paper's speed trade.
Gradients reach the encoder only through the query. A training row's own bank slot is
masked, so it never retrieves itself.

``fit_predict`` trains on ``X_tr`` and scores each matrix in ``X_out``. The bank is the
training fold alone. The cascade's held-out month and October are disjoint from ``X_tr``,
so those queries are not in the memory. A matrix that repeats training rows retrieves
those rows and sees their labels.

``W`` and ``T`` start at zero: the value begins as the neighbour's label embedding, and
the feature terms learn on top. Dropout sits on the predictor only, so a query and the
frozen bank live in one embedding space.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ringtrace.classes import N_CLASSES
from ringtrace.tabm import fit_prep, transform


@dataclass
class TabRSpec:
    seeds: tuple[int, ...] = (0, 1, 2)
    d: int = 256
    k: int = 32
    bank_legit: int = 200_000
    epochs: int = 10
    batch: int = 2048
    lr: float = 2e-3
    weight_decay: float = 3e-4
    dropout: float = 0.1
    device: str = "cuda"
    hidden: int = 512
    quantiles: int = 1000
    query_chunk: int = 8192  # queries per retrieval pass
    bank_chunk: int = 32768  # bank rows per cdist tile (caps the distance matrix)


def _lr_at(step: int, total: int) -> float:
    """Warm up over the first 5% of steps, then decay linearly to zero (same shape as TabM)."""
    return min(1.0, (step + 1) / (0.05 * total)) * max(0.0, 1 - step / total)


def _seed_all(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _device(spec: TabRSpec) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(spec.device)
    return torch.device("cpu")


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# context bank and exact chunked retrieval


def build_bank(y: np.ndarray, bank_legit: int, seed: int) -> np.ndarray:
    """Every rare row (label != 0), plus ``bank_legit`` class-0 rows, or all of them.

    The class-0 sample uses a ``numpy`` Generator seeded with ``seed``. Indices are sorted.
    """
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    rare = np.flatnonzero(labels != 0)
    legit = np.flatnonzero(labels == 0)
    if bank_legit < len(legit):
        if bank_legit <= 0:
            legit = legit[:0]
        else:
            rng = np.random.default_rng(seed)
            pick = np.sort(rng.choice(len(legit), size=bank_legit, replace=False))
            legit = legit[pick]
    parts = [part for part in (rare, legit) if len(part)]
    if not parts:
        raise ValueError("context bank is empty")
    bank = np.concatenate(parts)
    bank.sort()
    if len(bank) < 2:
        raise ValueError("context bank needs at least 2 rows so a query can exclude itself")
    return bank


def _mask_self(dist: torch.Tensor, exclude: torch.Tensor, start: int) -> None:
    """Set the column of each query's own bank row to +inf when it falls in this tile."""
    width = dist.shape[1]
    cols = exclude - start
    valid = (cols >= 0) & (cols < width)
    safe = cols.clamp(0, width - 1)
    rows = torch.arange(dist.shape[0], device=dist.device)
    current = dist[rows, safe]
    dist[rows, safe] = torch.where(valid, current.new_full((), float("inf")), current)


def neighbor_indices(
    query: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    exclude: torch.Tensor | None,
    bank_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices of the ``k`` nearest bank rows under L2, and those distances.

    ``exclude`` is a long tensor of bank positions to drop (one per query, ``-1`` skips
    the mask) or ``None``. The bank is scanned in tiles of ``bank_chunk`` and the
    per-tile top-k are merged, so the result matches one full ``cdist``. Requested ``k``
    is clipped to the number of legal candidates (bank size, minus one when exclusion
    is on). A tile holds ``query_rows * bank_chunk`` distances.
    """
    if not query.is_contiguous():
        query = query.contiguous()
    if not bank.is_contiguous():
        bank = bank.contiguous()
    rows, n_bank = query.shape[0], bank.shape[0]
    if exclude is not None:
        exclude = exclude.to(device=query.device, dtype=torch.long)
        k_eff = min(int(k), n_bank - 1)
    else:
        k_eff = min(int(k), n_bank)
    if k_eff < 1:
        raise ValueError(f"no retrieval candidates (bank={n_bank}, k={k})")
    bank_chunk = max(int(bank_chunk), 1)
    dev = query.device
    best_d = torch.full((rows, k_eff), float("inf"), device=dev, dtype=query.dtype)
    best_i = torch.zeros((rows, k_eff), dtype=torch.long, device=dev)
    for start in range(0, n_bank, bank_chunk):
        end = min(start + bank_chunk, n_bank)
        dist = torch.cdist(query, bank[start:end])
        if exclude is not None:
            _mask_self(dist, exclude, start)
        width = end - start
        take = min(k_eff, width)
        vals, loc = torch.topk(dist, take, dim=1, largest=False)
        glob = loc + start
        if take < k_eff:
            pad = k_eff - take
            vals = torch.cat(
                [vals, torch.full((rows, pad), float("inf"), device=dev, dtype=vals.dtype)],
                dim=1,
            )
            glob = torch.cat(
                [glob, torch.zeros((rows, pad), dtype=torch.long, device=dev)],
                dim=1,
            )
        merged_d = torch.cat([best_d, vals], dim=1)
        merged_i = torch.cat([best_i, glob], dim=1)
        best_d, which = torch.topk(merged_d, k_eff, dim=1, largest=False)
        best_i = merged_i.gather(1, which)
    return best_i, best_d


def retrieve(
    query: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    exclude: torch.Tensor | None,
    query_chunk: int,
    bank_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``neighbor_indices`` over query tiles, so a training batch cannot exceed the cap."""
    query_chunk = max(int(query_chunk), 1)
    with torch.no_grad():
        if query.shape[0] <= query_chunk:
            return neighbor_indices(query, bank, k, exclude, bank_chunk)
        idxs, dists = [], []
        for i in range(0, query.shape[0], query_chunk):
            ex = None if exclude is None else exclude[i : i + query_chunk]
            idx, dist = neighbor_indices(query[i : i + query_chunk], bank, k, ex, bank_chunk)
            idxs.append(idx)
            dists.append(dist)
        return torch.cat(idxs, 0), torch.cat(dists, 0)


# ---------------------------------------------------------------------------------------------
# the network


class Encoder(nn.Module):
    """MLP embedding with a linear skip, then L2 normalisation.

    The last MLP layer starts at zero, so a fresh encoder is ``normalize(skip(x))``:
    retrieval initially sees a linear image of the quantile features, and the MLP can
    still bend the metric once that layer moves.
    """

    def __init__(self, n_in: int, spec: TabRSpec):
        super().__init__()
        hidden, d = spec.hidden, spec.d
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, d)
        self.skip = nn.Linear(n_in, d, bias=False)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc1(x))
        hidden = torch.relu(self.fc2(hidden))
        return nn.functional.normalize(self.fc_out(hidden) + self.skip(x), dim=-1)


class TabRNet(nn.Module):
    def __init__(self, n_in: int, spec: TabRSpec):
        super().__init__()
        d = spec.d
        self.k = int(spec.k)
        self.query_chunk = int(spec.query_chunk)
        self.bank_chunk = int(spec.bank_chunk)
        self.encoder = Encoder(n_in, spec)
        self.cand_proj = nn.Linear(d, d, bias=False)  # W
        self.diff_proj = nn.Linear(d, d, bias=False)  # T, the map of E(x) - E(x_c)
        nn.init.zeros_(self.cand_proj.weight)
        nn.init.zeros_(self.diff_proj.weight)
        self.label = nn.Embedding(N_CLASSES, d)
        self.predictor = nn.Sequential(
            nn.Linear(2 * d, spec.hidden),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden, N_CLASSES),
        )

    def forward(
        self,
        x: torch.Tensor,
        bank_emb: torch.Tensor,
        bank_y: torch.Tensor,
        exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.encoder(x)
        idx, _ = retrieve(
            query.detach(), bank_emb, self.k, exclude, self.query_chunk, self.bank_chunk
        )
        cand = bank_emb[idx]
        diff = query[:, None, :] - cand
        attn = torch.softmax(-diff.square().sum(dim=-1), dim=-1)
        value = self.cand_proj(cand) + self.label(bank_y[idx]) + self.diff_proj(diff)
        mixed = (attn.unsqueeze(-1) * value).sum(dim=1)
        return self.predictor(torch.cat([query, mixed], dim=-1))


def _adamw(model: nn.Module, spec: TabRSpec) -> torch.optim.AdamW:
    decay, plain = [], []
    for name, param in model.named_parameters():
        if param.ndim == 1 or name.startswith("label"):
            plain.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": spec.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=spec.lr,
    )


def _bank_embeddings(
    model: TabRNet, features: torch.Tensor, bank_index: torch.Tensor, chunk: int
) -> torch.Tensor:
    chunk = max(int(chunk), 1)
    training = model.training
    model.eval()
    pieces = []
    with torch.no_grad():
        for i in range(0, bank_index.shape[0], chunk):
            pieces.append(model.encoder(features[bank_index[i : i + chunk]]))
    if training:
        model.train()
    return torch.cat(pieces, 0).contiguous()


def train_one(
    Z: np.ndarray,
    y: np.ndarray,
    spec: TabRSpec,
    seed: int,
    log=print,
) -> tuple[TabRNet, torch.Tensor, torch.Tensor]:
    """Fit one seed. Returns the eval-mode net, frozen bank embeddings and bank labels."""
    _seed_all(seed)
    dev = _device(spec)
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    bank = build_bank(labels, spec.bank_legit, seed)
    n_legit = int((labels[bank] == 0).sum())
    log(
        f"    seed {seed} bank {len(bank)} ({len(bank) - n_legit} rare + {n_legit} legit) "
        f"on {dev}"
    )
    model = TabRNet(Z.shape[1], spec).to(dev)
    features = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32)).to(dev)
    targets = torch.from_numpy(labels).to(dev)
    bank_index = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.int64)).to(dev)
    position = np.full(len(labels), -1, dtype=np.int64)
    position[bank] = np.arange(len(bank), dtype=np.int64)
    position_t = torch.from_numpy(position).to(dev)
    bank_y = targets[bank_index].contiguous()
    opt = _adamw(model, spec)
    steps = math.ceil(len(Z) / spec.batch)
    total = steps * spec.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_at(s, total))
    generator = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()
    bank_emb = _bank_embeddings(model, features, bank_index, spec.query_chunk)
    for epoch in range(spec.epochs):
        if epoch:
            bank_emb = _bank_embeddings(model, features, bank_index, spec.query_chunk)
        model.train()
        perm = torch.randperm(len(Z), device=dev, generator=generator)
        running = 0.0
        for i in range(steps):
            batch = perm[i * spec.batch : (i + 1) * spec.batch]
            logits = model(features[batch], bank_emb, bank_y, position_t[batch])
            loss = nn.functional.cross_entropy(logits, targets[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss.detach())
        log(
            f"    seed {seed} epoch {epoch + 1}/{spec.epochs} loss {running / steps:.4f} "
            f"({time.time() - t0:.0f}s)"
        )
    bank_emb = _bank_embeddings(model, features, bank_index, spec.query_chunk)
    model.eval()
    del features, targets, position_t, opt, sched, bank_index
    return model, bank_emb, bank_y


def predict(
    model: TabRNet,
    Z: np.ndarray,
    bank_emb: torch.Tensor,
    bank_y: torch.Tensor,
    spec: TabRSpec,
) -> np.ndarray:
    dev = bank_emb.device
    model.eval()
    out = np.empty((len(Z), N_CLASSES), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(Z), spec.query_chunk):
            block = Z[i : i + spec.query_chunk]
            x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).to(dev)
            logits = model(x, bank_emb, bank_y, None)
            out[i : i + len(block)] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    period_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec: TabRSpec,
    log=print,
) -> list[np.ndarray]:
    """Train on ``(X_tr, y_tr)`` and return 12-class probabilities for every matrix in ``X_out``.

    Averaged over ``spec.seeds``. ``period_tr`` is unused (kept for the cascade's signature).
    """
    t0 = time.time()
    prep = fit_prep(X_tr, spec.quantiles)
    Z = transform(prep, X_tr)
    held = [transform(prep, X) for X in X_out]
    log(
        f"  tabr: {Z.shape[1]} inputs ({len(prep.gap_cols)} gap flags), {len(Z)} rows "
        f"({time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        model, bank_emb, bank_y = train_one(Z, y_tr, spec, int(seed), log=log)
        for acc, block in zip(outs, held, strict=True):
            acc += predict(model, block, bank_emb, bank_y, spec)
        del model, bank_emb, bank_y
        _empty_cache()
    log(f"  tabr fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [acc / len(spec.seeds) for acc in outs]
