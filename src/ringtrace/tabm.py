"""TabM: an MLP whose k ensemble members share most weights (BatchEnsemble-style adapters).

Each linear layer holds one shared weight W and, per member i, an input scale r_i, an output scale
s_i and a bias b_i:  y_i = ((x_i * r_i) @ W) * s_i + b_i.  The k members are trained jointly on
the mean of their losses and predict the mean of their softmax probabilities.

Numeric inputs go through preprocessing fitted on the training rows only (a missing indicator for
every column that has gaps, median fill, a quantile map to a normal), then through a periodic
embedding per feature: x -> ReLU(Linear([sin(2 pi c x), cos(2 pi c x)])) with learned
frequencies c. The feature matrix holds only numeric columns (categorical columns are already
integer codes), so codes are embedded like any other number.

``fit_predict`` has the cascade's signature, so ``run`` and ``octlike`` can call either model.
Training uses a direct 12-class softmax; the decoder optimises macro-F1 on calibrated
probabilities, so class weights are off by default (``class_weight_power`` > 0 turns them on).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from ringtrace.cascade import forced_device
from ringtrace.classes import N_CLASSES


@dataclass
class TabMSpec:
    seeds: tuple[int, ...] = (0, 1, 2)
    k: int = 16
    width: int = 512
    blocks: int = 3
    dropout: float = 0.1
    d_embed: int = 8
    n_freq: int = 24
    freq_sigma: float = 0.5
    epochs: int = 10
    batch: int = 2048
    lr: float = 2e-3
    weight_decay: float = 3e-4
    class_weight_power: float = 0.0
    quantiles: int = 1000
    device: str = "cuda"


# ---------------------------------------------------------------------------------------------
# preprocessing (numpy; fitted on training rows only)


@dataclass
class Prep:
    gap_cols: np.ndarray
    fill: np.ndarray
    grid: np.ndarray  # (n_cols, q) quantile values per column
    normal: np.ndarray  # (q,) the matching standard-normal values


def fit_prep(X: np.ndarray, quantiles: int = 1000, sample: int = 200_000, seed: int = 0) -> Prep:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False) if len(X) > sample else None
    S = X if idx is None else X[np.sort(idx)]
    gaps = np.isnan(X).any(axis=0)
    fill = np.nan_to_num(np.nanmedian(np.where(np.isfinite(S), S, np.nan), axis=0), nan=0.0)
    S = np.where(np.isfinite(S), S, fill[None, :])
    levels = np.linspace(0.0, 1.0, quantiles)
    grid = np.quantile(S, levels, axis=0).T.astype(np.float64)
    from scipy.special import ndtri

    eps = 0.5 / len(S)
    normal = ndtri(np.clip(levels, eps, 1 - eps))
    return Prep(np.flatnonzero(gaps), fill.astype(np.float32), grid, normal)


def transform(prep: Prep, X: np.ndarray) -> np.ndarray:
    """Quantile-normal values for every column, then 0/1 gap indicators."""
    n, d = X.shape
    out = np.empty((n, d + len(prep.gap_cols)), dtype=np.float32)
    for j in range(d):
        col = X[:, j]
        bad = ~np.isfinite(col)
        col = np.where(bad, prep.fill[j], col)
        g = prep.grid[j]
        # ties in the grid (discrete columns): average the left and right interpolations
        lo = np.interp(col, g, prep.normal)
        hi = -np.interp(-col, -g[::-1], -prep.normal[::-1])
        out[:, j] = 0.5 * (lo + hi)
    for i, j in enumerate(prep.gap_cols):
        out[:, d + i] = ~np.isfinite(X[:, j])
    return out


# ---------------------------------------------------------------------------------------------
# the network


def _torch():
    import torch

    return torch


def build_model(n_in: int, spec: TabMSpec, n_out: int = N_CLASSES):
    torch = _torch()
    nn = torch.nn

    class Periodic(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Parameter(torch.randn(n_in, spec.n_freq) * spec.freq_sigma)
            self.w = nn.Parameter(torch.empty(n_in, 2 * spec.n_freq, spec.d_embed))
            self.b = nn.Parameter(torch.zeros(n_in, spec.d_embed))
            bound = 1 / math.sqrt(2 * spec.n_freq)
            nn.init.uniform_(self.w, -bound, bound)

        def forward(self, x):  # (B, n_in) -> (B, n_in * d_embed)
            z = 2 * math.pi * x[..., None] * self.c
            z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
            e = torch.einsum("bfq,fqd->bfd", z, self.w) + self.b
            return torch.relu(e).flatten(1)

    class EnsembleLinear(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.lin = nn.Linear(d_in, d_out, bias=False)
            self.r = nn.Parameter(torch.ones(spec.k, d_in))
            self.s = nn.Parameter(torch.ones(spec.k, d_out))
            self.b = nn.Parameter(torch.zeros(spec.k, d_out))
            # random-sign init of the adapters makes the members differ from the first step
            with torch.no_grad():
                self.r.copy_(torch.where(torch.rand_like(self.r) < 0.5, -1.0, 1.0))
                self.s.copy_(torch.where(torch.rand_like(self.s) < 0.5, -1.0, 1.0))
                bound = 1 / math.sqrt(d_in)
                self.b.uniform_(-bound, bound)

        def forward(self, x):  # (B, k, d_in) -> (B, k, d_out)
            return self.lin(x * self.r) * self.s + self.b

    class TabM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = Periodic()
            d = n_in * spec.d_embed
            layers = []
            for _ in range(spec.blocks):
                layers += [EnsembleLinear(d, spec.width), nn.ReLU(), nn.Dropout(spec.dropout)]
                d = spec.width
            self.body = nn.Sequential(*layers)
            self.head_w = nn.Parameter(torch.empty(spec.k, d, n_out))
            self.head_b = nn.Parameter(torch.zeros(spec.k, n_out))
            nn.init.uniform_(self.head_w, -1 / math.sqrt(d), 1 / math.sqrt(d))

        def forward(self, x):  # (B, n_in) -> logits (B, k, n_out)
            h = self.embed(x)[:, None, :].expand(-1, spec.k, -1)
            h = self.body(h)
            return torch.einsum("bkd,kdo->bko", h, self.head_w) + self.head_b

    return TabM()


def _seed_all(seed: int) -> None:
    torch = _torch()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def train_one(Z: np.ndarray, y: np.ndarray, spec: TabMSpec, seed: int, log=print):
    torch = _torch()
    _seed_all(seed)
    use_gpu = forced_device() is None and torch.cuda.is_available()
    dev = torch.device(spec.device if use_gpu else "cpu")
    model = build_model(Z.shape[1], spec).to(dev)
    Zt = torch.from_numpy(Z).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** spec.class_weight_power, 0)
    w = w / (w * counts).sum() * counts.sum()
    cw = torch.tensor(w, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    steps_per = math.ceil(len(Z) / spec.batch)
    total = steps_per * spec.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * total)) * max(0.0, 1 - s / total)
    )
    gen = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()
    for ep in range(spec.epochs):
        model.train()
        perm = torch.randperm(len(Z), device=dev, generator=gen)
        run = 0.0
        for i in range(steps_per):
            b = perm[i * spec.batch : (i + 1) * spec.batch]
            logits = model(Zt[b])
            yb = yt[b]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, N_CLASSES), yb.repeat_interleave(spec.k), weight=cw
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
        log(
            f"    seed {seed} epoch {ep + 1}/{spec.epochs} loss {run / steps_per:.4f} "
            f"({time.time() - t0:.0f}s)"
        )
    del Zt, yt
    return model


def predict(model, Z: np.ndarray, spec: TabMSpec, batch: int = 16384) -> np.ndarray:
    torch = _torch()
    dev = next(model.parameters()).device
    model.eval()
    out = np.empty((len(Z), N_CLASSES), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            x = torch.from_numpy(Z[i : i + batch]).to(dev)
            p = torch.softmax(model(x).float(), dim=-1).mean(dim=1)
            out[i : i + batch] = p.cpu().numpy()
    return out


def fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    period_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec: TabMSpec,
    log=print,
) -> list[np.ndarray]:
    """Train on (X_tr, y_tr) and return 12-class probabilities for every matrix in X_out.
    ``period_tr`` is unused (kept for the cascade's signature)."""
    t0 = time.time()
    prep = fit_prep(X_tr, spec.quantiles)
    Z = transform(prep, X_tr)
    Zo = [transform(prep, X) for X in X_out]
    log(
        f"  tabm: {Z.shape[1]} inputs ({len(prep.gap_cols)} gap flags), {len(Z)} rows "
        f"({time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        model = train_one(Z, y_tr, spec, seed, log=log)
        for o, z in zip(outs, Zo, strict=True):
            o += predict(model, z, spec)
        del model
        _torch().cuda.empty_cache()
    log(f"  tabm fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [o / len(spec.seeds) for o in outs]
