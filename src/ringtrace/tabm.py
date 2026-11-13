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
    embedding: str = "periodic"  # or "ple": piecewise-linear over quantile bins
    n_bins: int = 48
    two_stage: bool = False
    recent_weight: float = 1.0  # sample weight of the latest training month
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


def ple_edges(Z: np.ndarray, n_bins: int, sample: int = 200_000, seed: int = 0) -> np.ndarray:
    """(n_cols, n_bins + 1) bin edges at quantiles of the transformed training values; repeated
    edges (discrete columns) are spread by a tiny step so every bin has a positive width."""
    rng = np.random.default_rng(seed)
    S = Z[np.sort(rng.choice(len(Z), sample, replace=False))] if len(Z) > sample else Z
    edges = np.quantile(S, np.linspace(0, 1, n_bins + 1), axis=0).T.astype(np.float64)
    step = 1e-4 * np.arange(n_bins + 1)[None, :]
    return np.maximum.accumulate(edges + step, axis=1)


def build_model(n_in: int, spec: TabMSpec, n_out: int = N_CLASSES, edges=None):
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

    class PiecewiseLinear(nn.Module):
        """x -> ReLU(Linear(ple(x))): ple has one entry per bin, 0 below it, 1 above it and the
        position inside it, with bin edges at quantiles of the training values."""

        def __init__(self, edges):
            super().__init__()
            self.register_buffer("lo", torch.as_tensor(edges[:, :-1], dtype=torch.float32))
            width = edges[:, 1:] - edges[:, :-1]
            self.register_buffer(
                "inv", torch.as_tensor(1 / np.maximum(width, 1e-6), dtype=torch.float32)
            )
            nb = edges.shape[1] - 1
            self.w = nn.Parameter(torch.empty(n_in, nb, spec.d_embed))
            self.b = nn.Parameter(torch.zeros(n_in, spec.d_embed))
            nn.init.uniform_(self.w, -1 / math.sqrt(nb), 1 / math.sqrt(nb))

        def forward(self, x):
            z = ((x[..., None] - self.lo) * self.inv).clamp(0.0, 1.0)
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
            self.embed = PiecewiseLinear(edges) if spec.embedding == "ple" else Periodic()
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


def train_one(
    Z: np.ndarray,
    y: np.ndarray,
    spec: TabMSpec,
    seed: int,
    log=print,
    n_out: int = N_CLASSES,
    row_weight: np.ndarray | None = None,
):
    torch = _torch()
    _seed_all(seed)
    dev = torch.device(spec.device if torch.cuda.is_available() else "cpu")
    edges = ple_edges(Z, spec.n_bins, seed=seed) if spec.embedding == "ple" else None
    model = build_model(Z.shape[1], spec, n_out, edges).to(dev)
    Zt = torch.from_numpy(Z).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    counts = np.bincount(y, minlength=n_out).astype(np.float64)
    w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** spec.class_weight_power, 0)
    w = w / (w * counts).sum() * counts.sum()
    cw = torch.tensor(w, dtype=torch.float32, device=dev)
    rw = None if row_weight is None else torch.from_numpy(row_weight.astype(np.float32)).to(dev)
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
            per = (
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, n_out), yb.repeat_interleave(spec.k), reduction="none"
                )
                * cw[yb.repeat_interleave(spec.k)]
            )
            if rw is not None:
                per = per * rw[b].repeat_interleave(spec.k)
                loss = per.sum() / (rw[b].sum() * spec.k)
            else:
                loss = per.mean()
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
    out = None
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            x = torch.from_numpy(Z[i : i + batch]).to(dev)
            p = torch.softmax(model(x).float(), dim=-1).mean(dim=1)
            if out is None:
                out = np.empty((len(Z), p.shape[1]), dtype=np.float64)
            out[i : i + batch] = p.cpu().numpy()
    return out


def recency_weight(period: np.ndarray, w: float) -> np.ndarray | None:
    """Weight ``w`` on the rows of the latest month, 1 elsewhere (None when ``w`` is 1)."""
    if w == 1.0 or period is None or len(period) == 0:
        return None
    return np.where(period == period.max(), w, 1.0)


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
    if spec.two_stage:
        return _two_stage(Z, y_tr, period_tr, X_tr, Zo, spec, log, t0)
    log(
        f"  tabm: {Z.shape[1]} inputs ({len(prep.gap_cols)} gap flags), {len(Z)} rows "
        f"({time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    rw = recency_weight(period_tr, spec.recent_weight)
    for seed in spec.seeds:
        model = train_one(Z, y_tr, spec, seed, log=log, row_weight=rw)
        for o, z in zip(outs, Zo, strict=True):
            o += predict(model, z, spec)
        del model
        _torch().cuda.empty_cache()
    log(f"  tabm fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [o / len(spec.seeds) for o in outs]


def _two_stage(Z, y_tr, period_tr, X_tr, Zo, spec: TabMSpec, log, t0) -> list[np.ndarray]:
    """P(fraud) x P(type | fraud) with the cascade's screening rows: the XGBoost screen
    (cross-fitted over the training months) drops likely-fraud legit rows from the binary stage,
    and fraud rows it cannot see are legit there and left out of the type stage."""
    from ringtrace.cascade import CascadeSpec, screen_score

    cs = CascadeSpec()
    fraud = (y_tr > 0).astype(np.float32)
    score = screen_score(X_tr, fraud, period_tr, cs)
    keep_bin = (fraud == 1) | (score < cs.drop_legit_min_score)
    type_rows = (fraud == 1) & (score >= cs.type_min_score)
    log(
        f"  two-stage screen: binary rows {int(keep_bin.sum())}, type rows {int(type_rows.sum())}"
        f" ({time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(z), N_CLASSES), dtype=np.float64) for z in Zo]
    for seed in spec.seeds:
        mb = train_one(Z[keep_bin], type_rows[keep_bin].astype(np.int64), spec, seed, log, 2)
        pf = [predict(mb, z, spec)[:, 1] for z in Zo]
        del mb
        mt = train_one(Z[type_rows], y_tr[type_rows] - 1, spec, seed, log, N_CLASSES - 1)
        for o, z, f in zip(outs, Zo, pf, strict=True):
            o[:, 0] += 1 - f
            o[:, 1:] += f[:, None] * predict(mt, z, spec)
        del mt
        _torch().cuda.empty_cache()
    log(f"  two-stage tabm fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [o / len(spec.seeds) for o in outs]
