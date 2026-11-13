"""A RealMLP-style network, written from the published recipe (Holzmüller et al., 2024).

Preprocessing, fitted on the training rows only:

* median fill plus a gap flag for every column with gaps;
* robust scaling: x -> (x - median) / spread, where spread is the inter-quantile range
  (25-75 %, widened to 0-100 % / 2 when that is zero; columns with no spread are left at 0);
* smooth clipping: x -> x / sqrt(1 + (x / 3)^2).

Network:

* a learnable scale per input (initialised at 1);
* PBLD embeddings per feature: [x, Linear(cos(2 pi (w x + b)))] with learned periodic
  frequencies and phases, giving ``d_embed`` values per feature;
* an MLP of ``blocks`` x ``width`` with SELU and dropout, then a linear head.

Training: AdamW, warmup then cosine decay, cross-entropy with label smoothing (0.1 in the
recipe). The original also uses neural-tangent parametrisation, parametric activations and a
multi-cycle schedule over many epochs; at a million rows and a few epochs those are left out.
``fit_predict`` has the cascade's signature.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ringtrace.classes import N_CLASSES


@dataclass
class RealMLPSpec:
    seeds: tuple[int, ...] = (0, 1, 2)
    width: int = 256
    blocks: int = 3
    dropout: float = 0.15
    d_embed: int = 4
    n_freq: int = 8
    freq_sigma: float = 0.1
    epochs: int = 10
    batch: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    label_smoothing: float = 0.1
    device: str = "cuda"


@dataclass
class RobustPrep:
    gap_cols: np.ndarray
    median: np.ndarray
    spread: np.ndarray


def fit_prep(X: np.ndarray) -> RobustPrep:
    gaps = ~np.isfinite(X).all(axis=0)
    Xf = np.where(np.isfinite(X), X, np.nan)
    med = np.nan_to_num(np.nanmedian(Xf, axis=0), nan=0.0)
    q0, q25, q75, q100 = np.nanquantile(Xf, [0.0, 0.25, 0.75, 1.0], axis=0)
    spread = q75 - q25
    spread = np.where(spread > 0, spread, (q100 - q0) / 2)
    spread = np.nan_to_num(spread, nan=0.0)
    return RobustPrep(np.flatnonzero(gaps), med.astype(np.float32), spread.astype(np.float32))


def transform(prep: RobustPrep, X: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(X)
    x = np.where(bad, prep.median[None, :], X)
    safe = np.where(prep.spread > 0, prep.spread, 1.0)
    z = np.where(prep.spread[None, :] > 0, (x - prep.median[None, :]) / safe[None, :], 0.0)
    z = z / np.sqrt(1 + (z / 3) ** 2)
    flags = bad[:, prep.gap_cols].astype(np.float32)
    return np.concatenate([z.astype(np.float32), flags], axis=1)


def build_model(n_in: int, spec: RealMLPSpec, n_out: int = N_CLASSES):
    import torch

    nn = torch.nn

    class PBLD(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(n_in))
            self.w = nn.Parameter(torch.randn(n_in, spec.n_freq) * spec.freq_sigma)
            self.b = nn.Parameter(torch.rand(n_in, spec.n_freq))
            self.lin = nn.Parameter(torch.empty(n_in, spec.n_freq, spec.d_embed - 1))
            self.lb = nn.Parameter(torch.zeros(n_in, spec.d_embed - 1))
            nn.init.uniform_(self.lin, -1 / math.sqrt(spec.n_freq), 1 / math.sqrt(spec.n_freq))

        def forward(self, x):  # (B, n_in) -> (B, n_in * d_embed)
            x = x * self.scale
            z = torch.cos(2 * math.pi * (x[..., None] * self.w + self.b))
            e = torch.einsum("bfq,fqd->bfd", z, self.lin) + self.lb
            return torch.cat([x[..., None], e], dim=-1).flatten(1)

    layers: list = [PBLD()]
    d = n_in * spec.d_embed
    for _ in range(spec.blocks):
        layers += [nn.Linear(d, spec.width), nn.SELU(), nn.Dropout(spec.dropout)]
        d = spec.width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def train_one(Z: np.ndarray, y: np.ndarray, spec: RealMLPSpec, seed: int, log=print):
    import torch

    from ringtrace.tabm import _seed_all

    _seed_all(seed)
    dev = torch.device(spec.device if torch.cuda.is_available() else "cpu")
    model = build_model(Z.shape[1], spec).to(dev)
    Zt = torch.from_numpy(Z).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    steps_per = math.ceil(len(Z) / spec.batch)
    total = steps_per * spec.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warm
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        ),
    )
    gen = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()
    for ep in range(spec.epochs):
        model.train()
        perm = torch.randperm(len(Z), device=dev, generator=gen)
        run = 0.0
        for i in range(steps_per):
            b = perm[i * spec.batch : (i + 1) * spec.batch]
            loss = torch.nn.functional.cross_entropy(
                model(Zt[b]), yt[b], label_smoothing=spec.label_smoothing
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
    return model


def predict(model, Z: np.ndarray, batch: int = 16384) -> np.ndarray:
    import torch

    dev = next(model.parameters()).device
    model.eval()
    out = np.empty((len(Z), N_CLASSES), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            x = torch.from_numpy(Z[i : i + batch]).to(dev)
            out[i : i + batch] = torch.softmax(model(x).float(), dim=-1).cpu().numpy()
    return out


def fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    period_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec: RealMLPSpec,
    log=print,
    extras=None,
) -> list[np.ndarray]:
    """Train on (X_tr, y_tr), return 12-class probabilities for every matrix in X_out."""
    import torch

    prep = fit_prep(X_tr)
    Z = transform(prep, X_tr)
    Zo = [transform(prep, X) for X in X_out]
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        model = train_one(Z, y_tr, spec, seed, log=log)
        for o, z in zip(outs, Zo, strict=True):
            o += predict(model, z)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [o / len(spec.seeds) for o in outs]
