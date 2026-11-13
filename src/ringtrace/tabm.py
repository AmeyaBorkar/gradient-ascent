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
    device: str = "cuda"
    # training techniques (all off by default; see ``Extras`` for the arrays some of them need)
    loss: str = "ce"  # "ce", "focal" or "logit_adjust"
    focal_gamma: float = 2.0
    logit_tau: float = 1.0
    smooth_fraud: float = 0.0  # label smoothing across the legit / fraud boundary only
    distill_alpha: float = 0.0  # soft target = alpha * teacher + (1 - alpha) * one-hot
    qr_blank_p: float = 0.0  # chance a QR-to-person training row shows its blanked features
    mixup_alpha: float = 0.0  # mixup of the numeric embeddings (Beta(a, a))
    noise_std: float = 0.0  # gaussian noise on the quantile-normal inputs
    ema_decay: float = 0.0  # predict with an exponential moving average of the weights
    swa_last: int = 0  # predict with the mean of the weights at the end of the last n epochs
    pretrain_epochs: int = 0  # masked-feature (SCARF-style corruption) reconstruction first
    pretrain_mask: float = 0.3


@dataclass
class Extras:
    """Row-aligned arrays some techniques need, in the raw feature space of ``X_tr``.

    ``teacher``: out-of-fold class probabilities for the training rows (distillation);
    ``X_blank``/``blank_rows``: the training rows' features with the QR check blanked, and which
    rows that applies to (QR payments to a person); ``X_unlabelled``: extra rows used only as
    inputs by the pretraining step."""

    teacher: np.ndarray | None = None
    X_blank: np.ndarray | None = None
    blank_rows: np.ndarray | None = None
    X_unlabelled: np.ndarray | None = None


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

        def hidden(self, e):  # embedded (B, n_in * d_embed) -> (B, k, width)
            return self.body(e[:, None, :].expand(-1, spec.k, -1))

        def from_embed(self, e):
            return torch.einsum("bkd,kdo->bko", self.hidden(e), self.head_w) + self.head_b

        def forward(self, x):  # (B, n_in) -> logits (B, k, n_out)
            return self.from_embed(self.embed(x))

    return TabM()


def _seed_all(seed: int) -> None:
    torch = _torch()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _plain(spec: TabMSpec) -> bool:
    return (
        spec.loss == "ce"
        and spec.smooth_fraud == 0
        and spec.distill_alpha == 0
        and spec.mixup_alpha == 0
        and spec.noise_std == 0
        and spec.qr_blank_p == 0
    )


def soft_targets(
    y: np.ndarray, spec: TabMSpec, prior: np.ndarray, teacher: np.ndarray | None = None
) -> np.ndarray:
    """Training target distribution per row: one-hot, smoothed across the legit / fraud boundary
    (a legit row gives ``smooth_fraud`` to the fraud classes by their prior, a fraud row gives it
    to legit), then mixed with the teacher's probabilities by ``distill_alpha``."""
    q = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    q[np.arange(len(y)), y] = 1.0
    e = spec.smooth_fraud
    if e > 0:
        fraud_prior = np.asarray(prior, dtype=np.float64).copy()
        fraud_prior[0] = 0.0
        fraud_prior /= fraud_prior.sum()
        legit = y == 0
        q[legit] = (1 - e) * q[legit] + e * fraud_prior[None, :]
        q[~legit] *= 1 - e
        q[~legit, 0] += e
    if spec.distill_alpha > 0:
        if teacher is None:
            raise ValueError("distill_alpha needs the teacher's out-of-fold probabilities")
        a = spec.distill_alpha
        q = a * teacher.astype(np.float32) + (1 - a) * q
    return q


def _pretrain(model, Zall, spec: TabMSpec, seed: int, dev, log) -> None:
    """Reconstruct every input from a copy whose entries were swapped, with probability
    ``pretrain_mask``, for the same column of another random row (SCARF-style corruption)."""
    torch = _torch()
    head = torch.nn.Linear(spec.width, Zall.shape[1]).to(dev)
    params = list(model.embed.parameters()) + list(model.body.parameters())
    params += list(head.parameters())
    opt = torch.optim.AdamW(params, lr=spec.lr, weight_decay=spec.weight_decay)
    steps_per = math.ceil(len(Zall) / spec.batch)
    total = steps_per * spec.pretrain_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * total)) * max(0.0, 1 - s / total)
    )
    gen = torch.Generator(device=dev).manual_seed(10_000 + seed)
    t0 = time.time()
    for ep in range(spec.pretrain_epochs):
        model.train()
        perm = torch.randperm(len(Zall), device=dev, generator=gen)
        run = 0.0
        for i in range(steps_per):
            b = perm[i * spec.batch : (i + 1) * spec.batch]
            x = Zall[b]
            donor = Zall[torch.randint(0, len(Zall), (len(b),), device=dev, generator=gen)]
            m = torch.rand(x.shape, device=dev, generator=gen) < spec.pretrain_mask
            h = model.hidden(model.embed(torch.where(m, donor, x))).mean(dim=1)
            loss = torch.nn.functional.mse_loss(head(h), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
        log(
            f"    seed {seed} pretrain {ep + 1}/{spec.pretrain_epochs} "
            f"loss {run / steps_per:.4f} ({time.time() - t0:.0f}s)"
        )


def train_one(
    Z: np.ndarray,
    y: np.ndarray,
    spec: TabMSpec,
    seed: int,
    log=print,
    teacher: np.ndarray | None = None,
    Z_blank: np.ndarray | None = None,
    blank_rows: np.ndarray | None = None,
    Z_unlabelled: np.ndarray | None = None,
    n_numeric: int | None = None,
):
    """One TabM fit. ``Z_blank`` holds the blanked inputs of the rows ``blank_rows`` (same
    order); ``n_numeric`` is the number of leading quantile-normal columns (the noise target).
    With every technique off this is the plain cross-entropy fit, step for step."""
    torch = _torch()
    F = torch.nn.functional
    _seed_all(seed)
    dev = torch.device(spec.device if torch.cuda.is_available() else "cpu")
    model = build_model(Z.shape[1], spec).to(dev)
    if spec.pretrain_epochs > 0:
        Zall = Z if Z_unlabelled is None else np.concatenate([Z, Z_unlabelled])
        Zall_t = torch.from_numpy(Zall).to(dev)
        _pretrain(model, Zall_t, spec, seed, dev, log)
        del Zall_t
    Zt = torch.from_numpy(Z).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** spec.class_weight_power, 0)
    w = w / (w * counts).sum() * counts.sum()
    cw = torch.tensor(w, dtype=torch.float32, device=dev)
    plain = _plain(spec)
    blank_at = None
    if not plain:
        prior = counts / counts.sum()
        Qt = torch.from_numpy(soft_targets(y, spec, prior, teacher)).to(dev)
        log_prior = torch.tensor(np.log(np.maximum(prior, 1e-12)), dtype=torch.float32, device=dev)
        if spec.qr_blank_p > 0:
            if Z_blank is None or blank_rows is None:
                raise ValueError("qr_blank_p needs the blanked inputs")
            at = np.full(len(Z), -1, dtype=np.int64)
            at[blank_rows] = np.arange(len(blank_rows))
            blank_at = torch.from_numpy(at).to(dev)
            Zb = torch.from_numpy(Z_blank).to(dev)
        n_num = Z.shape[1] if n_numeric is None else n_numeric
        mix_rng = np.random.default_rng(20_000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    steps_per = math.ceil(len(Z) / spec.batch)
    total = steps_per * spec.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * total)) * max(0.0, 1 - s / total)
    )
    avg = None
    if spec.ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        avg = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(spec.ema_decay))
    elif spec.swa_last > 0:
        from torch.optim.swa_utils import AveragedModel

        avg = AveragedModel(model)
    gen = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()
    for ep in range(spec.epochs):
        model.train()
        perm = torch.randperm(len(Z), device=dev, generator=gen)
        run = 0.0
        for i in range(steps_per):
            b = perm[i * spec.batch : (i + 1) * spec.batch]
            if plain:
                logits = model(Zt[b])
                loss = F.cross_entropy(
                    logits.reshape(-1, N_CLASSES), yt[b].repeat_interleave(spec.k), weight=cw
                )
            else:
                x = Zt[b]
                if blank_at is not None:
                    j = blank_at[b]
                    coin = torch.rand(len(b), device=dev, generator=gen) < spec.qr_blank_p
                    x = torch.where(((j >= 0) & coin)[:, None], Zb[j.clamp(min=0)], x)
                if spec.noise_std > 0:
                    noise = torch.randn(x.shape, device=dev, generator=gen) * spec.noise_std
                    noise[:, n_num:] = 0
                    x = x + noise
                q = Qt[b]
                rw = cw[yt[b]]
                e = model.embed(x)
                if spec.mixup_alpha > 0:
                    lam = float(mix_rng.beta(spec.mixup_alpha, spec.mixup_alpha))
                    lam = max(lam, 1 - lam)
                    p2 = torch.randperm(len(b), device=dev, generator=gen)
                    e = lam * e + (1 - lam) * e[p2]
                    q = lam * q + (1 - lam) * q[p2]
                    rw = lam * rw + (1 - lam) * rw[p2]
                logits = model.from_embed(e)  # (B, k, C)
                if spec.loss == "logit_adjust":
                    logits = logits + spec.logit_tau * log_prior
                logp = F.log_softmax(logits, dim=-1)
                ce = -(q[:, None, :] * logp).sum(-1)  # (B, k)
                if spec.loss == "focal":
                    pt = (q[:, None, :] * logp.exp()).sum(-1)
                    ce = ce * (1 - pt).clamp(min=0) ** spec.focal_gamma
                loss = (ce.mean(dim=1) * rw).sum() / rw.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if spec.ema_decay > 0:
                avg.update_parameters(model)
            run += float(loss.detach())
        if avg is not None and spec.ema_decay == 0 and ep >= spec.epochs - spec.swa_last:
            avg.update_parameters(model)
        log(
            f"    seed {seed} epoch {ep + 1}/{spec.epochs} loss {run / steps_per:.4f} "
            f"({time.time() - t0:.0f}s)"
        )
    del Zt, yt
    return model if avg is None else avg.module


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
    extras: Extras | None = None,
) -> list[np.ndarray]:
    """Train on (X_tr, y_tr) and return 12-class probabilities for every matrix in X_out.
    ``period_tr`` is unused (kept for the cascade's signature)."""
    t0 = time.time()
    ex = extras or Extras()
    prep = fit_prep(X_tr, spec.quantiles)
    Z = transform(prep, X_tr)
    Zo = [transform(prep, X) for X in X_out]
    Zb = None
    if ex.X_blank is not None and ex.blank_rows is not None:
        Zb = transform(prep, ex.X_blank[ex.blank_rows])
    Zu = transform(prep, ex.X_unlabelled) if ex.X_unlabelled is not None else None
    log(
        f"  tabm: {Z.shape[1]} inputs ({len(prep.gap_cols)} gap flags), {len(Z)} rows "
        f"({time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        model = train_one(
            Z,
            y_tr,
            spec,
            seed,
            log=log,
            teacher=ex.teacher,
            Z_blank=Zb,
            blank_rows=ex.blank_rows,
            Z_unlabelled=Zu,
            n_numeric=X_tr.shape[1],
        )
        for o, z in zip(outs, Zo, strict=True):
            o += predict(model, z, spec)
        del model
        _torch().cuda.empty_cache()
    log(f"  tabm fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [o / len(spec.seeds) for o in outs]
