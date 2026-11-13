"""TabM with a batch metric term on top of cross-entropy.

The loop is ``tabm.train_one``: same preprocessing (fitted on the training rows only), same
seeds, same optimiser schedule. The batch loss is ``CE + lam * extra``. ``metric_loss="none"``
skips the extra term entirely, so ``fit_predict_metric`` matches ``tabm.fit_predict`` on CPU.

``extra`` uses the head-averaged softmax ``p = mean_k softmax(logits)``:

- ``softf1``: soft macro-F1 over the twelve classes. For each class, soft TP, FP and FN replace
  the hard counts with ``p_c``, ``F1_c = 2 TP / (2 TP + FP + FN + eps)``, and the penalty is
  ``1 - mean F1`` over classes that occur in the batch.
- ``pairwise``: ranking on the fraud score ``s = 1 - p(legit)``. Every fraud row is paired with
  the ``m`` legit rows of highest ``s``, ``m = min(4 * n_fraud, n_legit)``, and the penalty is
  the mean ``softplus(-(s_f - s_l))``. The cartesian product is capped at 50_000 pairs. This
  term does not separate the eleven fraud types from each other.
- ``reinforce``: draw one label per row from ``p``, take the batch macro-F1 against ``y`` as the
  reward, and apply a score-function step with an exponential-moving-average baseline:
  ``-(reward - baseline) * mean log p(sampled)``, averaged over the batch like the cross-entropy.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ringtrace import tabm
from ringtrace.classes import LEGIT, N_CLASSES

# Soft-F1 denominator offset. Keeps an empty soft count finite without moving a real F1.
_F1_EPS = 1e-6
# Floor under the sampled probability so a hard zero cannot make the score-function term NaN.
_LOGP_EPS = 1e-8
# Fraud rows times hard legits. A 2048-row batch with a quarter fraud is otherwise ~1.6e5 pairs.
_PAIR_CAP = 50_000
_LEGIT_PER_FRAUD = 4
# baseline <- EMA * baseline + (1 - EMA) * reward
_BASELINE_EMA = 0.9


def _torch():
    import torch

    return torch


def _head_mean_proba(logits):
    """Softmax probabilities averaged over the ensemble heads. logits: (B, k, C)."""
    return _torch().softmax(logits, dim=-1).mean(dim=1)


def _softf1_extra(p, y):
    """1 - mean soft F1 over classes present in ``y``. ``p`` is (B, C)."""
    torch = _torch()
    oh = torch.nn.functional.one_hot(y, N_CLASSES).to(dtype=p.dtype)
    tp = (p * oh).sum(dim=0)
    fp = (p * (1.0 - oh)).sum(dim=0)
    fn = ((1.0 - p) * oh).sum(dim=0)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + _F1_EPS)
    present = oh.sum(dim=0) > 0
    return 1.0 - f1[present].mean()


def _pair_budget(n_fraud: int, n_legit: int, cap: int = _PAIR_CAP) -> tuple[int, int]:
    """Fraud rows to keep, and how many hard legit rows to pair with each.

    Every fraud row is kept while ``n_fraud * m`` stays within ``cap`` by shrinking ``m``.
    Past that (more fraud rows than ``cap``), keep ``cap`` fraud rows and one legit each.
    """
    if n_fraud <= 0 or n_legit <= 0 or cap <= 0:
        return 0, 0
    m = min(_LEGIT_PER_FRAUD * n_fraud, n_legit)
    if n_fraud * m <= cap:
        return n_fraud, m
    m = cap // n_fraud
    if m >= 1:
        return n_fraud, min(m, n_legit)
    return min(n_fraud, cap), 1


def _pairwise_extra(p, y):
    """Mean softplus margin between fraud scores and the hardest legit scores."""
    torch = _torch()
    s = 1.0 - p[:, LEGIT]
    legit = y == LEGIT
    n_legit = int(legit.sum())
    n_fraud = int(s.shape[0] - n_legit)
    n_keep, m = _pair_budget(n_fraud, n_legit)
    if n_keep == 0 or m == 0:
        return p.new_zeros(())
    s_fraud = s[~legit]
    s_legit = s[legit]
    if n_keep < s_fraud.shape[0]:
        # Lowest fraud scores are the ones still ranked below legit.
        s_fraud = torch.topk(s_fraud, n_keep, largest=False).values
    s_hard = torch.topk(s_legit, m, largest=True).values
    margin = s_fraud[:, None] - s_hard[None, :]
    return torch.nn.functional.softplus(-margin).mean()


def _macro_f1_torch(y, pred):
    """Mean per-class F1 over all twelve classes. A class absent from both scores 0."""
    torch = _torch()
    k = N_CLASSES
    cm = torch.bincount(y * k + pred, minlength=k * k).reshape(k, k).to(dtype=torch.float32)
    tp = cm.diagonal()
    denom = cm.sum(dim=0) + cm.sum(dim=1)
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    return f1.mean()


def _reinforce_extra(p, y, baseline: float):
    """Score-function estimate of batch macro-F1. Returns ``(extra, new_baseline)``."""
    torch = _torch()
    # The draw is discrete; the score function backpropagates only through log p.
    sampled = torch.multinomial(p.detach(), num_samples=1).squeeze(1)
    reward = float(_macro_f1_torch(y, sampled).detach())
    advantage = reward - baseline
    baseline = _BASELINE_EMA * baseline + (1.0 - _BASELINE_EMA) * reward
    # Mean over the batch, so ``lam`` sits on the same per-row scale as the cross-entropy mean.
    logp = p.gather(1, sampled[:, None]).clamp_min(_LOGP_EPS).log().mean()
    extra = -(advantage) * logp
    return extra, baseline


def _extra(logits, y, metric_loss: str, baseline: float):
    p = _head_mean_proba(logits)
    if metric_loss == "softf1":
        return _softf1_extra(p, y), baseline
    if metric_loss == "pairwise":
        return _pairwise_extra(p, y), baseline
    if metric_loss == "reinforce":
        return _reinforce_extra(p, y, baseline)
    raise ValueError(f"unknown metric_loss {metric_loss!r}")


_LOSSES = ("none", "softf1", "pairwise", "reinforce")


def train_one_metric(
    Z: np.ndarray,
    y: np.ndarray,
    spec: tabm.TabMSpec,
    seed: int,
    metric_loss: str = "none",
    lam: float = 0.5,
    log=print,
):
    """One seeded TabM fit. ``metric_loss="none"`` is the ``tabm.train_one`` update."""
    torch = _torch()
    if metric_loss not in _LOSSES:
        raise ValueError(f"unknown metric_loss {metric_loss!r}")
    tabm._seed_all(seed)
    dev = torch.device(spec.device if torch.cuda.is_available() else "cpu")
    model = tabm.build_model(Z.shape[1], spec).to(dev)
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
    baseline = 0.0
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
            if metric_loss != "none":
                extra, baseline = _extra(logits, yb, metric_loss, baseline)
                loss = loss + lam * extra
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


def fit_predict_metric(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    period_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec: tabm.TabMSpec,
    metric_loss: str = "none",
    lam: float = 0.5,
    log=print,
) -> list[np.ndarray]:
    """Train on ``(X_tr, y_tr)`` and return 12-class probabilities for every matrix in ``X_out``.

    ``period_tr`` is unused (kept for the cascade's signature). Seeds are averaged.
    """
    t0 = time.time()
    prep = tabm.fit_prep(X_tr, spec.quantiles)
    Z = tabm.transform(prep, X_tr)
    Zo = [tabm.transform(prep, X) for X in X_out]
    log(
        f"  nn-metric ({metric_loss}, lam {lam:g}): {Z.shape[1]} inputs "
        f"({len(prep.gap_cols)} gap flags), {len(Z)} rows ({time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        model = train_one_metric(Z, y_tr, spec, seed, metric_loss=metric_loss, lam=lam, log=log)
        for o, z in zip(outs, Zo, strict=True):
            o += tabm.predict(model, z, spec)
        del model
        _torch().cuda.empty_cache()
    log(f"  nn-metric fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [o / len(spec.seeds) for o in outs]
