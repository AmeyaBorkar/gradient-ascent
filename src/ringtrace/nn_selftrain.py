"""Soft self-training for the TabM member.

Labelled rows are one-hot with weight 1. Unlabelled target-month rows keep the
teacher's probability vector and a shared weight ``w_pseudo``. ``min_conf``
drops an unconfident row; ``cap`` keeps only the most confident rows of each
predicted class. The preprocessing is ``tabm.fit_prep`` on the labelled rows
only, and the optimiser loop is ``tabm.train_one`` with a soft target.

An empty pseudo set, or ``w_pseudo == 0``, calls ``tabm.train_one`` on the
labelled rows, so the CPU probabilities match ``tabm.fit_predict``.

The student copies the teacher's mistakes (confirmation bias), and a pseudo
set whose class mix is wrong pulls the predicted prior off the target mix.
``min_conf``, ``cap`` and a ``w_pseudo`` below 1 are the brakes. Set
``cap[c]`` from a label-free count: ``floor(sum_i P[i, c])`` so that hardening
the teacher cannot inflate class ``c`` past its own expected count, or
``round(n * pi_c)`` when a base rate ``pi`` is known without target labels.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Mapping

import numpy as np

from ringtrace import tabm
from ringtrace.classes import N_CLASSES


def _torch():
    import torch

    return torch


def _class_weights(counts: np.ndarray, power: float) -> np.ndarray:
    """``tabm.train_one``'s class weights from (possibly fractional) counts."""
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    cw = np.where(counts > 0.0, (total / np.maximum(counts, 1.0)) ** power, 0.0)
    scale = float((cw * counts).sum())
    if scale == 0.0 or total == 0.0:
        return np.ones(N_CLASSES, dtype=np.float64)
    return cw / scale * total


def _expected_counts(target: np.ndarray, row_w: np.ndarray) -> np.ndarray:
    """Row-weighted expected class counts. One-hot unit weights reproduce ``bincount``."""
    mass = np.asarray(target, dtype=np.float64) * np.asarray(row_w, dtype=np.float64)[:, None]
    return mass.sum(axis=0)


def _one_hot(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-d, got shape {y.shape}")
    if y.size and (int(y.min()) < 0 or int(y.max()) >= N_CLASSES):
        raise ValueError(f"labels must be in 0..{N_CLASSES - 1}")
    out = np.zeros((y.shape[0], N_CLASSES), dtype=np.float32)
    if y.size:
        out[np.arange(y.shape[0]), y] = 1.0
    return out


def _cap_limits(cap: Mapping[int, int] | np.ndarray | None) -> np.ndarray | None:
    """Per-class caps. ``None`` means no cap. A missing dict key is unlimited (``-1``)."""
    if cap is None:
        return None
    if isinstance(cap, Mapping):
        limits = np.full(N_CLASSES, -1, dtype=np.int64)
        for key, value in cap.items():
            c = operator.index(key)
            n = operator.index(value)
            if c < 0 or c >= N_CLASSES:
                raise ValueError(f"cap class {c} is outside 0..{N_CLASSES - 1}")
            if n < 0:
                raise ValueError(f"cap[{c}] must be non-negative, got {n}")
            limits[c] = n
        return limits
    arr = np.asarray(cap)
    if arr.shape != (N_CLASSES,):
        raise ValueError(f"cap must have shape ({N_CLASSES},) or be a dict, got {arr.shape}")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"cap values must be integers, got dtype {arr.dtype}")
    if np.any(arr < 0):
        raise ValueError("cap counts must be non-negative")
    return arr.astype(np.int64, copy=False)


def select_pseudo(
    P: np.ndarray, min_conf: float = 0.0, cap: Mapping[int, int] | np.ndarray | None = None
) -> np.ndarray:
    """Indices of teacher rows kept for self-training, in the original order.

    A row is dropped when its largest probability is below ``min_conf`` or is
    not finite. ``cap[c]`` then keeps at most that many rows whose argmax is
    ``c``, choosing the highest probabilities. Equal probabilities keep the
    earlier row. The confidence cut runs first, so a row below ``min_conf``
    does not consume a cap slot. Argmax ties take the lowest class index.
    """
    P = np.asarray(P)
    if P.ndim != 2 or P.shape[1] != N_CLASSES:
        raise ValueError(f"P must have shape (n, {N_CLASSES}), got {P.shape}")
    if P.shape[0] == 0:
        return np.empty(0, dtype=np.intp)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    eligible = np.flatnonzero(np.isfinite(conf) & (conf >= min_conf))
    limits = _cap_limits(cap)
    if limits is None:
        return eligible
    parts: list[np.ndarray] = []
    for c in range(N_CLASSES):
        members = eligible[pred[eligible] == c]
        limit = int(limits[c])
        if limit < 0 or members.size <= limit:
            chosen = members
        elif limit == 0:
            continue
        else:
            # mergesort is stable: equal confidence keeps the earlier row
            order = np.argsort(-conf[members], kind="stable")
            chosen = members[order[:limit]]
        if chosen.size:
            parts.append(chosen)
    if not parts:
        return np.empty(0, dtype=np.intp)
    return np.sort(np.concatenate(parts))


def _weighted_soft_ce(logits, target, row_w, class_w):
    """Class-weighted soft cross-entropy, as a weighted mean over rows.

    ``logits`` is ``(batch, k, classes)``. Each ensemble member is its own
    term, as in ``tabm.train_one``. For a target distribution ``T`` the term is
    the expectation ``sum_c T_c * class_w_c * (-log p_c)``, and the denominator
    is the expected class weight ``sum_c T_c * class_w_c``. One-hot ``T`` with
    unit row weights is therefore ``F.cross_entropy(..., weight=class_w)``,
    which divides by the sum of class weights. PyTorch's soft-target reduction
    divides by the batch count instead, so it is not used here.

    ``row_w`` multiplies every member of that row on both sides. With uniform
    class weights this is a weighted mean of the per-row cross-entropy. A row
    weight of 0 adds nothing. A zero target mass contributes nothing, including
    when the matching log-probability is infinite.
    """
    log_p = _torch().log_softmax(logits, dim=-1)
    tw = target * class_w
    prod = log_p * tw[:, None, :]
    prod = _torch().where(tw[:, None, :] == 0, _torch().zeros_like(prod), prod)
    per = -prod.sum(dim=-1)
    numer = (per * row_w[:, None]).sum()
    denom = (row_w * tw.sum(dim=-1)).sum() * logits.shape[1]
    if float(denom) == 0.0:
        return logits.sum() * 0.0
    return numer / denom


def train_one_soft(
    Z: np.ndarray, T: np.ndarray, w: np.ndarray, spec: tabm.TabMSpec, seed: int, log=print
):
    """One seeded TabM fit on soft targets ``T`` ``(n, 12)`` and row weights ``w``.

    Rows with weight 0 are removed before the permutation, so they do not change
    the schedule, the dropout masks, or the loss. Features stay on CPU and one
    batch is copied to the device at a time: GPU memory follows the batch and
    the model, not the labelled-plus-pseudo matrix. Class weights use
    ``spec.class_weight_power`` on the row-weighted expected counts, which is
    ``tabm.train_one``'s formula when ``T`` is one-hot and ``w`` is 1.
    """
    torch = _torch()
    Z = np.asarray(Z, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    w = np.asarray(w, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-d, got shape {Z.shape}")
    if T.shape != (Z.shape[0], N_CLASSES):
        raise ValueError(f"T must have shape {(Z.shape[0], N_CLASSES)}, got {T.shape}")
    if w.shape != (Z.shape[0],):
        raise ValueError(f"w must have shape {(Z.shape[0],)}, got {w.shape}")
    if not np.isfinite(T).all():
        raise ValueError("T must be finite")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("w must be finite and non-negative")
    if not np.all(w > 0):
        keep = w > 0
        Z, T, w = Z[keep], T[keep], w[keep]
    if Z.shape[0] == 0:
        raise ValueError("no rows with a positive weight")

    cw_np = _class_weights(_expected_counts(T, w), spec.class_weight_power)
    tabm._seed_all(seed)
    dev = torch.device(spec.device if torch.cuda.is_available() else "cpu")
    model = tabm.build_model(Z.shape[1], spec).to(dev)
    # Share the numpy buffers. Indexing below copies one batch, then .to(dev).
    Zt = torch.from_numpy(np.ascontiguousarray(Z))
    Tt = torch.from_numpy(np.ascontiguousarray(T))
    wt = torch.from_numpy(np.ascontiguousarray(w.astype(np.float32)))
    cw = torch.tensor(cw_np, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    steps_per = math.ceil(len(Z) / spec.batch)
    total = steps_per * spec.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * total)) * max(0.0, 1 - s / total)
    )
    # CPU generator so the permutation does not depend on keeping Z on device.
    # On CPU this matches ``tabm.train_one``'s generator; dropout uses the global seed.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for ep in range(spec.epochs):
        model.train()
        perm = torch.randperm(len(Z), generator=gen)
        run = 0.0
        for i in range(steps_per):
            b = perm[i * spec.batch : (i + 1) * spec.batch]
            logits = model(Zt[b].to(dev))
            loss = _weighted_soft_ce(logits, Tt[b].to(dev), wt[b].to(dev), cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
        log(
            f"    seed {seed} epoch {ep + 1}/{spec.epochs} loss {run / steps_per:.4f} "
            f"({time.time() - t0:.0f}s)"
        )
    del Zt, Tt, wt
    return model


def _fit_labelled(Z, y, Zo, X_out, spec, log):
    """``tabm.fit_predict``'s seed loop, on an already transformed matrix."""
    torch = _torch()
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        model = tabm.train_one(Z, y, spec, seed, log=log)
        for o, z in zip(outs, Zo, strict=True):
            o += tabm.predict(model, z, spec)
        del model
        torch.cuda.empty_cache()
    return [o / len(spec.seeds) for o in outs]


def fit_predict_selftrain(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_pseudo: np.ndarray,
    P_pseudo: np.ndarray,
    X_out: list[np.ndarray],
    spec: tabm.TabMSpec,
    w_pseudo: float = 0.3,
    min_conf: float = 0.0,
    cap: Mapping[int, int] | np.ndarray | None = None,
    log=print,
) -> list[np.ndarray]:
    """Train on labelled rows plus soft pseudo rows; probabilities for each ``X_out``.

    ``tabm.fit_prep`` sees ``X_tr`` only. Pseudo rows that survive ``min_conf``
    and ``cap`` are appended with target ``P_pseudo`` and weight ``w_pseudo``
    (labelled rows: one-hot, weight 1). Seeds are averaged, as in
    ``tabm.fit_predict``. ``X_pseudo`` empty or ``w_pseudo == 0`` uses the
    labelled rows alone and matches ``tabm.fit_predict`` on CPU.

    ``cap[c]`` is a label-free budget for class ``c``. ``floor(P_pseudo[:, c].sum())``
    stops argmax hardening from exceeding the teacher's expected count. A base
    rate estimated without target labels is the alternative, ``round(n * pi_c)``,
    and is the one that corrects a teacher whose mix is shifted.
    """
    if not np.isfinite(w_pseudo) or w_pseudo < 0:
        raise ValueError(f"w_pseudo must be finite and non-negative, got {w_pseudo}")
    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr)
    X_pseudo = np.asarray(X_pseudo)
    P_pseudo = np.asarray(P_pseudo)
    if X_tr.ndim != 2:
        raise ValueError(f"X_tr must be 2-d, got shape {X_tr.shape}")
    if X_pseudo.ndim != 2 or P_pseudo.ndim != 2:
        raise ValueError(
            f"X_pseudo and P_pseudo must be 2-d, got {X_pseudo.shape} and {P_pseudo.shape}"
        )
    if X_pseudo.shape[0] != P_pseudo.shape[0]:
        raise ValueError(f"X_pseudo has {X_pseudo.shape[0]} rows, P_pseudo has {P_pseudo.shape[0]}")
    if P_pseudo.shape[1] != N_CLASSES:
        raise ValueError(f"P_pseudo must have {N_CLASSES} columns, got {P_pseudo.shape[1]}")
    if X_pseudo.shape[0] and X_pseudo.shape[1] != X_tr.shape[1]:
        raise ValueError(f"X_pseudo has {X_pseudo.shape[1]} columns, X_tr has {X_tr.shape[1]}")

    t0 = time.time()
    prep = tabm.fit_prep(X_tr, spec.quantiles)
    Z = tabm.transform(prep, X_tr)
    Zo = [tabm.transform(prep, X) for X in X_out]
    idx = (
        select_pseudo(P_pseudo, min_conf, cap)
        if w_pseudo > 0.0 and X_pseudo.shape[0] > 0
        else np.empty(0, dtype=np.intp)
    )
    log(
        f"  nn-selftrain: {Z.shape[1]} inputs ({len(prep.gap_cols)} gap flags), "
        f"{len(Z)} labelled + {idx.size} pseudo "
        f"(w {w_pseudo:g}, min_conf {min_conf:g}) ({time.time() - t0:.0f}s)"
    )
    if idx.size == 0:
        outs = _fit_labelled(Z, y_tr, Zo, X_out, spec, log)
        log(f"  nn-selftrain fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
        return outs

    n_lab = len(Z)
    Zp = tabm.transform(prep, X_pseudo[idx])
    Z_all = np.empty((n_lab + idx.size, Z.shape[1]), dtype=np.float32)
    Z_all[:n_lab] = Z
    Z_all[n_lab:] = Zp
    del Zp
    T_all = np.concatenate([_one_hot(y_tr), np.ascontiguousarray(P_pseudo[idx], dtype=np.float32)])
    w_all = np.concatenate(
        [np.ones(n_lab, dtype=np.float64), np.full(idx.size, float(w_pseudo), dtype=np.float64)]
    )
    torch = _torch()
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        model = train_one_soft(Z_all, T_all, w_all, spec, seed, log=log)
        for o, z in zip(outs, Zo, strict=True):
            o += tabm.predict(model, z, spec)
        del model
        torch.cuda.empty_cache()
    log(f"  nn-selftrain fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [o / len(spec.seeds) for o in outs]
