"""Entity inputs for the TabM member.

``past_pool`` is a causal mean of earlier rows of one entity (strictly smaller
timestamps; a tie is not history). ``fit_predict_entity`` hashes each id column
into its own embedding table, drops ids to a reserved unknown row during
training, and concatenates the vectors to the numeric features from
``tabm.fit_prep`` / ``tabm.transform``.

An id's embedding is fit on that id's training labels. For leave-one-month-out,
train on earlier months only and score the held-out month with those frozen
embeddings. Do not fit on the scored month or on later months. ``past_pool``
reads features, not labels: pass the rows that are allowed to be visible (earlier
months, plus the scored month when within-month history is allowed).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ringtrace import tabm
from ringtrace.classes import N_CLASSES


def past_pool(X, entity, ts, n=8, cols=None):
    """Mean of up to ``n`` strictly earlier rows of the same entity, plus a count.

    Sort key is ``(entity, ts, row index)``. Only rows with a smaller timestamp
    qualify, so same-timestamp rows are excluded. Among those, the ``n`` latest
    are averaged (index breaks timestamp ties). The last column is how many rows
    entered the mean, in ``0..n``. A row with no history gets NaN means and count 0.
    ``cols`` picks feature columns; the default is every column.
    """
    X = np.asarray(X)
    entity = np.asarray(entity)
    ts = np.asarray(ts)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-d, got shape {X.shape}")
    n_rows, n_feat = X.shape
    if entity.shape != (n_rows,) or ts.shape != (n_rows,):
        raise ValueError("entity and ts must have one entry per row")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    cols = np.arange(n_feat) if cols is None else np.asarray(cols, dtype=np.intp)
    n_cols = int(cols.shape[0])
    out = np.empty((n_rows, n_cols + 1), dtype=np.float64)
    if n_rows == 0:
        return out

    order = np.lexsort((np.arange(n_rows), ts, entity))
    ent_s = entity[order]
    ts_s = ts[order]
    vals = np.asarray(X[:, cols], dtype=np.float64)[order]

    change_ent = np.empty(n_rows, dtype=bool)
    change_ent[0] = True
    if n_rows > 1:
        change_ent[1:] = ent_s[1:] != ent_s[:-1]
    change_grp = change_ent.copy()
    if n_rows > 1:
        change_grp[1:] |= ts_s[1:] != ts_s[:-1]

    ent_start = np.flatnonzero(change_ent)[np.cumsum(change_ent) - 1]
    hist_end = np.flatnonzero(change_grp)[np.cumsum(change_grp) - 1]
    hist_start = np.maximum(ent_start, hist_end - np.intp(n))
    counts = hist_end - hist_start

    # NaNs are missing. A window that contains one is NaN, and later windows are not.
    missing = np.isnan(vals)
    csum = np.cumsum(np.where(missing, 0.0, vals), axis=0)
    bad = np.cumsum(missing.astype(np.int64), axis=0)
    csum = np.concatenate([np.zeros((1, n_cols), dtype=np.float64), csum], axis=0)
    bad = np.concatenate([np.zeros((1, n_cols), dtype=np.int64), bad], axis=0)
    window = csum[hist_end] - csum[hist_start]
    n_bad = bad[hist_end] - bad[hist_start]
    means = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    has = counts > 0
    means[has] = window[has] / counts[has, None]
    means[n_bad > 0] = np.nan

    out[order, :n_cols] = means
    out[order, n_cols] = counts
    return out


@dataclass
class EntitySpec(tabm.TabMSpec):
    buckets: int = 2**18
    d_ent: int = 8
    ent_dropout: float = 0.2


def _torch():
    import torch

    return torch


def _hash_columns(ids: np.ndarray, buckets: int) -> np.ndarray:
    """Hash each id column into ``[0, buckets)``. Column ``j`` has its own salt."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 2:
        raise ValueError(f"ids must have shape (n, E), got {ids.shape}")
    salts = np.arange(ids.shape[1], dtype=np.uint64) * np.uint64(0xD1B54A32D192ED03)
    x = ids.astype(np.uint64) + salts
    x *= np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x % np.uint64(buckets)).astype(np.int64)


def unseen_to_unknown(ids_tr: np.ndarray, ids: np.ndarray, buckets: int) -> np.ndarray:
    """Hashes of ``ids`` with every id absent from that column of ``ids_tr`` sent to the
    reserved unknown row ``buckets``.

    An id first seen in the scored rows has no trained embedding of its own: its hash slot is
    either untouched noise or another training id's slot (a collision). Either way it would
    feed the model something that is not this id's history, so it gets the unknown vector the
    model learned under ``ent_dropout``.
    """
    ids_tr = np.asarray(ids_tr, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 2 or ids_tr.ndim != 2 or ids.shape[1] != ids_tr.shape[1]:
        raise ValueError(f"id shapes differ: {ids_tr.shape} and {ids.shape}")
    out = _hash_columns(ids, buckets)
    for j in range(ids.shape[1]):
        out[~np.isin(ids[:, j], ids_tr[:, j]), j] = buckets
    return out


def _entity_net(n_num: int, n_ent: int, spec: EntitySpec):
    torch = _torch()
    nn = torch.nn

    class EntityTabM(nn.Module):
        def __init__(self):
            super().__init__()
            # Index ``buckets`` is the reserved unknown row. Slots ``0..buckets-1`` are hashes.
            self.embeds = nn.ModuleList(
                [nn.Embedding(spec.buckets + 1, spec.d_ent) for _ in range(n_ent)]
            )
            self.body = tabm.build_model(n_num + n_ent * spec.d_ent, spec)

        def forward(self, numeric, ids):
            if ids.shape[1] == 0:
                return self.body(numeric)
            emb = torch.cat([table(ids[:, j]) for j, table in enumerate(self.embeds)], dim=1)
            return self.body(torch.cat([numeric, emb], dim=1))

    return EntityTabM()


def _drop_ids(ids, spec: EntitySpec, gen):
    """Replace each id with the unknown row with probability ``ent_dropout``."""
    torch = _torch()
    p = float(spec.ent_dropout)
    if p <= 0.0:
        return ids
    unknown = spec.buckets
    if p >= 1.0:
        return ids.new_full(ids.shape, unknown)
    draw = torch.rand(*ids.shape, device=ids.device, generator=gen)
    return torch.where(draw < p, ids.new_full(ids.shape, unknown), ids)


def _train_one(Z, ids, y, spec: EntitySpec, seed: int, log):
    torch = _torch()
    tabm._seed_all(seed)
    dev = torch.device(spec.device if torch.cuda.is_available() else "cpu")
    model = _entity_net(Z.shape[1], ids.shape[1], spec).to(dev)
    Zt = torch.from_numpy(np.ascontiguousarray(Z)).to(dev)
    It = torch.from_numpy(np.ascontiguousarray(ids)).to(dev)
    yt = torch.from_numpy(y.astype(np.int64, copy=False)).to(dev)
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
            logits = model(Zt[b], _drop_ids(It[b], spec, gen))
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
    del Zt, It, yt
    return model


def _predict(model, Z, ids, batch: int = 16384) -> np.ndarray:
    torch = _torch()
    dev = next(model.parameters()).device
    model.eval()
    out = np.empty((len(Z), N_CLASSES), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            x = torch.from_numpy(np.ascontiguousarray(Z[i : i + batch])).to(dev)
            d = torch.from_numpy(np.ascontiguousarray(ids[i : i + batch])).to(dev)
            p = torch.softmax(model(x, d).float(), dim=-1).mean(dim=1)
            out[i : i + batch] = p.cpu().numpy()
    return out


def fit_predict_entity(
    X_tr,
    ids_tr,
    y_tr,
    period_tr,
    X_out,
    ids_out,
    spec: EntitySpec,
    log=print,
):
    """Train on ``(X_tr, ids_tr, y_tr)`` and return 12-class probabilities for each ``X_out``.

    ``period_tr`` is unused (kept for the cascade's signature). ``ids_*`` are int64
    arrays of shape ``(n, E)``. Seeds are averaged, as in ``tabm.fit_predict``.
    """
    del period_tr
    t0 = time.time()
    prep = tabm.fit_prep(X_tr, spec.quantiles)
    Z = tabm.transform(prep, X_tr)
    Zo = [tabm.transform(prep, X) for X in X_out]
    Hi = _hash_columns(ids_tr, spec.buckets)
    Ho = [unseen_to_unknown(ids_tr, np.asarray(ids), spec.buckets) for ids in ids_out]
    if Hi.shape[0] != len(Z):
        raise ValueError("ids_tr and X_tr must have the same number of rows")
    log(
        f"  nn-entity: {Z.shape[1]} numeric + {Hi.shape[1]}x{spec.d_ent} emb "
        f"({len(Z)} rows, {time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    n_seeds = len(spec.seeds)
    for seed in spec.seeds:
        model = _train_one(Z, Hi, y_tr, spec, int(seed), log)
        for o, z, h in zip(outs, Zo, Ho, strict=True):
            o += _predict(model, z, h)
        del model
        _torch().cuda.empty_cache()
    log(f"  nn-entity fitted with {n_seeds} seeds ({time.time() - t0:.0f}s)")
    return [o / n_seeds for o in outs]
