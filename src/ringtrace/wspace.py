"""Weight-space methods on the TabM member: change the trained weights, not the predictions.

    python -m ringtrace.wspace fit fold9 s13 ...     # train the endpoints, write every arm
    python -m ringtrace.wspace rebasin fold9 ...     # weight-matched seed average (saved weights)
    python -m ringtrace.wspace score                 # every arm against the base (fraud-only)

Four methods, all on the network of ``ringtrace.tabm``:

* **task arithmetic for the label regime** (``ta<alpha>``): theta_g is trained on the given labels
  and theta_c on event-cleaned labels of the fit months, from the same seed (same initial weights,
  same batch order). The arm is theta_g + alpha (theta_c - theta_g); alpha > 1 extrapolates.
* **interpolation across time** (``ti<alpha>``): theta_r is theta_g fine-tuned for a few epochs on
  the fit month nearest the scored month; the arm is theta_g + alpha (theta_r - theta_g).
* **re-basin averaging** (``rb_*``): independently trained seeds are aligned to the first by weight
  matching over every symmetry of the network (member order, hidden units, embedding units,
  frequencies and the sign gauges of the shared weights), then averaged into one model.
* **head re-optimisation** (``head_*``): the body is frozen and the legit column of the output
  layer is rescaled and shifted to maximise the decoded macro-F1 of the blend on two months.

A context is one fit: ``fold<m>`` (the normal ledger, month m held out, October predicted too) or
``s<seed>`` (an October-like frame, fitted on July + August, September held out). ``fit`` writes
``$RINGTRACE_OUT/wspace/<context>/``: the weights of every endpoint, ``arms.npz`` (the legit
probability of every arm on the held-out rows, mean over seeds) and ``report.json`` (loss along
each path, barriers, distances). Only the legit probability is kept: the member is used in the
fraud-only role, where the tree members set the split among fraud types.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ringtrace import tabm
from ringtrace.classes import N_CLASSES

ALPHAS = (0.5, 1.0, 1.5, 2.0)
PATH = (0.0, 0.25, 0.5, 0.75, 1.0)
RECENT = {7: 8, 8: 9, 9: 8}  # the fit month nearest the held-out month (the later one on a tie)
FRAME_RECENT = 8
RECIPE = "recipes/tabm.json"
CLEAN_MODE = "event_l2"
FT_EPOCHS = 2
FT_LR = 2e-4


# ---------------------------------------------------------------------------------------------
# weights as dictionaries of tensors


def state(model) -> dict:
    """A CPU copy of the model's weights."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def lerp(a: dict, b: dict, alpha: float) -> dict:
    """a + alpha (b - a); alpha outside [0, 1] extrapolates."""
    return {k: a[k] + alpha * (b[k] - a[k]) for k in a}


def add_vectors(base: dict, ends: list[dict], alphas: list[float]) -> dict:
    """base + sum_i alpha_i (end_i - base): task vectors added at their own scale."""
    out = {k: v.clone() for k, v in base.items()}
    for end, alpha in zip(ends, alphas, strict=True):
        for k in out:
            out[k] += alpha * (end[k] - base[k])
    return out


def mean_state(states: list[dict]) -> dict:
    return {k: sum(s[k] for s in states) / len(states) for k in states[0]}


def distance(a: dict, b: dict) -> float:
    return math.sqrt(sum(float(((a[k] - b[k]).double() ** 2).sum()) for k in a))


def norm(a: dict) -> float:
    return math.sqrt(sum(float((v.double() ** 2).sum()) for v in a.values()))


def cosine(base: dict, a: dict, b: dict) -> float:
    """Cosine of the two task vectors a - base and b - base."""
    dot = sum(float(((a[k] - base[k]).double() * (b[k] - base[k]).double()).sum()) for k in base)
    den = distance(a, base) * distance(b, base)
    return dot / den if den > 0 else 0.0


def load(sd: dict, spec: tabm.TabMSpec):
    """A model holding the weights ``sd`` on the spec's device."""
    torch = tabm._torch()
    dev = torch.device(spec.device if torch.cuda.is_available() else "cpu")
    model = tabm.build_model(sd["embed.c"].shape[0], spec).to(dev)
    model.load_state_dict(sd)
    return model


def proba(sd: dict, Z: np.ndarray, spec: tabm.TabMSpec) -> np.ndarray:
    model = load(sd, spec)
    out = tabm.predict(model, Z, spec)
    del model
    return out


def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def barrier(losses: list[float], alphas=PATH) -> float:
    """The largest excess of the loss over the straight line between the two ends (0 = linearly
    connected)."""
    a = np.asarray(alphas, dtype=np.float64)
    v = np.asarray(losses, dtype=np.float64)
    line = v[0] + (v[-1] - v[0]) * (a - a[0]) / (a[-1] - a[0])
    return float((v - line).max())


def path_losses(a: dict, b: dict, Z: np.ndarray, y: np.ndarray, spec, alphas=PATH) -> list:
    return [cross_entropy(proba(lerp(a, b, t), Z, spec), y) for t in alphas]


# ---------------------------------------------------------------------------------------------
# training


def fine_tune(
    sd: dict,
    Z: np.ndarray,
    y: np.ndarray,
    spec: tabm.TabMSpec,
    seed: int,
    epochs: int = FT_EPOCHS,
    lr: float = FT_LR,
    log=print,
) -> dict:
    """Continue from the weights ``sd`` on (Z, y): a fresh AdamW, the rate falling linearly from
    ``lr`` to 0. Returns the new weights."""
    if spec.class_weight_power:
        raise ValueError("fine_tune supports unweighted training only")
    torch = tabm._torch()
    tabm._seed_all(seed)
    model = load(sd, spec)
    dev = next(model.parameters()).device
    Zt = torch.from_numpy(Z).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=spec.weight_decay)
    steps_per = math.ceil(len(Z) / spec.batch)
    total = steps_per * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    gen = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Z), device=dev, generator=gen)
        run = 0.0
        for i in range(steps_per):
            b = perm[i * spec.batch : (i + 1) * spec.batch]
            loss = torch.nn.functional.cross_entropy(
                model(Zt[b]).reshape(-1, N_CLASSES), yt[b].repeat_interleave(spec.k)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
        log(
            f"    fine-tune seed {seed} epoch {ep + 1}/{epochs} loss {run / steps_per:.4f} "
            f"({time.time() - t0:.0f}s)"
        )
    out = state(model)
    del model, Zt, yt
    return out


def endpoints(
    Z: np.ndarray,
    y: np.ndarray,
    y_clean: np.ndarray,
    recent: np.ndarray,
    spec: tabm.TabMSpec,
    seed: int,
    log=print,
) -> dict:
    """One seed's three endpoints: ``g`` (given labels), ``c`` (cleaned labels, same seed, so
    the same initial weights and batch order), ``r`` (``g`` fine-tuned on the ``recent`` rows)."""
    out = {}
    for kind, target in (("g", y), ("c", y_clean)):
        model = tabm.train_one(Z, target, spec, seed, log=log)
        out[kind] = state(model)
        del model
    out["r"] = fine_tune(out["g"], Z[recent], y[recent], spec, seed, log=log)
    return out


def arm_weights(ends: dict, alphas=ALPHAS) -> dict:
    """Every arm's weights from one seed's endpoints."""
    arms = {"base": ends["g"]}
    for a in alphas:
        arms[f"ta{a:g}"] = lerp(ends["g"], ends["c"], a)
        arms[f"ti{a:g}"] = lerp(ends["g"], ends["r"], a)
    arms["ta1+ti1"] = add_vectors(ends["g"], [ends["c"], ends["r"]], [1.0, 1.0])
    return arms


# ---------------------------------------------------------------------------------------------
# re-basin: the network's symmetries and weight matching


def _member_keys(sd: dict) -> list[str]:
    keys = [k for k in ("head_w", "head_b") if k in sd]
    layer = 0
    while f"body.{3 * layer}.r" in sd:
        keys.extend(f"body.{3 * layer}.{suffix}" for suffix in ("r", "s", "b"))
        layer += 1
    return keys


def _sgn(v):
    """Sign of an inner product. Zero carries no gauge preference, so it counts as +1."""
    sign = v.sign()
    return v.new_ones(v.shape).where(sign == 0, sign)


def _assign(sim):
    """Column chosen for each row of a similarity matrix (maximum-weight matching)."""
    from scipy.optimize import linear_sum_assignment

    cost = np.ascontiguousarray(sim.detach().numpy())
    row, col = linear_sum_assignment(cost, maximize=True)
    perm = np.empty(cost.shape[0], dtype=np.int64)
    perm[row] = col
    return tabm._torch().from_numpy(perm)


def permute_members(sd: dict, pi) -> dict:
    """Reorder ensemble members. The shared weights have no member axis."""
    out = {k: v.clone() for k, v in sd.items()}
    for key in _member_keys(sd):
        out[key] = sd[key][pi]
    return out


def permute_freq(sd: dict, perm, sign) -> dict:
    """Reorder periodic frequencies. ``sign`` flips frequency and the odd (sine) row together."""
    out = {k: v.clone() for k, v in sd.items()}
    c, w = sd["embed.c"], sd["embed.w"]
    q = c.shape[1]
    sign = sign.to(dtype=c.dtype, device=c.device)
    perm = perm.to(device=c.device)
    out["embed.c"] = sign * c.gather(1, perm)
    index = perm.unsqueeze(-1).expand(-1, -1, w.shape[-1])
    moved = w.clone()
    moved[:, :q] = sign.unsqueeze(-1) * w[:, :q].gather(1, index)
    moved[:, q:] = w[:, q:].gather(1, index)
    out["embed.w"] = moved
    return out


def permute_embed(sd: dict, perm, sign) -> dict:
    """Reorder each feature's embedding units. ``sign`` is the first block's input gauge."""
    out = {k: v.clone() for k, v in sd.items()}
    w, bias = sd["embed.w"], sd["embed.b"]
    sign = sign.to(dtype=w.dtype, device=w.device)
    perm = perm.to(device=w.device)
    out["embed.w"] = w.gather(2, perm.unsqueeze(1).expand(-1, w.shape[1], -1))
    out["embed.b"] = bias.gather(1, perm)
    features, units = perm.shape
    for key in ("body.0.lin.weight", "body.0.r"):
        tensor = sd[key]
        grouped = tensor.reshape(tensor.shape[0], features, units)
        index = perm.unsqueeze(0).expand(grouped.shape[0], -1, -1)
        out[key] = (sign * grouped.gather(2, index)).reshape_as(tensor)
    return out


def permute_hidden(sd: dict, layer: int, perm, sign_in, sign_out) -> dict:
    """Reorder one block's outputs. ``sign_out`` gauges the next block's inputs, except on the
    last block, where the head has no input scale and the sign is ignored."""
    out = {k: v.clone() for k, v in sd.items()}
    key = f"body.{3 * layer}"
    weight = sd[f"{key}.lin.weight"]
    sign_in = sign_in.to(dtype=weight.dtype, device=weight.device)
    perm = perm.to(device=weight.device)
    out[f"{key}.lin.weight"] = sign_in[:, None] * weight[perm]
    out[f"{key}.s"] = sign_in * sd[f"{key}.s"][:, perm]
    out[f"{key}.b"] = sd[f"{key}.b"][:, perm]
    nxt = f"body.{3 * (layer + 1)}"
    if f"{nxt}.lin.weight" in sd:
        sign_out = sign_out.to(dtype=weight.dtype, device=weight.device)
        out[f"{nxt}.lin.weight"] = sign_out * sd[f"{nxt}.lin.weight"][:, perm]
        out[f"{nxt}.r"] = sign_out * sd[f"{nxt}.r"][:, perm]
    else:
        out["head_w"] = sd["head_w"][:, perm]
    return out


def _picked(sim, perm):
    return sim.gather(1, perm.unsqueeze(1)).squeeze(1)


def _match_frequencies(a, b):
    torch = tabm._torch()
    c_a, c_b = a["embed.c"], b["embed.c"]
    w_a, w_b = a["embed.w"], b["embed.w"]
    q = c_a.shape[1]
    paired = c_a[:, :, None] * c_b[:, None, :]
    paired = paired + torch.einsum("fqd,fpd->fqp", w_a[:, :q], w_b[:, :q])
    cosine = torch.einsum("fqd,fpd->fqp", w_a[:, q:], w_b[:, q:])
    sim = paired.abs() + cosine
    perm = torch.stack([_assign(sim[i]) for i in range(sim.shape[0])])
    return permute_freq(b, perm, _sgn(paired.gather(2, perm.unsqueeze(-1)).squeeze(-1)))


def _match_embedding(a, b):
    torch = tabm._torch()
    w_a, w_b = a["embed.w"], b["embed.w"]
    bias_a, bias_b = a["embed.b"], b["embed.b"]
    value = torch.einsum("fqd,fqe->fde", w_a, w_b) + bias_a[:, :, None] * bias_b[:, None, :]
    features, units = bias_a.shape
    cross = None
    for key in ("body.0.lin.weight", "body.0.r"):
        left = a[key].reshape(a[key].shape[0], features, units)
        right = b[key].reshape(b[key].shape[0], features, units)
        gram = torch.einsum("nfd,nfe->fde", left, right)
        cross = gram if cross is None else cross + gram
    perm = torch.stack([_assign((value + cross.abs())[i]) for i in range(features)])
    return permute_embed(b, perm, _sgn(cross.gather(2, perm.unsqueeze(-1)).squeeze(-1)))


def _match_hidden(a, b, layer):
    torch = tabm._torch()
    key = f"body.{3 * layer}"
    incoming = a[f"{key}.lin.weight"] @ b[f"{key}.lin.weight"].T
    incoming = incoming + a[f"{key}.s"].T @ b[f"{key}.s"]
    bias = a[f"{key}.b"].T @ b[f"{key}.b"]
    nxt = f"body.{3 * (layer + 1)}"
    if f"{nxt}.lin.weight" in a:
        outgoing = a[f"{nxt}.lin.weight"].T @ b[f"{nxt}.lin.weight"]
        outgoing = outgoing + a[f"{nxt}.r"].T @ b[f"{nxt}.r"]
        sim = incoming.abs() + bias + outgoing.abs()
    else:
        # No sign gauge on the head: its inner product is used as it stands.
        outgoing = torch.einsum("kxc,kyc->xy", a["head_w"], b["head_w"])
        sim = incoming.abs() + bias + outgoing
    perm = _assign(sim)
    sign_in = _sgn(_picked(incoming, perm))
    # sign_out is ignored on the last block; the head branch never reads it.
    sign_out = _sgn(_picked(outgoing, perm)) if f"{nxt}.lin.weight" in a else sign_in
    return permute_hidden(b, layer, perm, sign_in, sign_out)


def _match_members(a, b):
    sim = None
    for key in _member_keys(a):
        left, right = a[key], b[key]
        if left.ndim > 2:
            left = left.reshape(left.shape[0], -1)
            right = right.reshape(right.shape[0], -1)
        gram = left @ right.T
        sim = gram if sim is None else sim + gram
    return permute_members(b, _assign(sim))


def _as_float64(sd: dict) -> dict:
    torch = tabm._torch()
    return {k: v.detach().to(dtype=torch.float64, device="cpu").clone() for k, v in sd.items()}


def weight_matching(a: dict, b: dict, sweeps: int = 20) -> tuple[dict, list[float]]:
    """Align ``b`` to ``a`` by coordinate descent over the symmetries above.

    The history starts with the distance before any sweep and gains one entry after every sweep.
    """
    ref, cur = _as_float64(a), _as_float64(b)
    n_layers = 0
    while f"body.{3 * n_layers}.lin.weight" in cur:
        n_layers += 1
    history = [distance(ref, cur)]
    for _ in range(sweeps):
        before = history[-1]
        # members first: the +-1 member scales carry most of the weight norm, so every later
        # similarity is dominated by them and needs the member order settled
        cur = _match_members(ref, cur)
        cur = _match_frequencies(ref, cur)
        cur = _match_embedding(ref, cur)
        for layer in range(n_layers):
            cur = _match_hidden(ref, cur, layer)
        history.append(distance(ref, cur))
        if before == 0.0 or (before - history[-1]) / before < 1e-9:
            break
    aligned = {k: cur[k].to(dtype=b[k].dtype, device=b[k].device) for k in b}
    return aligned, history


def merge_seeds(states: list[dict], align: bool = True) -> tuple[dict, list]:
    """Average states. With ``align``, every later state is weight-matched to the first."""
    if not align:
        return mean_state(states), []
    aligned = [states[0]]
    histories = []
    for other in states[1:]:
        matched, hist = weight_matching(states[0], other)
        aligned.append(matched)
        histories.append(hist)
    return mean_state(aligned), histories


# ---------------------------------------------------------------------------------------------
# contexts


def wspace_dir() -> Path:
    from ringtrace import paths

    d = paths.out_dir() / "wspace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def context(name: str, recipe_path: str = RECIPE) -> dict:
    """The matrices and row sets of one fit (see the module text)."""
    from ringtrace import eventclean, octlike, store
    from ringtrace import run as runner

    recipe = runner.load_recipe(recipe_path)
    if name.startswith("fold"):
        held = int(name[4:])
        store.use_frame(None)
        led = store.ledger()
        X, _ = runner.feature_matrix(recipe)
        is_test = led["is_test"].to_numpy()
        period = led["period"].to_numpy()
        fit = ~is_test & (period != held)
        hold = ~is_test & (period == held)
        rows = np.flatnonzero(hold)
        test = np.flatnonzero(is_test)
        recent = RECENT[held]
    elif name.startswith("s"):
        frame = octlike.Frame(int(name[1:]))
        led = frame.led
        X, _ = runner.feature_matrix(recipe)
        period = frame.period
        fit, hold = frame.fit, frame.hold
        rows = frame.stacked[hold]
        test = None
        recent = FRAME_RECENT
    else:
        raise ValueError(f"unknown context {name!r}")
    y = led["y"].to_numpy().astype(np.int64)
    y_clean = octlike.train_target({"train_on": CLEAN_MODE}, y, fit, None, eventclean.ledger_events)
    spec, _ = runner.model(recipe)
    return {
        "name": name,
        "X": X,
        "y": y,
        "y_clean": y_clean,
        "fit": fit,
        "hold": hold,
        "rows": rows,
        "test": test,
        "recent": period[fit] == recent,
        "recent_month": recent,
        "spec": spec,
    }


def matrices(ctx: dict) -> dict:
    """Preprocessing fitted on the fit rows, applied to the fit, held-out and October rows."""
    X, spec = ctx["X"], ctx["spec"]
    prep = tabm.fit_prep(X[ctx["fit"]], spec.quantiles)
    out = {"fit": tabm.transform(prep, X[ctx["fit"]]), "hold": tabm.transform(prep, X[ctx["hold"]])}
    if ctx["test"] is not None:
        out["test"] = tabm.transform(prep, X[ctx["test"]])
    return out


def save_state(path: Path, sd: dict) -> None:
    torch = tabm._torch()
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(sd, tmp)
    tmp.replace(path)


def load_state(path: Path) -> dict:
    return tabm._torch().load(path, map_location="cpu", weights_only=True)


def fit_context(name: str, seeds=(0, 1, 2), log=print) -> dict:
    """Train the endpoints of every seed and write every arm of the context."""
    torch = tabm._torch()
    t0 = time.time()
    ctx = context(name)
    spec = ctx["spec"]
    out = wspace_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    Z = matrices(ctx)
    y_fit, yc_fit = ctx["y"][ctx["fit"]], ctx["y_clean"][ctx["fit"]]
    y_hold = ctx["y"][ctx["hold"]]
    report: dict = {
        "context": name,
        "inputs": int(Z["fit"].shape[1]),
        "fit_rows": int(len(y_fit)),
        "hold_rows": int(len(y_hold)),
        "relabelled": int((y_fit != yc_fit).sum()),
        "legit_to_fraud": int(((y_fit == 0) & (yc_fit > 0)).sum()),
        "recent_month": int(ctx["recent_month"]),
        "recent_rows": int(ctx["recent"].sum()),
        "fine_tune": {"epochs": FT_EPOCHS, "lr": FT_LR},
        "seeds": {},
    }
    log(f"[wspace] {name}: {json.dumps(report)} ({time.time() - t0:.0f}s)")
    sums: dict = {}
    for n_done, seed in enumerate(seeds, start=1):
        t1 = time.time()
        ends = endpoints(Z["fit"], y_fit, yc_fit, ctx["recent"], spec, seed, log=log)
        for kind, sd in ends.items():
            save_state(out / f"sd_{kind}_seed{seed}.pt", sd)
        rep = {
            "norm_g": norm(ends["g"]),
            "dist_c": distance(ends["g"], ends["c"]),
            "dist_r": distance(ends["g"], ends["r"]),
            "cos_c_r": cosine(ends["g"], ends["c"], ends["r"]),
            "loss": {},
        }
        path: dict = {"c": {}, "r": {}}
        for arm, sd in arm_weights(ends).items():
            p = proba(sd, Z["hold"], spec)
            rep["loss"][arm] = cross_entropy(p, y_hold)
            sums[arm] = sums.get(arm, 0.0) + p[:, 0]
            if "test" in Z:
                key = f"{arm}__test"
                sums[key] = sums.get(key, 0.0) + proba(sd, Z["test"], spec)[:, 0]
        for kind in ("c", "r"):
            mid = [
                cross_entropy(proba(lerp(ends["g"], ends[kind], t), Z["hold"], spec), y_hold)
                for t in (0.25, 0.75)
            ]
            tag = "ta" if kind == "c" else "ti"
            v = [
                rep["loss"]["base"],
                mid[0],
                rep["loss"][f"{tag}0.5"],
                mid[1],
                rep["loss"][f"{tag}1"],
            ]
            path[kind] = {"losses": v, "barrier": barrier(v)}
        rep["path"] = path
        rep["wall_s"] = time.time() - t1
        report["seeds"][str(seed)] = rep
        log(
            f"[wspace] {name} seed {seed}: |g| {rep['norm_g']:.2f}, |c-g| {rep['dist_c']:.2f}, "
            f"|r-g| {rep['dist_r']:.2f}, cos {rep['cos_c_r']:+.3f}, barrier c "
            f"{path['c']['barrier']:+.5f} r {path['r']['barrier']:+.5f}, held-out loss "
            + " ".join(f"{a} {v:.5f}" for a, v in rep["loss"].items())
            + f" ({time.time() - t1:.0f}s)"
        )
        arrays = {k: (v / n_done).astype(np.float32) for k, v in sums.items()}
        arrays["rows"] = ctx["rows"]
        if ctx["test"] is not None:
            arrays["test_rows"] = ctx["test"]
        tmp = out / ".arms.tmp.npz"
        np.savez(tmp, **arrays)
        tmp.replace(out / "arms.npz")
        report["seeds_done"] = list(seeds[:n_done])
        report["wall_s"] = time.time() - t0
        (out / "report.json").write_text(json.dumps(report, indent=1))
        del ends
        torch.cuda.empty_cache()
    log(f"[wspace] {name} done in {time.time() - t0:.0f}s")
    return report


# ---------------------------------------------------------------------------------------------
# scoring: every arm in the fraud-only role

from ringtrace import final  # noqa: E402

TREES = (
    ("champion", "blend-champion-m1", "champion"),
    ("lgbm", "lgbm", "lgbm"),
    ("catboost", "catboost", "catboost"),
    ("clean-event", "clean-event", "clean-event"),
    ("clean-l2", "clean-l2", "clean-l2"),
)
_FOLD_MONTHS = (7, 8, 9)
_FRAME_SEEDS = (13, 14, 15)


def fraud_only(p_legit) -> np.ndarray:
    """(n, 12): column 0 is the legit probability and the other columns share the rest."""
    p = np.asarray(p_legit, dtype=np.float64)
    out = np.empty((len(p), N_CLASSES), dtype=np.float64)
    out[:, 0] = p
    out[:, 1:] = (1.0 - p)[:, None] / (N_CLASSES - 1)
    return out


def load_arms(ctx_dir, files=("arms.npz", "rebasin.npz", "head.npz")) -> dict:
    """Merge the npz files that exist. Keys ending in ``__test`` are October legit probabilities.

    ``rows`` must be identical in every file that carries them, and so must ``test_rows``. A later
    file overwrites an arm of the same name."""
    ctx_dir = Path(ctx_dir)
    rows = None
    test_rows = None
    arms: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    for name in files:
        path = ctx_dir / name
        if not path.is_file():
            continue
        with np.load(path) as blob:
            if "rows" in blob:
                got = blob["rows"].copy()
                if rows is not None and not np.array_equal(rows, got):
                    raise ValueError(f"{name}: rows do not match an earlier file")
                rows = got if rows is None else rows
            if "test_rows" in blob:
                got = blob["test_rows"].copy()
                if test_rows is not None and not np.array_equal(test_rows, got):
                    raise ValueError(f"{name}: test_rows do not match an earlier file")
                test_rows = got if test_rows is None else test_rows
            for key in blob.files:
                if key in ("rows", "test_rows"):
                    continue
                vec = blob[key].copy()
                if key.endswith("__test"):
                    test[key.removesuffix("__test")] = vec
                else:
                    arms[key] = vec
    return {"rows": rows, "test_rows": test_rows, "arms": arms, "test": test}


def arm_matrix(base_full, te, folds, arm) -> np.ndarray:
    """A copy of the stored member, with this arm written on the held-out rows it covers.

    October is the mean of the three folds' test vectors when every fold has one, and stays as
    stored otherwise. Each fold's ``test_rows`` must be ``te``."""
    out = np.array(base_full, dtype=np.float64, copy=True)
    for loaded in folds.values():
        if arm in loaded["arms"]:
            out[np.asarray(loaded["rows"])] = fraud_only(loaded["arms"][arm])
    if all(m in folds and arm in folds[m]["test"] for m in _FOLD_MONTHS):
        vecs = []
        for month in _FOLD_MONTHS:
            got = folds[month]["test_rows"]
            if got is None or not np.array_equal(got, te):
                raise ValueError(f"fold{month} test_rows do not match the October rows")
            vecs.append(np.asarray(folds[month]["test"][arm], dtype=np.float64))
        out[np.asarray(te)] = fraud_only(np.mean(vecs, axis=0))
    return out


class ArmBench(final.Bench):
    """Bench that serves a weight-space arm (run ``wspace:<name>``) from ``arm_arrays``."""

    def __init__(self, led, targets, groups, runs_dir, cells_dir=None, log=print):
        super().__init__(led, targets, groups, runs_dir, cells_dir, log=log)
        self.arm_arrays = {}

    def member(self, m: dict) -> np.ndarray:
        run = m["run"]
        if run.startswith("wspace:"):
            return self.arm_arrays[run.removeprefix("wspace:")]
        return super().member(m)


def arm_recipe(arm: str) -> dict:
    """Final recipe: five tree members at 0.1 and this arm as the fraud-only network at 0.5."""
    if arm == "stored":
        run, cells = "tabm", "tabm"
    else:
        run, cells = f"wspace:{arm}", f"wspace-{arm}"
    members = [
        {"name": name, "run": tree_run, "variant": "fold", "cells": tree_cells, "role": "full"}
        for name, tree_run, tree_cells in TREES
    ]
    members.append({"name": "tabm", "run": run, "variant": "fold", "cells": cells, "role": "fraud"})
    values = {name: 0.1 for name, _, _ in TREES}
    values["tabm"] = 0.5
    return final.check_recipe(
        {
            "name": f"wspace-{arm}",
            "members": members,
            "weights": {"mode": "fixed", "values": values},
            "decoder": {"fraud_scale": 0.875},
            "rules": ["anchored_templates"],
            "base_rate_rules": [{"name": "blank_merchant_qr", "max_october_rows": 40}],
        }
    )


def write_cell(cells_dir, arm, seed, rows, p_legit) -> None:
    """One October-like cell (set 0) of this arm's fraud-only probabilities."""
    cells_dir = Path(cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cells_dir / f"wspace-{arm}_s{seed}_r0.npz",
        rows=np.asarray(rows),
        prob=fraud_only(p_legit).astype(np.float32),
    )


def _contexts(root: Path, prefix: str, keys) -> dict:
    out = {}
    for key in keys:
        directory = root / f"{prefix}{key}"
        if (directory / "arms.npz").is_file():
            out[key] = load_arms(directory)
    return out


def _month_line(scores: dict) -> str:
    parts = []
    for name, block in scores.items():
        months = " / ".join(f"{v:.5f}" for v in block["months"])
        parts.append(f"{name} {months}")
    return "  ".join(parts)


def score(
    base="base",
    arms=None,
    base_run="tabm",
    cells_dir=None,
    draws=2000,
    oct_draws=1000,
    log=print,
) -> dict:
    """Score every arm against ``base`` and write ``score_<base>.json``."""
    from ringtrace import paths

    root = wspace_dir()
    folds = _contexts(root, "fold", _FOLD_MONTHS)
    frames = _contexts(root, "s", _FRAME_SEEDS)
    cells = root / "cells" if cells_dir is None else Path(cells_dir)
    if arms is None:
        found = {name for loaded in folds.values() for name in loaded["arms"]}
        found -= {base, "stored"}
        arms = [*sorted(found), "stored"]
    bench = ArmBench.from_store(paths.out_dir() / "runs", cells, log=log)
    base_full = bench.member({"run": base_run, "variant": "fold"})
    for arm in dict.fromkeys(a for a in (base, *arms) if a != "stored"):
        bench.arm_arrays[arm] = arm_matrix(base_full, bench.te, folds, arm)
        for seed, loaded in frames.items():
            if arm in loaded["arms"]:
                write_cell(cells, arm, seed, loaded["rows"], loaded["arms"][arm])
    base_recipe = arm_recipe(base)
    chals = [arm_recipe(arm) for arm in arms]
    octs = bench.octlike(base_recipe, chals, sets=(0,), draws=oct_draws)
    report: dict = {}
    for arm, chal, oct in zip(arms, chals, octs, strict=True):
        lo = bench.compare(base_recipe, chal, draws)
        verdict = final.gate_verdict(lo, oct)
        log("[wspace] " + final.fmt_line(chal["name"], lo, oct, verdict))
        report[arm] = {"lomo": lo, "octlike": oct, "verdict": verdict, "scores": bench.lomo(chal)}
    base_scores = bench.lomo(base_recipe)
    report["base"] = {
        "arm": base,
        "scores": base_scores,
        "folds": sorted(folds),
        "frames": sorted(frames),
    }
    log(f"[wspace] {base}: {_month_line(base_scores)}")
    (root / f"score_{base}.json").write_text(
        json.dumps(report, indent=1, default=final._json_default)
    )
    return report


# ---------------------------------------------------------------------------------------------
# command line


def main(argv=None):
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.wspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("contexts", nargs="+")
    fit.add_argument("--seeds", default="0,1,2")
    sc = sub.add_parser("score")
    sc.add_argument("--base", default="base")
    sc.add_argument("--arms", default=None)
    sc.add_argument("--base-run", default="tabm")
    sc.add_argument("--cells", default=None)
    sc.add_argument("--draws", type=int, default=2000)
    sc.add_argument("--oct-draws", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        seeds = tuple(int(s) for s in args.seeds.split(",") if s)
        for name in args.contexts:
            fit_context(name, seeds)
    elif args.cmd == "score":
        chosen = None
        if args.arms is not None:
            chosen = [part.strip() for part in args.arms.split(",") if part.strip()]
        score(
            base=args.base,
            arms=chosen,
            base_run=args.base_run,
            cells_dir=args.cells,
            draws=args.draws,
            oct_draws=args.oct_draws,
        )


if __name__ == "__main__":
    main()
