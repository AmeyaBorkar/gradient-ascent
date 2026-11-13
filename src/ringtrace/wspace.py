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
# command line


def main(argv=None):
    from ringtrace import threads

    threads.pin()
    ap = argparse.ArgumentParser(prog="python -m ringtrace.wspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("contexts", nargs="+")
    f.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        seeds = tuple(int(s) for s in args.seeds.split(",") if s)
        for name in args.contexts:
            fit_context(name, seeds)


if __name__ == "__main__":
    main()
