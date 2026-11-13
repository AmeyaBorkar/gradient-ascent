"""Pure helpers for ``analysis/counter.py`` (no data access, tested on synthetic arrays)."""

from __future__ import annotations

import numpy as np


def paired(diffs: np.ndarray) -> dict:
    """Mean, sd, standard error and mean / se of per-seed paired differences."""
    d = np.asarray(diffs, dtype=np.float64)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    se = sd / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    mean = float(d.mean())
    z = mean / se if se > 0 else float(np.sign(mean)) * float("inf") if mean != 0 else 0.0
    return {"n": int(len(d)), "mean": mean, "sd": sd, "se": float(se), "z": float(z)}


def choose(
    stats: dict[str, dict], keys: tuple[str, ...] = ("agreement",), guards: tuple[str, ...] = ()
) -> str:
    """The setting whose worst mean difference over ``keys`` is largest, among those whose mean
    difference in every ``guards`` metric is at least 0; ``stats[setting][metric]`` holds
    ``paired`` output. Ties go to the first setting in insertion order (simpler settings first)."""
    best, best_v = None, -np.inf
    for name, s in stats.items():
        if any(s[g]["mean"] < 0 for g in guards):
            continue
        v = min(s[k]["mean"] for k in keys)
        if v > best_v:
            best, best_v = name, v
    if best is None:
        raise ValueError("no setting passes the guards")
    return best


def passes(s: dict, rise: float = 0.02, z: float = 2.0) -> bool:
    """The first gate's agreement test: a rise of at least ``rise``, more than ``z`` se above 0."""
    return s["mean"] >= rise and s["z"] > z


def class_moves(before: np.ndarray, after: np.ndarray, names=None) -> dict[str, int]:
    """Counts of ``a->b`` for every row where the call changed (``names[a]`` when given)."""
    moved = before != after
    keys, n = np.unique(
        np.stack([before[moved], after[moved]], axis=1).reshape(-1, 2), axis=0, return_counts=True
    )
    label = (lambda c: str(int(c))) if names is None else (lambda c: names[int(c)])
    return {f"{label(a)}->{label(b)}": int(c) for (a, b), c in zip(keys, n, strict=True)}
