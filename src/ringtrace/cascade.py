"""Two-stage model: P(fraud) from a binary booster, times P(type | fraud) from an 11-way booster.

Training targets come from a cross-fitted fraud score computed inside the training months only:

* the binary stage drops legit-labelled rows that the score calls likely fraud (they are mostly
  unreported fraud, which would teach the model to doubt real patterns);
* fraud-labelled rows the score cannot see (the ordinary-looking ones: no row or relational
  feature separates them from legit traffic) are legit for the binary stage and are left out of
  the type stage, where they would only blur the type boundaries.

Validation is leave-one-month-out; test probabilities are the mean of the month-fold models.

The two stages can use XGBoost (``BoosterSpec``) or LightGBM (``LgbmSpec``); the screening score
has its own spec, so a LightGBM cascade can keep the XGBoost screen of the champion. The two
stages can also use CatBoost (``stages="catboost"``, ``CatBoostSpec``); the screening score then
still comes from ``booster``, so every option trains on the same rows.

An optional per-row sample weight is applied to both stage fits and not to the screening
score, so a class-mix weight changes what the stages learn without changing which rows they see.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb

from ringtrace import threads
from ringtrace.classes import N_CLASSES


@dataclass
class BoosterSpec:
    rounds: int = 300
    eta: float = 0.08
    depth: int = 8
    subsample: float = 0.8
    colsample: float = 0.6
    min_child_weight: float = 2.0
    l2: float = 2.0
    max_bin: int = 256
    device: str = "cuda"

    def params(self, seed: int, **extra) -> dict:
        p = {
            "tree_method": "hist",
            "device": self.device,
            "eta": self.eta,
            "max_depth": self.depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample,
            "min_child_weight": self.min_child_weight,
            "lambda": self.l2,
            "max_bin": self.max_bin,
            "seed": seed,
            "nthread": threads.budget(),
            "verbosity": 0,
        }
        p.update(extra)
        return p


@dataclass
class LgbmSpec:
    """LightGBM on the CPU; every random seed of a model derives from its cascade seed."""

    kind: str = "lightgbm"
    rounds: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 30
    feature_fraction: float = 0.5
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 2.0
    max_bin: int = 255

    def params(self, seed: int, **extra) -> dict:
        p = {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "max_bin": self.max_bin,
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "device_type": "cpu",
            "num_threads": threads.budget(),
            "verbosity": -1,
        }
        p.update(extra)
        return p


@dataclass
class CatBoostSpec:
    iterations: int = 300
    depth: int = 8
    learning_rate: float = 0.08
    border_count: int = 254
    task_type: str = "GPU"
    devices: str = "0"
    gpu_ram_part: float | None = None  # share of GPU memory CatBoost may take (None: its default)

    def params(self, seed: int, loss: str) -> dict:
        p = {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "border_count": self.border_count,
            "task_type": self.task_type,
            "loss_function": loss,
            "random_seed": seed,
            "thread_count": threads.budget(),
            "verbose": False,
            "allow_writing_files": False,
        }
        if self.task_type == "GPU":
            p["devices"] = self.devices
            if self.gpu_ram_part is not None:
                p["gpu_ram_part"] = self.gpu_ram_part
            else:
                # share of GPU memory CatBoost may reserve (set when it shares the GPU)
                cap = os.environ.get("RINGTRACE_CATBOOST_GPU_RAM")
                if cap:
                    p["gpu_ram_part"] = float(cap)
        return p


Spec = BoosterSpec | LgbmSpec


def booster_spec(d: dict | None) -> Spec:
    """Booster spec from a recipe dict: ``{"kind": "lightgbm", ...}`` or XGBoost (default)."""
    d = dict(d or {})
    if d.get("kind", "xgboost") == "lightgbm":
        return LgbmSpec(**d)
    d.pop("kind", None)
    return BoosterSpec(**d)


@dataclass
class CascadeSpec:
    booster: Spec = field(default_factory=BoosterSpec)
    stages: str = "booster"  # "booster" (or "xgboost"): use ``booster``; "catboost"
    catboost: CatBoostSpec = field(default_factory=CatBoostSpec)
    seeds: tuple[int, ...] = (0, 1, 2)
    screen_rounds: int = 150
    type_min_score: float = 0.10
    drop_legit_min_score: float = 0.70
    screen: Spec | None = None  # booster of the screening score; None = same as ``booster``
    use_screen: bool = True  # False: every fraud row is a type row and no legit row is dropped


def _fit(X, y, spec: Spec, seed: int, rounds: int, n_class: int = 0, weight=None):
    if isinstance(spec, LgbmSpec):
        import lightgbm as lgb

        extra = (
            {"objective": "multiclass", "num_class": n_class}
            if n_class
            else {"objective": "binary"}
        )
        # ``weight`` omitted keeps the unweighted Dataset, including its default of ones.
        kw = {}
        if weight is not None:
            # copy: a booster may keep the array, and the next seed must see the same weights
            kw["weight"] = np.array(weight, dtype=np.float64, copy=True)
        d = lgb.Dataset(
            X,
            label=y,
            params={"max_bin": spec.max_bin, "verbosity": -1},
            free_raw_data=True,
            **kw,
        )
        return lgb.train(spec.params(seed, **extra), d, num_boost_round=rounds)
    extra = (
        {"objective": "multi:softprob", "num_class": n_class}
        if n_class
        else {"objective": "binary:logistic"}
    )
    if weight is None:
        d = xgb.QuantileDMatrix(X, label=y, max_bin=spec.max_bin)
    else:
        d = xgb.QuantileDMatrix(
            X,
            label=y,
            weight=np.array(weight, dtype=np.float64, copy=True),
            max_bin=spec.max_bin,
        )
    return xgb.train(spec.params(seed, **extra), d, rounds)


def _fit_binary(X, y, spec: Spec, seed: int, rounds: int, weight=None):
    return _fit(X, y, spec, seed, rounds, weight=weight)


def _fit_types(X, y_type, spec: Spec, seed: int, rounds: int, weight=None):
    return _fit(X, y_type, spec, seed, rounds, n_class=N_CLASSES - 1, weight=weight)


def _fit_cat(X, y, spec: CatBoostSpec, seed: int, loss: str, weight=None):
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(**spec.params(seed, loss))
    if weight is None:
        m.fit(X, y)
    else:
        m.fit(X, y, sample_weight=np.array(weight, dtype=np.float64, copy=True))
    return m


def _fit_stages(
    X_bin,
    y_bin,
    X_type,
    y_type,
    spec: CascadeSpec,
    seed: int,
    w_bin: np.ndarray | None = None,
    w_type: np.ndarray | None = None,
):
    if spec.stages == "catboost":
        return (
            _fit_cat(X_bin, y_bin.astype(np.int64), spec.catboost, seed, "Logloss", w_bin),
            _fit_cat(X_type, y_type.astype(np.int64), spec.catboost, seed, "MultiClass", w_type),
        )
    if spec.stages not in ("booster", "xgboost"):  # "xgboost": the name before LightGBM
        raise ValueError(f"unknown stage booster {spec.stages!r}")
    return (
        _fit_binary(X_bin, y_bin, spec.booster, seed, spec.booster.rounds, w_bin),
        _fit_types(X_type, y_type, spec.booster, seed, spec.booster.rounds, w_type),
    )


def _predict(model, X) -> np.ndarray:
    if isinstance(model, xgb.Booster):
        return model.inplace_predict(X)
    if hasattr(model, "predict_proba"):  # CatBoost
        p = model.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else p
    return model.predict(X, num_threads=threads.budget())


def screen_score(
    X: np.ndarray, fraud: np.ndarray, period: np.ndarray, spec: CascadeSpec, seed: int = 0
) -> np.ndarray:
    """Cross-fitted P(fraud label) over the given rows: each period scored by a model trained on
    the other periods."""
    out = np.zeros(len(fraud), dtype=np.float32)
    for p in np.unique(period):
        tr = period != p
        m = _fit_binary(
            X[tr],
            fraud[tr].astype(np.float32),
            spec.screen or spec.booster,
            seed,
            spec.screen_rounds,
        )
        out[period == p] = _predict(m, X[period == p])
    return out


def fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    period_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec: CascadeSpec,
    log=print,
    sample_weight: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Train on (X_tr, y_tr) and return 12-class probabilities for every matrix in X_out.

    ``sample_weight`` is one non-negative weight per training row. It is applied to the
    stage-one fit (rows kept after the screen) and the stage-two fit (type rows). The
    screening score stays unweighted: it chooses which rows those stages see, and that
    choice is about label noise rather than the class mix. ``None`` leaves both stages on
    the unweighted path. An all-ones weight is that same fit: it is the booster default.
    Weights are not rescaled, because a global multiplier is not free under L2.
    """
    if sample_weight is not None:
        sample_weight = np.ascontiguousarray(
            np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        if sample_weight.shape != (len(y_tr),):
            raise ValueError(
                f"sample_weight has length {sample_weight.shape[0]}, training rows are {len(y_tr)}"
            )
        if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0):
            raise ValueError("sample_weight must be finite and non-negative")
        if float(sample_weight.sum()) <= 0:
            raise ValueError("sample_weight sums to zero")
    t0 = time.time()
    fraud = (y_tr > 0).astype(np.float32)
    # screen off (one month cannot be cross-fitted): every label is taken as it is
    score = screen_score(X_tr, fraud, period_tr, spec) if spec.use_screen else fraud.copy()
    keep_bin = (fraud == 1) | (score < spec.drop_legit_min_score)
    type_rows = (fraud == 1) & (score >= spec.type_min_score)
    # fraud-labelled rows the screen cannot see are the ordinary-looking ones: legit for stage one
    target_bin = type_rows.astype(np.float32)
    log(
        f"  screen: dropped {int((~keep_bin).sum())} legit rows, type rows {int(type_rows.sum())}"
        f" of {int(fraud.sum())} fraud ({time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    # Slice only when a weight was asked for, so the unweighted call stays the one below.
    # Stage one weights only its positives (the fraud rows the screen can see). Its negatives,
    # ordinary-looking fraud rows included, keep weight 1: they stand for legit traffic there,
    # and a fraud-class weight on them would push their class towards legit.
    w_bin = (
        None
        if sample_weight is None
        else np.where(target_bin[keep_bin] == 1, sample_weight[keep_bin], 1.0)
    )
    w_type = None if sample_weight is None else sample_weight[type_rows]
    for seed in spec.seeds:
        mb, mt = _fit_stages(
            X_tr[keep_bin],
            target_bin[keep_bin],
            X_tr[type_rows],
            (y_tr[type_rows] - 1).astype(np.float32),
            spec,
            seed,
            w_bin,
            w_type,
        )
        for o, X in zip(outs, X_out, strict=True):
            pf = _predict(mb, X).astype(np.float64)
            pt = _predict(mt, X).astype(np.float64).reshape(len(X), N_CLASSES - 1)
            o[:, 0] += 1 - pf
            o[:, 1:] += pf[:, None] * pt
    log(f"  cascade fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [o / len(spec.seeds) for o in outs]
