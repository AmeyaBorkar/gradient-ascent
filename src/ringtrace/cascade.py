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

``RINGTRACE_DEVICE=cpu`` runs every XGBoost and CatBoost fit on the CPU whatever the recipe says
(unset: each spec's own device). CPU and GPU fits of the same seed differ numerically.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb

from ringtrace import threads
from ringtrace.classes import N_CLASSES


def device_override() -> str | None:
    """``"cpu"`` when ``RINGTRACE_DEVICE=cpu`` forces every booster fit onto the CPU; ``None`` when
    the variable is unset or empty. Any other value is refused, so a typo cannot reach a GPU."""
    v = os.environ.get("RINGTRACE_DEVICE", "").strip().lower()
    if not v:
        return None
    if v != "cpu":
        raise ValueError(f"RINGTRACE_DEVICE={v!r}: only 'cpu' (or unset) is understood")
    return "cpu"


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
            "device": device_override() or self.device,
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
        if device_override() == "cpu":
            # the CUDA build of XGBoost fails in the default multiclass metric when no GPU is
            # visible; no fit here is evaluated, so the trees and predictions are unchanged
            p["disable_default_eval_metric"] = 1
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
        task_type = "CPU" if device_override() == "cpu" else self.task_type
        p = {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "border_count": self.border_count,
            "task_type": task_type,
            "loss_function": loss,
            "random_seed": seed,
            "thread_count": threads.budget(),
            "verbose": False,
            "allow_writing_files": False,
        }
        if task_type == "GPU":
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


def _fit(X, y, spec: Spec, seed: int, rounds: int, n_class: int = 0):
    if isinstance(spec, LgbmSpec):
        import lightgbm as lgb

        extra = (
            {"objective": "multiclass", "num_class": n_class}
            if n_class
            else {"objective": "binary"}
        )
        d = lgb.Dataset(
            X, label=y, params={"max_bin": spec.max_bin, "verbosity": -1}, free_raw_data=True
        )
        return lgb.train(spec.params(seed, **extra), d, num_boost_round=rounds)
    extra = (
        {"objective": "multi:softprob", "num_class": n_class}
        if n_class
        else {"objective": "binary:logistic"}
    )
    d = xgb.QuantileDMatrix(X, label=y, max_bin=spec.max_bin)
    return xgb.train(spec.params(seed, **extra), d, rounds)


def _fit_binary(X, y, spec: Spec, seed: int, rounds: int):
    return _fit(X, y, spec, seed, rounds)


def _fit_types(X, y_type, spec: Spec, seed: int, rounds: int):
    return _fit(X, y_type, spec, seed, rounds, n_class=N_CLASSES - 1)


def _fit_cat(X, y, spec: CatBoostSpec, seed: int, loss: str):
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(**spec.params(seed, loss))
    m.fit(X, y)
    return m


def _fit_stages(X_bin, y_bin, X_type, y_type, spec: CascadeSpec, seed: int):
    if spec.stages == "catboost":
        return (
            _fit_cat(X_bin, y_bin.astype(np.int64), spec.catboost, seed, "Logloss"),
            _fit_cat(X_type, y_type.astype(np.int64), spec.catboost, seed, "MultiClass"),
        )
    if spec.stages not in ("booster", "xgboost"):  # "xgboost": the name before LightGBM
        raise ValueError(f"unknown stage booster {spec.stages!r}")
    return (
        _fit_binary(X_bin, y_bin, spec.booster, seed, spec.booster.rounds),
        _fit_types(X_type, y_type, spec.booster, seed, spec.booster.rounds),
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


def training_rows(
    fraud: np.ndarray, score: np.ndarray, spec: CascadeSpec
) -> tuple[np.ndarray, np.ndarray]:
    """(stage-one rows, stage-two rows) from the labels and the cross-fitted screening score.

    Stage one keeps every fraud-labelled row and the legit-labelled rows scoring below
    ``drop_legit_min_score``; its target is 1 only on the stage-two rows (fraud-labelled rows
    scoring at least ``type_min_score``), so fraud-labelled rows the screen cannot see are legit
    for stage one and absent from stage two."""
    keep_bin = (fraud == 1) | (score < spec.drop_legit_min_score)
    type_rows = (fraud == 1) & (score >= spec.type_min_score)
    return keep_bin, type_rows


def seed_outputs(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    period_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec: CascadeSpec,
    log=print,
):
    """Yield ``(seed, [12-class probabilities for every matrix in X_out])`` seed by seed; the
    screening score is computed once and shared by every seed."""
    t0 = time.time()
    fraud = (y_tr > 0).astype(np.float32)
    # screen off (one month cannot be cross-fitted): every label is taken as it is
    score = screen_score(X_tr, fraud, period_tr, spec) if spec.use_screen else fraud.copy()
    keep_bin, type_rows = training_rows(fraud, score, spec)
    # fraud-labelled rows the screen cannot see are the ordinary-looking ones: legit for stage one
    target_bin = type_rows.astype(np.float32)
    log(
        f"  screen: dropped {int((~keep_bin).sum())} legit rows, type rows {int(type_rows.sum())}"
        f" of {int(fraud.sum())} fraud ({time.time() - t0:.0f}s)"
    )
    for seed in spec.seeds:
        mb, mt = _fit_stages(
            X_tr[keep_bin],
            target_bin[keep_bin],
            X_tr[type_rows],
            (y_tr[type_rows] - 1).astype(np.float32),
            spec,
            seed,
        )
        res = []
        for X in X_out:
            pf = _predict(mb, X).astype(np.float64)
            pt = _predict(mt, X).astype(np.float64).reshape(len(X), N_CLASSES - 1)
            o = np.empty((len(X), N_CLASSES), dtype=np.float64)
            o[:, 0] = 1 - pf
            o[:, 1:] = pf[:, None] * pt
            res.append(o)
        yield seed, res
    log(f"  cascade fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")


def fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    period_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec: CascadeSpec,
    log=print,
) -> list[np.ndarray]:
    """Train on (X_tr, y_tr) and return 12-class probabilities for every matrix in X_out (the
    mean over ``spec.seeds``)."""
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for _, res in seed_outputs(X_tr, y_tr, period_tr, X_out, spec, log=log):
        for o, r in zip(outs, res, strict=True):
            o += r
    return [o / len(spec.seeds) for o in outs]
