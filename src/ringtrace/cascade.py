"""Two-stage model: P(fraud) from a binary booster, times P(type | fraud) from an 11-way booster.

Training targets come from a cross-fitted fraud score computed inside the training months only:

* the binary stage drops legit-labelled rows that the score calls likely fraud (they are mostly
  unreported fraud, which would teach the model to doubt real patterns);
* fraud-labelled rows the score cannot see (the ordinary-looking ones: no row or relational
  feature separates them from legit traffic) are legit for the binary stage and are left out of
  the type stage, where they would only blur the type boundaries.

Validation is leave-one-month-out; test probabilities are the mean of the month-fold models.
"""

from __future__ import annotations

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
class CascadeSpec:
    booster: BoosterSpec = field(default_factory=BoosterSpec)
    seeds: tuple[int, ...] = (0, 1, 2)
    screen_rounds: int = 150
    type_min_score: float = 0.10
    drop_legit_min_score: float = 0.70
    screen: bool = True  # False: every fraud row is a type row and no legit row is dropped


def _fit_binary(X, y, spec: BoosterSpec, seed: int, rounds: int, weight=None) -> xgb.Booster:
    d = xgb.QuantileDMatrix(X, label=y, weight=weight, max_bin=spec.max_bin)
    return xgb.train(spec.params(seed, objective="binary:logistic"), d, rounds)


def _fit_types(X, y_type, spec: BoosterSpec, seed: int, rounds: int, weight=None):
    d = xgb.QuantileDMatrix(X, label=y_type, weight=weight, max_bin=spec.max_bin)
    return xgb.train(
        spec.params(seed, objective="multi:softprob", num_class=N_CLASSES - 1), d, rounds
    )


def _predict(model: xgb.Booster, X) -> np.ndarray:
    return model.inplace_predict(X)


def screen_score(
    X: np.ndarray, fraud: np.ndarray, period: np.ndarray, spec: CascadeSpec, seed: int = 0
) -> np.ndarray:
    """Cross-fitted P(fraud label) over the given rows: each period scored by a model trained on
    the other periods."""
    out = np.zeros(len(fraud), dtype=np.float32)
    for p in np.unique(period):
        tr = period != p
        m = _fit_binary(X[tr], fraud[tr].astype(np.float32), spec.booster, seed, spec.screen_rounds)
        out[period == p] = _predict(m, X[period == p])
    return out


def fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    period_tr: np.ndarray,
    X_out: list[np.ndarray],
    spec: CascadeSpec,
    log=print,
) -> list[np.ndarray]:
    """Train on (X_tr, y_tr) and return 12-class probabilities for every matrix in X_out."""
    t0 = time.time()
    fraud = (y_tr > 0).astype(np.float32)
    if spec.screen:
        score = screen_score(X_tr, fraud, period_tr, spec)
    else:  # a single month cannot be cross-fitted: train on the labels as they are
        score = np.ones(len(fraud), dtype=np.float32)
        score[fraud == 0] = 0.0
    keep_bin = (fraud == 1) | (score < spec.drop_legit_min_score)
    type_rows = (fraud == 1) & (score >= spec.type_min_score)
    # fraud-labelled rows the screen cannot see are the ordinary-looking ones: legit for stage one
    target_bin = type_rows.astype(np.float32)
    log(
        f"  screen: dropped {int((~keep_bin).sum())} legit rows, type rows {int(type_rows.sum())}"
        f" of {int(fraud.sum())} fraud ({time.time() - t0:.0f}s)"
    )
    outs = [np.zeros((len(X), N_CLASSES), dtype=np.float64) for X in X_out]
    for seed in spec.seeds:
        mb = _fit_binary(
            X_tr[keep_bin], target_bin[keep_bin], spec.booster, seed, spec.booster.rounds
        )
        mt = _fit_types(
            X_tr[type_rows],
            (y_tr[type_rows] - 1).astype(np.float32),
            spec.booster,
            seed,
            spec.booster.rounds,
        )
        for o, X in zip(outs, X_out, strict=True):
            pf = _predict(mb, X).astype(np.float64)
            pt = _predict(mt, X).astype(np.float64).reshape(len(X), N_CLASSES - 1)
            o[:, 0] += 1 - pf
            o[:, 1:] += pf[:, None] * pt
    log(f"  cascade fitted with {len(spec.seeds)} seeds ({time.time() - t0:.0f}s)")
    return [o / len(spec.seeds) for o in outs]
