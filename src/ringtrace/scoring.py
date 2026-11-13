"""Macro-F1 over the twelve classes, and the per-class pieces it is made of."""

from __future__ import annotations

import numpy as np

from ringtrace.classes import N_CLASSES


def confusion(truth: np.ndarray, pred: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    return np.bincount(truth * k + pred, minlength=k * k).reshape(k, k)


def per_class_f1(truth: np.ndarray, pred: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    cm = confusion(truth, pred, k)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return f1


def macro_f1(truth: np.ndarray, pred: np.ndarray, k: int = N_CLASSES) -> float:
    """Mean F1 over all k classes; a class absent from both truth and prediction scores 0,
    which matches scikit-learn's ``f1_score(average='macro', labels=range(k))``."""
    return float(per_class_f1(truth, pred, k).mean())
