"""Pure helpers for the findings analyses (no file access; tested on synthetic data)."""

from __future__ import annotations

import numpy as np

from ringtrace.scoring import confusion, per_class_f1

OLD_APP_FRAUD = 0.618  # old-app rate of fraud-pass train rows
OLD_APP_LEGIT = 0.150  # old-app rate of legit-pass train rows


def macro_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def slice_scores(
    y: np.ndarray, preds: list[np.ndarray], frac: float, draws: int, seed: int = 0, k: int = 12
) -> dict[str, np.ndarray]:
    """Macro-F1 of each prediction on ``draws`` random public slices (size ``frac``) and on the
    complementary private slice. Returns arrays of shape (draws, len(preds))."""
    rng = np.random.default_rng(seed)
    n = len(y)
    n_pub = round(frac * n)
    codes = [np.asarray(y) * k + np.asarray(p) for p in preds]
    full = [np.bincount(c, minlength=k * k) for c in codes]
    pub = np.zeros((draws, len(preds)))
    priv = np.zeros((draws, len(preds)))
    for d in range(draws):
        idx = rng.choice(n, n_pub, replace=False)
        for j, c in enumerate(codes):
            cm_pub = np.bincount(c[idx], minlength=k * k)
            pub[d, j] = macro_from_cm(cm_pub.reshape(k, k))
            priv[d, j] = macro_from_cm((full[j] - cm_pub).reshape(k, k))
    return {"public": pub, "private": priv}


def one_row_gain(n_true: np.ndarray, n_pred: np.ndarray, f1: np.ndarray) -> np.ndarray:
    """Macro-F1 change when one more row of class k is called k (it was called something
    else before; the loss on that other class is ignored). TP is recovered from F1."""
    n_true = np.asarray(n_true, float)
    n_pred = np.asarray(n_pred, float)
    tp = np.asarray(f1, float) * (n_true + n_pred) / 2
    new = 2 * (tp + 1) / (n_true + n_pred + 1)
    old = np.where(n_true + n_pred > 0, 2 * tp / np.maximum(n_true + n_pred, 1e-12), 0)
    return (new - old) / len(n_true)


def headroom(clean: np.ndarray, gen: np.ndarray, pred: np.ndarray, k: int = 12) -> dict:
    """Per-class F1 of ``pred`` and of the ceiling (calls = the gen-only labels), on cleaned."""
    now = per_class_f1(clean, pred, k)
    ceil = per_class_f1(clean, gen, k)
    fraud_rows = np.bincount(clean, minlength=k)
    ordinary = np.bincount(clean[(clean > 0) & (gen == 0)], minlength=k)
    share = np.where(fraud_rows > 0, ordinary / np.maximum(fraud_rows, 1), 0.0)
    share[0] = 0.0
    return {
        "now": now,
        "ceiling": ceil,
        "gap": ceil - now,
        "ordinary_share": share,
        "macro_now": float(now.mean()),
        "macro_ceiling": float(ceil.mean()),
    }


def disagreement(
    ref: np.ndarray, other: np.ndarray, truth: np.ndarray | None = None, k: int = 12, top: int = 8
) -> dict:
    """Rows where two call vectors differ, who the truth sides with, and the largest cells."""
    diff = ref != other
    out: dict = {"rows": int(diff.sum())}
    if truth is not None:
        t = truth[diff]
        out["truth_other"] = int((t == other[diff]).sum())
        out["truth_ref"] = int((t == ref[diff]).sum())
        out["truth_neither"] = int(diff.sum()) - out["truth_other"] - out["truth_ref"]
    cm = confusion(ref[diff], other[diff], k)
    flat = np.argsort(-cm, axis=None, kind="stable")[:top]
    cells = []
    for f in flat:
        r, c = divmod(int(f), k)
        if cm[r, c] == 0:
            break
        cell = {"ref": r, "other": c, "rows": int(cm[r, c])}
        if truth is not None:
            m = diff & (ref == r) & (other == c)
            cell["truth_other"] = int((truth[m] == c).sum())
            cell["truth_ref"] = int((truth[m] == r).sum())
        cells.append(cell)
    out["cells"] = cells
    return out


def fraud_pass_share(old_app: np.ndarray) -> dict:
    """Estimated fraud-pass share of a set of train rows from its old-app rate, with a
    binomial 95% interval."""
    old_app = np.asarray(old_app, bool)
    n = len(old_app)
    if n == 0:
        return {"rows": 0, "old_app": float("nan"), "share": float("nan"), "ci": [None, None]}
    r = float(old_app.mean())
    se = np.sqrt(max(r * (1 - r), 1e-12) / n)
    span = OLD_APP_FRAUD - OLD_APP_LEGIT

    def to_share(x: float) -> float:
        return float(np.clip((x - OLD_APP_LEGIT) / span, 0.0, 1.0))

    return {
        "rows": n,
        "old_app": r,
        "share": to_share(r),
        "ci": [to_share(r - 1.96 * se), to_share(r + 1.96 * se)],
    }
