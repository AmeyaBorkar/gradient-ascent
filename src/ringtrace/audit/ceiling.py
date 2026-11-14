"""Label-free composition of a month, and the macro-F1 a perfect caller of the reachable rows gets.

Per fraud class k of a month:

* reachable rows R_k: what a label-free detector can find (template events, marks, fraud-phase
  phones);
* hidden rows H_k: ordinary fraud-pass rows on home or borrowed phones, which carry no signal.

H comes from the device counter. An ordinary row that drew a new phone sits on a one-row
fraud-phase phone in the segment of its class (a probe row, ``ringtrace.audit.probes``). If a share
``r`` of ordinary rows draws a new phone (measured on train labels), a month with ``p_k`` probe rows
of class k holds ``p_k / r`` ordinary rows of that class, of which ``p_k (1 / r - 1)`` are hidden.

A caller that labels every reachable row right and nothing else scores F1_k = 2 R_k / (2 R_k + H_k)
on class k, and 2 T_0 / (T_0 + P_0) on legit (T_0 legit rows, P_0 rows called legit). Pure
functions; counts are arrays over the 11 fraud classes in ``CLASSES`` order (legit excluded).
"""

from __future__ import annotations

import numpy as np


def new_device_rate(probe_fraud: float, other_fraud: float) -> float:
    """Share of ordinary fraud rows that drew a new phone: fraud-labelled probe rows over all
    fraud-labelled rows without evidence (probe rows plus those on home or legit-block phones).
    Unreported rows drop out of both counts alike."""
    total = probe_fraud + other_fraud
    return float(probe_fraud / total) if total > 0 else float("nan")


def hidden_rows(probe_per_class: np.ndarray, rate: float) -> np.ndarray:
    """Ordinary rows per class that no detector reaches (on home or borrowed phones)."""
    return np.asarray(probe_per_class, dtype=np.float64) * (1.0 / rate - 1.0)


def class_f1(tp: np.ndarray, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    tp, pred, true = (np.asarray(a, dtype=np.float64) for a in (tp, pred, true))
    den = pred + true
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1.0), 0.0)


def expected_scores(pred: np.ndarray, tp: np.ndarray, true: np.ndarray, n_rows: int) -> dict:
    """Per-class F1 (legit first) and macro of a file with ``pred`` calls and ``tp`` correct calls
    per fraud class, against ``true`` rows per fraud class, in a month of ``n_rows`` rows. Wrong
    fraud calls are taken to be legit rows."""
    pred, tp, true = (np.asarray(a, dtype=np.float64) for a in (pred, tp, true))
    legit_true = n_rows - true.sum()
    legit_pred = n_rows - pred.sum()
    legit_tp = legit_true - (pred.sum() - tp.sum())
    f1 = np.concatenate(
        [class_f1([legit_tp], [legit_pred], [legit_true]), class_f1(tp, pred, true)]
    )
    return {"f1": f1, "macro": float(f1.mean())}


def ceiling(reachable: np.ndarray, hidden: np.ndarray, n_rows: int) -> dict:
    """Scores of a caller that labels every reachable row right and nothing else."""
    r = np.asarray(reachable, dtype=np.float64)
    h = np.asarray(hidden, dtype=np.float64)
    res = expected_scores(r, r, r + h, n_rows)
    true = r + h
    res["fraud_rows"] = float(true.sum())
    res["hidden_share"] = float(h.sum() / true.sum()) if true.sum() > 0 else float("nan")
    return res


def extra_rows_for(
    target: float,
    pred: np.ndarray,
    tp: np.ndarray,
    true: np.ndarray,
    hidden: np.ndarray,
    n_rows: int,
    tol: float = 1e-7,
) -> dict:
    """Correct extra calls needed for the macro to reach ``target``, spread over the classes in
    proportion to their hidden rows (each extra call is a hidden row found and typed right).
    ``feasible`` is False when finding every hidden row is still not enough."""
    pred, tp, true, hidden = (np.asarray(a, dtype=np.float64) for a in (pred, tp, true, hidden))

    def macro_at(s: float) -> float:
        x = s * hidden
        return expected_scores(pred + x, tp + x, true, n_rows)["macro"]

    now = macro_at(0.0)
    if now >= target:
        return {
            "feasible": True,
            "now": now,
            "share_of_hidden": 0.0,
            "extra_rows": 0.0,
            "per_class": [0.0] * len(hidden),
        }
    top = macro_at(1.0)
    if top < target:
        return {"feasible": False, "now": now, "max_macro": top}
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if macro_at(mid) < target else (lo, mid)
    x = hi * hidden
    return {
        "feasible": True,
        "now": now,
        "share_of_hidden": hi,
        "extra_rows": float(x.sum()),
        "per_class": x.tolist(),
    }


def split_scores(
    pred: np.ndarray,
    tp: np.ndarray,
    true: np.ndarray,
    n_rows: int,
    n_public: int,
    front_share: float,
) -> dict:
    """Expected public and private macro of one file when the public part is the first
    ``n_public`` rows *of the file* and a share ``front_share`` of its fraud calls sits there.

    Rows called legit fill the rest of the public part at random, so a random row order is
    ``front_share = n_public / n_rows`` and gives the file's full-month score on both parts. Under
    an id-based public part the row order does not matter at all; this models the other case."""
    pred, tp, true = (np.asarray(a, dtype=np.float64) for a in (pred, tp, true))
    f = float(front_share)
    legit_called = n_rows - pred.sum()
    frac = (n_public - f * pred.sum()) / legit_called  # share of legit-called rows in public
    missed = true - tp
    public = expected_scores(f * pred, f * tp, f * tp + frac * missed, n_public)["macro"]
    private = expected_scores(
        (1 - f) * pred, (1 - f) * tp, (1 - f) * tp + (1 - frac) * missed, n_rows - n_public
    )["macro"]
    return {"front_share": f, "public": public, "private": private}
