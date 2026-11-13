"""Pure helpers for ``analysis/fresh.py`` (no file access; tested on synthetic data).

F1 of class k is 2 TP / (2 TP + FN + FP), so its loss 1 - F1 = (FN + FP) / (2 TP + FN + FP) splits
additively over the rows that make up FN and FP. Divided by the class count it is the macro-F1 at
stake in each pocket:

* (a) ``a_ordinary``: true-k rows with no generator evidence (gen-only target legit), missed;
* (b) ``b_evidence_legit``: true-k generator rows called legit;
* (c) ``c_evidence_wrongtype``: true-k generator rows called another fraud type;
* (d) ``d_legit_called``: true-legit rows called k;
* (e) ``e_otherfraud_called``: rows of another fraud class called k (the mirror of (a)-(c)).
"""

from __future__ import annotations

import numpy as np

from ringtrace.scoring import per_class_f1


def band(x: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin index of each value in ``edges`` (left-closed), clipped to the outer bins."""
    return np.clip(
        np.searchsorted(np.asarray(edges, float), x, side="right") - 1, 0, len(edges) - 2
    )


def _source(i: np.ndarray, anchor, tmpl, fraud_block) -> list[str]:
    out = []
    for j in i:
        if tmpl[j] > 0:
            out.append(f"tmpl{int(tmpl[j])}")
        elif anchor[j] > 0:
            out.append(f"anchor{int(anchor[j])}")
        elif fraud_block[j]:
            out.append("fraud_block")
        else:
            out.append("none")
    return out


def _tally(names: list[str]) -> dict[str, int]:
    u, c = np.unique(np.asarray(names, dtype=object), return_counts=True) if names else ([], [])
    return {str(a): int(b) for a, b in zip(u, c, strict=True)}


def anatomy(y, pred, gen, anchor, tmpl, fraud_block, k: int = 12) -> dict:
    y, pred, gen = (np.asarray(a, dtype=np.int64) for a in (y, pred, gen))
    f1 = per_class_f1(y, pred, k)
    stake, pairs, legit_by, ev_legit_by = {}, {}, {}, {}
    for c in range(1, k):
        tp = int(((y == c) & (pred == c)).sum())
        miss = (y == c) & (pred != c)
        fp = (pred == c) & (y != c)
        denom = 2 * tp + int(miss.sum()) + int(fp.sum())
        scale = 1.0 / (denom * k) if denom else 0.0
        a = miss & (gen == 0)
        b = miss & (gen > 0) & (pred == 0)
        cc = miss & (gen > 0) & (pred > 0)
        d = fp & (y == 0)
        e = fp & (y > 0)
        stake[c] = {
            "a_ordinary": a.sum() * scale,
            "b_evidence_legit": b.sum() * scale,
            "c_evidence_wrongtype": cc.sum() * scale,
            "d_legit_called": d.sum() * scale,
            "e_otherfraud_called": e.sum() * scale,
        }
        stake[c] = {kk: float(v) for kk, v in stake[c].items()}
        pairs[c] = {
            int(t): int(n) for t, n in zip(*np.unique(pred[cc], return_counts=True), strict=True)
        }
        legit_by[c] = _tally(_source(np.flatnonzero(d), anchor, tmpl, fraud_block))
        ev_legit_by[c] = _tally(_source(np.flatnonzero(b), anchor, tmpl, fraud_block))
    return {
        "macro": float(f1.mean()),
        "f1": f1.tolist(),
        "stake": stake,
        "wrongtype_pairs": pairs,
        "legit_called_by": legit_by,
        "evidence_legit_by": ev_legit_by,
    }


def _share(b: np.ndarray, nb: int) -> np.ndarray:
    h = np.bincount(b, minlength=nb).astype(float)
    return h / h.sum() if h.sum() else h


def profile_shift(b, y, calls, tr, te, nb: int, k: int = 12) -> dict:
    """Per fraud class, the bin mix of train labels, train calls and October calls, and the total
    variation distance between October and train calls and between train calls and labels."""
    out = {}
    for c in range(1, k):
        lab = _share(b[tr & (y == c)], nb)
        oof = _share(b[tr & (calls == c)], nb)
        octo = _share(b[te & (calls == c)], nb)
        out[c] = {
            "lab": np.round(lab, 3).tolist(),
            "oof": np.round(oof, 3).tolist(),
            "oct": np.round(octo, 3).tolist(),
            "tv_oct_oof": float(0.5 * np.abs(octo - oof).sum()),
            "tv_oof_lab": float(0.5 * np.abs(oof - lab).sum()),
            "n_oct": int((te & (calls == c)).sum()),
        }
    return out
