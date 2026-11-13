"""Cleaned ("silver") labels for scoring and diagnosis.

The train labels carry two small per-row noises: about 12 % of generator-made fraud rows are
labelled legit (unreported), and about 3 % of fraud rows carry a random other type. The test
labels are clean, so we also score every run against a cleaned version of the train labels.

How a row is judged "generator-made" (G), without using the scored model:

1. An independent teacher (a plain multiclass booster on the relational and mark columns, out of
   fold by month) gives q = P(fraud label | x), calibrated by isotonic regression.
2. Invert the noise: q = g (1 - u) + (1 - g) rho, so g = (q - rho) / (1 - u - rho), with u the
   unreported share and rho the rate of fraud labels on legit-made rows.
3. Multiply the prior odds g / (1 - g) by likelihood ratios of evidence the models never see:
   the label itself, and the two columns that die in October (old app version, IP risk bin),
   whose rates among G and legit-made rows we estimate here from confident rows.
4. p_gen = posterior. Legit-labelled rows with p_gen >= 0.5 become fraud (their event's majority
   type, else the teacher's type); fraud rows whose type disagrees with a clear event majority, or
   sits at the teacher's noise floor, take the majority / teacher type. Fraud-labelled rows with
   p_gen < 0.5 ("ordinary" rows) keep their label in ``label`` and become legit in ``structured``.

Look-alike events (a whole event carrying a partner type) are never corrected: only rows that
disagree with their own event are touched.

The result is frozen in the cache on first build so that every run is scored on the same labels.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from ringtrace import paths

RHO = 0.0238  # fraud labels per legit-made row (the ordinary-row rate)
UNREPORTED = 0.122  # generator-made fraud rows labelled legit
NOISE_FLOOR = 0.0068  # teacher probability of the given type below which it is a random flip
EVENT_MAJORITY = 2 / 3


def _dir():
    d = paths.cache_dir() / "silver"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(led: pl.DataFrame, which: str = "label") -> np.ndarray | None:
    """Silver labels for every ledger row (-1 on test), or None if not built yet."""
    p = _dir() / f"silver_{which}.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    return arr if len(arr) == led.height else None


def _isotonic(score: np.ndarray, target: np.ndarray) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return iso.fit(score, target).predict(score)


def _ratio_table(
    values: np.ndarray, gen: np.ndarray, made_legit: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Likelihood ratio P(bin | G) / P(bin | legit-made) per bin, smoothed and made monotone."""
    b = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 1)
    k = len(edges)
    pg = (np.bincount(b[gen], minlength=k) + 0.5) / (gen.sum() + 0.5 * k)
    pl_ = (np.bincount(b[made_legit], minlength=k) + 0.5) / (made_legit.sum() + 0.5 * k)
    lr = np.maximum.accumulate(pg / pl_)
    return lr[b]


def build(
    led: pl.DataFrame,
    teacher: np.ndarray,
    events: np.ndarray,
    log=print,
    mask: np.ndarray | None = None,
) -> dict:
    """Compute silver labels from the teacher's out-of-fold probabilities, which are given for
    the rows in ``mask`` (default: every train row), in ledger order. Other rows get -1."""
    train = ~led["is_test"].to_numpy() if mask is None else np.asarray(mask, dtype=bool)
    y = led["y"].to_numpy()[train].astype(np.int64)
    ev = events[train]
    p = np.asarray(teacher, dtype=np.float64)
    fraud_label = y > 0
    q = _isotonic(1.0 - p[:, 0], fraud_label.astype(float))
    g = np.clip((q - RHO) / (1 - UNREPORTED - RHO), 1e-6, 1 - 1e-6)

    app_old = led["app_version"].to_numpy()[train] < 4.0
    ip = led["ip_risk_score"].to_numpy()[train]
    confident_gen = fraud_label & (q >= 0.8)
    confident_legit = ~fraud_label & (q < 0.01)
    lr_app = np.where(
        app_old,
        app_old[confident_gen].mean() / app_old[confident_legit].mean(),
        (1 - app_old[confident_gen].mean()) / (1 - app_old[confident_legit].mean()),
    )
    edges = np.r_[0.0, np.arange(0.15, 0.80, 0.05)]
    lr_ip = _ratio_table(ip, confident_gen, confident_legit, edges)
    lr_label = np.where(fraud_label, (1 - UNREPORTED) / RHO, UNREPORTED / (1 - RHO))
    odds = g / (1 - g) * lr_label * lr_app * lr_ip
    p_gen = odds / (1 + odds)
    gen = p_gen >= 0.5

    silver = y.copy()
    teacher_type = 1 + p[:, 1:].argmax(axis=1)
    # event majority type among confident generator-made fraud rows
    majority = np.full(len(y), -1, dtype=np.int64)
    share = np.zeros(len(y))
    others = np.zeros(len(y), dtype=np.int64)
    has_ev = ev >= 0
    voters = has_ev & fraud_label & gen
    if voters.any():
        df = pl.DataFrame({"e": ev[voters], "t": y[voters]})
        cnt = df.group_by("e", "t").len()
        tot = cnt.group_by("e").agg(pl.col("len").sum().alias("tot"))
        top = (
            cnt.sort(["e", "len", "t"], descending=[False, True, False])
            .group_by("e", maintain_order=True)
            .first()
            .join(tot, on="e")
        )
        lut = pl.DataFrame({"e": ev}).with_row_index("i").join(top, on="e", how="left").sort("i")
        majority = lut["t"].fill_null(-1).to_numpy().astype(np.int64)
        share = (lut["len"] / lut["tot"]).fill_null(0).to_numpy()
        others = lut["tot"].fill_null(0).to_numpy().astype(np.int64) - voters.astype(np.int64)

    unreported = ~fraud_label & gen
    use_event = unreported & (majority > 0)
    silver[use_event] = majority[use_event]
    silver[unreported & ~use_event] = teacher_type[unreported & ~use_event]

    structured = fraud_label & gen
    flip_event = (
        structured & (majority > 0) & (majority != y) & (share >= EVENT_MAJORITY) & (others >= 2)
    )
    fraud_p = p[:, 1:] / np.maximum(p[:, 1:].sum(axis=1, keepdims=True), 1e-12)
    given_p = fraud_p[np.arange(len(y)), np.clip(y - 1, 0, None)]
    flip_floor = structured & ~flip_event & (given_p < NOISE_FLOOR) & (teacher_type != y)
    silver[flip_event] = majority[flip_event]
    silver[flip_floor] = teacher_type[flip_floor]

    ordinary = fraud_label & ~gen
    structured_only = silver.copy()
    structured_only[ordinary] = 0

    stats = {
        "rows": int(len(y)),
        "changed": int((silver != y).sum()),
        "unreported_to_fraud": int(unreported.sum()),
        "unreported_by_event": int(use_event.sum()),
        "wrong_type_by_event": int(flip_event.sum()),
        "wrong_type_by_floor": int(flip_floor.sum()),
        "ordinary_fraud_rows": int(ordinary.sum()),
        "ordinary_share_of_fraud": float(ordinary.sum() / fraud_label.sum()),
        "p_gen_middle": int(((p_gen > 0.2) & (p_gen < 0.8)).sum()),
        "lr_app_old_new": [
            float(lr_app[app_old][0]) if app_old.any() else None,
            float(lr_app[~app_old][0]) if (~app_old).any() else None,
        ],
    }
    log(f"[silver] {json.dumps(stats)}")
    n = led.height
    full = {k: np.full(n, -1, dtype=np.int64) for k in ("label", "structured")}
    full["label"][train] = silver
    full["structured"][train] = structured_only
    pgen_full = np.full(n, np.nan, dtype=np.float32)
    pgen_full[train] = p_gen
    return {
        "label": full["label"],
        "structured": full["structured"],
        "p_gen": pgen_full,
        "stats": stats,
    }


def save(result: dict) -> None:
    d = _dir()
    for k in ("label", "structured", "p_gen"):
        np.save(d / f"silver_{k}.npy", result[k])
    (d / "silver_stats.json").write_text(json.dumps(result["stats"], indent=1))
