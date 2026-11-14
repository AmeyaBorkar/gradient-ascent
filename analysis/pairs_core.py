"""Pure helpers for ``analysis/pairs.py`` (no file access; tested on synthetic data).

The sim_swap / account_takeover drain template on the victim's own phone (drain shape 1: fresh
SIM, 0.25 balance steps) is written by both routines. The account_takeover routine's events in
this template differ from the sim_swap routine's own events in a few generation-time parameters:
they are shorter (2 / 3 / 4 rows in about 50 / 37 / 13 % against a uniform third each), their
payees are always new and their logins always 3 to 25 s (the sim_swap routine's own events break
either on about 9 % of events), and their steps come faster. ``drain_event_features`` turns those
into a label-free event table; ``event_scores`` fits a logistic model for the takeover side
(leave one month out on train, all train months for October); ``flip`` moves the sim_swap calls
of the selected events to account_takeover.
"""

from __future__ import annotations

import numpy as np
import polars as pl

LOGIN_TEMPLATE_MAX = 25.0  # seconds; the drain template's login range is 3 to 25 s
NIGHT_LAST_HOUR = 5

FEATURES = (
    "size",
    "login_hi",
    "old_payee",
    "home_phone",
    "pin_share",
    "log_login",
    "log_gap",
    "night",
    "log_balance",
    "age_minus1",
    "sim_age",
    "size2",
    "size4",
)


def drain_event_features(rows: pl.DataFrame) -> pl.DataFrame:
    """One line per drain event (``drn_event``) with the label-free features in ``FEATURES``.

    ``rows`` holds the event's rows with columns ``drn_event``, ``ts`` (seconds), ``hour``,
    ``sender``, ``device``, ``login_to_txn_sec``, ``new_payee``, ``pin_retries``,
    ``sim_age_hours``, ``balance_before`` and ``age_err`` (stated sender account age minus the age
    computed from the opening date). Events are sorted by id.
    """
    size = pl.len()
    out = rows.group_by("drn_event").agg(
        size.alias("size"),
        (pl.col("login_to_txn_sec") > LOGIN_TEMPLATE_MAX).any().cast(pl.Float64).alias("login_hi"),
        (pl.col("new_payee") == 0).any().cast(pl.Float64).alias("old_payee"),
        (pl.col("device") == pl.col("sender")).all().cast(pl.Float64).alias("home_phone"),
        (pl.col("pin_retries") > 0).mean().alias("pin_share"),
        pl.col("login_to_txn_sec").mean().log1p().alias("log_login"),
        ((pl.col("ts").max() - pl.col("ts").min()) / (size - 1).clip(lower_bound=1))
        .log1p()
        .alias("log_gap"),
        (pl.col("hour") <= NIGHT_LAST_HOUR).mean().alias("night"),
        pl.col("balance_before").max().log1p().alias("log_balance"),
        (pl.col("age_err") == -1).mean().alias("age_minus1"),
        pl.col("sim_age_hours").mean().alias("sim_age"),
    )
    out = out.with_columns(
        (pl.col("size") == 2).cast(pl.Float64).alias("size2"),
        (pl.col("size") == 4).cast(pl.Float64).alias("size4"),
        pl.col("size").cast(pl.Float64),
    )
    return out.sort("drn_event")


def matrix(events: pl.DataFrame) -> np.ndarray:
    """The feature matrix in ``FEATURES`` order."""
    return events.select([pl.col(c).cast(pl.Float64) for c in FEATURES]).to_numpy()


def event_label(n_own: np.ndarray, n_partner: np.ndarray) -> np.ndarray:
    """1 when the partner label is the event majority, 0 for the own label, -1 for a tie."""
    return np.where(n_partner > n_own, 1, np.where(n_own > n_partner, 0, -1))


def _fit_predict(X, y, fit, pred, c):
    from sklearn.linear_model import LogisticRegression

    mu = X[fit].mean(axis=0)
    sd = X[fit].std(axis=0) + 1e-9
    model = LogisticRegression(C=c, max_iter=5000)
    model.fit((X[fit] - mu) / sd, y[fit])
    return model.predict_proba((X[pred] - mu) / sd)[:, 1]


def event_scores(
    X: np.ndarray,
    label: np.ndarray,
    period: np.ndarray,
    is_test: np.ndarray,
    months=(7, 8, 9),
    c: float = 1.0,
) -> np.ndarray:
    """P(partner) per event: out of fold by month on train events (fit on the other months),
    and from one fit on every train month for test events. Tied events never enter a fit."""
    s = np.full(len(label), np.nan)
    train = ~is_test
    usable = train & (label >= 0)
    for m in months:
        fit = usable & (period != m)
        pred = train & (period == m)
        if pred.any():
            s[pred] = _fit_predict(X, label, fit, pred, c)
    if is_test.any():
        s[is_test] = _fit_predict(X, label, usable, is_test, c)
    return s


def flip(
    calls: np.ndarray,
    event_rows: list[np.ndarray],
    select: np.ndarray,
    from_cls: int,
    to_cls: int,
) -> np.ndarray:
    """A copy of ``calls`` with the ``from_cls`` calls of the selected events set to ``to_cls``.
    Calls of any other class (legit included) are left alone."""
    out = calls.copy()
    picked = [event_rows[i] for i in np.flatnonzero(select)]
    if not picked:
        return out
    rows = np.concatenate(picked).astype(np.int64)
    rows = rows[out[rows] == from_cls]
    out[rows] = to_cls
    return out


def month_gains(
    target: np.ndarray, base: np.ndarray, new: np.ndarray, period: np.ndarray, months=(7, 8, 9)
):
    """Macro-F1 of ``new`` minus that of ``base`` per month (rows with ``target`` >= 0)."""
    from ringtrace.scoring import macro_f1

    out = []
    for m in months:
        r = (period == m) & (target >= 0)
        out.append(macro_f1(target[r], new[r]) - macro_f1(target[r], base[r]))
    return out


def passes_bar(gains: dict[str, list[float]], tol: float = 1e-12) -> bool:
    """Non-negative on given and cleaned in every month, positive in at least one."""
    vals = [v for t in ("given", "cleaned") for v in gains[t]]
    return all(v >= -tol for v in vals) and any(v > tol for v in vals)
