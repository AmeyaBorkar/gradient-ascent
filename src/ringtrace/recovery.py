"""A recovery stage on legit calls, driven by operator roles in other months.

The bait rules showed that an account's label-free operator role in another month
(``ringtrace.baitroles``) carries information the row models never see. This stage generalises
that: a LightGBM multiclass model is fitted on the rows the current calls leave legit, with three
groups of inputs:

* the blend's 12-class probabilities for the row;
* for the sender and for a person receiver, whether the account holds each operator role in a
  train month other than the row's own (an October row: any train month);
* a few row fields (amount and its roundness, payment type, person receiver, new payee, the QR
  mismatch flag, hour).

Its target is the cleaned label. A legit call becomes class k when the stage gives k a
probability of at least ``threshold`` (k the most likely fraud class). Train months are handled
out of fold: month m's calls come from a model fitted on the other train months' legit calls, so
the scores of a recipe that uses the rule stay honest. October's come from a model fitted on all
three train months.

Used as a base-rate rule after ``blank_merchant_qr`` (``recipes/recovery.json``). The fit is
deterministic: fixed seed, fixed thread count, LightGBM's deterministic mode.
"""

from __future__ import annotations

import numpy as np
import polars as pl

TRAIN_MONTHS = (7, 8, 9)
TXN_CODES = {"PAY": 0, "QR": 1, "COLLECT": 2, "AUTOPAY": 3, "REFUND": 4}


def role_flags(
    sender: np.ndarray,
    receiver: np.ndarray,
    to_person: np.ndarray,
    roles: pl.DataFrame,
    names: list[str],
    months,
) -> np.ndarray:
    """Per row: the sender, and a person receiver, holds each role in one of ``months``."""
    r = roles.filter(pl.col("period").is_in(list(months)))
    out = np.zeros((len(sender), 2 * len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        accts = np.unique(r.filter(pl.col("role") == name)["acct"].to_numpy())
        out[:, j] = np.isin(sender, accts)
        out[:, len(names) + j] = to_person & np.isin(receiver, accts)
    return out


def row_fields(led: pl.DataFrame) -> np.ndarray:
    """The row fields the stage reads, one row per ledger row."""
    amt = led["amount"].to_numpy().astype(np.float64)
    txn = led["txn_type"].fill_null("").to_list()
    return np.column_stack(
        [
            np.log1p(amt),
            np.isclose(amt % 1, 0),
            np.isclose(amt % 10, 0),
            np.isclose(amt % 100, 0),
            np.array([TXN_CODES.get(t, 5) for t in txn], dtype=np.float64),
            led["to_person"].fill_null(False).to_numpy().astype(np.float64),
            led["new_payee"].fill_null(0).to_numpy().astype(np.float64),
            led["qr_vpa_mismatch"].fill_null(-1).to_numpy().astype(np.float64),
            led["timestamp"].dt.hour().to_numpy().astype(np.float64),
        ]
    ).astype(np.float32)


def _params(seed: int, threads: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": 12,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": threads,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
    }


def stage(
    pred: np.ndarray,
    proba: np.ndarray,
    target: np.ndarray,
    period: np.ndarray,
    is_test: np.ndarray,
    fields: np.ndarray,
    flags_for,
    threshold: float = 0.5,
    rounds: int = 300,
    seed: int = 0,
    threads: int = 4,
    log=print,
) -> np.ndarray:
    """The rule's core on plain arrays. ``flags_for(rows, months)`` returns the role flags of
    ``rows`` for roles held in ``months``; ``target`` is the training label (-1 where unknown)."""
    import lightgbm as lgb

    pred = np.asarray(pred, dtype=np.int64)
    out = pred.copy()
    legit = pred == 0
    train = ~is_test & (target >= 0)

    def features(rows: np.ndarray, months) -> np.ndarray:
        return np.column_stack([proba[rows], flags_for(rows, months), fields[rows]]).astype(
            np.float32
        )

    def fit(months_fit) -> object:
        parts, ys = [], []
        for j in months_fit:
            rows = np.flatnonzero(train & legit & (period == j))
            parts.append(features(rows, [o for o in TRAIN_MONTHS if o != j]))
            ys.append(target[rows])
        data = lgb.Dataset(np.vstack(parts), label=np.concatenate(ys))
        return lgb.train(_params(seed, threads), data, num_boost_round=rounds)

    def assign(model, rows: np.ndarray, months_roles) -> int:
        if not len(rows):
            return 0
        p = model.predict(features(rows, months_roles))
        k = p[:, 1:].argmax(axis=1) + 1
        add = p[np.arange(len(p)), k] >= threshold
        out[rows[add]] = k[add]
        return int(add.sum())

    present = [m for m in TRAIN_MONTHS if (train & (period == m)).any()]
    for m in present:
        others = [o for o in present if o != m]
        if not others:
            continue
        model = fit(others)
        n = assign(model, np.flatnonzero(~is_test & legit & (period == m)), others)
        log(f"[recovery] month {m}: {n} legit calls retyped (model fitted on {others})")
    if is_test.any() and present:
        model = fit(present)
        n = assign(model, np.flatnonzero(is_test & legit), present)
        log(f"[recovery] October: {n} legit calls retyped")
    return out


def recovery(
    pred: np.ndarray,
    proba: np.ndarray,
    led: pl.DataFrame,
    log=print,
    threshold: float = 0.5,
    rounds: int = 300,
    seed: int = 0,
    threads: int = 4,
) -> np.ndarray:
    """The rule: see the module docstring."""
    from ringtrace import baitroles, silver

    target = silver.load(led, "label")
    if target is None:
        # An October-like frame (a thinned September) carries no aligned cleaned labels and no
        # calls on its fit months, so the stage cannot be fitted there; it leaves the calls as
        # they are and the frame is judged without it.
        log("[recovery] no cleaned labels aligned with this ledger: rule skipped")
        return np.asarray(pred, dtype=np.int64).copy()
    cols = baitroles.columns(led)
    roles = baitroles.operator_roles(cols).select("acct", "period", "role")
    names = sorted(roles["role"].unique().to_list())
    sender = led["sender"].to_numpy()
    receiver = led["receiver"].to_numpy()
    to_person = led["to_person"].fill_null(False).to_numpy()

    def flags_for(rows: np.ndarray, months) -> np.ndarray:
        return role_flags(sender[rows], receiver[rows], to_person[rows], roles, names, months)

    return stage(
        pred,
        np.asarray(proba, dtype=np.float32),
        np.asarray(target, dtype=np.int64),
        led["period"].to_numpy(),
        led["is_test"].to_numpy(),
        row_fields(led),
        flags_for,
        threshold=threshold,
        rounds=rounds,
        seed=seed,
        threads=threads,
        log=log,
    )
