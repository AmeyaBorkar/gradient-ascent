"""Festival scaling of ordinary payments, then a recipe run on that ledger.

October's ordinary payments are about 1.166 times the training months' amounts, and balances
about 1.02 times, while the shares of round amounts stay the same. This module builds that
shift on any chosen months. Ordinary-routine rows (structured silver label 0) are rescaled,
fraud-routine rows keep their amounts, and the scored month (``is_test``) is never scaled.
Labels and every other column stay as they are.

The scaled ledger is installed with ``store.use_frame`` under its own cache key: the base
file digest, the factors, the periods, a hash of the mask and a hash of the scaled amounts.
Feature families are cached per ledger key, so a run on the scaled ledger cannot read families
built on the unscaled one.

    python -m ringtrace.festival show --periods 7,8,9
    python -m ringtrace.festival run recipes/champion.json --periods 9 --folds 9
    python -m ringtrace.festival run recipes/champion.json --periods 7,8,9

The default run name is ``<recipe stem>-fest-<periods>`` with the months sorted and joined by
hyphens (``champion-fest-7-8-9``). ``--name`` and ``--folds`` are forwarded to the recipe run.
``run`` still appends its usual fold suffix when only some months are held out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from ringtrace import silver, store

# October's ordinary payments, from the measured month-to-month amount and balance ratios.
DEFAULT_AMOUNT_FACTOR = 1.166
DEFAULT_BALANCE_FACTOR = 1.02
# Coarsest first would also work; the loop below writes fine units first so a coarser hit wins.
_UNITS = (0.01, 1.0, 10.0, 100.0)
# Slack on amount/unit against the nearest integer. Tight enough that one extra paisa does not
# count as a whole rupee, loose enough for binary noise on a paise grid.
_UNIT_TOL = 1e-6


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _normalise_periods(periods: Iterable[int]) -> tuple[int, ...]:
    try:
        out = tuple(sorted({int(p) for p in periods}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"periods must be integers, got {periods!r}") from exc
    if not out:
        raise ValueError("at least one period is required")
    return out


def _coarsest_unit(amount: np.ndarray) -> np.ndarray:
    """Coarsest of 100, 10, 1, 0.01 that divides ``amount`` within ``_UNIT_TOL``."""
    unit = np.full(amount.shape, 0.01, dtype=np.float64)
    for u in _UNITS:
        quot = amount / u
        unit = np.where(np.abs(quot - np.round(quot)) <= _UNIT_TOL, u, unit)
    return unit


def _next_ratio(unit: np.ndarray) -> np.ndarray:
    """Steps of ``unit`` in the next coarser unit (0.01 -> 100, 1 -> 10, 10 -> 10, 100 -> 0)."""
    ratio = np.zeros(unit.shape, dtype=np.float64)
    for fine, coarse in zip(_UNITS[:-1], _UNITS[1:], strict=True):
        ratio = np.where(np.isclose(unit, fine), np.rint(coarse / fine), ratio)
    return ratio


def _round_amounts(amount: np.ndarray, factor: float, seed: int = 0) -> np.ndarray:
    """Scale amounts and snap each one back onto its own roundness class.

    Each scaled amount lies between two multiples of its coarsest unit. It takes the upper one
    with probability equal to its position between them (seeded stochastic rounding), so the
    mean of every class scales by ``factor``; nearest rounding would leave a small multiple of
    100 such as 100, 200 or 300 where it was. Of two neighbouring multiples, at most one is also
    a multiple of the next coarser unit. Landing there would move the row into a rounder class
    (about one multiple of 10 in ten would become a multiple of 100, one paise amount in a
    hundred a whole rupee), so the row takes the other neighbour. The share of each class
    therefore stays exactly as it was.

    One unit is the floor, so a tiny round amount does not collapse to a zero that would leave
    its class. A zero amount is a multiple of 100, so the floor lifts it to 100.
    """
    amount = np.asarray(amount, dtype=np.float64)
    unit = _coarsest_unit(amount)
    target = (amount * factor) / unit
    low = np.floor(target)
    up = np.random.default_rng(seed).random(len(target)) < (target - low)
    steps = np.maximum(low + up, 1.0)
    ratio = _next_ratio(unit)
    coarser = (ratio > 0) & (np.fmod(steps, np.where(ratio > 0, ratio, 1.0)) == 0)
    other = np.where(steps > low, low, low + 1.0)
    steps = np.where(coarser, np.maximum(other, 1.0), steps)
    unit_paise = np.rint(unit * 100.0).astype(np.int64)
    paise = np.rint(steps).astype(np.int64) * unit_paise
    return paise.astype(np.float64) / 100.0


def _round_balance(balance: np.ndarray, factor: float) -> np.ndarray:
    balance = np.asarray(balance, dtype=np.float64)
    return np.rint(balance * factor * 100.0) / 100.0


def scale_rows(
    led: pl.DataFrame,
    mask: np.ndarray,
    amount_factor: float,
    balance_factor: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Copy ``led`` with money columns rescaled where ``mask`` is true.

    ``amount`` keeps the coarsest unit the original value already lands on (100, 10, 1 or 0.01)
    and is rounded, stochastically with ``seed``, to a neighbouring multiple of that unit that
    is not a multiple of a coarser one, never below one unit. ``balance_before``
    is rounded to the paisa. Every other row stays bit-identical, every other column is copied
    through, and ``led`` itself is not modified. Callers that must skip the scored month pass a
    mask from ``festival_mask``.
    """
    amount_factor = _finite("amount_factor", amount_factor)
    balance_factor = _finite("balance_factor", balance_factor)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (led.height,):
        raise ValueError(f"mask length {mask.size} does not match the ledger ({led.height} rows)")
    amount = np.asarray(led["amount"].to_numpy(), dtype=np.float64).copy()
    balance = np.asarray(led["balance_before"].to_numpy(), dtype=np.float64).copy()
    if bool(mask.any()):
        amount[mask] = _round_amounts(amount[mask], amount_factor, seed)
        balance[mask] = _round_balance(balance[mask], balance_factor)
    return led.with_columns(
        pl.Series("amount", amount, dtype=led.schema["amount"]),
        pl.Series("balance_before", balance, dtype=led.schema["balance_before"]),
    )


def festival_mask(led: pl.DataFrame, structured: np.ndarray, periods: Iterable[int]) -> np.ndarray:
    """Ordinary-routine rows of the chosen months.

    True where the row is not ``is_test``, its ``period`` is one of ``periods``, and its
    structured silver label is 0. Non-zero structured labels are the fraud routine and stay
    unscaled. Test rows are -1 in that array and are excluded here as well.
    """
    structured = np.asarray(structured)
    if structured.shape != (led.height,):
        raise ValueError(
            f"structured labels have shape {structured.shape}, ledger has {led.height} rows"
        )
    chosen = np.isin(led["period"].to_numpy(), _normalise_periods(periods))
    is_test = np.asarray(led["is_test"].to_numpy(), dtype=bool)
    return (~is_test) & chosen & (structured == 0)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:12]


def _cache_key(
    base_key: str,
    amount_factor: float,
    balance_factor: float,
    periods: tuple[int, ...],
    mask: np.ndarray,
    scaled: pl.DataFrame,
) -> str:
    """Key for the feature cache.

    Families are stored under the ledger key. Reusing the base file digest on its own would
    serve features built on the unscaled ledger, so the factors, periods and mask hash name
    this scaling and the scaled amount and balance bytes distinguish two money columns whose
    factor text would look the same. ``base_key`` is the source-file digest, so a change to
    the underlying files does not reuse a festival cache either.
    """
    tag = "-".join(str(p) for p in periods)
    mask_bytes = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    digest = hashlib.sha256()
    for name in ("amount", "balance_before"):
        values = np.ascontiguousarray(scaled[name].to_numpy(), dtype=np.float64)
        digest.update(name.encode())
        digest.update(values.tobytes())
    return (
        f"fest-{base_key}-a{amount_factor:.6g}-b{balance_factor:.6g}-p{tag}-"
        f"{_sha(mask_bytes)}-{digest.hexdigest()[:12]}"
    )


def _summary(
    base: pl.DataFrame,
    scaled: pl.DataFrame,
    mask: np.ndarray,
    periods: tuple[int, ...],
    amount_factor: float,
    balance_factor: float,
    key: str,
) -> dict:
    period = base["period"].to_numpy()
    old = base["amount"].to_numpy()
    new = scaled["amount"].to_numpy()
    rows = {int(p): int(np.count_nonzero(mask & (period == p))) for p in periods}
    old_sum = float(old[mask].sum()) if bool(mask.any()) else 0.0
    ratio = float(new[mask].sum() / old_sum) if old_sum != 0.0 else None
    return {
        "periods": list(periods),
        "amount_factor": float(amount_factor),
        "balance_factor": float(balance_factor),
        "rows_scaled": rows,
        "mean_amount_ratio": ratio,
        "key": key,
    }


def install(
    periods: Iterable[int],
    amount_factor: float = DEFAULT_AMOUNT_FACTOR,
    balance_factor: float = DEFAULT_BALANCE_FACTOR,
) -> dict:
    """Reset to the base ledger, scale ordinary rows in ``periods``, and serve that frame.

    Raises if the structured silver labels are missing. They say which train rows the fraud
    routine wrote. The summary counts scaled rows per period and gives the realised mean amount
    ratio on those rows (sum of scaled amounts over sum of original amounts). Rounding pulls
    that ratio slightly off ``amount_factor``. Calling ``install`` again starts from the base
    ledger, so the factors are not applied twice.
    """
    amount_factor = _finite("amount_factor", amount_factor)
    balance_factor = _finite("balance_factor", balance_factor)
    periods = _normalise_periods(periods)
    store.use_frame(None)
    base = store.ledger()
    structured = silver.load(base, "structured")
    if structured is None:
        raise RuntimeError(
            "structured silver labels are missing; build them before festival scaling"
        )
    mask = festival_mask(base, structured, periods)
    if bool(np.any(mask & np.asarray(base["is_test"].to_numpy(), dtype=bool))):
        raise RuntimeError("festival scaling tried to change the scored month")
    scaled = scale_rows(base, mask, amount_factor, balance_factor)
    # _KEY is still clear, so this is the source-file digest and not a frame key.
    base_key = store._ledger_key()
    key = _cache_key(base_key, amount_factor, balance_factor, periods, mask, scaled)
    store.use_frame(scaled, key=key)
    return _summary(base, scaled, mask, periods, amount_factor, balance_factor, key)


def _emit(summary: dict) -> None:
    payload = {
        "periods": summary["periods"],
        "amount_factor": summary["amount_factor"],
        "balance_factor": summary["balance_factor"],
        "rows_scaled": {str(p): summary["rows_scaled"][p] for p in summary["periods"]},
        "mean_amount_ratio": summary["mean_amount_ratio"],
        "key": summary["key"],
    }
    print(json.dumps(payload, indent=1), flush=True)


def _parse_periods(text: str) -> tuple[int, ...]:
    try:
        return _normalise_periods(part for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _run_argv(
    recipe: str, periods: tuple[int, ...], name: str | None, folds: str | None
) -> list[str]:
    run_name = name or f"{Path(recipe).stem}-fest-{'-'.join(str(p) for p in periods)}"
    argv = [recipe, "--name", run_name]
    if folds is not None:
        argv.extend(["--folds", folds])
    return argv


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m ringtrace.festival")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="scale the ledger, then run a recipe on it")
    run_p.add_argument("recipe")
    show_p = sub.add_parser("show", help="scale the ledger and print the summary")
    for parser in (run_p, show_p):
        parser.add_argument("--periods", required=True, type=_parse_periods)
        parser.add_argument("--amount-factor", type=float, default=DEFAULT_AMOUNT_FACTOR)
        parser.add_argument("--balance-factor", type=float, default=DEFAULT_BALANCE_FACTOR)
    run_p.add_argument("--name", default=None, help="run name; default <recipe>-fest-<periods>")
    run_p.add_argument("--folds", default=None, help="forwarded to the recipe run")
    args = ap.parse_args(argv)

    summary = install(args.periods, args.amount_factor, args.balance_factor)
    if args.cmd == "show":
        _emit(summary)
        return summary
    _emit(summary)
    from ringtrace.run import main as run_main

    return run_main(_run_argv(args.recipe, args.periods, args.name, args.folds))


if __name__ == "__main__":
    main()
