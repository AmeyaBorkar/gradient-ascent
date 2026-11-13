"""Feature families, each built once and cached by content hash.

A family is a module with ``build(ledger) -> polars.DataFrame`` (one row per ledger row, in row
order). Its cache key combines the ledger's input digests with the hash of the module's own
source, so editing a family's code rebuilds only that family.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import time

import polars as pl

from ringtrace import paths, tables

FAMILIES = {
    "context": "ringtrace.context",
    "marks": "ringtrace.evidence.marks",
    "bursts": "ringtrace.evidence.bursts",
    "drains": "ringtrace.evidence.drains",
    "flows": "ringtrace.evidence.flows",
    "requests": "ringtrace.evidence.requests",
    "merchant": "ringtrace.evidence.merchant",
    "repeats": "ringtrace.evidence.repeats",
    "phones": "ringtrace.evidence.phones",
}

_LEDGER: pl.DataFrame | None = None


def ledger() -> pl.DataFrame:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = tables.ledger()
    return _LEDGER


def _source_hash(module) -> str:
    src = inspect.getsource(module)
    deps = getattr(module, "DEPENDS_ON", ())
    for d in deps:
        src += inspect.getsource(importlib.import_module(d))
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def _ledger_key() -> str:
    names = ("train", "test", "accounts", "merchants")
    return "-".join(tables.file_digest(paths.data_dir() / f"{n}.csv", 6) for n in names)


def family(name: str, verbose: bool = True) -> pl.DataFrame:
    module = importlib.import_module(FAMILIES[name])
    key = f"{name}-{_ledger_key()}-{_source_hash(module)}"
    target = paths.cache_dir() / "features" / f"{key}.parquet"
    if target.exists():
        return pl.read_parquet(target)
    t = time.time()
    df = module.build(ledger())
    if df.height != ledger().height:
        raise ValueError(f"family {name} returned {df.height} rows, ledger has {ledger().height}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(target)
    if verbose:
        print(f"[store] built {name}: {df.width} columns in {time.time() - t:.1f}s", flush=True)
    return df


def frame(names: list[str]) -> pl.DataFrame:
    parts = [family(n) for n in names]
    return pl.concat(parts, how="horizontal")
