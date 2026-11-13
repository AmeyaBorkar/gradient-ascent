"""One place that decides how many CPU threads every library may use."""

from __future__ import annotations

import os

DEFAULT_BUDGET = 16  # the box allows 20 in total; keep headroom for the shell and I/O

_VARS = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def budget() -> int:
    return int(os.environ.get("RINGTRACE_THREADS", DEFAULT_BUDGET))


def pin(n: int | None = None) -> int:
    """Set the thread variables (unless the caller already set them) and return the budget."""
    n = n or budget()
    for var in _VARS:
        os.environ.setdefault(var, str(n))
    return n
