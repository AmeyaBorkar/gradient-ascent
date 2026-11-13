"""Where inputs, caches and outputs live. Everything can be overridden by environment."""

from __future__ import annotations

import os
from pathlib import Path


def _dir(var: str, default: str) -> Path:
    p = Path(os.environ.get(var, default)).expanduser()
    return p


def data_dir() -> Path:
    return _dir("RINGTRACE_DATA", "~/data")


def cache_dir() -> Path:
    p = _dir("RINGTRACE_CACHE", "~/cache/ringtrace")
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir() -> Path:
    p = _dir("RINGTRACE_OUT", "~/outputs")
    p.mkdir(parents=True, exist_ok=True)
    return p
