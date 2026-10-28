"""Canonical JSON and content hashing.

Hashes must be identical on every machine, so floats are rounded, keys are sorted
and Python's randomised ``hash()`` is never used.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

FLOAT_DECIMALS = 9


def canonicalize(obj: Any) -> Any:
    """Convert ``obj`` into a structure whose JSON text is unique for equal content."""
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite float cannot be hashed")
        rounded = round(obj, FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(obj, dict):
        return {str(k): canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [canonicalize(v) for v in obj]
    raise TypeError(f"cannot canonicalize type {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Deterministic compact JSON text."""
    return json.dumps(
        canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_object(obj: Any) -> str:
    """SHA-256 of the canonical JSON of ``obj``."""
    return sha256_hex(canonical_json(obj))
