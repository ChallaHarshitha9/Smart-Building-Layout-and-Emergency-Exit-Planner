"""Reproducible seed derivation. No global random state is ever used."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(base: int, *keys: str | int) -> int:
    """Derive a child seed from ``base`` and a path of keys.

    Keys are stringified, so ``1`` and ``"1"`` give the same seed.
    """
    payload = "|".join([str(int(base)), *map(str, keys)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def make_rng(base: int, *keys: str | int) -> np.random.Generator:
    """A NumPy generator seeded from ``derive_seed(base, *keys)``."""
    return np.random.default_rng(derive_seed(base, *keys))
