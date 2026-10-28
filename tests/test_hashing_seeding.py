import pytest

from smartbuilding.core.hashing import canonical_json, hash_object
from smartbuilding.core.seeding import derive_seed, make_rng


def test_key_order_does_not_matter():
    assert hash_object({"a": 1, "b": [1, 2]}) == hash_object({"b": [1, 2], "a": 1})


def test_floats_are_rounded_so_tiny_noise_is_ignored():
    assert hash_object({"x": 0.1 + 0.2}) == hash_object({"x": 0.3})


def test_negative_zero_is_normalised():
    assert canonical_json({"x": -0.0}) == canonical_json({"x": 0.0})


def test_non_finite_and_unknown_types_rejected():
    with pytest.raises(ValueError):
        hash_object({"x": float("nan")})
    with pytest.raises(TypeError):
        hash_object({"x": {1, 2}})


def test_different_content_gives_different_hash():
    assert hash_object({"a": 1}) != hash_object({"a": 2})


def test_derive_seed_is_deterministic_and_key_sensitive():
    assert derive_seed(1, "a", 2) == derive_seed(1, "a", 2)
    assert derive_seed(1, "a", 2) != derive_seed(1, "a", 3)
    assert derive_seed(1, "a") != derive_seed(2, "a")
    assert 0 <= derive_seed(7, "x") < 2**63


def test_make_rng_reproducible():
    a = make_rng(5, "k").random(3)
    b = make_rng(5, "k").random(3)
    assert (a == b).all()
