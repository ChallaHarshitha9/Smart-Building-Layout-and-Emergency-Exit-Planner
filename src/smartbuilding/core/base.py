"""Base classes for all contracts.

``Part``      nested value object: strict, immutable.
``Contract``  top-level artefact: adds schema version and an optional content hash.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from smartbuilding.core.hashing import canonical_json, hash_object
from smartbuilding.core.primitives import Hash
from smartbuilding.core.version import SCHEMA_VERSION, major


class Part(BaseModel):
    """Strict immutable value object. Unknown fields are rejected."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class Contract(Part):
    """A versioned top-level object exchanged between modules."""

    schema_version: str = SCHEMA_VERSION
    content_hash: Hash | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        try:
            compatible = major(value) == major(SCHEMA_VERSION)
        except ValueError:
            compatible = False
        if not compatible:
            raise ValueError(f"schema_version {value!r} is not compatible with {SCHEMA_VERSION!r}")
        return value

    @model_validator(mode="after")
    def _check_hash(self) -> Self:
        if self.content_hash is not None and self.content_hash != self.compute_hash():
            raise ValueError("content_hash does not match the content of the object")
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """JSON-compatible dict without the hash field."""
        data = self.model_dump(mode="json")
        data.pop("content_hash", None)
        return data

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_dict())

    def compute_hash(self) -> str:
        return hash_object(self.canonical_dict())

    def with_hash(self) -> Self:
        """Return a copy that carries its own content hash."""
        return self.model_copy(update={"content_hash": self.compute_hash()})
