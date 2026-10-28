"""Loading and saving contracts from YAML/JSON with friendly errors.

* YAML is read with ``safe_load`` and duplicate keys are rejected (PyYAML would silently keep
  the last one).
* Validation errors name the file and the field path.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from smartbuilding.core.base import Contract

T = TypeVar("T", bound=BaseModel)


class ContractLoadError(ValueError):
    """Raised when a file cannot be parsed or does not match its contract."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def load_data(path: str | Path) -> Any:
    """Parse a YAML or JSON file into plain Python data."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractLoadError(f"{p}: cannot read file ({exc})") from exc
    try:
        if p.suffix.lower() in {".yaml", ".yml"}:
            return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 (SafeLoader subclass)
        if p.suffix.lower() == ".json":
            return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (yaml.YAMLError, ValueError) as exc:
        raise ContractLoadError(f"{p}: {exc}") from exc
    raise ContractLoadError(f"{p}: unsupported file type {p.suffix!r} (use .yaml, .yml or .json)")


def format_validation_error(path: str | Path, exc: ValidationError) -> str:
    lines = [f"{path}: {exc.error_count()} validation error(s)"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_model(path: str | Path, model: type[T]) -> T:
    """Load and validate ``path`` as ``model``."""
    data = load_data(path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ContractLoadError(format_validation_error(path, exc)) from exc


def save_json(obj: Contract, path: str | Path, with_hash: bool = True) -> None:
    """Write a contract as stable, human-readable JSON (sorted keys, trailing newline)."""
    target = obj.with_hash() if with_hash else obj
    text = json.dumps(target.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ParseResult(Generic[T]):
    """Outcome of a lenient parse: either an object or a list of readable errors."""

    obj: T | None
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.obj is not None


def try_validate(model: type[T], data: Any, label: str = "<input>") -> ParseResult[T]:
    """Validate ``data`` without raising.

    Used for LLM-only and random baselines: output that cannot even form a valid object is
    counted as structurally invalid and reported, instead of crashing the pipeline.
    """
    try:
        return ParseResult(model.model_validate(data), ())
    except ValidationError as exc:
        errors = tuple(
            f"{label}: {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        return ParseResult(None, errors)
