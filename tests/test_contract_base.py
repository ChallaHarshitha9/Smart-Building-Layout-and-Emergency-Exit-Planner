import pytest
from pydantic import ValidationError

from smartbuilding.core.layout import Layout
from tests.factories import make_layout


def test_unknown_field_rejected(layout):
    data = layout.model_dump(mode="json")
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        Layout.model_validate(data)


def test_models_are_immutable(layout):
    with pytest.raises(ValidationError):
        layout.grid_w = 5  # type: ignore[misc]


def test_incompatible_schema_version_rejected(layout):
    data = layout.model_dump(mode="json")
    data["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError, match="schema_version"):
        Layout.model_validate(data)
    data["schema_version"] = "1.7.3"  # same major is accepted
    Layout.model_validate(data)


def test_content_hash_verified_on_load(layout):
    good = layout.with_hash()
    Layout.model_validate(good.model_dump(mode="json"))
    data = good.model_dump(mode="json")
    data["grid_w"] = 61
    with pytest.raises(ValidationError, match="content_hash"):
        Layout.model_validate(data)


def test_hash_stable_and_content_sensitive(layout):
    assert layout.compute_hash() == make_layout().compute_hash()
    assert layout.compute_hash() != make_layout(grid_w=61).compute_hash()
    assert layout.with_hash().compute_hash() == layout.compute_hash()  # hash excludes itself


def test_roundtrip_json(layout):
    again = Layout.model_validate_json(layout.model_dump_json())
    assert again == layout
