import pytest
from pydantic import ValidationError

from smartbuilding.core.bundle import (
    check_spec_against_catalogue,
    check_spec_against_profiles,
    load_bundle,
)
from smartbuilding.core.spec import BuildingSpec, Relation
from tests.factories import CONFIGS, EXAMPLES, load_example_spec


def _data():
    return load_example_spec().model_dump(mode="json") | {"content_hash": None}


def test_examples_validate_and_roundtrip():
    for name in ("school_spec.json", "office_spec.json"):
        spec = load_example_spec(name)
        assert BuildingSpec.model_validate_json(spec.model_dump_json()) == spec
        assert spec.content_hash == spec.compute_hash()


def test_instance_ids_and_resolution():
    spec = load_example_spec()
    assert spec.all_room_ids()[:2] == ("R_classroom_1", "R_classroom_2")
    assert spec.resolve_ref("lab").room_type == "lab"
    assert spec.resolve_ref("R_classroom_6").room_type == "classroom"
    with pytest.raises(ValueError):
        spec.resolve_ref("R_classroom_7")
    with pytest.raises(ValueError):
        spec.resolve_ref("gym")


def test_relation_to_missing_room_rejected():
    d = _data()
    d["relations"] = [{"a": "lab", "b": "gym", "type": "near"}]
    with pytest.raises(ValidationError, match="gym"):
        BuildingSpec.model_validate(d)


def test_duplicate_room_types_rejected():
    d = _data()
    d["rooms"] = [d["rooms"][0], d["rooms"][0]]
    with pytest.raises(ValidationError, match="unique"):
        BuildingSpec.model_validate(d)


def test_bad_area_range_rejected():
    d = _data()
    d["rooms"][0]["min_area_m2"] = 100
    with pytest.raises(ValidationError, match="min_area"):
        BuildingSpec.model_validate(d)


def test_future_change_must_be_flexible():
    d = _data()
    d["future_changes"] = [{"room_ref": "lab", "to_type": "classroom", "count": 1}]
    with pytest.raises(ValidationError, match="flexible_to"):
        BuildingSpec.model_validate(d)


def test_future_change_count_limited():
    d = _data()
    d["future_changes"] = [{"room_ref": "classroom", "to_type": "lab", "count": 7}]
    with pytest.raises(ValidationError, match="exceeds"):
        BuildingSpec.model_validate(d)


def test_bad_ids_and_values_rejected():
    d = _data()
    d["rooms"][0]["room_type"] = "Class Room"
    with pytest.raises(ValidationError):
        BuildingSpec.model_validate(d)
    d = _data()
    d["floors"] = 11
    with pytest.raises(ValidationError):
        BuildingSpec.model_validate(d)


def test_incompatible_relation_is_hard_by_default():
    assert Relation(a="lab", b="canteen", type="incompatible").hard is True
    assert Relation(a="lab", b="classroom", type="near").hard is False
    assert Relation(a="lab", b="canteen", type="incompatible", hard=False).hard is False


def test_relation_rules():
    with pytest.raises(ValidationError):
        Relation(a="lab", b="lab", type="near")
    with pytest.raises(ValidationError):
        Relation(a="lab", b="classroom", type="adjacent", max_distance_m=5)
    with pytest.raises(ValidationError):
        Relation(a="lab", b="classroom", type="near", min_distance_m=5)


def test_spec_against_catalogue_and_profiles():
    bundle = load_bundle(CONFIGS / "run_default.yaml")
    spec = load_example_spec()
    assert check_spec_against_catalogue(spec, bundle.catalogue) == []
    assert check_spec_against_profiles(spec, bundle.profiles) == []
    office = load_example_spec("office_spec.json")
    d = office.model_dump(mode="json") | {"content_hash": None}
    d["occupancy"]["profile_ref"] = "profile_school"
    wrong = BuildingSpec.model_validate(d)
    assert "school" in check_spec_against_profiles(wrong, bundle.profiles)[0]
    d["rooms"][0]["room_type"] = "gym"
    d["relations"] = []
    d["future_changes"] = []
    unknown = BuildingSpec.model_validate(d)
    assert any("gym" in p for p in check_spec_against_catalogue(unknown, bundle.catalogue))


def test_example_files_exist():
    assert (EXAMPLES / "school_spec.json").exists()
