import json

import pytest

from smartbuilding.core.loaders import ContractLoadError, load_data, load_model, save_json
from smartbuilding.core.spec import BuildingSpec
from tests.factories import EXAMPLES


def test_yaml_duplicate_keys_rejected(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("a: 1\na: 2\n", encoding="utf-8")
    with pytest.raises(ContractLoadError, match="duplicate key"):
        load_data(f)


def test_json_duplicate_keys_rejected(tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ContractLoadError, match="duplicate key"):
        load_data(f)


def test_unsupported_and_missing_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(ContractLoadError, match="unsupported"):
        load_data(tmp_path / "a.txt")
    with pytest.raises(ContractLoadError, match="cannot read"):
        load_data(tmp_path / "missing.json")


def test_error_message_has_file_and_field_path(tmp_path):
    data = json.loads((EXAMPLES / "school_spec.json").read_text())
    data.pop("content_hash")
    data["floors"] = 0
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(data))
    with pytest.raises(ContractLoadError) as err:
        load_model(f, BuildingSpec)
    assert "bad.json" in str(err.value) and "floors" in str(err.value)


def test_save_and_reload_roundtrip_with_hash(tmp_path, layout):
    f = tmp_path / "layout.json"
    save_json(layout, f)
    again = load_model(f, type(layout))
    assert again.content_hash == layout.compute_hash()
    assert again.with_hash() == layout.with_hash()
    assert f.read_text().endswith("\n")


def test_yaml_yes_no_is_not_silently_a_vocabulary_value(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("side: no\n", encoding="utf-8")
    assert load_data(f) == {"side": False}  # documents the YAML quirk we avoid in vocabularies
