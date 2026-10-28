import pytest
from pydantic import ValidationError

from smartbuilding.core.bundle import (
    CrossCheckError,
    check_bundle,
    check_scenarios_against_profile,
    load_bundle,
    raise_if_problems,
)
from smartbuilding.core.config import (
    DoseConfig,
    HazardModel,
    OccupantProfile,
    RoomCatalogue,
    RunConfig,
    Sourcing,
    Thresholds,
)
from smartbuilding.core.loaders import load_model
from smartbuilding.core.scenario import ScenarioFamilies
from tests.factories import CONFIGS


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(CONFIGS / "run_default.yaml")


def test_default_bundle_loads_and_is_consistent(bundle):
    assert check_bundle(bundle) == []


def test_scenario_expansion_counts_and_split(bundle):
    s = bundle.scenarios()
    n_fam = len(bundle.families.families)
    assert len(s.scenarios) == n_fam * (3 + 30)
    assert len(s.by_split("repair")) == n_fam * 3
    assert len(s.by_split("test")) == n_fam * 30


def test_scenario_expansion_is_deterministic(bundle):
    from smartbuilding.core.scenario import expand_families

    a = expand_families(bundle.families, 42)
    b = expand_families(bundle.families, 42)
    c = expand_families(bundle.families, 43)
    assert a.compute_hash() == b.compute_hash()
    assert a.compute_hash() != c.compute_hash()


def test_assumptions_are_reported(bundle):
    entries = bundle.assumptions()
    assert "thresholds.max_travel_distance_m" in entries
    assert "hazard.A" in entries and "sensors.default_sensors" in entries
    assert any(e.startswith("catalogue.") for e in entries)


def test_sourcing_requires_citation_unless_assumption():
    Sourcing(basis="assumption")
    with pytest.raises(ValidationError, match="source"):
        Sourcing(basis="code")
    Sourcing(basis="code", source="Some Code, clause 1")


def test_thresholds_file_rejects_missing_source(tmp_path):
    text = (
        (CONFIGS / "thresholds.yaml")
        .read_text()
        .replace(
            'sourcing: &asm {basis: assumption, note: "PLACEHOLDER - replace with cited value"}',
            "sourcing: &asm {basis: code}",
        )
    )
    f = tmp_path / "t.yaml"
    f.write_text(text)
    with pytest.raises(Exception, match="source"):
        load_model(f, Thresholds)


def test_catalogue_cross_references_checked():
    data = load_model(CONFIGS / "room_catalogue.yaml", RoomCatalogue).model_dump(mode="json")
    data["content_hash"] = None
    data["room_types"][0]["flexible_to"] = ["nonexistent"]
    with pytest.raises(ValidationError, match="unknown type"):
        RoomCatalogue.model_validate(data)


def test_profile_shares_must_sum_to_one(bundle):
    p = bundle.profiles.profiles[0].model_dump(mode="json")
    p["groups"][0]["share"] = 0.5
    with pytest.raises(ValidationError, match="sum to 1"):
        OccupantProfile.model_validate(p)


def test_dose_thresholds_ordered():
    with pytest.raises(ValidationError, match="increasing"):
        DoseConfig(
            alert_min=3,
            critical_min=2,
            severe_min=8,
            penalty_alert=1,
            penalty_critical=2,
            penalty_severe=3,
        )
    with pytest.raises(ValidationError, match="non-decreasing"):
        DoseConfig(
            alert_min=1,
            critical_min=2,
            severe_min=8,
            penalty_alert=5,
            penalty_critical=2,
            penalty_severe=3,
        )


def test_hazard_state_mapping_ordered(bundle):
    d = bundle.hazard_a.model_dump(mode="json")
    d["content_hash"] = None
    d["state_mapping"] = {"smoky_min": 0.9, "fire_min": 0.5}
    with pytest.raises(ValidationError, match="smaller"):
        HazardModel.model_validate(d)


def test_run_config_paths_must_be_relative(bundle):
    d = bundle.run.model_dump(mode="json")
    d["content_hash"] = None
    for bad in ("/etc/catalogue.yaml", "C:\\data\\c.yaml", "../outside.yaml"):
        d["paths"]["catalogue"] = bad
        with pytest.raises(ValidationError, match="relative|leave the project"):
            RunConfig.model_validate(d)
    d["paths"]["catalogue"] = "room_catalogue.yaml"
    d["llm"]["cache_dir"] = "/tmp/cache"
    with pytest.raises(ValidationError, match="relative"):
        RunConfig.model_validate(d)


def test_cross_file_problems_are_found(bundle):
    d = bundle.families.model_dump(mode="json")
    d["content_hash"] = None
    d["families"][0]["sensor_model_ref"] = "nope"
    d["families"][1]["occupancy_period"] = "weekend"
    bad = ScenarioFamilies.model_validate(d)
    problems = check_bundle(type(bundle)(**{**bundle.__dict__, "families": bad}))
    assert any("nope" in p for p in problems) and any("weekend" in p for p in problems)
    with pytest.raises(CrossCheckError):
        raise_if_problems(problems)
    raise_if_problems([])


def test_scenario_templates_checked_against_profile(bundle):
    profile = bundle.profiles.get("profile_school")
    assert check_scenarios_against_profile(bundle.families, profile) == []
    d = bundle.families.model_dump(mode="json")
    d["content_hash"] = None
    d["families"][0]["groups"] = [
        {"group_id": "G_x", "template_id": "robot", "placement": {"mode": "random_room"}}
    ]
    bad = ScenarioFamilies.model_validate(d)
    assert "robot" in check_scenarios_against_profile(bad, profile)[0]
