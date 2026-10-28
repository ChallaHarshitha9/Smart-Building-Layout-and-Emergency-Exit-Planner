"""Versioning, conventions, stable ids, devices, lenient parsing and neutral wording."""

import pytest
from pydantic import ValidationError

from smartbuilding.core.base import Contract
from smartbuilding.core.devices import device_for_region, device_kind, make_device_id
from smartbuilding.core.diff import check_stable_room_ids, layout_edit_distance
from smartbuilding.core.layout import FloorPlan, Layout, Rect, RoomPlacement
from smartbuilding.core.loaders import try_validate
from smartbuilding.core.navgraph import NavGraph
from smartbuilding.core.primitives import COORDINATE_CONVENTION, DeviceKind
from smartbuilding.core.runtime import BeliefState
from smartbuilding.core.schemas import CONTRACTS
from smartbuilding.core.terminology import CHECKED_PHRASE, find_banned_terms
from smartbuilding.core.version import SCHEMA_VERSION
from tests.factories import make_layout

H = "a" * 64


# ---- versioning, units, coordinate convention


def test_every_contract_carries_a_schema_version():
    for name, cls in CONTRACTS.items():
        if issubclass(cls, Contract):
            assert "schema_version" in cls.model_fields, name
            assert cls.model_fields["schema_version"].default == SCHEMA_VERSION


def test_physical_fields_name_their_unit():
    """Any field that holds a physical quantity must carry a unit suffix."""
    suffixes = ("_m", "_m2", "_s", "_mps", "_days", "_units", "_cells", "_pers", "_per_min")
    physical = ("width", "depth", "length", "area", "time", "speed", "distance", "duration")
    from pydantic import BaseModel

    def models(cls):
        seen, stack = set(), [cls]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            if c.__name__.endswith("Weights"):  # dimensionless weights, not quantities
                continue
            for f in c.model_fields.values():
                for t in getattr(f.annotation, "__args__", ()) or (f.annotation,):
                    if isinstance(t, type) and issubclass(t, BaseModel):
                        stack.append(t)
            yield c

    allowed = {"w", "d", "x", "y", "width_cells", "dispersion", "share", "width"}
    offenders = []
    for cls in CONTRACTS.values():
        for model in models(cls):
            for name in model.model_fields:
                if any(p in name for p in physical) and not name.endswith(suffixes):
                    if name not in allowed and not name.endswith("_weight"):
                        offenders.append(f"{model.__name__}.{name}")
    assert sorted(set(offenders)) == []


def test_coordinate_convention_is_in_the_published_schema():
    text = Layout.model_json_schema()["description"]
    assert "x grows east" in text and "y grows north" in text and "south-west" in text
    assert "x grows east" in COORDINATE_CONVENTION
    assert "metres" in NavGraph.model_json_schema()["description"]


# ---- stable ids and edit distance


def _converted(layout: Layout) -> Layout:
    plan = layout.floors[0]
    room = plan.rooms[0]
    changed = room.model_copy(update={"room_type": "lab", "converted_from": "office"})
    return make_layout(
        layout_id="LY_conv", floors=(plan.model_copy(update={"rooms": (changed, plan.rooms[1])}),)
    )


def test_conversion_keeps_room_ids_stable(layout):
    child = _converted(layout)
    assert child.room_ids() == layout.room_ids()
    assert check_stable_room_ids(layout, child) == []


def test_unstable_ids_are_reported(layout):
    plan = layout.floors[0]
    renamed = RoomPlacement(room_id="R_office_9", room_type="office", rect=plan.rooms[0].rect)
    doors = tuple(
        d.model_copy(update={"a": "R_office_9"}) if d.a == "R_office_1" else d for d in plan.doors
    )
    child = make_layout(
        floors=(plan.model_copy(update={"rooms": (renamed, plan.rooms[1]), "doors": doors}),),
    )
    problems = check_stable_room_ids(layout, child)
    assert "room R_office_1 disappeared" in problems and "room R_office_9 appeared" in problems
    assert (
        check_stable_room_ids(
            layout, child, allow_removed=["R_office_1"], allow_added=["R_office_9"]
        )
        == []
    )


def test_edit_distance_counts_changed_entities(layout):
    assert layout_edit_distance(layout, layout).total == 0
    moved = layout.floors[0].rooms[0].model_copy(update={"rect": Rect(x=2, y=20, w=20, d=16)})
    plan = layout.floors[0].model_copy(update={"rooms": (moved, layout.floors[0].rooms[1])})
    d = layout_edit_distance(layout, make_layout(floors=(plan,)))
    assert (d.rooms, d.total) == (1, 1)
    d2 = layout_edit_distance(layout, _converted(layout))
    assert d2.rooms == 1
    empty = make_layout(floors=(FloorPlan(floor=0),))
    assert layout_edit_distance(layout, empty).total == 2 + 1 + 2 + 2


# ---- devices in beliefs


def test_device_ids_encode_their_kind():
    assert device_kind("DV_lift_1") == DeviceKind.lift
    assert make_device_id(DeviceKind.detector, 4) == "DV_detector_4"
    assert device_for_region("L_2") == "DV_lift_2"
    assert device_for_region("D_7") == "DV_door_7"
    with pytest.raises(ValueError):
        device_for_region("C_1")
    with pytest.raises(ValueError):
        device_kind("N_room_1")


def test_belief_tracks_lifts_doors_detectors_equipment_and_alarms():
    devices = {
        "DV_lift_1": 0.95,
        "DV_door_3": 0.9,
        "DV_detector_2": 0.7,
        "DV_equipment_1": 0.99,
        "DV_alarm_1": 0.98,
    }
    b = BeliefState(graph_hash=H, t_s=0, node_belief={"N_room_1": (1.0, 0.0, 0.0)}, devices=devices)
    assert {device_kind(k) for k in b.devices} == set(DeviceKind)
    assert BeliefState.model_validate_json(b.model_dump_json()) == b
    with pytest.raises(ValidationError):
        BeliefState(
            graph_hash=H,
            t_s=0,
            node_belief={"N_room_1": (1.0, 0.0, 0.0)},
            devices={"DV_toaster_1": 0.5},
        )
    with pytest.raises(ValidationError):
        BeliefState(
            graph_hash=H,
            t_s=0,
            node_belief={"N_room_1": (1.0, 0.0, 0.0)},
            devices={"DV_lift_1": 1.5},
        )


# ---- lenient parsing (design-invalid vs structurally invalid)


def test_try_validate_returns_object_or_errors_without_raising(layout):
    good = try_validate(Layout, layout.model_dump(mode="json"), "llm-1")
    assert good.ok and good.errors == ()
    data = layout.model_dump(mode="json")
    data["floors"][0]["doors"][0]["b"] = "C_99"
    bad = try_validate(Layout, data, "llm-2")
    assert not bad.ok and bad.obj is None
    assert any("llm-2" in e and "unknown region" in e for e in bad.errors)
    garbage = try_validate(Layout, {"nonsense": 1}, "llm-3")
    assert not garbage.ok and garbage.errors


def test_overlapping_layout_is_loadable_not_an_error(layout):
    data = layout.model_dump(mode="json")
    data["floors"][0]["rooms"][1]["rect"] = data["floors"][0]["rooms"][0]["rect"]
    result = try_validate(Layout, data)
    assert result.ok  # design-invalid but structurally valid: the rule checker flags it


# ---- neutral wording


def test_banned_terms_are_found_as_whole_words_only():
    assert find_banned_terms("The building is safe and compliant.") == ["compliant", "safe"]
    assert find_banned_terms("Approved and certified; unsafe.") == [
        "approved",
        "certified",
        "unsafe",
    ]
    assert find_banned_terms("Fire safety equipment, safeguards, compliance-free?") == [
        "compliance"
    ]
    assert find_banned_terms(CHECKED_PHRASE) == []
    assert find_banned_terms("SAFE") == ["safe"]
    assert find_banned_terms("It passes all configured checks.") == []
