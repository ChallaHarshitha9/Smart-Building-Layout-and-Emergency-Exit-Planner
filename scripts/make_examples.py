"""Regenerate the example specs in data/examples/ (run: python scripts/make_examples.py)."""

from pathlib import Path

from smartbuilding.core.loaders import save_json
from smartbuilding.core.spec import (
    AccessibilityRules,
    BuildingSpec,
    FutureChange,
    OccupancySpec,
    Relation,
    RoomRequest,
)

OUT = Path(__file__).resolve().parent.parent / "data" / "examples"

SCHOOL_TEXT = (
    "School wing, 30 m x 18 m, 6 classrooms, 2 labs, staff room, toilets, lift. "
    "Classrooms near labs. Wheelchair accessible. About 200 occupants. "
    "Two classrooms may later become labs."
)


def school() -> BuildingSpec:
    return BuildingSpec(
        spec_id="school_wing",
        building_type="school",
        footprint={"width_m": 30.0, "depth_m": 18.0},
        floors=2,
        entrance_side="south",
        window_sides=("north", "south"),
        rooms=(
            RoomRequest(
                room_type="classroom",
                count=6,
                min_area_m2=40,
                max_area_m2=60,
                needs_window=True,
                occupant_load=30,
                flexible_to=("lab",),
            ),
            RoomRequest(
                room_type="lab",
                count=2,
                min_area_m2=50,
                max_area_m2=80,
                needs_window=True,
                occupant_load=20,
            ),
            RoomRequest(
                room_type="staff_room",
                count=1,
                min_area_m2=20,
                max_area_m2=40,
                needs_window=True,
                occupant_load=8,
            ),
            RoomRequest(
                room_type="toilet",
                count=2,
                min_area_m2=8,
                max_area_m2=20,
                wet=True,
                functional=False,
            ),
        ),
        relations=(Relation(a="lab", b="classroom", type="near", weight=2.0),),
        accessibility=AccessibilityRules(step_free_all_floors=True, lift_required=True),
        future_changes=(FutureChange(room_ref="classroom", to_type="lab", count=2),),
        occupancy=OccupancySpec(total=200, profile_ref="profile_school"),
        source_text=SCHOOL_TEXT,
    )


def office() -> BuildingSpec:
    return BuildingSpec(
        spec_id="small_office",
        building_type="office",
        footprint={"width_m": 24.0, "depth_m": 14.0},
        floors=1,
        entrance_side="south",
        window_sides=("north", "south"),
        rooms=(
            RoomRequest(
                room_type="reception",
                count=1,
                min_area_m2=10,
                max_area_m2=30,
                needs_window=True,
                occupant_load=4,
            ),
            RoomRequest(
                room_type="open_office",
                count=1,
                min_area_m2=60,
                max_area_m2=120,
                needs_window=True,
                occupant_load=20,
            ),
            RoomRequest(
                room_type="office",
                count=4,
                min_area_m2=10,
                max_area_m2=20,
                needs_window=True,
                occupant_load=3,
            ),
            RoomRequest(
                room_type="meeting_room",
                count=2,
                min_area_m2=15,
                max_area_m2=30,
                needs_window=True,
                occupant_load=10,
                flexible_to=("lab",),
            ),
            RoomRequest(
                room_type="toilet",
                count=1,
                min_area_m2=8,
                max_area_m2=16,
                wet=True,
                functional=False,
            ),
        ),
        relations=(Relation(a="reception", b="open_office", type="near"),),
        accessibility=AccessibilityRules(step_free_all_floors=True),
        future_changes=(FutureChange(room_ref="meeting_room", to_type="lab", count=1),),
        occupancy=OccupancySpec(total=40, profile_ref="profile_office"),
    )


if __name__ == "__main__":
    save_json(school(), OUT / "school_spec.json")
    save_json(office(), OUT / "office_spec.json")
    print("written", OUT)
