"""Hand-made reference layouts and deliberately flawed variants.

The 'ok' layouts must pass every configured hard check. Each flawed variant breaks exactly the
checks listed in its ``expected`` set, which tests compare with the checker's real output.
Flawed layouts are structurally valid (they load) but design-invalid, like LLM-only or random
baseline output.
"""

from dataclasses import dataclass

from smartbuilding.core.layout import Door, Layout, Rect
from smartbuilding.core.primitives import Side
from smartbuilding.core.spec import BuildingSpec, Relation
from smartbuilding.layout.template import ExitPlan, SlotRoom, assemble_corridor_layout


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    layout: Layout
    expected: tuple[str, ...]  # exact set of failing hard-check ids
    spec: BuildingSpec | None = None  # a different spec, if the flaw is in the requirements


def _slots(spec: list[tuple[str, str, int]]) -> list[SlotRoom]:
    return [SlotRoom(room_id=i, room_type=t, width_cells=w) for i, t, w in spec]


def build_office(spec: BuildingSpec, layout_id: str = "LY_office_ok") -> Layout:
    south = _slots(
        [
            ("R_reception_1", "reception", 8),
            ("R_open_office_1", "open_office", 25),
            ("R_toilet_1", "toilet", 5),
            ("R_meeting_room_1", "meeting_room", 10),
        ]
    )
    north = _slots(
        [
            ("R_office_1", "office", 6),
            ("R_office_2", "office", 6),
            ("R_office_3", "office", 6),
            ("R_office_4", "office", 6),
            ("R_meeting_room_2", "meeting_room", 8),
        ]
    )
    return assemble_corridor_layout(layout_id=layout_id, spec=spec, south=south, north=north)


def build_school(spec: BuildingSpec, layout_id: str = "LY_school_ok") -> Layout:
    south = _slots(
        [
            ("R_classroom_1", "classroom", 12),
            ("R_classroom_2", "classroom", 12),
            ("R_classroom_3", "classroom", 12),
            ("R_lab_1", "lab", 16),
            ("R_toilet_1", "toilet", 4),
            ("R_toilet_2", "toilet", 4),
        ]
    )
    north = _slots(
        [
            ("R_classroom_4", "classroom", 12),
            ("R_classroom_5", "classroom", 12),
            ("R_classroom_6", "classroom", 12),
            ("R_lab_2", "lab", 16),
            ("R_staff_room_1", "staff_room", 6),
        ]
    )
    return assemble_corridor_layout(layout_id=layout_id, spec=spec, south=south, north=north)


# --------------------------------------------------------------------------- mutations


def _replace_room_rect(layout: Layout, room_id: str, rect: Rect, layout_id: str) -> Layout:
    plan = layout.floors[0]
    rooms = tuple(
        r.model_copy(update={"rect": rect}) if r.room_id == room_id else r for r in plan.rooms
    )
    return layout.model_copy(
        update={"layout_id": layout_id, "floors": (plan.model_copy(update={"rooms": rooms}),)}
    )


def _replace_door(layout: Layout, room_id: str, layout_id: str, **changes: int) -> Layout:
    plan = layout.floors[0]
    doors = tuple(d.model_copy(update=changes) if d.a == room_id else d for d in plan.doors)
    return layout.model_copy(
        update={"layout_id": layout_id, "floors": (plan.model_copy(update={"doors": doors}),)}
    )


def _drop_room(layout: Layout, room_id: str, layout_id: str) -> Layout:
    plan = layout.floors[0]
    return layout.model_copy(
        update={
            "layout_id": layout_id,
            "floors": (
                plan.model_copy(
                    update={
                        "rooms": tuple(r for r in plan.rooms if r.room_id != room_id),
                        "doors": tuple(d for d in plan.doors if d.a != room_id),
                    }
                ),
            ),
        }
    )


def _drop_door(layout: Layout, room_id: str, layout_id: str) -> Layout:
    plan = layout.floors[0]
    return layout.model_copy(
        update={
            "layout_id": layout_id,
            "floors": (
                plan.model_copy(update={"doors": tuple(d for d in plan.doors if d.a != room_id)}),
            ),
        }
    )


def _room(layout: Layout, room_id: str):
    return next(r for r in layout.floors[0].rooms if r.room_id == room_id)


def _door(layout: Layout, room_id: str) -> Door:
    return next(d for d in layout.floors[0].doors if d.a == room_id)


def office_variants(spec: BuildingSpec) -> list[Variant]:
    base = build_office(spec)
    out: list[Variant] = []

    r = _room(base, "R_office_2").rect
    out.append(
        Variant(
            "overlap",
            "R_office_2 is moved onto R_office_1",
            _replace_room_rect(
                base, "R_office_2", Rect(x=4, y=r.y, w=r.w, d=r.d), "LY_office_overlap"
            ),
            ("HC_no_overlap",),
        )
    )

    r = _room(base, "R_meeting_room_2").rect
    out.append(
        Variant(
            "out_of_bounds",
            "R_meeting_room_2 is moved partly outside the footprint",
            _replace_room_rect(
                base, "R_meeting_room_2", Rect(x=44, y=r.y, w=r.w, d=r.d), "LY_office_out_of_bounds"
            ),
            ("HC_connectivity", "HC_door_placement", "HC_inside_footprint"),
        )
    )

    narrow = assemble_corridor_layout(
        layout_id="LY_office_narrow_corridor",
        spec=spec,
        corridor_width_cells=2,
        south_depth_cells=12,
        north_depth_cells=12,
        south=_slots(
            [
                ("R_reception_1", "reception", 8),
                ("R_open_office_1", "open_office", 25),
                ("R_toilet_1", "toilet", 5),
                ("R_meeting_room_1", "meeting_room", 10),
            ]
        ),
        north=_slots(
            [
                ("R_office_1", "office", 6),
                ("R_office_2", "office", 6),
                ("R_office_3", "office", 6),
                ("R_office_4", "office", 6),
                ("R_meeting_room_2", "meeting_room", 8),
            ]
        ),
        exits=(ExitPlan(Side.west, 2), ExitPlan(Side.east, 2)),
    )
    out.append(
        Variant("narrow_corridor", "the corridor is 1.0 m wide", narrow, ("HC_corridor_width",))
    )

    one_exit = assemble_corridor_layout(
        layout_id="LY_office_single_exit",
        spec=spec,
        exits=(ExitPlan(Side.west),),
        south=_slots(
            [
                ("R_reception_1", "reception", 8),
                ("R_open_office_1", "open_office", 25),
                ("R_toilet_1", "toilet", 5),
                ("R_meeting_room_1", "meeting_room", 10),
            ]
        ),
        north=_slots(
            [
                ("R_office_1", "office", 6),
                ("R_office_2", "office", 6),
                ("R_office_3", "office", 6),
                ("R_office_4", "office", 6),
                ("R_meeting_room_2", "meeting_room", 8),
            ]
        ),
    )
    out.append(Variant("single_exit", "only one exit", one_exit, ("HC_min_exit_count",)))

    out.append(
        Variant(
            "disconnected_room",
            "R_office_3 has no door",
            _drop_door(base, "R_office_3", "LY_office_disconnected"),
            ("HC_connectivity",),
        )
    )
    out.append(
        Variant(
            "narrow_door",
            "the door of R_office_1 is 0.5 m wide",
            _replace_door(base, "R_office_1", "LY_office_narrow_door", width_cells=1),
            ("HC_door_width",),
        )
    )
    out.append(
        Variant(
            "missing_room",
            "R_office_4 is missing",
            _drop_room(base, "R_office_4", "LY_office_missing_room"),
            ("HC_required_rooms",),
        )
    )

    r = _room(base, "R_office_2").rect
    small = _replace_room_rect(
        base, "R_office_2", Rect(x=r.x, y=r.y, w=3, d=r.d), "LY_office_small_room"
    )
    small = _replace_door(small, "R_office_2", "LY_office_small_room", x=r.x)
    out.append(Variant("small_room", "R_office_2 is 9 square metres", small, ("HC_room_area",)))

    d = _door(base, "R_office_1")
    out.append(
        Variant(
            "misplaced_door",
            "the door of R_office_1 is off the wall",
            _replace_door(base, "R_office_1", "LY_office_misplaced_door", y=d.y + 1),
            ("HC_connectivity", "HC_door_placement"),
        )
    )

    plan = base.floors[0]
    moved = tuple(e.model_copy(update={"x": 40}) if e.exit_id == "E_2" else e for e in plan.exits)
    out.append(
        Variant(
            "exit_off_boundary",
            "E_2 is inside the building, not on its boundary",
            base.model_copy(
                update={
                    "layout_id": "LY_office_exit_off_boundary",
                    "floors": (plan.model_copy(update={"exits": moved}),),
                }
            ),
            ("HC_exit_placement", "HC_min_exit_count"),
        )
    )
    return out


def school_variants(spec: BuildingSpec) -> list[Variant]:
    base = build_school(spec)
    south = _slots(
        [
            ("R_classroom_1", "classroom", 12),
            ("R_classroom_2", "classroom", 12),
            ("R_classroom_3", "classroom", 12),
            ("R_lab_1", "lab", 16),
            ("R_toilet_1", "toilet", 4),
            ("R_toilet_2", "toilet", 4),
        ]
    )
    north = _slots(
        [
            ("R_classroom_4", "classroom", 12),
            ("R_classroom_5", "classroom", 12),
            ("R_classroom_6", "classroom", 12),
            ("R_lab_2", "lab", 16),
            ("R_staff_room_1", "staff_room", 6),
        ]
    )
    long_travel = assemble_corridor_layout(
        layout_id="LY_school_single_exit_long_travel",
        spec=spec,
        south=south,
        north=north,
        exits=(ExitPlan(Side.west),),
    )
    variants = [
        Variant(
            "single_exit_long_travel",
            "one exit at the west end of a 30 m corridor",
            long_travel,
            ("HC_min_exit_count", "HC_travel_distance"),
        )
    ]
    bad_spec = spec.model_copy(
        update={
            "spec_id": "school_1f_incompatible",
            "relations": (
                *spec.relations,
                Relation(a="lab", b="toilet", type="incompatible", min_distance_m=12.0),
            ),
            "content_hash": None,
        }
    )
    bad_layout = base.model_copy(
        update={"layout_id": "LY_school_incompatible", "spec_hash": bad_spec.compute_hash()}
    )
    variants.append(
        Variant(
            "incompatible_labs_toilets",
            "labs must be 12 m from toilets, but R_lab_1 is next to R_toilet_1",
            bad_layout,
            ("HC_incompatible_separation",),
            spec=bad_spec,
        )
    )
    return variants
