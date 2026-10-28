"""Layout: rooms, corridors, doors, exits, stairs and lifts on an integer cell grid.

A Layout may describe a *design-invalid* building (overlaps, out-of-bounds rooms). That is
deliberate: LLM-only and random baselines must be representable so the checker can reject
them. The contract enforces structural well-formedness only (unique ids, references exist).
"""

from typing import Self

from pydantic import Field, model_validator

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.hashing import hash_object
from smartbuilding.core.primitives import (
    ROOM_INSTANCE_RE,
    ConstraintKind,
    CorridorId,
    DoorId,
    DoorState,
    ExitId,
    Finite,
    Hash,
    LayoutId,
    LiftId,
    MoveId,
    NonNegFloat,
    NonNegInt,
    PosFloat,
    PosInt,
    RegionId,
    RoomId,
    RoomTypeId,
    Side,
    StairId,
    VerificationStatus,
)


class Rect(Part):
    """Axis-aligned rectangle in grid cells. (x, y) is the south-west corner."""

    x: NonNegInt
    y: NonNegInt
    w: PosInt
    d: PosInt

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.d

    @property
    def area_cells(self) -> int:
        return self.w * self.d

    def overlaps(self, other: "Rect") -> bool:
        return not (
            self.x2 <= other.x or other.x2 <= self.x or self.y2 <= other.y or other.y2 <= self.y
        )


class RoomPlacement(Part):
    """A placed room.

    ``room_id`` is a stable identity: it never changes during repair or conversion. Its type
    part records the type the room was *created* as. ``room_type`` is the current type; after a
    conversion (for example classroom to lab) the id keeps the original type and
    ``converted_from`` records it.
    """

    room_id: RoomId
    room_type: RoomTypeId
    floor: NonNegInt = 0
    rect: Rect
    converted_from: RoomTypeId | None = None

    @model_validator(mode="after")
    def _id_and_type_consistent(self) -> Self:
        match = ROOM_INSTANCE_RE.match(self.room_id)
        if match is None:  # pragma: no cover - the RoomId pattern already guarantees a match
            raise ValueError(f"malformed room id {self.room_id!r}")
        origin = match.group(1)
        if self.room_type == origin:
            if self.converted_from is not None:
                raise ValueError("converted_from must be empty when the type is unchanged")
        elif self.converted_from != origin:
            raise ValueError(
                f"room {self.room_id!r} has type {self.room_type!r} but was created as "
                f"{origin!r}; set converted_from={origin!r} to record the conversion"
            )
        return self


class CorridorPlacement(Part):
    corridor_id: CorridorId
    floor: NonNegInt = 0
    rect: Rect

    @property
    def width_cells(self) -> int:
        return min(self.rect.w, self.rect.d)


class Door(Part):
    door_id: DoorId
    floor: NonNegInt = 0
    a: RegionId
    b: RegionId
    x: NonNegInt
    y: NonNegInt
    width_cells: PosInt
    state: DoorState = DoorState.open

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if self.a == self.b:
            raise ValueError("a door must connect two different regions")
        return self


class Exit(Part):
    exit_id: ExitId
    floor: NonNegInt = 0
    x: NonNegInt
    y: NonNegInt
    width_cells: PosInt
    connects_to: RegionId
    side: Side | None = None


class Stair(Part):
    stair_id: StairId
    rect: Rect
    floors: tuple[NonNegInt, ...] = Field(min_length=2)
    width_cells: PosInt


class Lift(Part):
    lift_id: LiftId
    rect: Rect
    floors: tuple[NonNegInt, ...] = Field(min_length=2)
    accessible: bool = True


class VerticalCore(Part):
    """Stairs and lifts. One rectangle per shaft, shared by all floors it serves."""

    stairs: tuple[Stair, ...] = ()
    lifts: tuple[Lift, ...] = ()


class FloorPlan(Part):
    floor: NonNegInt
    rooms: tuple[RoomPlacement, ...] = ()
    corridors: tuple[CorridorPlacement, ...] = ()
    doors: tuple[Door, ...] = ()
    exits: tuple[Exit, ...] = ()


class ConstraintResult(Part):
    constraint_id: str = Field(min_length=1)
    kind: ConstraintKind
    satisfied: bool
    value: Finite | None = None
    limit: Finite | None = None
    detail: str = ""
    entity_ids: tuple[str, ...] = ()


class SoftScore(Part):
    name: str = Field(min_length=1)
    cost: NonNegFloat
    weight: NonNegFloat = 1.0


class ConflictSet(Part):
    """A minimal group of constraints that cannot all hold, with a readable explanation."""

    constraint_ids: tuple[str, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class Verification(Part):
    status: VerificationStatus = VerificationStatus.unchecked
    hard: tuple[ConstraintResult, ...] = ()
    soft: tuple[SoftScore, ...] = ()
    conflicts: tuple[ConflictSet, ...] = ()

    @model_validator(mode="after")
    def _status_consistent(self) -> Self:
        if self.status == VerificationStatus.unchecked and (self.hard or self.soft):
            raise ValueError("an unchecked verification must not contain results")
        return self

    @property
    def all_hard_satisfied(self) -> bool:
        return self.status == VerificationStatus.checked and all(r.satisfied for r in self.hard)

    @property
    def total_soft_cost(self) -> float:
        return sum(s.cost * s.weight for s in self.soft)


class LayoutProvenance(Part):
    generator: str = Field(min_length=1)
    seed: NonNegInt = 0
    parent_layout_id: LayoutId | None = None
    repair_move_id: MoveId | None = None


class Layout(Contract):
    """Rooms, corridors, doors, exits, stairs and lifts on an integer cell grid.

    Conventions: plan view, x grows east, y grows north, origin at the south-west corner of
    the footprint; floor is an integer index (0 = ground). Lengths in metres, areas in square
    metres, times in seconds; geometry is stored as integer cells of cell_size_m metres.
    Room ids are stable identities and never change across repair.
    """

    layout_id: LayoutId
    spec_hash: Hash
    cell_size_m: PosFloat = 0.5
    grid_w: PosInt
    grid_d: PosInt
    floors: tuple[FloorPlan, ...] = Field(min_length=1)
    vertical_core: VerticalCore = Field(default_factory=VerticalCore)
    verification: Verification = Field(default_factory=Verification)
    provenance: LayoutProvenance

    # ---- helpers
    def floor_plan(self, floor: int) -> FloorPlan:
        for plan in self.floors:
            if plan.floor == floor:
                return plan
        raise KeyError(f"no floor {floor}")

    def room_ids(self) -> tuple[str, ...]:
        return tuple(r.room_id for plan in self.floors for r in plan.rooms)

    def region_ids(self) -> frozenset[str]:
        ids: set[str] = set()
        for plan in self.floors:
            ids.update(r.room_id for r in plan.rooms)
            ids.update(c.corridor_id for c in plan.corridors)
            ids.update(e.exit_id for e in plan.exits)
        ids.update(s.stair_id for s in self.vertical_core.stairs)
        ids.update(lift.lift_id for lift in self.vertical_core.lifts)
        return frozenset(ids)

    @property
    def geometry_hash(self) -> str:
        """Hash of geometry only. Ignores ids, verification and provenance.

        Used as the cache key for evaluation results and as the link from NavGraph
        and ConstructionPlan back to the layout they were derived from.
        """
        data = self.canonical_dict()
        keep = {k: data[k] for k in ("cell_size_m", "grid_w", "grid_d", "floors", "vertical_core")}
        return hash_object(keep)

    # ---- structural validation
    @model_validator(mode="after")
    def _check_structure(self) -> Self:
        floor_ids = [plan.floor for plan in self.floors]
        if len(set(floor_ids)) != len(floor_ids):
            raise ValueError("floor indices must be unique")
        all_ids: list[str] = []
        door_ids: list[str] = []
        for plan in self.floors:
            for room in plan.rooms:
                if room.floor != plan.floor:
                    raise ValueError(
                        f"room {room.room_id} has floor {room.floor} in floor {plan.floor}"
                    )
            for corridor in plan.corridors:
                if corridor.floor != plan.floor:
                    raise ValueError(f"corridor {corridor.corridor_id} is on the wrong floor")
            for door in plan.doors:
                if door.floor != plan.floor:
                    raise ValueError(f"door {door.door_id} is on the wrong floor")
            for ex in plan.exits:
                if ex.floor != plan.floor:
                    raise ValueError(f"exit {ex.exit_id} is on the wrong floor")
            all_ids += [r.room_id for r in plan.rooms]
            all_ids += [c.corridor_id for c in plan.corridors]
            all_ids += [e.exit_id for e in plan.exits]
            door_ids += [d.door_id for d in plan.doors]
        all_ids += [s.stair_id for s in self.vertical_core.stairs]
        all_ids += [lift.lift_id for lift in self.vertical_core.lifts]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("region ids (rooms, corridors, exits, stairs, lifts) must be unique")
        if len(set(door_ids)) != len(door_ids):
            raise ValueError("door ids must be unique")

        regions = self.region_ids()
        for plan in self.floors:
            for door in plan.doors:
                for end in (door.a, door.b):
                    if end not in regions:
                        raise ValueError(f"door {door.door_id} refers to unknown region {end}")
            for ex in plan.exits:
                if ex.connects_to not in regions:
                    raise ValueError(f"exit {ex.exit_id} refers to unknown region {ex.connects_to}")
        known = set(floor_ids)
        shaft_floors = [s.floors for s in self.vertical_core.stairs]
        shaft_floors += [lift.floors for lift in self.vertical_core.lifts]
        for floors in shaft_floors:
            if not set(floors) <= known:
                raise ValueError("a stair or lift serves a floor that does not exist")
            if list(floors) != sorted(set(floors)):
                raise ValueError("stair and lift floors must be sorted and unique")
        return self
