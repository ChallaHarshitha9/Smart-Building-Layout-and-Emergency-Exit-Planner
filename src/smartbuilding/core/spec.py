"""BuildingSpec: the structured form of the user's requirements (output of the L1 parser)."""

from typing import Any, Self

from pydantic import Field, model_validator

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    ROOM_INSTANCE_RE,
    BuildingType,
    Name,
    NonNegFloat,
    NonNegInt,
    PosFloat,
    PosInt,
    RelationType,
    RoomRef,
    RoomTypeId,
    Side,
)


class Footprint(Part):
    width_m: PosFloat
    depth_m: PosFloat


class RoomRequest(Part):
    """A request for ``count`` rooms of one type. Instances are named ``R_<type>_<k>``."""

    room_type: RoomTypeId
    count: PosInt = 1
    min_area_m2: PosFloat
    max_area_m2: PosFloat
    needs_window: bool = False
    wet: bool = False
    occupant_load: NonNegInt = 0  # persons per room
    functional: bool = True  # counts towards usable floor area
    flexible_to: tuple[RoomTypeId, ...] = ()

    @model_validator(mode="after")
    def _check_area(self) -> Self:
        if self.min_area_m2 > self.max_area_m2:
            raise ValueError("min_area_m2 must not exceed max_area_m2")
        return self

    def instance_ids(self) -> tuple[str, ...]:
        return tuple(f"R_{self.room_type}_{k}" for k in range(1, self.count + 1))


class Relation(Part):
    """Spatial relation between two room types or two room instances."""

    a: RoomRef
    b: RoomRef
    type: RelationType
    hard: bool = False
    weight: PosFloat = 1.0
    max_distance_m: PosFloat | None = None  # only for 'near'
    min_distance_m: PosFloat | None = None  # only for 'far' / 'incompatible'

    @model_validator(mode="before")
    @classmethod
    def _incompatible_is_hard_by_default(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("hard") is None:
            data = {**data, "hard": str(data.get("type")) == RelationType.incompatible.value}
        return data

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.a == self.b:
            raise ValueError("a relation must connect two different rooms or room types")
        if self.max_distance_m is not None and self.type != RelationType.near:
            raise ValueError("max_distance_m is only allowed for 'near' relations")
        if self.min_distance_m is not None and self.type not in (
            RelationType.far,
            RelationType.incompatible,
        ):
            raise ValueError("min_distance_m is only allowed for 'far' or 'incompatible'")
        if self.hard and self.type == RelationType.near and self.max_distance_m is None:
            raise ValueError("a hard 'near' relation needs max_distance_m")
        if self.hard and self.type == RelationType.far and self.min_distance_m is None:
            raise ValueError("a hard 'far' relation needs min_distance_m")
        return self


class AccessibilityRules(Part):
    step_free_all_floors: bool = True
    lift_required: bool = False
    min_door_width_m: PosFloat | None = None  # overrides the global threshold if given


class Preferences(Part):
    adjacency_weight: NonNegFloat = 1.0
    daylight_weight: NonNegFloat = 1.0
    ventilation_weight: NonNegFloat = 1.0
    usable_area_weight: NonNegFloat = 1.0
    energy_weight: NonNegFloat = 1.0
    adaptability_weight: NonNegFloat = 1.0


class FutureChange(Part):
    """A planned conversion, for example 'two classrooms may later become labs'."""

    room_ref: RoomRef
    to_type: RoomTypeId
    count: PosInt = 1


class OccupancySpec(Part):
    total: PosInt
    profile_ref: Name


class BuildingSpec(Contract):
    """Structured requirements. Lengths in metres, areas in square metres."""

    spec_id: Name
    building_type: BuildingType
    footprint: Footprint
    floors: int = Field(ge=1, le=10)
    entrance_side: Side
    window_sides: tuple[Side, ...] = Field(min_length=1)
    rooms: tuple[RoomRequest, ...] = Field(min_length=1)
    relations: tuple[Relation, ...] = ()
    accessibility: AccessibilityRules = Field(default_factory=AccessibilityRules)
    preferences: Preferences = Field(default_factory=Preferences)
    future_changes: tuple[FutureChange, ...] = ()
    occupancy: OccupancySpec
    source_text: str | None = None  # the original natural-language requirement, if any

    # ---- helpers
    def request_for(self, room_type: str) -> RoomRequest | None:
        return next((r for r in self.rooms if r.room_type == room_type), None)

    def all_room_ids(self) -> tuple[str, ...]:
        return tuple(i for r in self.rooms for i in r.instance_ids())

    def resolve_ref(self, ref: str) -> RoomRequest:
        """Return the request a room ref points to; raise ValueError if it does not exist."""
        match = ROOM_INSTANCE_RE.match(ref)
        if match:
            request = self.request_for(match.group(1))
            if request is None or not 1 <= int(match.group(2)) <= request.count:
                raise ValueError(f"room reference {ref!r} does not exist in the spec")
            return request
        request = self.request_for(ref)
        if request is None:
            raise ValueError(f"room type {ref!r} does not exist in the spec")
        return request

    # ---- structural validation
    @model_validator(mode="after")
    def _check_structure(self) -> Self:
        types = [r.room_type for r in self.rooms]
        if len(set(types)) != len(types):
            raise ValueError("room types must be unique; use 'count' for several rooms")
        if len(set(self.window_sides)) != len(self.window_sides):
            raise ValueError("window_sides must not repeat")
        for rel in self.relations:
            self.resolve_ref(rel.a)
            self.resolve_ref(rel.b)
        for change in self.future_changes:
            request = self.resolve_ref(change.room_ref)
            if change.to_type not in request.flexible_to:
                raise ValueError(
                    f"future change {change.room_ref}->{change.to_type}: "
                    f"{request.room_type!r} is not flexible_to {change.to_type!r}"
                )
            limit = 1 if ROOM_INSTANCE_RE.match(change.room_ref) else request.count
            if change.count > limit:
                raise ValueError(f"future change count {change.count} exceeds available {limit}")
        return self
