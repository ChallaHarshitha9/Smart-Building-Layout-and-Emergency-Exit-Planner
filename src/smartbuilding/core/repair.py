"""Findings from the bottleneck analysis and typed repair moves.

Every proposer (rules, LLM agents, search) must emit a ``RepairMove``. The action is a
discriminated union, so an LLM can only propose one of the allowed move types with exactly
the allowed parameters. Anything else fails validation.
"""

from typing import Annotated, Literal

from pydantic import Field

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    CorridorId,
    ExitId,
    FindingId,
    FindingKind,
    Finite,
    MoveId,
    NonNegFloat,
    NonNegInt,
    PosInt,
    Proposer,
    RegionId,
    RoomId,
    RoomTypeId,
)


class Finding(Part):
    finding_id: FindingId
    kind: FindingKind
    severity: NonNegFloat
    entity_ids: tuple[str, ...] = Field(min_length=1)
    value: Finite | None = None
    detail: str = ""


class WidenCorridor(Part):
    type: Literal["widen_corridor"] = "widen_corridor"
    corridor_id: CorridorId
    new_width_cells: PosInt


class AddExit(Part):
    type: Literal["add_exit"] = "add_exit"
    floor: NonNegInt
    x: NonNegInt
    y: NonNegInt
    width_cells: PosInt
    connects_to: RegionId


class MoveExit(Part):
    type: Literal["move_exit"] = "move_exit"
    exit_id: ExitId
    x: NonNegInt
    y: NonNegInt


class AddDoor(Part):
    type: Literal["add_door"] = "add_door"
    floor: NonNegInt
    a: RegionId
    b: RegionId
    x: NonNegInt
    y: NonNegInt
    width_cells: PosInt


class SwapRooms(Part):
    type: Literal["swap_rooms"] = "swap_rooms"
    room_a: RoomId
    room_b: RoomId


class MoveRoom(Part):
    type: Literal["move_room"] = "move_room"
    room_id: RoomId
    floor: NonNegInt
    x: NonNegInt
    y: NonNegInt


class ResizeRoom(Part):
    type: Literal["resize_room"] = "resize_room"
    room_id: RoomId
    w: PosInt
    d: PosInt


class ReassignFlexUse(Part):
    type: Literal["reassign_flex_use"] = "reassign_flex_use"
    room_id: RoomId
    to_type: RoomTypeId


MoveAction = Annotated[
    WidenCorridor
    | AddExit
    | MoveExit
    | AddDoor
    | SwapRooms
    | MoveRoom
    | ResizeRoom
    | ReassignFlexUse,
    Field(discriminator="type"),
]


class CostEstimate(Part):
    cost_units: NonNegFloat
    duration_days: NonNegFloat


class RepairMove(Contract):
    move_id: MoveId
    proposer: Proposer
    action: MoveAction
    finding_ids: tuple[FindingId, ...] = ()
    cost_estimate: CostEstimate | None = None
    rationale: str = ""
