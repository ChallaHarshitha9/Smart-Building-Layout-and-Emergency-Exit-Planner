"""Enums, identifier patterns and constrained numeric types shared by all contracts."""

import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints

# --------------------------------------------------------------------------- enums


class BuildingType(StrEnum):
    office = "office"
    school = "school"
    hostel = "hostel"
    hospital = "hospital"


class RelationType(StrEnum):
    adjacent = "adjacent"
    near = "near"
    far = "far"
    incompatible = "incompatible"


class Side(StrEnum):
    north = "north"
    south = "south"
    east = "east"
    west = "west"


class NodeKind(StrEnum):
    room = "room"
    door = "door"
    corridor = "corridor"
    stair = "stair"
    lift = "lift"
    exit = "exit"


class HazardState(StrEnum):
    clear = "clear"
    smoky = "smoky"
    fire = "fire"


HAZARD_ORDER: tuple[HazardState, ...] = (HazardState.clear, HazardState.smoky, HazardState.fire)


class Channel(StrEnum):
    smoke = "smoke"
    occupancy = "occupancy"
    exit = "exit"
    lift = "lift"
    device = "device"
    report = "report"


class Split(StrEnum):
    repair = "repair"
    test = "test"


class Proposer(StrEnum):
    rule = "rule"
    llm = "llm"
    search = "search"
    rl = "rl"


class MoveType(StrEnum):
    widen_corridor = "widen_corridor"
    add_exit = "add_exit"
    move_exit = "move_exit"
    add_door = "add_door"
    swap_rooms = "swap_rooms"
    move_room = "move_room"
    resize_room = "resize_room"
    reassign_flex_use = "reassign_flex_use"


class HazardVariant(StrEnum):
    A = "A"
    B = "B"


class SpreadKernel(StrEnum):
    neighbour = "neighbour"
    two_hop = "two_hop"


class DoorState(StrEnum):
    open = "open"
    closed = "closed"
    stuck = "stuck"


class EventType(StrEnum):
    exit_blocked = "exit_blocked"
    lift_out = "lift_out"
    second_fire = "second_fire"
    detector_fail = "detector_fail"
    door_stuck = "door_stuck"


class Basis(StrEnum):
    """Where a number comes from. ``assumption`` is allowed but is always reported."""

    code = "code"
    reference = "reference"
    assumption = "assumption"


class LLMBackend(StrEnum):
    cached = "cached"
    api = "api"
    local = "local"


class Baseline(StrEnum):
    B1 = "B1"  # nearest-exit Dijkstra (static)
    B2 = "B2"  # SafeStep-style threshold Dijkstra
    B3 = "B3"  # dose-aware planner on ground truth (oracle)
    B4 = "B4"  # dose-aware planner on raw observations
    B5 = "B5"  # dose-aware planner on belief
    B6 = "B6"  # RL on raw observations
    B7 = "B7"  # RL on belief


class ObjectiveMode(StrEnum):
    pareto = "pareto"
    weighted = "weighted"


class Algorithm(StrEnum):
    dqn = "dqn"
    ppo = "ppo"


class FindingKind(StrEnum):
    congested_edge = "congested_edge"
    exit_overloaded = "exit_overloaded"
    high_dose_room = "high_dose_room"
    travel_distance_violation = "travel_distance_violation"
    accessibility_gap = "accessibility_gap"


class ConstraintKind(StrEnum):
    hard = "hard"
    soft = "soft"


class VerificationStatus(StrEnum):
    unchecked = "unchecked"
    checked = "checked"


class SelectorMode(StrEnum):
    explicit_id = "explicit_id"
    room_type = "room_type"
    random_room = "random_room"
    random_room_near_exit = "random_room_near_exit"
    random_exit = "random_exit"
    random_lift = "random_lift"
    random_corridor = "random_corridor"
    nearest_exit_to_ignition = "nearest_exit_to_ignition"
    distributed_by_load = "distributed_by_load"


class MetricScope(StrEnum):
    run = "run"
    aggregate = "aggregate"
    layout = "layout"


class ConstructionTaskType(StrEnum):
    structure = "structure"
    prepare_space = "prepare_space"
    build_walls = "build_walls"
    install_door = "install_door"
    place_utilities = "place_utilities"
    install_safety_equipment = "install_safety_equipment"
    build_corridor = "build_corridor"
    build_exit = "build_exit"


class DecisionSource(StrEnum):
    planner = "planner"
    rl = "rl"
    baseline = "baseline"


class DeviceKind(StrEnum):
    """Kinds of devices whose working/failed status the belief tracks."""

    lift = "lift"
    door = "door"
    detector = "detector"
    equipment = "equipment"  # safety equipment such as extinguishers and emergency lighting
    alarm = "alarm"


class CaseExpectation(StrEnum):
    passes = "passes"  # every configured hard check passes
    fails = "fails"  # the listed checks (and only those) do not pass


class SensorPlacement(StrEnum):
    all_rooms = "all_rooms"
    corridors_only = "corridors_only"
    rooms_and_corridors = "rooms_and_corridors"


# --------------------------------------------------------------------------- ids


def _sc(pattern: str) -> StringConstraints:
    return StringConstraints(pattern=pattern, min_length=1, max_length=96)


_TYPE = r"[a-z][a-z0-9_]*"
_ROOM = rf"R_{_TYPE}_[0-9]+"

RoomTypeId = Annotated[str, _sc(rf"^{_TYPE}$")]
RoomId = Annotated[str, _sc(rf"^{_ROOM}$")]
RoomRef = Annotated[str, _sc(rf"^({_ROOM}|{_TYPE})$")]  # an instance id or a whole room type
NodeId = Annotated[str, _sc(r"^N_[a-z]+_[0-9]+$")]
ExitId = Annotated[str, _sc(r"^E_[0-9]+$")]
DoorId = Annotated[str, _sc(r"^D_[0-9]+$")]
CorridorId = Annotated[str, _sc(r"^C_[0-9]+$")]
StairId = Annotated[str, _sc(r"^S_[0-9]+$")]
LiftId = Annotated[str, _sc(r"^L_[0-9]+$")]
RegionId = Annotated[str, _sc(rf"^({_ROOM}|C_[0-9]+|E_[0-9]+|S_[0-9]+|L_[0-9]+)$")]
DeviceId = Annotated[str, _sc(r"^DV_(lift|door|detector|equipment|alarm)_[0-9]+$")]
GroupId = Annotated[str, _sc(r"^G_[a-z0-9_]+$")]
ScenarioId = Annotated[str, _sc(r"^SC_[A-Za-z0-9_\-]+$")]
LayoutId = Annotated[str, _sc(r"^LY_[A-Za-z0-9_\-]+$")]
MoveId = Annotated[str, _sc(r"^MV_[0-9]+$")]
FindingId = Annotated[str, _sc(r"^F_[0-9]+$")]
TaskId = Annotated[str, _sc(r"^T_[a-z0-9_]+$")]
MetricId = Annotated[str, _sc(r"^[a-z][a-z0-9_]*$")]
FactId = Annotated[str, _sc(r"^FA_[0-9]+$")]
Hash = Annotated[str, _sc(r"^[0-9a-f]{64}$")]
Name = Annotated[
    str, _sc(r"^[A-Za-z][A-Za-z0-9_\-]*$")
]  # generic ids for catalogues, profiles, sets

COORDINATE_CONVENTION = (
    "Plan view: x grows east, y grows north, origin at the south-west corner of the footprint; "
    "floor is a non-negative integer index (0 = ground). Lengths are in metres, areas in square "
    "metres, times in seconds. Layout geometry is stored as integer cells of cell_size_m metres."
)

ROOM_INSTANCE_RE = re.compile(rf"^R_({_TYPE})_([0-9]+)$")

# --------------------------------------------------------------------------- numbers

Prob = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PosFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
PosInt = Annotated[int, Field(ge=1)]
NonNegInt = Annotated[int, Field(ge=0)]


def _check_relative(path: str) -> str:
    normalised = path.replace("\\", "/")
    if normalised.startswith("/") or re.match(r"^[A-Za-z]:", normalised):
        raise ValueError("path must be relative, not absolute")
    if ".." in PurePosixPath(normalised).parts:
        raise ValueError("path must not leave the project ('..' not allowed)")
    return path


RelPath = Annotated[str, StringConstraints(min_length=1), AfterValidator(_check_relative)]
