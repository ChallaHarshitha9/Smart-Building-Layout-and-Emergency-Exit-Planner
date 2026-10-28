"""What decision modules may know: observations and belief.

Ground truth is NOT defined here. It lives in ``smartbuilding.sim.truth`` so that no module
that only imports from ``core`` can reach it. ``tests/test_schemas_cli_barrier.py`` enforces
that planners, RL, belief and GenAI code never import ``smartbuilding.sim``.
"""

from typing import Self

from pydantic import model_validator

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    Channel,
    DeviceId,
    ExitId,
    Finite,
    GroupId,
    Hash,
    HazardState,
    NodeId,
    NonNegFloat,
    Prob,
    RoomId,
)


class Observation(Part):
    t_s: NonNegFloat
    channel: Channel
    source_id: str
    target_id: str | None = None
    value: Finite
    reliability: Prob


class PublicGroup(Part):
    """What a decision maker may know about one group: where it is and what it can do.

    ``node`` is the node the group is at, or (when ``in_transit``) the node it is walking towards;
    ``came_from`` is the node it left, so a planner may turn it around.
    """

    group_id: GroupId
    template_id: str
    size: int
    node: NodeId
    in_transit: bool = False
    came_from: NodeId | None = None
    waiting: bool = False
    evacuated: bool = False
    speed_mps: float
    can_use_stairs: bool = True
    can_use_lift: bool = True


class PublicState(Part):
    """The only simulator state that reaches planners, belief filters and RL: no hazard, no truth."""

    t_s: float
    tick: int
    groups: tuple[PublicGroup, ...]

    def active(self) -> tuple[PublicGroup, ...]:
        return tuple(g for g in self.groups if not g.evacuated)


class OccupancyBelief(Part):
    mean: NonNegFloat
    var: NonNegFloat


class BeliefState(Contract):
    graph_hash: Hash
    t_s: NonNegFloat
    node_belief: dict[NodeId, tuple[Prob, Prob, Prob]]  # (clear, smoky, fire)
    exit_avail: dict[ExitId, Prob] = {}
    # P(device is working) for lifts, doors, detectors, safety equipment and alarms.
    devices: dict[DeviceId, Prob] = {}
    occupancy: dict[RoomId, OccupancyBelief] = {}

    @model_validator(mode="after")
    def _normalised(self) -> Self:
        for node, triple in self.node_belief.items():
            if abs(sum(triple) - 1.0) > 1e-6:
                raise ValueError(f"belief for {node} must sum to 1, got {sum(triple)}")
        return self

    def p_hazard(self, node_id: str, state: HazardState) -> float:
        order = (HazardState.clear, HazardState.smoky, HazardState.fire)
        return self.node_belief[node_id][order.index(state)]
