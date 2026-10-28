"""Decisions produced by planners and RL policies. Reads belief only, never ground truth."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    Algorithm,
    Baseline,
    DecisionSource,
    ExitId,
    GroupId,
    Hash,
    Name,
    NodeId,
    NonNegFloat,
    NonNegInt,
    RelPath,
)
from smartbuilding.core.runtime import Observation, PublicState


class Route(Part):
    nodes: tuple[NodeId, ...] = Field(min_length=1)
    est_time_s: NonNegFloat = 0.0
    est_dose: NonNegFloat = 0.0


class Decision(Part):
    group_id: GroupId
    t_s: NonNegFloat
    target_exit: ExitId
    route: Route
    alternatives: tuple[Route, ...] = ()  # ranked, best first
    source: DecisionSource = DecisionSource.planner


class Plan(Contract):
    plan_id: Name
    graph_hash: Hash
    method: Baseline
    decisions: tuple[Decision, ...] = ()


class Policy(Contract):
    policy_id: Name
    algorithm: Algorithm
    checkpoint_path: RelPath
    obs_spec_version: str = Field(min_length=1)
    layout_geometry_hash: Hash
    training_seeds: tuple[NonNegInt, ...] = Field(min_length=1)
    uses_belief: bool


class Controller(Protocol):
    """Anything that steers evacuees: baselines, planners, RL policies.

    The simulator calls ``observe`` after every tick with that tick's observations and ``decide`` on
    every decision tick. Neither receives ground truth.
    """

    def observe(self, t_s: float, observations: Sequence[Observation]) -> None: ...

    def decide(self, public: PublicState) -> Plan | None: ...
