"""Engineering inputs I3-I8: catalogue, thresholds, hazard, sensors, profiles, run config.

Every number that is not a pure design variable carries a ``Sourcing`` record. Values that
are team assumptions are allowed but are listed by ``assumption_entries()`` so they stay
visible in reports.
"""

from typing import Self

from pydantic import Field, model_validator

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    Algorithm,
    Baseline,
    Basis,
    BuildingType,
    HazardVariant,
    LLMBackend,
    Name,
    NonNegFloat,
    NonNegInt,
    ObjectiveMode,
    PosFloat,
    PosInt,
    Prob,
    RelationType,
    RelPath,
    RoomTypeId,
    SensorPlacement,
    SpreadKernel,
)


class Sourcing(Part):
    basis: Basis
    source: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _source_required(self) -> Self:
        if self.basis != Basis.assumption and not self.source:
            raise ValueError("a 'source' citation is required unless basis is 'assumption'")
        return self


# --------------------------------------------------------------------------- catalogue


class RoomTypeDef(Part):
    type_id: RoomTypeId
    label: str = Field(min_length=1)
    min_area_m2: PosFloat
    max_area_m2: PosFloat
    needs_window: bool = False
    wet: bool = False
    occupant_load: NonNegInt = 0
    functional: bool = True
    flexible_to: tuple[RoomTypeId, ...] = ()
    sourcing: Sourcing

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min_area_m2 > self.max_area_m2:
            raise ValueError("min_area_m2 must not exceed max_area_m2")
        return self


class DefaultRelation(Part):
    a: RoomTypeId
    b: RoomTypeId
    type: RelationType
    hard: bool = False
    weight: PosFloat = 1.0
    sourcing: Sourcing


class RoomCatalogue(Contract):
    catalogue_id: Name
    room_types: tuple[RoomTypeDef, ...] = Field(min_length=1)
    default_relations: tuple[DefaultRelation, ...] = ()

    def get(self, type_id: str) -> RoomTypeDef | None:
        return next((t for t in self.room_types if t.type_id == type_id), None)

    def assumption_entries(self) -> tuple[str, ...]:
        out = [
            f"catalogue.{t.type_id}"
            for t in self.room_types
            if t.sourcing.basis == Basis.assumption
        ]
        out += [
            f"catalogue.relation.{r.a}-{r.b}"
            for r in self.default_relations
            if r.sourcing.basis == Basis.assumption
        ]
        return tuple(out)

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [t.type_id for t in self.room_types]
        if len(set(ids)) != len(ids):
            raise ValueError("room type ids must be unique")
        known = set(ids)
        for t in self.room_types:
            for target in t.flexible_to:
                if target not in known:
                    raise ValueError(f"{t.type_id}.flexible_to refers to unknown type {target!r}")
        for r in self.default_relations:
            if r.a not in known or r.b not in known:
                raise ValueError(f"default relation {r.a}-{r.b} refers to an unknown type")
        return self


# --------------------------------------------------------------------------- thresholds


class Threshold(Part):
    value: PosFloat
    unit: str = Field(min_length=1)
    sourcing: Sourcing


class CountThreshold(Part):
    value: PosInt
    unit: str = "count"
    sourcing: Sourcing


class Thresholds(Contract):
    """Numeric limits used by the rule checker. Never hard-coded elsewhere."""

    thresholds_id: Name
    min_corridor_width_m: Threshold
    min_door_width_m: Threshold
    min_exit_width_m: Threshold
    max_travel_distance_m: Threshold
    incompatible_min_separation_m: Threshold
    specific_flow_pers_per_m_s: Threshold
    near_reference_distance_m: Threshold  # 'near' means within this walking distance by default
    far_reference_distance_m: Threshold  # 'far' means at least this walking distance by default
    min_exit_count: CountThreshold

    def assumption_entries(self) -> tuple[str, ...]:
        return tuple(
            f"thresholds.{name}"
            for name in type(self).model_fields
            if isinstance(getattr(self, name), Threshold | CountThreshold)
            and getattr(self, name).sourcing.basis == Basis.assumption
        )


# --------------------------------------------------------------------------- hazard


class HazardParams(Part):
    """Hazard dynamics.

    Every probability-like value below is expressed *per ``rate_interval_s``* (default 10 s), never per
    tick, so the physics does not change when ``SimConfig.tick_s`` changes. ``spread_prob_*`` is the
    chance that fire crosses an edge of that type within one interval; ``smoke_generation`` is the
    fraction of the gap to full smoke closed at a fire node per interval; ``smoke_diffusion`` is the
    fraction of a smoke-intensity difference exchanged along an open edge per interval and
    ``smoke_decay`` the fraction of smoke lost per interval at a node that is not burning.
    """

    spread_kernel: SpreadKernel = SpreadKernel.neighbour
    rate_interval_s: PosFloat = 10.0
    spread_prob_open_door: Prob
    spread_prob_closed_door: Prob
    spread_prob_open_space: Prob
    smoke_generation: NonNegFloat
    smoke_diffusion: Prob
    smoke_decay: Prob


class StateMapping(Part):
    """Maps continuous smoke intensity in [0, 1] to the discrete hazard state."""

    smoky_min: Prob
    fire_min: Prob

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.smoky_min < self.fire_min:
            raise ValueError("smoky_min must be smaller than fire_min")
        return self


class HazardModel(Contract):
    hazard_id: Name
    variant: HazardVariant
    description: str = ""
    params: HazardParams
    state_mapping: StateMapping
    sourcing: Sourcing

    def assumption_entries(self) -> tuple[str, ...]:
        return (f"hazard.{self.variant.value}",) if self.sourcing.basis == Basis.assumption else ()


class SensorModel(Contract):
    sensor_model_id: Name
    smoke_placement: SensorPlacement = SensorPlacement.rooms_and_corridors
    smoke_detection_prob: Prob
    smoke_false_alarm_prob: Prob
    # P(alarm | smoky) = smoke_detection_prob * smoky_detection_factor; P(alarm | fire) = smoke_detection_prob.
    # The belief filter uses the same emission model as the simulator.
    smoky_detection_factor: Prob = 0.7
    # Exit, lift, door and occupancy reports are issued once per this many seconds.
    report_period_s: PosFloat = 6.0
    occupancy_noise_sd: NonNegFloat
    occupancy_sensor_coverage: Prob
    exit_report_reliability: Prob
    lift_report_reliability: Prob
    device_report_reliability: Prob
    user_report_reliability: Prob
    user_report_rate_per_min: NonNegFloat
    sourcing: Sourcing

    def assumption_entries(self) -> tuple[str, ...]:
        return (
            (f"sensors.{self.sensor_model_id}",) if self.sourcing.basis == Basis.assumption else ()
        )


# --------------------------------------------------------------------------- occupants


class GroupTemplate(Part):
    template_id: Name
    label: str = Field(min_length=1)
    share: Prob
    speed_mps: PosFloat
    can_use_stairs: bool = True
    can_use_lift: bool = True
    needs_assistance: bool = False
    response_delay_s: NonNegFloat = 0.0


class OccupancyPeriod(Part):
    name: Name
    mean_fraction: Prob
    dispersion: NonNegFloat = 0.0


class OccupantProfile(Part):
    profile_id: Name
    building_type: BuildingType
    groups: tuple[GroupTemplate, ...] = Field(min_length=1)
    periods: tuple[OccupancyPeriod, ...] = Field(min_length=1)
    sourcing: Sourcing

    @model_validator(mode="after")
    def _check(self) -> Self:
        total = sum(g.share for g in self.groups)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"group shares must sum to 1, got {total}")
        ids = [g.template_id for g in self.groups]
        if len(set(ids)) != len(ids):
            raise ValueError("group template ids must be unique")
        names = [p.name for p in self.periods]
        if len(set(names)) != len(names):
            raise ValueError("occupancy period names must be unique")
        return self


class OccupantProfiles(Contract):
    profiles_id: Name
    profiles: tuple[OccupantProfile, ...] = Field(min_length=1)

    def get(self, profile_id: str) -> OccupantProfile | None:
        return next((p for p in self.profiles if p.profile_id == profile_id), None)

    def assumption_entries(self) -> tuple[str, ...]:
        return tuple(
            f"profiles.{p.profile_id}"
            for p in self.profiles
            if p.sourcing.basis == Basis.assumption
        )

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [p.profile_id for p in self.profiles]
        if len(set(ids)) != len(ids):
            raise ValueError("profile ids must be unique")
        return self


# --------------------------------------------------------------------------- run config


class LLMConfig(Part):
    backend: LLMBackend = LLMBackend.cached
    llm_name: str = "none"
    temperature: NonNegFloat = 0.0
    cache_dir: RelPath = "data/llm_cache"
    prompt_version: str = "v1"
    max_retries: int = Field(default=3, ge=0, le=10)


class DoseConfig(Part):
    """Dose proxy. NOT the physical FED; thresholds are on the proxy scale."""

    k: PosFloat = 1.0
    alert_min: PosFloat
    critical_min: PosFloat
    severe_min: PosFloat
    penalty_alert: NonNegFloat
    penalty_critical: NonNegFloat
    penalty_severe: NonNegFloat

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.alert_min < self.critical_min < self.severe_min:
            raise ValueError("dose level thresholds must be strictly increasing")
        if not self.penalty_alert <= self.penalty_critical <= self.penalty_severe:
            raise ValueError("dose penalties must be non-decreasing")
        return self


class SimConfig(Part):
    tick_s: PosFloat = 2.0
    decision_period_ticks: PosInt = 3
    # Smoke is integrated with sub-steps of at most this length so results barely depend on tick_s.
    max_substep_s: PosFloat = 0.5
    lifts_disabled_in_fire: bool = True


class BeliefParams(Part):
    """Assumptions of the Bayes filter itself (what the *decision maker* believes about the world).

    Kept separate from the simulator's ``HazardModel`` on purpose: the filter derives its spread rates
    from a hazard model it is *given* (variant A when tuned) and adds the small background rates below.
    Rates are per second.
    """

    ignition_prob_per_s: Prob = 1e-4  # a clear node starts burning with no burning neighbour
    spontaneous_smoke_prob_per_s: Prob = (
        1e-4  # a clear node becomes smoky with no hazardous neighbour
    )
    smoke_recovery_per_s: Prob = 0.01  # a smoky node clears when nothing feeds it
    escalation_per_s: Prob = 0.005  # a smoky node starts burning
    exit_fail_prob_per_s: Prob = 5e-4
    device_fail_prob_per_s: Prob = 5e-4
    occupancy_process_var_per_s: NonNegFloat = 0.05  # random-walk variance of room occupancy
    min_prob: Prob = 1e-6  # floor so no state is ever ruled out for good
    prior_available: Prob = 0.99  # prior P(working) for each exit, lift and door
    occupancy_prior_fraction: Prob = 0.7  # prior mean occupancy as a fraction of a room's load


class RewardWeights(Part):
    time: NonNegFloat = 0.01
    dose: NonNegFloat = 1.0
    congestion: NonNegFloat = 0.1
    evacuated: NonNegFloat = 0.5
    completion_bonus: NonNegFloat = 10.0


class ObjectiveWeights(Part):
    dose: NonNegFloat = 1.0
    evac_time: NonNegFloat = 0.5
    congestion: NonNegFloat = 0.5
    construction_cost: NonNegFloat = 0.2
    usable_area: NonNegFloat = 0.2


class ObjectiveConfig(Part):
    mode: ObjectiveMode = ObjectiveMode.pareto
    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)


class RLConfig(Part):
    algorithm: Algorithm = Algorithm.dqn
    reward: RewardWeights = Field(default_factory=RewardWeights)
    training_seeds: PosInt = 5
    episodes: PosInt = 2000


class Budgets(Part):
    solver_time_limit_s: PosFloat = 60.0
    max_layouts: PosInt = 5
    repair_rounds: NonNegInt = 3
    n_repair_scenarios: PosInt = 10
    n_test_scenarios: PosInt = 30


class ConfigPaths(Part):
    """All paths are relative to the directory of the run-config file."""

    catalogue: RelPath
    thresholds: RelPath
    profiles: RelPath
    hazard_a: RelPath
    hazard_b: RelPath
    sensors: RelPath
    scenario_families: RelPath


class RunConfig(Contract):
    run_id: Name
    base_seed: NonNegInt = 0
    paths: ConfigPaths
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    belief: BeliefParams = Field(default_factory=BeliefParams)
    dose: DoseConfig
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    budgets: Budgets = Field(default_factory=Budgets)
    baselines: tuple[Baseline, ...] = Field(default=tuple(Baseline), min_length=1)
