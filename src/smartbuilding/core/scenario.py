"""Scenarios.

``ScenarioFamily``   layout-agnostic recipe (selectors such as 'random room of type lab').
``Scenario``         one seeded instance of a family; still layout-agnostic.
``ResolvedScenario`` a scenario bound to one NavGraph (concrete node ids). Built in Step 4.
"""

from typing import Annotated, Self

from pydantic import Field, model_validator

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    EventType,
    GroupId,
    Hash,
    HazardVariant,
    Name,
    NodeId,
    NonNegFloat,
    NonNegInt,
    PosFloat,
    PosInt,
    ScenarioId,
    SelectorMode,
    Split,
)
from smartbuilding.core.seeding import derive_seed

_REF_MODES = {SelectorMode.explicit_id, SelectorMode.room_type}


class Selector(Part):
    """Layout-agnostic choice of a room, exit, lift or corridor, resolved later per layout.

    ``random_room_near_exit`` picks, with the scenario seed, one room among the
    ``nearest_fraction`` of rooms closest to an exit by graph distance (ties broken by id).
    The fraction is relative so the meaning does not depend on building size.
    """

    mode: SelectorMode
    ref: str | None = None
    nearest_fraction: Annotated[float, Field(gt=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def _ref(self) -> Self:
        if self.mode in _REF_MODES and not self.ref:
            raise ValueError(f"selector mode {self.mode.value!r} needs a 'ref'")
        if self.mode not in _REF_MODES and self.ref is not None:
            raise ValueError(f"selector mode {self.mode.value!r} must not have a 'ref'")
        near = self.mode == SelectorMode.random_room_near_exit
        if near and self.nearest_fraction is None:
            raise ValueError("selector mode 'random_room_near_exit' needs 'nearest_fraction'")
        if not near and self.nearest_fraction is not None:
            raise ValueError("'nearest_fraction' is only allowed for 'random_room_near_exit'")
        return self


class IgnitionSpec(Part):
    where: Selector
    time_s: NonNegFloat = 0.0


class EventSpec(Part):
    type: EventType
    time_s: NonNegFloat
    target: Selector


ALL_TEMPLATES = "all"  # reserved: one group per template of the building's occupant profile


class GroupSpec(Part):
    group_id: GroupId
    template_id: Name  # a template id from the occupant profile, or the reserved name 'all'
    placement: Selector
    size_override: PosInt | None = None

    @model_validator(mode="after")
    def _all_has_no_override(self) -> Self:
        if self.template_id == ALL_TEMPLATES and self.size_override is not None:
            raise ValueError("size_override is not allowed with template 'all'")
        return self


class ScenarioFamily(Part):
    family_id: Name
    description: str = ""
    hazard_variant: HazardVariant = HazardVariant.A
    ignition: IgnitionSpec
    events: tuple[EventSpec, ...] = ()
    groups: tuple[GroupSpec, ...] = Field(min_length=1)
    sensor_model_ref: Name
    occupancy_period: Name | None = None
    horizon_s: PosFloat = 900.0
    n_repair: NonNegInt = 3
    n_test: NonNegInt = 30

    @model_validator(mode="after")
    def _check(self) -> Self:
        times = [e.time_s for e in self.events]
        if times != sorted(times):
            raise ValueError("events must be sorted by time_s")
        ids = [g.group_id for g in self.groups]
        if len(set(ids)) != len(ids):
            raise ValueError("group ids must be unique within a family")
        return self


class ScenarioFamilies(Contract):
    families_id: Name
    families: tuple[ScenarioFamily, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [f.family_id for f in self.families]
        if len(set(ids)) != len(ids):
            raise ValueError("family ids must be unique")
        return self


class Scenario(Contract):
    """One seeded scenario instance. Times in seconds."""

    scenario_id: ScenarioId
    family_id: Name
    split: Split
    hazard_variant: HazardVariant
    ignition: IgnitionSpec
    events: tuple[EventSpec, ...] = ()
    groups: tuple[GroupSpec, ...] = Field(min_length=1)
    sensor_model_ref: Name
    occupancy_period: Name | None = None
    seed: NonNegInt
    horizon_s: PosFloat


class ScenarioSet(Contract):
    set_id: Name
    scenarios: tuple[Scenario, ...] = Field(min_length=1)

    def by_split(self, split: Split) -> tuple[Scenario, ...]:
        return tuple(s for s in self.scenarios if s.split == split)

    @model_validator(mode="after")
    def _no_leakage(self) -> Self:
        ids = [s.scenario_id for s in self.scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError("scenario ids must be unique")
        seeds = [s.seed for s in self.scenarios]
        if len(set(seeds)) != len(seeds):
            raise ValueError("scenario seeds must be unique (no leakage between splits)")
        return self


def expand_families(families: ScenarioFamilies, base_seed: int) -> ScenarioSet:
    """Deterministically expand families into seeded scenarios for both splits."""
    out: list[Scenario] = []
    for fam in families.families:
        for split, n in ((Split.repair, fam.n_repair), (Split.test, fam.n_test)):
            for k in range(n):
                out.append(
                    Scenario(
                        scenario_id=f"SC_{fam.family_id}_{split.value}_{k:03d}",
                        family_id=fam.family_id,
                        split=split,
                        hazard_variant=fam.hazard_variant,
                        ignition=fam.ignition,
                        events=fam.events,
                        groups=fam.groups,
                        sensor_model_ref=fam.sensor_model_ref,
                        occupancy_period=fam.occupancy_period,
                        seed=derive_seed(base_seed, fam.family_id, split.value, k),
                        horizon_s=fam.horizon_s,
                    )
                )
    return ScenarioSet(set_id=f"{families.families_id}_expanded", scenarios=tuple(out))


class ResolvedEvent(Part):
    type: EventType
    time_s: NonNegFloat
    target_id: str = Field(min_length=1)


class ResolvedGroup(Part):
    group_id: GroupId
    template_id: Name
    size: PosInt
    start_node: NodeId


class ResolvedScenario(Contract):
    scenario_id: ScenarioId
    graph_hash: Hash
    hazard_variant: HazardVariant
    ignition_node: NodeId
    ignition_time_s: NonNegFloat
    events: tuple[ResolvedEvent, ...] = ()
    groups: tuple[ResolvedGroup, ...] = Field(min_length=1)
    sensor_model_ref: Name
    seed: NonNegInt
    horizon_s: PosFloat
