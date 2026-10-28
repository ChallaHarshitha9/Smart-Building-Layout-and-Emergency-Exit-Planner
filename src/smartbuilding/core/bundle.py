"""Loading the whole set of engineering inputs and checking them against each other.

Per-file structure is validated by the contracts. This module checks *references between
files* (for example a scenario naming a sensor model that does not exist).
"""

from dataclasses import dataclass
from pathlib import Path

from smartbuilding.core.config import (
    HazardModel,
    OccupantProfile,
    OccupantProfiles,
    RoomCatalogue,
    RunConfig,
    SensorModel,
    Thresholds,
)
from smartbuilding.core.loaders import load_model
from smartbuilding.core.primitives import HazardVariant
from smartbuilding.core.scenario import (
    ALL_TEMPLATES,
    ScenarioFamilies,
    ScenarioSet,
    expand_families,
)
from smartbuilding.core.spec import BuildingSpec


class CrossCheckError(ValueError):
    """Raised when files are individually valid but inconsistent with each other."""


@dataclass(frozen=True)
class ConfigBundle:
    root: Path
    run: RunConfig
    catalogue: RoomCatalogue
    thresholds: Thresholds
    profiles: OccupantProfiles
    hazard_a: HazardModel
    hazard_b: HazardModel
    sensors: SensorModel
    families: ScenarioFamilies

    def scenarios(self) -> ScenarioSet:
        return expand_families(self.families, self.run.base_seed)

    def assumptions(self) -> tuple[str, ...]:
        """Every input value that is a team assumption rather than a cited source."""
        return (
            *self.catalogue.assumption_entries(),
            *self.thresholds.assumption_entries(),
            *self.profiles.assumption_entries(),
            *self.hazard_a.assumption_entries(),
            *self.hazard_b.assumption_entries(),
            *self.sensors.assumption_entries(),
        )


def load_bundle(run_config_path: str | Path) -> ConfigBundle:
    """Load a run config and every file it references. Paths are relative to its folder."""
    run_path = Path(run_config_path)
    root = run_path.parent
    run = load_model(run_path, RunConfig)
    p = run.paths
    return ConfigBundle(
        root=root,
        run=run,
        catalogue=load_model(root / p.catalogue, RoomCatalogue),
        thresholds=load_model(root / p.thresholds, Thresholds),
        profiles=load_model(root / p.profiles, OccupantProfiles),
        hazard_a=load_model(root / p.hazard_a, HazardModel),
        hazard_b=load_model(root / p.hazard_b, HazardModel),
        sensors=load_model(root / p.sensors, SensorModel),
        families=load_model(root / p.scenario_families, ScenarioFamilies),
    )


def check_bundle(bundle: ConfigBundle) -> list[str]:
    problems: list[str] = []
    if bundle.hazard_a.variant != HazardVariant.A:
        problems.append("hazard_a file must have variant 'A'")
    if bundle.hazard_b.variant != HazardVariant.B:
        problems.append("hazard_b file must have variant 'B'")
    period_names = {per.name for prof in bundle.profiles.profiles for per in prof.periods}
    for fam in bundle.families.families:
        if fam.sensor_model_ref != bundle.sensors.sensor_model_id:
            problems.append(
                f"family {fam.family_id}: unknown sensor model {fam.sensor_model_ref!r}"
            )
        if fam.occupancy_period is not None and fam.occupancy_period not in period_names:
            problems.append(
                f"family {fam.family_id}: unknown occupancy period {fam.occupancy_period!r}"
            )
    return problems


def check_spec_against_catalogue(spec: BuildingSpec, catalogue: RoomCatalogue) -> list[str]:
    problems: list[str] = []
    for req in spec.rooms:
        if catalogue.get(req.room_type) is None:
            problems.append(
                f"room type {req.room_type!r} is not in catalogue {catalogue.catalogue_id!r}"
            )
        for target in req.flexible_to:
            if catalogue.get(target) is None:
                problems.append(f"{req.room_type}.flexible_to: unknown type {target!r}")
    return problems


def check_spec_against_profiles(spec: BuildingSpec, profiles: OccupantProfiles) -> list[str]:
    profile = profiles.get(spec.occupancy.profile_ref)
    if profile is None:
        return [f"occupancy profile {spec.occupancy.profile_ref!r} does not exist"]
    if profile.building_type != spec.building_type:
        return [
            f"profile {profile.profile_id!r} is for {profile.building_type.value}, "
            f"spec is {spec.building_type.value}"
        ]
    return []


def check_scenarios_against_profile(
    families: ScenarioFamilies, profile: OccupantProfile
) -> list[str]:
    known = {g.template_id for g in profile.groups}
    problems: list[str] = []
    for fam in families.families:
        for grp in fam.groups:
            if grp.template_id != ALL_TEMPLATES and grp.template_id not in known:
                problems.append(
                    f"family {fam.family_id}, group {grp.group_id}: template "
                    f"{grp.template_id!r} not in profile {profile.profile_id!r}"
                )
    return problems


def raise_if_problems(problems: list[str]) -> None:
    if problems:
        raise CrossCheckError("\n".join(f"- {p}" for p in problems))
