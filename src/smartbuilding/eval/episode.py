"""Run one evacuation episode end to end and summarise it. Used by experiments and the UI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from smartbuilding.core.bundle import ConfigBundle
from smartbuilding.core.config import HazardModel
from smartbuilding.core.decision import Baseline, Controller
from smartbuilding.core.layout import Layout
from smartbuilding.core.navgraph import NavGraph
from smartbuilding.core.primitives import HazardVariant
from smartbuilding.core.runtime import Observation
from smartbuilding.core.scenario import ResolvedScenario, Scenario
from smartbuilding.core.spec import BuildingSpec
from smartbuilding.eval.oracle import OracleTracker
from smartbuilding.layout.graph import build_navgraph
from smartbuilding.planning.controllers import DoseAwareController, make_controller
from smartbuilding.planning.dose_aware import DoseAwareParams
from smartbuilding.sim.engine import SimResult, SimState, Simulator
from smartbuilding.sim.resolve import resolve_scenario
from smartbuilding.uncertainty.belief import BeliefFilter


def room_loads(layout: Layout, spec: BuildingSpec) -> dict[str, int]:
    out: dict[str, int] = {}
    for plan in layout.floors:
        for r in plan.rooms:
            req = spec.request_for(r.room_type)
            out[r.room_id] = req.occupant_load if req else 0
    return out


def navgraph_for(layout: Layout, spec: BuildingSpec) -> NavGraph:
    return build_navgraph(layout, room_loads(layout, spec))


def resolve_for(
    bundle: ConfigBundle, scenario: Scenario, graph: NavGraph, spec: BuildingSpec
) -> ResolvedScenario:
    profile = bundle.profiles.get(spec.occupancy.profile_ref)
    if profile is None:
        raise ValueError(f"unknown occupant profile {spec.occupancy.profile_ref!r}")
    return resolve_scenario(scenario, graph, profile, spec.occupancy.total)


def hazard_for(bundle: ConfigBundle, variant: HazardVariant) -> HazardModel:
    return bundle.hazard_a if variant == HazardVariant.A else bundle.hazard_b


def make_simulator(
    bundle: ConfigBundle, graph: NavGraph, resolved: ResolvedScenario, spec: BuildingSpec
) -> Simulator:
    return Simulator(
        graph,
        resolved,
        hazard_for(bundle, resolved.hazard_variant),
        bundle.sensors,
        bundle.profiles,
        bundle.run.sim,
        bundle.run.dose,
        np.random.default_rng(resolved.seed),
        profile_id=spec.occupancy.profile_ref,
        specific_flow_pers_per_m_s=bundle.thresholds.specific_flow_pers_per_m_s.value,
    )


def make_run(
    bundle: ConfigBundle,
    graph: NavGraph,
    resolved: ResolvedScenario,
    spec: BuildingSpec,
    method: Baseline,
    params: DoseAwareParams | None = None,
) -> tuple[Simulator, Controller]:
    """A simulator and the controller for ``method``, ready for ``sim.run(controller)``.

    The belief filter always uses hazard variant A (the tuned model); the simulator uses the
    scenario's variant, so on variant B belief and truth genuinely differ (RQ6).
    """
    sim = make_simulator(bundle, graph, resolved, spec)
    flow = bundle.thresholds.specific_flow_pers_per_m_s.value
    tick = bundle.run.sim.tick_s
    ctrl: Controller
    if method == Baseline.B3:
        filt = BeliefFilter(graph, bundle.hazard_a, bundle.sensors, tick, bundle.run.belief)
        ctrl = DoseAwareController(
            graph,
            OracleTracker(sim),
            Baseline.B3,
            bundle.hazard_a,
            bundle.run.dose,
            flow,
            params,
            predict_model=filt.model,
        )
    else:
        ctrl = make_controller(
            method,
            graph,
            bundle.hazard_a,
            bundle.sensors,
            bundle.run.dose,
            tick,
            flow,
            bundle.run.belief,
            params,
        )
    return sim, ctrl


def run_episode(
    bundle: ConfigBundle,
    graph: NavGraph,
    resolved: ResolvedScenario,
    spec: BuildingSpec,
    method: Baseline,
    params: DoseAwareParams | None = None,
    on_tick: Callable[[SimState, Sequence[Observation]], None] | None = None,
) -> tuple[SimResult, Simulator]:
    """Simulate ``method`` on one resolved scenario."""
    sim, ctrl = make_run(bundle, graph, resolved, spec, method, params)
    return sim.run(ctrl, on_tick), sim


@dataclass(frozen=True)
class EpisodeSummary:
    people: int
    evacuated: int
    completion: float
    mean_evac_s: float | None  # person-weighted, over those who got out
    max_evac_s: float | None
    mean_dose: float  # person-weighted dose proxy (NOT FED)
    max_dose: float
    dose_levels: dict[str, int]  # persons per level: none / alert / critical / severe
    peak_queue_persons: int
    peak_queue_delay_s: float
    person_wait_s: float
    invalid_decisions: int


def summarise(result: SimResult, bundle: ConfigBundle) -> EpisodeSummary:
    d = bundle.run.dose
    people = sum(result.sizes.values())
    done = [g for g, t in result.evac_times_s.items() if t is not None]
    ev_people = sum(result.sizes[g] for g in done)
    levels = {"none": 0, "alert": 0, "critical": 0, "severe": 0}
    for g, dose in result.doses.items():
        lv = (
            "severe"
            if dose >= d.severe_min
            else "critical"
            if dose >= d.critical_min
            else "alert"
            if dose >= d.alert_min
            else "none"
        )
        levels[lv] += result.sizes[g]
    return EpisodeSummary(
        people=people,
        evacuated=ev_people,
        completion=result.completion_rate,
        mean_evac_s=(
            sum(result.sizes[g] * float(result.evac_times_s[g] or 0.0) for g in done) / ev_people
            if ev_people
            else None
        ),
        max_evac_s=max((float(result.evac_times_s[g] or 0.0) for g in done), default=None),
        mean_dose=sum(result.sizes[g] * x for g, x in result.doses.items()) / people
        if people
        else 0.0,
        max_dose=max(result.doses.values(), default=0.0),
        dose_levels=levels,
        peak_queue_persons=result.peak_queue_persons,
        peak_queue_delay_s=result.peak_queue_delay_s,
        person_wait_s=result.person_wait_s,
        invalid_decisions=result.invalid_decisions,
    )
