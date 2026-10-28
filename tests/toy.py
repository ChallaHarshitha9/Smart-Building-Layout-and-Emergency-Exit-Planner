"""Shared fixtures: the E0 toy corridor (west exit E_1, room 1 at x=6, room 2 at x=16, east exit E_2)."""

from __future__ import annotations

import numpy as np

from smartbuilding.core.config import BeliefParams, DoseConfig, HazardModel, SimConfig
from smartbuilding.core.primitives import HazardVariant
from smartbuilding.core.scenario import ResolvedEvent, ResolvedGroup, ResolvedScenario
from smartbuilding.eval.e0 import _toy_config, build_e0_toy_layout
from smartbuilding.layout.graph import build_navgraph
from smartbuilding.sim.engine import Simulator

DOSE = DoseConfig(
    k=1.0,
    alert_min=0.5,
    critical_min=3.0,
    severe_min=8.0,
    penalty_alert=100.0,
    penalty_critical=300.0,
    penalty_severe=800.0,
)
GRAPH = build_navgraph(build_e0_toy_layout())
HZ, SENS, PROFS = _toy_config()
BP = BeliefParams()


def node(region: str) -> str:
    return next(n.node_id for n in GRAPH.nodes if n.region_id == region)


def no_fire(hz: HazardModel) -> HazardModel:
    """Same smoke physics, fire never spreads (and ignites far in the future in the scenario)."""
    return hz.model_copy(
        update={
            "params": hz.params.model_copy(
                update={
                    "spread_prob_open_door": 0.0,
                    "spread_prob_closed_door": 0.0,
                    "spread_prob_open_space": 0.0,
                }
            )
        }
    )


def scenario(
    groups=None, events=(), ignite="R_office_1", ignition_time_s=0.0, seed=1, horizon_s=240.0
):
    groups = groups or (
        ResolvedGroup(group_id="G_a", template_id="early", size=1, start_node=node("R_office_2")),
    )
    return ResolvedScenario(
        scenario_id="SC_toy",
        graph_hash=GRAPH.derived_from,
        hazard_variant=HazardVariant.A,
        ignition_node=node(ignite),
        ignition_time_s=ignition_time_s,
        events=tuple(events),
        groups=tuple(groups),
        sensor_model_ref="sensor_e0",
        seed=seed,
        horizon_s=horizon_s,
    )


def sim(sc=None, hz=HZ, tick=2.0, flow=1.3, seed=1, **kw) -> Simulator:
    return Simulator(
        GRAPH,
        sc or scenario(),
        hz,
        SENS,
        PROFS,
        SimConfig(tick_s=tick),
        DOSE,
        np.random.default_rng(seed),
        profile_id="prof_e0",
        specific_flow_pers_per_m_s=flow,
        **kw,
    )


def blocked_event(exit_id="E_1", t=0.0) -> ResolvedEvent:
    from smartbuilding.core.primitives import EventType

    return ResolvedEvent(type=EventType.exit_blocked, time_s=t, target_id=node(exit_id))
