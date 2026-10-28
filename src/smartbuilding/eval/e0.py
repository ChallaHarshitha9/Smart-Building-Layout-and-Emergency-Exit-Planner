"""E0 sanity benchmark: the SafeStep toy corridor (40 m, two rooms, two exits, fire in one room).

A group in the centre room must leave. The fire is in the room next to the west exit, so the
nearest exit (west) is the hazardous one. Compared at an early and a late response time:
B1 nearest-exit Dijkstra, B2 threshold Dijkstra, B5 dose-aware planner (all on the belief filter).

This is a sanity check of the simulator and planners, not a result: one toy scenario over a
few seeds. `check_passed` only states whether the expected qualitative pattern appeared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartbuilding.core.config import (
    BeliefParams,
    DoseConfig,
    GroupTemplate,
    HazardModel,
    HazardParams,
    OccupancyPeriod,
    OccupantProfile,
    OccupantProfiles,
    SensorModel,
    SimConfig,
    Sourcing,
    StateMapping,
)
from smartbuilding.core.decision import Baseline
from smartbuilding.core.layout import (
    CorridorPlacement,
    Door,
    Exit,
    FloorPlan,
    Layout,
    LayoutProvenance,
    Rect,
    RoomPlacement,
    VerticalCore,
)
from smartbuilding.core.navgraph import NavGraph
from smartbuilding.core.primitives import Basis, BuildingType, HazardVariant, SensorPlacement
from smartbuilding.core.scenario import ResolvedGroup, ResolvedScenario
from smartbuilding.layout.graph import build_navgraph
from smartbuilding.planning.controllers import make_controller
from smartbuilding.sim.engine import SimState, Simulator


def build_e0_toy_layout() -> Layout:
    """Build a 40m corridor with Room 1 near West exit and Room 2 in center."""
    # 0.5 m cells: 40 m = 80 cells wide, 12 m = 24 cells deep.
    # Corridor C_1: y in [8, 12) (width 80, depth 4 -> 2m wide)
    # Room 1: South side, near west exit: x in [4, 20), y in [0, 8)
    # Room 2: North side, center: x in [24, 40), y in [12, 20)
    # Exit 1: West end of corridor (x=0, y=9)
    # Exit 2: East end of corridor (x=80, y=9)
    plan = FloorPlan(
        floor=0,
        rooms=(
            RoomPlacement(room_id="R_office_1", room_type="office", rect=Rect(x=4, y=0, w=16, d=8)),
            RoomPlacement(
                room_id="R_office_2", room_type="office", rect=Rect(x=24, y=12, w=16, d=8)
            ),
        ),
        corridors=(CorridorPlacement(corridor_id="C_1", rect=Rect(x=0, y=8, w=80, d=4)),),
        doors=(
            Door(door_id="D_1", a="R_office_1", b="C_1", x=12, y=8, width_cells=2),
            Door(door_id="D_2", a="R_office_2", b="C_1", x=32, y=12, width_cells=2),
        ),
        exits=(
            Exit(exit_id="E_1", x=0, y=9, width_cells=3, connects_to="C_1"),
            Exit(exit_id="E_2", x=80, y=9, width_cells=3, connects_to="C_1"),
        ),
    )
    return Layout(
        layout_id="LY_e0_safestep_toy",
        spec_hash="e0" * 32,
        grid_w=80,
        grid_d=24,
        floors=(plan,),
        vertical_core=VerticalCore(),
        provenance=LayoutProvenance(generator="e0_benchmark", seed=0),
    )


_ASSUME = Sourcing(basis=Basis.assumption)


@dataclass(frozen=True)
class E0Row:
    seed: int
    method: str
    response: str  # "early" | "late"
    exit_used: str | None
    dose: float
    evac_time_s: float | None
    evacuated: bool


@dataclass(frozen=True)
class E0Result:
    rows: tuple[E0Row, ...]
    mean_dose: dict[str, float]  # "B1/late" -> mean dose over seeds
    exit_counts: dict[str, dict[str, int]]  # "B5/late" -> {exit id: count}
    check_passed: bool
    note: str


def _toy_config() -> tuple[HazardModel, SensorModel, OccupantProfiles]:
    hz = HazardModel(
        hazard_id="hazard_e0",
        variant=HazardVariant.A,
        params=HazardParams(
            spread_prob_open_door=0.08,
            spread_prob_closed_door=0.02,
            spread_prob_open_space=0.1,
            smoke_generation=0.6,
            smoke_diffusion=0.5,
            smoke_decay=0.02,
        ),
        state_mapping=StateMapping(smoky_min=0.15, fire_min=0.5),
        sourcing=_ASSUME,
    )
    sens = SensorModel(
        sensor_model_id="sensor_e0",
        smoke_placement=SensorPlacement.rooms_and_corridors,
        smoke_detection_prob=0.95,
        smoke_false_alarm_prob=0.02,
        occupancy_noise_sd=1.0,
        occupancy_sensor_coverage=0.9,
        exit_report_reliability=0.95,
        lift_report_reliability=0.9,
        device_report_reliability=0.9,
        user_report_reliability=0.8,
        user_report_rate_per_min=0.5,
        sourcing=_ASSUME,
    )
    profs = OccupantProfiles(
        profiles_id="profiles_e0",
        profiles=(
            OccupantProfile(
                profile_id="prof_e0",
                building_type=BuildingType.office,
                groups=(
                    GroupTemplate(template_id="early", label="Early", share=0.5, speed_mps=1.2),
                    GroupTemplate(
                        template_id="late",
                        label="Late",
                        share=0.5,
                        speed_mps=1.2,
                        response_delay_s=40.0,
                    ),
                ),
                periods=(OccupancyPeriod(name="typical", mean_fraction=1.0),),
                sourcing=_ASSUME,
            ),
        ),
    )
    return hz, sens, profs


def run_e0(seed: int = 42, n_seeds: int = 20) -> E0Result:
    layout = build_e0_toy_layout()
    graph = build_navgraph(layout)
    hz, sens, profs = _toy_config()
    room1 = next(n.node_id for n in graph.nodes if n.region_id == "R_office_1")
    room2 = next(n.node_id for n in graph.nodes if n.region_id == "R_office_2")
    sim_cfg = SimConfig()
    dose = DoseConfig(
        k=1.0,
        alert_min=0.5,
        critical_min=3.0,
        severe_min=8.0,
        penalty_alert=100.0,
        penalty_critical=300.0,
        penalty_severe=800.0,
    )
    rows: list[E0Row] = []
    for sd in range(seed, seed + n_seeds):
        for response in ("early", "late"):
            scenario = ResolvedScenario(
                scenario_id="SC_e0",
                graph_hash=graph.derived_from,
                hazard_variant=HazardVariant.A,
                ignition_node=room1,
                ignition_time_s=0.0,
                events=(),
                groups=(
                    ResolvedGroup(group_id="G_e0", template_id=response, size=1, start_node=room2),
                ),
                sensor_model_ref="sensor_e0",
                seed=sd,
                horizon_s=240.0,
            )
            for method in (Baseline.B1, Baseline.B2, Baseline.B5):
                sim = Simulator(
                    graph,
                    scenario,
                    hz,
                    sens,
                    profs,
                    sim_cfg,
                    dose,
                    np.random.default_rng(sd),
                    profile_id="prof_e0",
                    specific_flow_pers_per_m_s=1.3,
                )
                ctrl = make_controller(
                    method, graph, hz, sens, dose, sim_cfg.tick_s, 1.3, BeliefParams()
                )
                last: list[SimState] = []

                def record(st: SimState, _obs: object, store: list[SimState] = last) -> None:
                    store.append(st)

                res = sim.run(ctrl, record)
                g = last[-1].groups[0]
                rows.append(
                    E0Row(
                        seed=sd,
                        method=method.value,
                        response=response,
                        exit_used=graph_exit_id(graph, g.node) if g.evacuated else None,
                        dose=res.doses["G_e0"],
                        evac_time_s=res.evac_times_s["G_e0"],
                        evacuated=g.evacuated,
                    )
                )
    mean_dose: dict[str, float] = {}
    exit_counts: dict[str, dict[str, int]] = {}
    for m in ("B1", "B2", "B5"):
        for resp in ("early", "late"):
            sel = [r for r in rows if r.method == m and r.response == resp]
            mean_dose[f"{m}/{resp}"] = sum(r.dose for r in sel) / len(sel)
            counts: dict[str, int] = {}
            for r in sel:
                counts[r.exit_used or "none"] = counts.get(r.exit_used or "none", 0) + 1
            exit_counts[f"{m}/{resp}"] = counts
    passed = (
        mean_dose["B5/late"] < mean_dose["B1/late"] and mean_dose["B2/late"] < mean_dose["B1/late"]
    )
    note = (
        f"over {n_seeds} seeds, a late-responding group took less dose with B2 and B5 than with B1"
        if passed
        else f"over {n_seeds} seeds B2/B5 did not beat B1 at a late start; inspect the rows"
    )
    return E0Result(
        rows=tuple(rows),
        mean_dose=mean_dose,
        exit_counts=exit_counts,
        check_passed=passed,
        note=note,
    )


def graph_exit_id(graph: NavGraph, node_id: str) -> str | None:
    return next((n.region_id for n in graph.nodes if n.node_id == node_id), None)
