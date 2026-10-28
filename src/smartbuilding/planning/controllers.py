"""Controllers: glue between a belief tracker and a planning function. No ground truth here."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from smartbuilding.core.config import (
    BeliefParams,
    DoseConfig,
    HazardModel,
    SensorModel,
)
from smartbuilding.core.decision import Baseline, Plan
from smartbuilding.core.navgraph import NavGraph
from smartbuilding.core.runtime import BeliefState, Observation, PublicState
from smartbuilding.planning.dose_aware import DoseAwareParams, plan_dose_aware
from smartbuilding.planning.evacuation import plan_b1, plan_b2
from smartbuilding.uncertainty.belief import BeliefFilter, BeliefModel, predict_series
from smartbuilding.uncertainty.raw import RawTracker


class Tracker(Protocol):
    def observe(self, t_s: float, observations: Sequence[Observation]) -> None: ...
    def belief(self) -> BeliefState: ...


class B1Controller:
    """Nearest-exit Dijkstra. Ignores observations."""

    def __init__(self, graph: NavGraph) -> None:
        self.graph = graph

    def observe(self, t_s: float, observations: Sequence[Observation]) -> None:
        return None

    def decide(self, public: PublicState) -> Plan | None:
        return plan_b1(self.graph, public.active(), public.t_s)


class B2Controller:
    def __init__(self, graph: NavGraph, tracker: Tracker, threshold: float = 0.5) -> None:
        self.graph, self.tracker, self.threshold = graph, tracker, threshold

    def observe(self, t_s: float, observations: Sequence[Observation]) -> None:
        self.tracker.observe(t_s, observations)

    def decide(self, public: PublicState) -> Plan | None:
        return plan_b2(
            self.graph, public.active(), self.tracker.belief(), self.threshold, public.t_s
        )


class DoseAwareController:
    """B3 (truth tracker), B4 (RawTracker) or B5 (BeliefFilter): the same planner, different input.

    ``model`` enables forward prediction of the belief (used for B5 only).
    """

    def __init__(
        self,
        graph: NavGraph,
        tracker: Tracker,
        method: Baseline,
        hazard_model: HazardModel,
        dose: DoseConfig,
        flow_pers_per_m_s: float | None,
        params: DoseAwareParams | None = None,
        predict_model: BeliefModel | None = None,
        horizon_s: float = 120.0,
    ) -> None:
        self.graph, self.tracker, self.method = graph, tracker, method
        self.mapping = hazard_model.state_mapping
        self.dose, self.flow = dose, flow_pers_per_m_s
        self.params = params or DoseAwareParams()
        self.predict_model, self.horizon_s = predict_model, horizon_s

    def observe(self, t_s: float, observations: Sequence[Observation]) -> None:
        self.tracker.observe(t_s, observations)

    def decide(self, public: PublicState) -> Plan | None:
        belief = self.tracker.belief()
        if self.predict_model is not None:
            series = predict_series(
                belief, self.predict_model, self.horizon_s, self.params.series_step_s
            )
        else:
            series = [belief]
        return plan_dose_aware(
            self.graph,
            public.active(),
            series,
            self.mapping,
            self.dose.k,
            self.params,
            self.flow,
            self.method,
            public.t_s,
        )


def make_controller(
    method: Baseline,
    graph: NavGraph,
    hazard_belief_model: HazardModel,
    sensors: SensorModel,
    dose: DoseConfig,
    tick_s: float,
    flow_pers_per_m_s: float | None,
    belief_params: BeliefParams | None = None,
    params: DoseAwareParams | None = None,
    occupancy_prior: dict[str, float] | None = None,
) -> B1Controller | B2Controller | DoseAwareController:
    """B1, B2 (on belief), B4 (raw observations) and B5 (belief). B3 needs ground truth: see eval.oracle."""
    if method == Baseline.B1:
        return B1Controller(graph)
    if method == Baseline.B2:
        return B2Controller(
            graph,
            BeliefFilter(
                graph, hazard_belief_model, sensors, tick_s, belief_params, occupancy_prior
            ),
        )
    if method == Baseline.B4:
        return DoseAwareController(
            graph, RawTracker(graph), method, hazard_belief_model, dose, flow_pers_per_m_s, params
        )
    if method == Baseline.B5:
        filt = BeliefFilter(
            graph, hazard_belief_model, sensors, tick_s, belief_params, occupancy_prior
        )
        return DoseAwareController(
            graph,
            filt,
            method,
            hazard_belief_model,
            dose,
            flow_pers_per_m_s,
            params,
            predict_model=filt.model,
        )
    raise ValueError(f"{method.value} is not a planner baseline built here")
