"""Dose-aware, risk-aware evacuation planner (B3 oracle / B4 raw / B5 belief use this one planner).

Edge cost (seconds-equivalent) for walking u -> v, arriving at u after t seconds:

    walk time  +  lambda_dose * dose estimate  +  lambda_queue * queue delay estimate

* walk time = length / group speed.
* dose estimate = k * walk time * mean expected smoke over the edge, taken from the belief
  *predicted forward* to the time the group would be there (time-dependent costs).
* queue delay = people currently waiting at u (other groups) / edge flow capacity. The planner
  routes every group on its own: it does not coordinate groups (that is what RL is tested for).
* exits: an exit whose availability belief is below ``min_exit_avail`` is excluded; others are
  penalised by (1 - P(available)) * exit_penalty_s. Doors believed stuck are impassable.

Because queue-dependent costs break strict optimality this is a risk-aware heuristic planner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from smartbuilding.core.config import StateMapping
from smartbuilding.core.decision import Baseline, Plan
from smartbuilding.core.devices import device_for_region
from smartbuilding.core.navgraph import NavEdge, NavGraph
from smartbuilding.core.runtime import BeliefState, PublicGroup
from smartbuilding.planning.evacuation import GroupRef, as_public, build_plan, exit_nodes, passable
from smartbuilding.planning.routing import EdgeCost
from smartbuilding.uncertainty.belief import expected_smoke


@dataclass(frozen=True)
class DoseAwareParams:
    lambda_dose: float = 10.0  # seconds-equivalent per unit of dose proxy
    lambda_queue: float = 1.0
    exit_penalty_s: float = 120.0
    min_exit_avail: float = 0.2
    door_ok_min: float = 0.3
    series_step_s: float = 6.0  # spacing of the predicted beliefs


def plan_dose_aware(
    graph: NavGraph,
    groups: Sequence[GroupRef],
    series: Sequence[BeliefState],
    mapping: StateMapping,
    dose_k: float,
    params: DoseAwareParams | None = None,
    flow_pers_per_m_s: float | None = None,
    method: Baseline = Baseline.B5,
    t_s: float = 0.0,
    plan_id: str = "plan_dose_aware",
) -> Plan:
    """Plan with ``series`` = [belief now, belief +step, ...]. A one-element series means a static belief."""
    p = params or DoseAwareParams()
    now = series[0]
    kinds = {n.node_id: n.kind for n in graph.nodes}
    pubs = [as_public(r) for r in groups]
    at_node: dict[str, int] = {}
    for g in pubs:
        if not g.evacuated and not g.in_transit:
            at_node[g.node] = at_node.get(g.node, 0) + g.size

    def smoke(node: str, t: float) -> float:
        idx = min(int(t / p.series_step_s + 0.5), len(series) - 1)
        return expected_smoke(series[idx], node, mapping)

    def door_ok(e: NavEdge) -> bool:
        return (
            e.door_id is None or now.devices.get(device_for_region(e.door_id), 1.0) >= p.door_ok_min
        )

    def edge_dose(g: PublicGroup, u: str, v: str, e: NavEdge, t: float) -> float:
        te = e.length_m / g.speed_mps
        return dose_k * te * 0.5 * (smoke(u, t) + smoke(v, t + te))

    def factory(g: PublicGroup) -> EdgeCost:
        def cost(u: str, v: str, e: NavEdge, t: float) -> float | None:
            if not passable(kinds, g, u, v, e) or not door_ok(e):
                return None
            te = e.length_m / g.speed_mps
            q = 0.0
            if flow_pers_per_m_s:
                others = at_node.get(u, 0) - (g.size if (not g.in_transit and g.node == u) else 0)
                q = max(0, others) / (e.width_m * flow_pers_per_m_s)
            return te + p.lambda_dose * edge_dose(g, u, v, e, t) + p.lambda_queue * q

        return cost

    def route_dose(g: PublicGroup, nodes: tuple[str, ...]) -> float:
        total, t = 0.0, 0.0
        edges = {(e.u, e.v): e for e in graph.edges} | {(e.v, e.u): e for e in graph.edges}
        for u, v in zip(nodes, nodes[1:], strict=False):
            total += edge_dose(g, u, v, edges[(u, v)], t)
            t += edges[(u, v)].length_m / g.speed_mps
        return total

    allowed: dict[str, float] = {}
    for node_id, exit_id in exit_nodes(graph).items():
        avail = now.exit_avail.get(exit_id, 1.0)
        if avail >= p.min_exit_avail:
            allowed[node_id] = (1.0 - avail) * p.exit_penalty_s
    if not allowed:  # believed all blocked: still try, least doubtful first
        allowed = {
            n: (1.0 - now.exit_avail.get(x, 1.0)) * p.exit_penalty_s
            for n, x in exit_nodes(graph).items()
        }
    return build_plan(graph, pubs, method, plan_id, t_s, factory, allowed, route_dose)
