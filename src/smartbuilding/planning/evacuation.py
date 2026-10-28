"""Evacuation baselines B1 and B2, and the shared plan builder.

B1  nearest-exit Dijkstra on length. Static: ignores hazards and exit status.
B2  SafeStep-style threshold Dijkstra: nodes whose hazard probability is at least ``threshold`` are
    removed, then the shortest path is taken. If that disconnects the group from every exit it falls
    back to B1. Here it is re-run at every decision tick on the same belief the other planners get,
    which makes it a stronger baseline than the paper's one-shot version and isolates the value of
    dose-aware costing, exit status and queue awareness.

This package never imports the simulator (information barrier).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from smartbuilding.core.decision import Baseline, Decision, Plan, Route
from smartbuilding.core.navgraph import NavEdge, NavGraph
from smartbuilding.core.primitives import NodeKind
from smartbuilding.core.runtime import BeliefState, PublicGroup
from smartbuilding.planning.routing import (
    EdgeCost,
    RouteResult,
    adjacency,
    least_cost_routes,
)

GroupRef = PublicGroup | tuple[str, str]  # (group id, node) is accepted for convenience

DEFAULT_SPEED_MPS = 1.2


def as_public(ref: GroupRef) -> PublicGroup:
    if isinstance(ref, PublicGroup):
        return ref
    gid, node = ref
    return PublicGroup(
        group_id=gid, template_id="adult", size=1, node=node, speed_mps=DEFAULT_SPEED_MPS
    )


def exit_nodes(graph: NavGraph) -> dict[str, str]:
    """Exit node id -> exit id (E_n)."""
    return {
        n.node_id: n.region_id
        for n in graph.nodes
        if n.kind == NodeKind.exit and n.region_id is not None
    }


def passable(
    graph_kinds: Mapping[str, NodeKind], g: PublicGroup, u: str, v: str, e: NavEdge
) -> bool:
    """Static passability for one group: stairs and lifts (lifts are never relied on in a fire)."""
    if e.stairs and not g.can_use_stairs:
        return False
    return NodeKind.lift not in (graph_kinds[u], graph_kinds[v])


CostFactory = Callable[[PublicGroup], EdgeCost]
RouteDose = Callable[[PublicGroup, tuple[str, ...]], float]


def build_plan(
    graph: NavGraph,
    groups: Sequence[GroupRef],
    method: Baseline,
    plan_id: str,
    t_s: float,
    cost_factory: CostFactory,
    allowed_exits: Mapping[str, float] | None = None,
    route_dose: RouteDose | None = None,
) -> Plan:
    """Least-cost route per group to the best exit, with ranked alternatives.

    ``allowed_exits`` maps exit node id -> extra cost added when ranking exits (use it to rule exits
    out by leaving them out, or to penalise doubtful ones). None means every exit, no penalty.
    """
    adj = adjacency(graph)
    all_exits = exit_nodes(graph)
    exits = {n: 0.0 for n in all_exits} if allowed_exits is None else dict(allowed_exits)
    decisions: list[Decision] = []
    for ref in groups:
        g = as_public(ref)
        if g.evacuated or not all_exits:
            continue
        routes = least_cost_routes(adj, g.node, exits.keys(), cost_factory(g), g.speed_mps)
        ranked = sorted(routes.items(), key=lambda kv: (kv[1].cost + exits[kv[0]], kv[0]))

        def to_route(r: RouteResult, grp: PublicGroup = g) -> Route:
            dose = route_dose(grp, r.nodes) if route_dose else 0.0
            return Route(nodes=r.nodes, est_time_s=r.time_s, est_dose=dose)

        if ranked:
            best_node, best = ranked[0]
            target = all_exits[best_node]
            main = to_route(best)
            alts = tuple(to_route(r) for _, r in ranked[1:])
        else:  # no usable exit: hold position (a decision is still recorded)
            target = sorted(all_exits.values())[0]
            main = Route(nodes=(g.node,))
            alts = ()
        decisions.append(
            Decision(
                group_id=g.group_id, t_s=t_s, target_exit=target, route=main, alternatives=alts
            )
        )
    return Plan(
        plan_id=plan_id, graph_hash=graph.derived_from, method=method, decisions=tuple(decisions)
    )


def plan_b1(
    graph: NavGraph, groups: Sequence[GroupRef], t_s: float = 0.0, plan_id: str = "plan_b1"
) -> Plan:
    kinds = {n.node_id: n.kind for n in graph.nodes}

    def factory(g: PublicGroup) -> EdgeCost:
        def cost(u: str, v: str, e: NavEdge, t: float) -> float | None:
            return e.length_m if passable(kinds, g, u, v, e) else None

        return cost

    return build_plan(graph, groups, Baseline.B1, plan_id, t_s, factory)


def plan_b2(
    graph: NavGraph,
    groups: Sequence[GroupRef],
    belief: BeliefState,
    threshold: float = 0.5,
    t_s: float = 0.0,
    plan_id: str = "plan_b2",
) -> Plan:
    kinds = {n.node_id: n.kind for n in graph.nodes}
    p_haz = {nid: 1.0 - tri[0] for nid, tri in belief.node_belief.items()}

    def factory(g: PublicGroup) -> EdgeCost:
        def cost(u: str, v: str, e: NavEdge, t: float) -> float | None:
            if not passable(kinds, g, u, v, e) or p_haz.get(v, 0.0) >= threshold:
                return None
            return e.length_m

        return cost

    plan = build_plan(graph, groups, Baseline.B2, plan_id, t_s, factory)
    # fall back to B1 for groups that the blocking cut off from every exit
    stuck = {
        d.group_id
        for d in plan.decisions
        if len(d.route.nodes) == 1 and d.route.nodes[0] not in exit_nodes(graph)
    }
    if not stuck:
        return plan
    fallback = {
        d.group_id: d
        for d in plan_b1(
            graph, [as_public(r) for r in groups if as_public(r).group_id in stuck], t_s
        ).decisions
    }
    return plan.model_copy(
        update={"decisions": tuple(fallback.get(d.group_id, d) for d in plan.decisions)}
    )
