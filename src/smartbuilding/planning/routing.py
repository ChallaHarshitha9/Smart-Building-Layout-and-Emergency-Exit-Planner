"""Generic least-cost routing over a NavGraph (own Dijkstra; ties broken by node id).

The edge cost function may depend on the time the walker reaches the edge, which makes this a
time-dependent label-setting search. With costs that are not FIFO-consistent (queues) it is a
heuristic, which is how the planner is described in the report.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from smartbuilding.core.navgraph import NavEdge, NavGraph

# (from node, to node, edge, seconds until the walker is at `from`) -> cost, or None if impassable
EdgeCost = Callable[[str, str, NavEdge, float], float | None]


@dataclass(frozen=True)
class RouteResult:
    nodes: tuple[str, ...]
    cost: float
    time_s: float


def adjacency(graph: NavGraph) -> dict[str, list[tuple[str, NavEdge]]]:
    adj: dict[str, list[tuple[str, NavEdge]]] = {n.node_id: [] for n in graph.nodes}
    for e in graph.edges:
        adj[e.u].append((e.v, e))
        adj[e.v].append((e.u, e))
    for lst in adj.values():
        lst.sort(key=lambda x: x[0])
    return adj


def least_cost_routes(
    adj: dict[str, list[tuple[str, NavEdge]]],
    start: str,
    targets: Iterable[str],
    edge_cost: EdgeCost,
    speed_mps: float,
) -> dict[str, RouteResult]:
    """Best route from ``start`` to every reachable target. A target equal to start gets a 1-node route."""
    wanted = set(targets)
    best: dict[str, tuple[float, float]] = {}
    parent: dict[str, str | None] = {}
    heap: list[tuple[float, str, float, str | None]] = [(0.0, start, 0.0, None)]
    while heap:
        cost, node, t, par = heapq.heappop(heap)
        if node in best:
            continue
        best[node] = (cost, t)
        parent[node] = par
        for nxt, e in adj[node]:
            if nxt in best:
                continue
            c = edge_cost(node, nxt, e, t)
            if c is None:
                continue
            heapq.heappush(heap, (cost + c, nxt, t + e.length_m / speed_mps, node))
    out: dict[str, RouteResult] = {}
    for tgt in sorted(wanted & best.keys()):
        path: list[str] = []
        cur: str | None = tgt
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        out[tgt] = RouteResult(tuple(path), best[tgt][0], best[tgt][1])
    return out
