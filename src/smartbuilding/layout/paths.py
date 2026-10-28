"""Shortest walking distances (multi-source Dijkstra). Written for this project, no libraries."""

import heapq
from collections.abc import Iterable, Mapping, Sequence

INF = float("inf")


def shortest_distances(
    adjacency: Mapping[str, Sequence[tuple[str, float]]], sources: Iterable[str]
) -> dict[str, float]:
    """Distance from the nearest source to every reachable node.

    Ties are broken by node id, so the result is identical on every run.
    """
    dist: dict[str, float] = {}
    heap: list[tuple[float, str]] = []
    for s in sorted(set(sources)):
        if s in adjacency:
            heapq.heappush(heap, (0.0, s))
    while heap:
        d, node = heapq.heappop(heap)
        if node in dist:
            continue
        dist[node] = d
        for nxt, w in adjacency[node]:
            if nxt not in dist:
                heapq.heappush(heap, (d + w, nxt))
    return dist
