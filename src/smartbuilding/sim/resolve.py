"""Bind a layout-agnostic Scenario to one NavGraph: concrete nodes, groups and group sizes.

Occupancy: total people = ``total_occupants`` x the scenario's occupancy-period fraction, split over
group templates by their ``share`` and over rooms by room load (``distributed_by_load``).
Sizes are rounded with the largest-remainder method so the total is exact.
"""

from __future__ import annotations

import math
import re

import numpy as np

from smartbuilding.core.config import OccupantProfile
from smartbuilding.core.navgraph import NavGraph, NavNode
from smartbuilding.core.primitives import EventType, NodeKind, SelectorMode
from smartbuilding.core.scenario import (
    ALL_TEMPLATES,
    ResolvedEvent,
    ResolvedGroup,
    ResolvedScenario,
    Scenario,
    Selector,
)
from smartbuilding.layout.paths import shortest_distances


class ResolveError(ValueError):
    """A selector or group could not be bound to the layout."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower())


def _kind_nodes(graph: NavGraph, kind: NodeKind) -> list[NavNode]:
    return sorted((n for n in graph.nodes if n.kind == kind), key=lambda n: n.node_id)


def _dist_to_exits(graph: NavGraph) -> dict[str, float]:
    adj: dict[str, list[tuple[str, float]]] = {n.node_id: [] for n in graph.nodes}
    for e in graph.edges:
        adj[e.u].append((e.v, e.length_m))
        adj[e.v].append((e.u, e.length_m))
    exits = [n.node_id for n in graph.nodes if n.kind == NodeKind.exit]
    return shortest_distances(adj, exits)


def resolve_selector(
    sel: Selector,
    graph: NavGraph,
    rng: np.random.Generator,
    ignition_node: str | None = None,
) -> str:
    """Return a node id (or, for explicit ids that are not nodes such as doors, the id itself)."""
    mode = sel.mode
    if mode == SelectorMode.explicit_id:
        assert sel.ref is not None
        for n in graph.nodes:
            if sel.ref in (n.node_id, n.region_id):
                return n.node_id
        return sel.ref
    if mode == SelectorMode.room_type:
        rooms = [
            n
            for n in _kind_nodes(graph, NodeKind.room)
            if n.region_id and n.region_id.startswith(f"R_{sel.ref}_")
        ]
        if not rooms:
            raise ResolveError(f"no room of type {sel.ref!r} in this layout")
        return rooms[int(rng.integers(len(rooms)))].node_id
    if mode in (SelectorMode.random_room, SelectorMode.distributed_by_load):
        rooms = _kind_nodes(graph, NodeKind.room)
        if not rooms:
            raise ResolveError("layout has no rooms")
        return rooms[int(rng.integers(len(rooms)))].node_id
    if mode == SelectorMode.random_room_near_exit:
        assert sel.nearest_fraction is not None
        dist = _dist_to_exits(graph)
        rooms = sorted(
            _kind_nodes(graph, NodeKind.room),
            key=lambda n: (dist.get(n.node_id, math.inf), n.node_id),
        )
        if not rooms:
            raise ResolveError("layout has no rooms")
        k = max(1, math.ceil(sel.nearest_fraction * len(rooms)))
        return rooms[int(rng.integers(k))].node_id
    kinds = {
        SelectorMode.random_exit: NodeKind.exit,
        SelectorMode.random_lift: NodeKind.lift,
        SelectorMode.random_corridor: NodeKind.corridor,
    }
    if mode in kinds:
        nodes = _kind_nodes(graph, kinds[mode])
        if not nodes:
            raise ResolveError(f"layout has no {kinds[mode].value} for selector {mode.value!r}")
        return nodes[int(rng.integers(len(nodes)))].node_id
    if mode == SelectorMode.nearest_exit_to_ignition:
        if ignition_node is None:
            raise ResolveError("nearest_exit_to_ignition needs an ignition node")
        adj: dict[str, list[tuple[str, float]]] = {n.node_id: [] for n in graph.nodes}
        for e in graph.edges:
            adj[e.u].append((e.v, e.length_m))
            adj[e.v].append((e.u, e.length_m))
        d = shortest_distances(adj, [ignition_node])
        exits = _kind_nodes(graph, NodeKind.exit)
        if not exits:
            raise ResolveError("layout has no exits")
        return min(exits, key=lambda n: (d.get(n.node_id, math.inf), n.node_id)).node_id
    raise ResolveError(f"unsupported selector mode {mode.value!r}")


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    s = sum(weights)
    if s <= 0 or total <= 0:
        return [0] * len(weights)
    raw = [w / s * total for w in weights]
    base = [math.floor(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[: total - sum(base)]:
        base[i] += 1
    return base


def resolve_scenario(
    scenario: Scenario,
    graph: NavGraph,
    profile: OccupantProfile,
    total_occupants: int,
    rng: np.random.Generator | None = None,
) -> ResolvedScenario:
    rng = rng or np.random.default_rng(scenario.seed)
    ign = resolve_selector(scenario.ignition.where, graph, rng)

    events: list[ResolvedEvent] = []
    for ev in scenario.events:
        if ev.type == EventType.lift_out and not _kind_nodes(graph, NodeKind.lift):
            continue  # single-floor layout: nothing to fail
        events.append(
            ResolvedEvent(
                type=ev.type,
                time_s=ev.time_s,
                target_id=resolve_selector(ev.target, graph, rng, ignition_node=ign),
            )
        )

    fraction = 1.0
    if scenario.occupancy_period is not None:
        period = next((p for p in profile.periods if p.name == scenario.occupancy_period), None)
        if period is None:
            raise ResolveError(f"profile has no occupancy period {scenario.occupancy_period!r}")
        fraction = period.mean_fraction
    people = round(total_occupants * fraction)
    shares = {g.template_id: g.share for g in profile.groups}

    rooms = _kind_nodes(graph, NodeKind.room)
    groups: list[ResolvedGroup] = []
    seen: set[str] = set()

    def add(gid: str, tpl: str, size: int, node: str) -> None:
        if size <= 0:
            return
        gid = "G_" + _slug(gid.removeprefix("G_"))
        if gid in seen:
            raise ResolveError(f"duplicate group id {gid!r}")
        seen.add(gid)
        groups.append(ResolvedGroup(group_id=gid, template_id=tpl, size=size, start_node=node))

    for spec in scenario.groups:
        templates = (
            [g.template_id for g in profile.groups]
            if spec.template_id == ALL_TEMPLATES
            else [spec.template_id]
        )
        for tpl in templates:
            if tpl not in shares:
                raise ResolveError(
                    f"template {tpl!r} is not in occupant profile {profile.profile_id!r}"
                )
        if spec.size_override is not None:
            add(
                f"{spec.group_id}",
                templates[0],
                spec.size_override,
                resolve_selector(spec.placement, graph, rng),
            )
        elif spec.placement.mode == SelectorMode.distributed_by_load:
            load = [max(r.capacity_pers, 0.0) for r in rooms]
            if sum(load) <= 0:
                load = [1.0] * len(rooms)
            cells = [(r, t) for r in rooms for t in templates]
            w = [load[rooms.index(r)] * shares[t] for r, t in cells]
            target = round(people * sum(shares[t] for t in templates))
            for (r, t), n in zip(cells, _largest_remainder(w, target), strict=True):
                add(f"G_{r.region_id or r.node_id}_{t}", t, n, r.node_id)
        else:
            node = resolve_selector(spec.placement, graph, rng)
            for t in templates:
                add(f"{spec.group_id}_{t}", t, max(1, round(people * shares[t])), node)
    if not groups:
        raise ResolveError("scenario resolved to zero occupants")

    return ResolvedScenario(
        scenario_id=scenario.scenario_id,
        graph_hash=graph.derived_from,
        hazard_variant=scenario.hazard_variant,
        ignition_node=ign,
        ignition_time_s=scenario.ignition.time_s,
        events=tuple(events),
        groups=tuple(groups),
        sensor_model_ref=scenario.sensor_model_ref,
        seed=scenario.seed,
        horizon_s=scenario.horizon_s,
    )
