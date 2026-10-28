"""Deriving the navigation graph from a Layout.

Design rules
* Deterministic: node ids follow sorted region ids, so equal layouts give equal graphs.
* Robust: design-invalid layouts never crash the builder. A door that is not on the shared
  wall of the two regions it names, or an exit that is not on the footprint boundary next to
  its region, simply does not connect anything (and is reported by the rule checker).
* Corridors become chains of nodes placed where doors and exits attach, so walking distance
  follows the corridor centreline.
"""

from dataclasses import dataclass
from typing import Literal

from smartbuilding.core.layout import Layout, Rect
from smartbuilding.core.navgraph import NavEdge, NavGraph, NavNode
from smartbuilding.core.primitives import NodeKind
from smartbuilding.layout.geometry import (
    area_m2,
    centroid_m,
    dist,
    door_center_m,
    door_fits_boundary,
    exit_center_m,
    exit_fits_region,
    shared_boundary,
)

RegionKind = Literal["room", "corridor", "stair", "lift"]


@dataclass(frozen=True)
class GraphOptions:
    stair_length_m: float = 7.0  # walking length of one stair flight between floors (placeholder)
    lift_length_m: float = 3.5  # travel length between floors by lift (placeholder)


@dataclass(frozen=True)
class RegionInfo:
    region_id: str
    kind: RegionKind
    floors: tuple[int, ...]
    rect: Rect


def region_index(layout: Layout) -> dict[str, RegionInfo]:
    """Every rectangle-bearing region (rooms, corridors, stairs, lifts) by id."""
    out: dict[str, RegionInfo] = {}
    for plan in layout.floors:
        for r in plan.rooms:
            out[r.room_id] = RegionInfo(r.room_id, "room", (plan.floor,), r.rect)
        for c in plan.corridors:
            out[c.corridor_id] = RegionInfo(c.corridor_id, "corridor", (plan.floor,), c.rect)
    for s in layout.vertical_core.stairs:
        out[s.stair_id] = RegionInfo(s.stair_id, "stair", tuple(s.floors), s.rect)
    for lift in layout.vertical_core.lifts:
        out[lift.lift_id] = RegionInfo(lift.lift_id, "lift", tuple(lift.floors), lift.rect)
    return out


@dataclass
class _EdgeAcc:
    length: float
    width: float
    accessible: bool
    stairs: bool
    door_id: str | None


@dataclass(frozen=True)
class Topology:
    """Graph plus bookkeeping the rule checker needs. Never empty-graph-limited."""

    nodes: tuple[NavNode, ...]
    edges: tuple[NavEdge, ...]
    room_node: dict[str, str]
    exit_node: dict[str, str]
    invalid_doors: tuple[tuple[str, str], ...]
    invalid_exits: tuple[tuple[str, str], ...]

    def adjacency(self, accessible_only: bool = False) -> dict[str, list[tuple[str, float]]]:
        adj: dict[str, list[tuple[str, float]]] = {n.node_id: [] for n in self.nodes}
        for e in self.edges:
            if accessible_only and not e.accessible:
                continue
            adj[e.u].append((e.v, e.length_m))
            adj[e.v].append((e.u, e.length_m))
        return adj


def _axis_horizontal(rect: Rect) -> bool:
    return rect.w >= rect.d


def build_topology(
    layout: Layout,
    room_loads: dict[str, int] | None = None,
    options: GraphOptions | None = None,
) -> Topology:
    opts = options or GraphOptions()
    cs = layout.cell_size_m
    regions = region_index(layout)
    loads = room_loads or {}

    # ---- validate doors and exits geometrically
    doors_ok: list[tuple[str, str, str, tuple[float, float], int, int]] = []
    invalid_doors: list[tuple[str, str]] = []
    for plan in layout.floors:
        for d in sorted(plan.doors, key=lambda x: x.door_id):
            ra, rb = regions.get(d.a), regions.get(d.b)
            if ra is None or rb is None:
                invalid_doors.append((d.door_id, "connects a region that has no rectangle"))
                continue
            if plan.floor not in ra.floors or plan.floor not in rb.floors:
                invalid_doors.append((d.door_id, "is on a floor its regions do not serve"))
                continue
            boundary = shared_boundary(ra.rect, rb.rect)
            if not door_fits_boundary(d, boundary):
                invalid_doors.append((d.door_id, "is not on the shared wall of its two regions"))
                continue
            assert boundary is not None
            doors_ok.append(
                (d.door_id, d.a, d.b, door_center_m(d, boundary, cs), d.width_cells, plan.floor)
            )

    exits_ok: list[tuple[str, str, tuple[float, float], int, int]] = []
    invalid_exits: list[tuple[str, str]] = []
    for plan in layout.floors:
        for e in sorted(plan.exits, key=lambda x: x.exit_id):
            region = regions.get(e.connects_to)
            if region is None or plan.floor not in region.floors:
                invalid_exits.append((e.exit_id, "connects to a region that is not on its floor"))
                continue
            side = exit_fits_region(e, region.rect, layout.grid_w, layout.grid_d)
            if side is None:
                invalid_exits.append(
                    (e.exit_id, "is not on the footprint boundary next to its region")
                )
                continue
            exits_ok.append(
                (e.exit_id, e.connects_to, exit_center_m(e, side, cs), e.width_cells, plan.floor)
            )

    # ---- corridor attachment coordinates (metres along the corridor axis)
    attach: dict[str, set[float]] = {}

    def axis_coord(region: RegionInfo, point: tuple[float, float]) -> float:
        return round(point[0] if _axis_horizontal(region.rect) else point[1], 6)

    for _, a, b, centre, _, _ in doors_ok:
        for rid in (a, b):
            if regions[rid].kind == "corridor":
                attach.setdefault(rid, set()).add(axis_coord(regions[rid], centre))
    for _, rid, centre, _, _ in exits_ok:
        if regions[rid].kind == "corridor":
            attach.setdefault(rid, set()).add(axis_coord(regions[rid], centre))

    # ---- nodes
    nodes: list[NavNode] = []
    counters: dict[str, int] = {}

    def add_node(
        kind: NodeKind,
        floor: int,
        pos: tuple[float, float],
        region_id: str | None,
        capacity: float = 0.0,
        area: float = 0.0,
    ) -> str:
        counters[kind.value] = counters.get(kind.value, 0) + 1
        node_id = f"N_{kind.value}_{counters[kind.value]}"
        nodes.append(
            NavNode(
                node_id=node_id,
                kind=kind,
                floor=floor,
                x_m=pos[0],
                y_m=pos[1],
                capacity_pers=capacity,
                area_m2=area,
                region_id=region_id,
            )
        )
        return node_id

    room_node: dict[str, str] = {}
    node_pos: dict[str, tuple[float, float]] = {}
    for plan in sorted(layout.floors, key=lambda p: p.floor):
        for room in sorted(plan.rooms, key=lambda r: r.room_id):
            pos = centroid_m(room.rect, cs)
            nid = add_node(
                NodeKind.room,
                plan.floor,
                pos,
                room.room_id,
                float(loads.get(room.room_id, 0)),
                area_m2(room.rect, cs),
            )
            room_node[room.room_id] = nid
            node_pos[nid] = pos

    shaft_node: dict[tuple[str, int], str] = {}
    for rid in sorted(k for k, v in regions.items() if v.kind in ("stair", "lift")):
        info = regions[rid]
        kind = NodeKind.stair if info.kind == "stair" else NodeKind.lift
        for f in info.floors:
            pos = centroid_m(info.rect, cs)
            nid = add_node(kind, f, pos, rid, 0.0, area_m2(info.rect, cs))
            shaft_node[(rid, f)] = nid
            node_pos[nid] = pos

    corridor_nodes: dict[str, list[tuple[float, str]]] = {}
    for rid in sorted(k for k, v in regions.items() if v.kind == "corridor"):
        info = regions[rid]
        cx, cy = centroid_m(info.rect, cs)
        chain: list[tuple[float, str]] = []
        for coord in sorted(attach.get(rid, ())):
            pos = (coord, cy) if _axis_horizontal(info.rect) else (cx, coord)
            nid = add_node(NodeKind.corridor, info.floors[0], pos, rid)
            node_pos[nid] = pos
            chain.append((coord, nid))
        corridor_nodes[rid] = chain

    def node_for(rid: str, floor: int, centre: tuple[float, float]) -> str:
        info = regions[rid]
        if info.kind == "room":
            return room_node[rid]
        if info.kind == "corridor":
            coord = axis_coord(info, centre)
            return next(n for c, n in corridor_nodes[rid] if c == coord)
        return shaft_node[(rid, floor)]

    # ---- edges (merged when two openings join the same pair of nodes)
    edge_data: dict[tuple[str, str], _EdgeAcc] = {}

    def add_edge(
        u: str,
        v: str,
        length: float,
        width: float,
        accessible: bool,
        stairs: bool,
        door_id: str | None,
    ) -> None:
        if u == v:
            return
        key = (u, v) if u < v else (v, u)
        length = max(length, 1e-6)
        old = edge_data.get(key)
        if old is None:
            edge_data[key] = _EdgeAcc(length, width, accessible, stairs, door_id)
            return
        old.length = min(old.length, length)
        old.width += width
        old.accessible = old.accessible or accessible
        old.stairs = old.stairs and stairs
        if old.door_id is None or (door_id is not None and door_id < old.door_id):
            old.door_id = door_id

    for door_id, a, b, centre, width_cells, floor in doors_ok:
        ua, ub = node_for(a, floor, centre), node_for(b, floor, centre)
        length = dist(node_pos[ua], centre) + dist(centre, node_pos[ub])
        add_edge(ua, ub, length, width_cells * cs, True, False, door_id)

    for rid, chain in corridor_nodes.items():
        width_m = min(regions[rid].rect.w, regions[rid].rect.d) * cs
        for (_, n1), (_, n2) in zip(chain, chain[1:], strict=False):
            add_edge(n1, n2, dist(node_pos[n1], node_pos[n2]), width_m, True, False, None)

    exit_node: dict[str, str] = {}
    for exit_id, rid, centre, width_cells, floor in exits_ok:
        enid = add_node(NodeKind.exit, floor, centre, exit_id)
        node_pos[enid] = centre
        exit_node[exit_id] = enid
        target = node_for(rid, floor, centre)
        add_edge(enid, target, dist(centre, node_pos[target]), width_cells * cs, True, False, None)

    for rid in sorted(k for k, v in regions.items() if v.kind in ("stair", "lift")):
        info = regions[rid]
        width_m = min(info.rect.w, info.rect.d) * cs
        is_stair = info.kind == "stair"
        accessible = False if is_stair else True
        if not is_stair:
            lift = next(x for x in layout.vertical_core.lifts if x.lift_id == rid)
            accessible = lift.accessible
        length = opts.stair_length_m if is_stair else opts.lift_length_m
        for f1, f2 in zip(info.floors, info.floors[1:], strict=False):
            add_edge(
                shaft_node[(rid, f1)],
                shaft_node[(rid, f2)],
                length,
                width_m,
                accessible,
                is_stair,
                None,
            )

    edges = tuple(
        NavEdge(
            u=k[0],
            v=k[1],
            length_m=e.length,
            width_m=e.width,
            accessible=e.accessible,
            stairs=e.stairs,
            door_id=e.door_id,
        )
        for k, e in sorted(edge_data.items())
    )
    return Topology(
        tuple(nodes), edges, room_node, exit_node, tuple(invalid_doors), tuple(invalid_exits)
    )


def build_navgraph(
    layout: Layout,
    room_loads: dict[str, int] | None = None,
    options: GraphOptions | None = None,
) -> NavGraph:
    """The NavGraph contract for a layout. Raises ValueError if the layout has no nodes."""
    topo = build_topology(layout, room_loads, options)
    if not topo.nodes:
        raise ValueError("layout has no rooms, corridors or exits, so there is no graph")
    return NavGraph(
        derived_from=layout.geometry_hash,
        cell_size_m=layout.cell_size_m,
        nodes=topo.nodes,
        edges=topo.edges,
    )
