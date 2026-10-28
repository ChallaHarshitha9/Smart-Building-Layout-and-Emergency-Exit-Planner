"""Deterministic rule checker.

Checks a Layout against a BuildingSpec and the configured thresholds, and reports which
configured checks pass. It never claims safety or legal compliance: the result is
"constraint-checked against the configured rules". Nothing here is random, nothing calls an
LLM, and output order does not depend on the order of the input lists.

Design-invalid layouts (overlaps, out of bounds, misplaced doors) are ordinary inputs: they
are reported as failing checks, never as exceptions.
"""

import math
from dataclasses import dataclass, field

from smartbuilding.core.config import RoomCatalogue, Thresholds
from smartbuilding.core.layout import (
    ConflictSet,
    ConstraintResult,
    Layout,
    RoomPlacement,
    SoftScore,
    Verification,
)
from smartbuilding.core.metrics import MetricSet, MetricValue
from smartbuilding.core.primitives import (
    ConstraintKind,
    MetricScope,
    RelationType,
    Side,
    VerificationStatus,
)
from smartbuilding.core.spec import BuildingSpec, Relation
from smartbuilding.layout.geometry import (
    area_m2,
    exterior_contacts,
    inside_grid,
    overlap_cells,
    shared_boundary,
)
from smartbuilding.layout.graph import (
    GraphOptions,
    RegionInfo,
    Topology,
    build_topology,
    region_index,
)
from smartbuilding.layout.paths import INF, shortest_distances
from smartbuilding.layout.relations import (
    effective_relations,
    members,
    pair_quality,
)

TOL = 1e-9

HARD_RULE_ORDER: tuple[str, ...] = (
    "HC_floor_count",
    "HC_required_rooms",
    "HC_room_area",
    "HC_inside_footprint",
    "HC_no_overlap",
    "HC_door_placement",
    "HC_exit_placement",
    "HC_connectivity",
    "HC_min_exit_count",
    "HC_exit_width",
    "HC_corridor_width",
    "HC_door_width",
    "HC_travel_distance",
    "HC_step_free_access",
    "HC_vertical_access",
    "HC_incompatible_separation",
    "HC_hard_relations",
)


@dataclass(frozen=True)
class CheckerOptions:
    """Checker settings that are not part of the cited thresholds. All are placeholders."""

    min_window_contact_m: float = 1.0  # wall contact needed for a window
    adjacent_min_contact_m: float = 1.0  # shared wall length that counts as 'adjacent'
    energy_proxy_reference: float = 0.25  # exterior wall metres per room square metre
    graph: GraphOptions = field(default_factory=GraphOptions)

    def assumption_notes(self) -> tuple[str, ...]:
        return (
            f"min_window_contact_m={self.min_window_contact_m}",
            f"adjacent_min_contact_m={self.adjacent_min_contact_m}",
            f"energy_proxy_reference={self.energy_proxy_reference}",
            f"stair_length_m={self.graph.stair_length_m}",
            f"lift_length_m={self.graph.lift_length_m}",
        )


@dataclass(frozen=True)
class LayoutAnalysis:
    verification: Verification
    quality: dict[str, float]  # quantities used for metrics, for example 'usable_area_ratio'
    topology: Topology
    room_distance_to_exit_m: dict[str, float]


@dataclass
class _Ctx:
    layout: Layout
    spec: BuildingSpec
    thresholds: Thresholds
    catalogue: RoomCatalogue
    options: CheckerOptions
    cs: float
    rooms: dict[str, RoomPlacement]
    regions: dict[str, RegionInfo]
    topo: Topology
    relations: tuple[Relation, ...]
    exit_dist: dict[str, float]
    exit_dist_accessible: dict[str, float]
    room_types: dict[str, str]
    _room_dist: dict[str, dict[str, float]] = field(default_factory=dict)

    def room_distance(self, a: str, b: str) -> float:
        """Walking distance between two rooms in metres (inf if not connected)."""
        if a not in self._room_dist:
            self._room_dist[a] = shortest_distances(self.topo.adjacency(), [self.topo.room_node[a]])
        return self._room_dist[a].get(self.topo.room_node[b], INF)

    def contact_cells(self, a: str, b: str) -> int:
        boundary = shared_boundary(self.rooms[a].rect, self.rooms[b].rect)
        return boundary.length_cells if boundary else 0

    @property
    def window_cells(self) -> int:
        return max(1, math.ceil(self.options.min_window_contact_m / self.cs - TOL))

    @property
    def adjacent_cells(self) -> int:
        return max(1, math.ceil(self.options.adjacent_min_contact_m / self.cs - TOL))


def _f(x: float) -> str:
    return f"{x:.2f}"


def _hard(
    cid: str,
    violators: list[str],
    detail_pass: str,
    detail_fail: str,
    value: float | None = None,
    limit: float | None = None,
) -> ConstraintResult:
    ok = not violators
    return ConstraintResult(
        constraint_id=cid,
        kind=ConstraintKind.hard,
        satisfied=ok,
        value=value,
        limit=limit,
        detail=detail_pass if ok else detail_fail,
        entity_ids=tuple(sorted(set(violators))),
    )


# --------------------------------------------------------------------------- hard rules


def _floor_count(c: _Ctx) -> ConstraintResult:
    floors = sorted(p.floor for p in c.layout.floors)
    ok = floors == list(range(c.spec.floors))
    return _hard(
        "HC_floor_count",
        [] if ok else ["floors"],
        f"layout has the {c.spec.floors} required floor(s)",
        f"layout has floors {floors}, the spec requires {c.spec.floors}",
        value=float(len(floors)),
        limit=float(c.spec.floors),
    )


def _required_rooms(c: _Ctx) -> ConstraintResult:
    placed: dict[str, int] = {}
    for t in c.room_types.values():
        placed[t] = placed.get(t, 0) + 1
    required = {r.room_type: r.count for r in c.spec.rooms}
    bad: list[str] = []
    notes: list[str] = []
    for t in sorted(set(placed) | set(required)):
        if placed.get(t, 0) != required.get(t, 0):
            bad.append(t)
            notes.append(f"{t}: placed {placed.get(t, 0)}, required {required.get(t, 0)}")
    return _hard(
        "HC_required_rooms",
        bad,
        "room counts match the spec",
        "room counts differ from the spec (" + "; ".join(notes) + ")",
    )


def _room_area(c: _Ctx) -> ConstraintResult:
    bad: list[str] = []
    notes: list[str] = []
    checked = 0
    for rid in sorted(c.rooms):
        req = c.spec.request_for(c.room_types[rid])
        if req is None:
            continue
        checked += 1
        a = area_m2(c.rooms[rid].rect, c.cs)
        if a < req.min_area_m2 - TOL or a > req.max_area_m2 + TOL:
            bad.append(rid)
            notes.append(f"{rid} {_f(a)} m2 outside {_f(req.min_area_m2)}-{_f(req.max_area_m2)}")
    return _hard(
        "HC_room_area",
        bad,
        f"{checked} room area(s) are within the requested ranges",
        "room areas outside the requested range: " + "; ".join(notes),
    )


def _inside_footprint(c: _Ctx) -> ConstraintResult:
    lay = c.layout
    bad: list[str] = []
    for rid, info in sorted(c.regions.items()):
        if not inside_grid(info.rect, lay.grid_w, lay.grid_d):
            bad.append(rid)
    notes = [f"{b} extends beyond the grid" for b in bad]
    width_ok = abs(lay.grid_w * c.cs - c.spec.footprint.width_m) <= c.cs / 2 + TOL
    depth_ok = abs(lay.grid_d * c.cs - c.spec.footprint.depth_m) <= c.cs / 2 + TOL
    if not (width_ok and depth_ok):
        bad.append("layout_grid")
        notes.append(
            f"grid is {_f(lay.grid_w * c.cs)} x {_f(lay.grid_d * c.cs)} m, spec footprint is "
            f"{_f(c.spec.footprint.width_m)} x {_f(c.spec.footprint.depth_m)} m"
        )
    return _hard(
        "HC_inside_footprint",
        bad,
        "everything lies inside the footprint grid, and the grid matches the spec",
        "outside the footprint: " + "; ".join(notes),
    )


def _no_overlap(c: _Ctx) -> ConstraintResult:
    items = sorted(c.regions.items())
    bad: list[str] = []
    notes: list[str] = []
    total = 0.0
    for i, (ida, a) in enumerate(items):
        for idb, b in items[i + 1 :]:
            if not set(a.floors) & set(b.floors):
                continue
            cells = overlap_cells(a.rect, b.rect)
            if cells:
                area = cells * c.cs * c.cs
                total += area
                bad += [ida, idb]
                notes.append(f"{ida} and {idb} overlap by {_f(area)} m2")
    return _hard(
        "HC_no_overlap",
        bad,
        "no rooms, corridors, stairs or lifts overlap",
        "overlapping: " + "; ".join(notes[:6]) + (" ..." if len(notes) > 6 else ""),
        value=total,
        limit=0.0,
    )


def _door_placement(c: _Ctx) -> ConstraintResult:
    bad = [d for d, _ in c.topo.invalid_doors]
    notes = [f"{d} {why}" for d, why in c.topo.invalid_doors]
    n = sum(len(p.doors) for p in c.layout.floors)
    return _hard(
        "HC_door_placement",
        bad,
        f"all {n} door(s) sit on the shared wall of the two regions they connect",
        "door placement does not pass: " + "; ".join(notes),
    )


def _exit_placement(c: _Ctx) -> ConstraintResult:
    bad = [e for e, _ in c.topo.invalid_exits]
    notes = [f"{e} {why}" for e, why in c.topo.invalid_exits]
    n = sum(len(p.exits) for p in c.layout.floors)
    return _hard(
        "HC_exit_placement",
        bad,
        f"all {n} exit(s) sit on the footprint boundary next to the region they serve",
        "exit placement does not pass: " + "; ".join(notes),
    )


def _connectivity(c: _Ctx) -> ConstraintResult:
    bad = [r for r in sorted(c.rooms) if c.topo.room_node[r] not in c.exit_dist]
    return _hard(
        "HC_connectivity",
        bad,
        f"all {len(c.rooms)} room(s) are connected to an exit",
        "rooms with no connection to any exit: " + ", ".join(bad),
    )


def _min_exit_count(c: _Ctx) -> ConstraintResult:
    floor0 = {n.node_id for n in c.topo.nodes if n.floor == 0}
    n = sum(1 for node in c.topo.exit_node.values() if node in floor0)
    limit = c.thresholds.min_exit_count.value
    return _hard(
        "HC_min_exit_count",
        [] if n >= limit else ["exits"],
        f"{n} valid exit(s) on the ground floor, at least {limit} configured",
        f"{n} valid exit(s) on the ground floor, at least {limit} configured",
        value=float(n),
        limit=float(limit),
    )


def _exit_width(c: _Ctx) -> ConstraintResult:
    limit = c.thresholds.min_exit_width_m.value
    widths = {
        e.exit_id: e.width_cells * c.cs
        for p in c.layout.floors
        for e in p.exits
        if e.exit_id in c.topo.exit_node
    }
    bad = [e for e, w in sorted(widths.items()) if w < limit - TOL]
    return _hard(
        "HC_exit_width",
        bad,
        f"exit width is at least {_f(limit)} m" if widths else "no valid exits to measure",
        f"exits narrower than {_f(limit)} m: " + ", ".join(bad),
        value=min(widths.values()) if widths else None,
        limit=limit,
    )


def _corridor_width(c: _Ctx) -> ConstraintResult:
    limit = c.thresholds.min_corridor_width_m.value
    widths = {
        cor.corridor_id: min(cor.rect.w, cor.rect.d) * c.cs
        for p in c.layout.floors
        for cor in p.corridors
    }
    bad = [k for k, w in sorted(widths.items()) if w < limit - TOL]
    return _hard(
        "HC_corridor_width",
        bad,
        f"corridor width is at least {_f(limit)} m" if widths else "no corridors to measure",
        f"corridors narrower than {_f(limit)} m: " + ", ".join(bad),
        value=min(widths.values()) if widths else None,
        limit=limit,
    )


def _door_width(c: _Ctx) -> ConstraintResult:
    # A spec may raise the limit but never lower it.
    limit = max(c.thresholds.min_door_width_m.value, c.spec.accessibility.min_door_width_m or 0.0)
    widths = {d.door_id: d.width_cells * c.cs for p in c.layout.floors for d in p.doors}
    bad = [k for k, w in sorted(widths.items()) if w < limit - TOL]
    return _hard(
        "HC_door_width",
        bad,
        f"door width is at least {_f(limit)} m" if widths else "no doors to measure",
        f"doors narrower than {_f(limit)} m: " + ", ".join(bad),
        value=min(widths.values()) if widths else None,
        limit=limit,
    )


def _travel_distance(c: _Ctx) -> ConstraintResult:
    limit = c.thresholds.max_travel_distance_m.value
    d = {r: c.exit_dist[c.topo.room_node[r]] for r in c.rooms if c.topo.room_node[r] in c.exit_dist}
    bad = [r for r, v in sorted(d.items()) if v > limit + TOL]
    worst = max(d.values()) if d else None
    return _hard(
        "HC_travel_distance",
        bad,
        f"walking distance to the nearest exit is at most {_f(limit)} m"
        + (f" (longest {_f(worst)} m)" if worst is not None else ""),
        f"walking distance to the nearest exit exceeds {_f(limit)} m for: "
        + ", ".join(f"{r} ({_f(d[r])} m)" for r in bad),
        value=worst,
        limit=limit,
    )


def _step_free(c: _Ctx) -> ConstraintResult | None:
    if not c.spec.accessibility.step_free_all_floors:
        return None
    bad = [
        r
        for r in sorted(c.rooms)
        if c.topo.room_node[r] in c.exit_dist and c.topo.room_node[r] not in c.exit_dist_accessible
    ]
    return _hard(
        "HC_step_free_access",
        bad,
        "every connected room has a step-free route to an exit",
        "rooms reachable only by stairs: " + ", ".join(bad),
    )


def _vertical_access(c: _Ctx) -> ConstraintResult | None:
    if c.spec.floors <= 1:
        return None
    need = set(range(c.spec.floors))
    bad: list[str] = []
    if not any(need <= set(s.floors) for s in c.layout.vertical_core.stairs):
        bad.append("stair_serving_all_floors")
    wants_lift = c.spec.accessibility.lift_required or c.spec.accessibility.step_free_all_floors
    if wants_lift and not any(
        lift.accessible and need <= set(lift.floors) for lift in c.layout.vertical_core.lifts
    ):
        bad.append("accessible_lift_serving_all_floors")
    return _hard(
        "HC_vertical_access",
        bad,
        "stairs (and an accessible lift where required) serve every floor",
        "missing vertical access: " + ", ".join(bad),
    )


def _incompatible(c: _Ctx) -> ConstraintResult | None:
    rels = [r for r in c.relations if r.type == RelationType.incompatible and r.hard]
    if not rels:
        return None
    bad: list[str] = []
    notes: list[str] = []
    smallest = INF
    limit_used = c.thresholds.incompatible_min_separation_m.value
    for rel in rels:
        need = rel.min_distance_m or c.thresholds.incompatible_min_separation_m.value
        limit_used = need
        for ra in members(c.room_types, rel.a):
            for rb in members(c.room_types, rel.b):
                if ra == rb:
                    continue
                d = c.room_distance(ra, rb)
                smallest = min(smallest, d)
                if d < need - TOL:
                    bad += [ra, rb]
                    notes.append(f"{ra} and {rb} are {_f(d)} m apart, {_f(need)} m configured")
    return _hard(
        "HC_incompatible_separation",
        bad,
        "incompatible room pairs are separated by the configured walking distance",
        "incompatible rooms too close: " + "; ".join(sorted(set(notes))),
        value=None if smallest == INF else smallest,
        limit=limit_used,
    )


def _hard_relations(c: _Ctx) -> ConstraintResult | None:
    rels = [r for r in c.relations if r.hard and r.type != RelationType.incompatible]
    if not rels:
        return None
    bad: list[str] = []
    notes: list[str] = []
    for rel in rels:
        A, B = members(c.room_types, rel.a), members(c.room_types, rel.b)
        if not A or not B:
            continue  # a missing room type is reported by HC_required_rooms
        for ra in A:
            partners = [rb for rb in B if rb != ra]
            if rel.type == RelationType.far:
                need = rel.min_distance_m or c.thresholds.far_reference_distance_m.value
                for rb in partners:
                    d = c.room_distance(ra, rb)
                    if d < need - TOL:
                        bad += [ra, rb]
                        notes.append(f"{ra} and {rb} are {_f(d)} m apart, {_f(need)} m wanted")
            else:
                if rel.type == RelationType.adjacent:
                    ok = any(c.contact_cells(ra, rb) >= c.adjacent_cells for rb in partners)
                else:
                    need = rel.max_distance_m or c.thresholds.near_reference_distance_m.value
                    ok = any(c.room_distance(ra, rb) <= need + TOL for rb in partners)
                if not ok:
                    bad.append(ra)
                    notes.append(f"{ra} has no {rel.type.value} {rel.b}")
    return _hard(
        "HC_hard_relations",
        bad,
        "all mandatory room relations are met",
        "mandatory relations not met: " + "; ".join(sorted(set(notes))),
    )


_RULES = (
    _floor_count,
    _required_rooms,
    _room_area,
    _inside_footprint,
    _no_overlap,
    _door_placement,
    _exit_placement,
    _connectivity,
    _min_exit_count,
    _exit_width,
    _corridor_width,
    _door_width,
    _travel_distance,
    _step_free,
    _vertical_access,
    _incompatible,
    _hard_relations,
)

# --------------------------------------------------------------------------- soft scores


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def _soft_adjacency(c: _Ctx) -> tuple[float, float | None, float | None]:
    """(cost, quality, mean related-room distance) for relations that are preferences."""
    num = den = 0.0
    dists: list[float] = []
    for rel in c.relations:
        if rel.hard:
            continue
        A, B = members(c.room_types, rel.a), members(c.room_types, rel.b)
        for ra in A:
            partners = [rb for rb in B if rb != ra]
            if not partners:
                continue
            qs = [
                pair_quality(
                    rel,
                    c.room_distance(ra, rb),
                    c.contact_cells(ra, rb),
                    c.adjacent_cells,
                    c.thresholds,
                )
                for rb in partners
            ]
            q = min(qs) if rel.type in (RelationType.far, RelationType.incompatible) else max(qs)
            num += rel.weight * q
            den += rel.weight
            if rel.type in (RelationType.adjacent, RelationType.near):
                finite = [c.room_distance(ra, rb) for rb in partners]
                finite = [x for x in finite if x != INF]
                if finite:
                    dists.append(min(finite))
    if den == 0:
        return 0.0, None, None
    quality = num / den
    return _clamp(1 - quality), quality, (sum(dists) / len(dists) if dists else None)


def _window_ok(c: _Ctx, rid: str, sides: set[Side] | None) -> bool:
    contacts = exterior_contacts(c.rooms[rid].rect, c.layout.grid_w, c.layout.grid_d)
    return any(
        length >= c.window_cells and (sides is None or side in sides)
        for side, length in contacts.items()
    )


def _soft_daylight(c: _Ctx) -> float:
    need = [
        r
        for r in sorted(c.rooms)
        if (req := c.spec.request_for(c.room_types[r])) is not None and req.needs_window
    ]
    if not need:
        return 0.0
    sides = set(c.spec.window_sides)
    return _clamp(1 - sum(_window_ok(c, r, sides) for r in need) / len(need))


def _soft_ventilation(c: _Ctx) -> float:
    need = [
        r
        for r in sorted(c.rooms)
        if (req := c.spec.request_for(c.room_types[r])) is not None
        and (req.needs_window or req.wet)
    ]
    if not need:
        return 0.0
    return _clamp(1 - sum(_window_ok(c, r, None) for r in need) / len(need))


def _energy_proxy(c: _Ctx) -> float:
    wall = sum(
        sum(exterior_contacts(room.rect, c.layout.grid_w, c.layout.grid_d).values()) * c.cs
        for room in c.rooms.values()
    )
    total = sum(area_m2(room.rect, c.cs) for room in c.rooms.values())
    return wall / total if total > 0 else 0.0


def _convertible(c: _Ctx, rid: str, target: str) -> bool:
    tdef = c.catalogue.get(target)
    if tdef is None:
        return False
    a = area_m2(c.rooms[rid].rect, c.cs)
    if not (tdef.min_area_m2 - TOL <= a <= tdef.max_area_m2 + TOL):
        return False
    if tdef.needs_window and not _window_ok(c, rid, set(c.spec.window_sides)):
        return False
    return True


def _adaptability(c: _Ctx) -> float | None:
    """Fraction of planned conversions the layout allows; None if nothing is flexible."""
    if c.spec.future_changes:
        scores = []
        for ch in c.spec.future_changes:
            rooms = members(c.room_types, ch.room_ref)
            ok = sum(1 for r in rooms if _convertible(c, r, ch.to_type))
            scores.append(min(ok, ch.count) / ch.count)
        return sum(scores) / len(scores)
    flex = [
        (r, req.flexible_to)
        for r in sorted(c.rooms)
        if (req := c.spec.request_for(c.room_types[r])) is not None and req.flexible_to
    ]
    if not flex:
        return None
    return sum(any(_convertible(c, r, t) for t in targets) for r, targets in flex) / len(flex)


def usable_area_ratio(layout: Layout, spec: BuildingSpec) -> float:
    """Area of functional rooms divided by gross floor area (footprint times floors)."""
    cs = layout.cell_size_m
    functional = 0.0
    for plan in layout.floors:
        for room in plan.rooms:
            req = spec.request_for(room.room_type)
            if req is not None and req.functional:
                functional += area_m2(room.rect, cs)
    gross = spec.footprint.width_m * spec.footprint.depth_m * spec.floors
    return functional / gross if gross > 0 else 0.0


# --------------------------------------------------------------------------- grouping


def group_violations(results: tuple[ConstraintResult, ...]) -> tuple[ConflictSet, ...]:
    """Group failing checks that involve the same entities, with a plain explanation."""
    failing = [r for r in results if not r.satisfied]
    parent = list(range(len(failing)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(failing):
        for j in range(i + 1, len(failing)):
            if set(a.entity_ids) & set(failing[j].entity_ids):
                parent[find(i)] = find(j)
    groups: dict[int, list[ConstraintResult]] = {}
    for i, r in enumerate(failing):
        groups.setdefault(find(i), []).append(r)
    out = []
    for g in sorted(groups.values(), key=lambda g: g[0].constraint_id):
        ids = tuple(sorted(r.constraint_id for r in g))
        entities = sorted({e for r in g for e in r.entity_ids})
        shared = f" for {', '.join(entities)}" if entities else ""
        text = f"These configured checks do not pass{shared}: {', '.join(ids)}. " + " ".join(
            r.detail + "." for r in sorted(g, key=lambda r: r.constraint_id)
        )
        out.append(ConflictSet(constraint_ids=ids, explanation=text))
    return tuple(out)


# --------------------------------------------------------------------------- entry points


def analyse_layout(
    layout: Layout,
    spec: BuildingSpec,
    thresholds: Thresholds,
    catalogue: RoomCatalogue,
    options: CheckerOptions | None = None,
) -> LayoutAnalysis:
    opts = options or CheckerOptions()
    rooms = {r.room_id: r for p in layout.floors for r in p.rooms}
    loads = {
        rid: (
            spec.request_for(r.room_type).occupant_load  # type: ignore[union-attr]
            if spec.request_for(r.room_type)
            else 0
        )
        for rid, r in rooms.items()
    }
    topo = build_topology(layout, loads, opts.graph)
    adj = topo.adjacency()
    exit_nodes = list(topo.exit_node.values())
    ctx = _Ctx(
        layout=layout,
        spec=spec,
        thresholds=thresholds,
        catalogue=catalogue,
        options=opts,
        cs=layout.cell_size_m,
        rooms=rooms,
        regions=region_index(layout),
        topo=topo,
        relations=effective_relations(spec, catalogue, thresholds),
        exit_dist=shortest_distances(adj, exit_nodes),
        exit_dist_accessible=shortest_distances(topo.adjacency(True), exit_nodes),
        room_types={rid: r.room_type for rid, r in rooms.items()},
    )

    hard = tuple(r for rule in _RULES if (r := rule(ctx)) is not None)
    adj_cost, adj_quality, related_dist = _soft_adjacency(ctx)
    energy = _energy_proxy(ctx)
    adapt = _adaptability(ctx)
    usable = usable_area_ratio(layout, spec)
    prefs = spec.preferences
    soft = (
        SoftScore(name="adjacency", cost=adj_cost, weight=prefs.adjacency_weight),
        SoftScore(name="daylight", cost=_soft_daylight(ctx), weight=prefs.daylight_weight),
        SoftScore(name="ventilation", cost=_soft_ventilation(ctx), weight=prefs.ventilation_weight),
        SoftScore(
            name="energy",
            cost=_clamp(energy / opts.energy_proxy_reference),
            weight=prefs.energy_weight,
        ),
        SoftScore(
            name="adaptability",
            cost=0.0 if adapt is None else _clamp(1 - adapt),
            weight=prefs.adaptability_weight,
        ),
        SoftScore(name="usable_area", cost=_clamp(1 - usable), weight=prefs.usable_area_weight),
    )
    verification = Verification(
        status=VerificationStatus.checked, hard=hard, soft=soft, conflicts=group_violations(hard)
    )

    n_rooms = len(rooms)
    accessible = sum(1 for r in rooms if topo.room_node[r] in ctx.exit_dist_accessible)
    quality: dict[str, float] = {
        "constraint_satisfaction_rate": sum(r.satisfied for r in hard) / len(hard),
        "usable_area_ratio": usable,
        "accessibility_coverage": accessible / n_rooms if n_rooms else 0.0,
        "energy_proxy": energy,
    }
    if adj_quality is not None:
        quality["adjacency_quality"] = adj_quality
    if related_dist is not None:
        quality["avg_related_room_distance_m"] = related_dist
    if adapt is not None:
        quality["adaptability"] = adapt
    dist_exit = {
        r: ctx.exit_dist[topo.room_node[r]] for r in rooms if topo.room_node[r] in ctx.exit_dist
    }
    return LayoutAnalysis(verification, quality, topo, dist_exit)


def check_layout(
    layout: Layout,
    spec: BuildingSpec,
    thresholds: Thresholds,
    catalogue: RoomCatalogue,
    options: CheckerOptions | None = None,
) -> Verification:
    """The constraint report for a layout."""
    return analyse_layout(layout, spec, thresholds, catalogue, options).verification


def verify_layout(
    layout: Layout,
    spec: BuildingSpec,
    thresholds: Thresholds,
    catalogue: RoomCatalogue,
    options: CheckerOptions | None = None,
) -> Layout:
    """A copy of the layout with its verification attached (hash cleared)."""
    v = check_layout(layout, spec, thresholds, catalogue, options)
    return layout.model_copy(update={"verification": v, "content_hash": None})


_METRIC_UNITS = {"avg_related_room_distance_m": "m", "energy_proxy": "m/m2"}


def layout_metrics(analysis: LayoutAnalysis, layout: Layout) -> MetricSet:
    """Layout-scope MetricSet derived from an analysis (CSR, usable area, adjacency, ...)."""
    values = tuple(
        MetricValue(metric_id=k, value=v, unit=_METRIC_UNITS.get(k, "fraction"))
        for k, v in sorted(analysis.quality.items())
    )
    return MetricSet(
        scope=MetricScope.layout, layout_geometry_hash=layout.geometry_hash, values=values
    )
