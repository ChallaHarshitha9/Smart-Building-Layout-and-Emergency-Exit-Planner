"""Scene data for the 3D interface.

Everything here comes from the real modules: the CSP solver, the rule checker, the construction
planner, the simulator, the Bayes belief filter and the B1/B2 planners. Nothing is canned.
The front end only draws what this module returns.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from smartbuilding.core.bundle import ConfigBundle
from smartbuilding.core.decision import Baseline
from smartbuilding.core.layout import Layout
from smartbuilding.core.loaders import load_model
from smartbuilding.core.primitives import EventType, HazardVariant, SelectorMode
from smartbuilding.core.scenario import EventSpec, IgnitionSpec, Scenario, Selector
from smartbuilding.core.spec import BuildingSpec
from smartbuilding.eval.episode import make_run, navgraph_for, resolve_for, room_loads, summarise
from smartbuilding.layout.checker import check_layout
from smartbuilding.layout.geometry import (
    door_center_m,
    exit_center_m,
    exit_side,
    shared_boundary,
)
from smartbuilding.layout.graph import build_navgraph, build_topology, region_index
from smartbuilding.planning.construction import plan_construction
from smartbuilding.solver.csp import solve

METHODS = ("B1", "B2", "B4", "B5", "B3")


def list_specs(examples_dir: Path) -> list[dict[str, Any]]:
    """Example specs with labels taken from the data, never hard-coded."""
    out = []
    for f in sorted(examples_dir.glob("*_spec.json")):
        spec = load_model(f, BuildingSpec)
        out.append(
            {
                "file": f.name,
                "id": spec.spec_id,
                "type": spec.building_type.value,
                "label": f"{spec.spec_id.replace('_', ' ').title()} "
                f"({spec.footprint.width_m:g} m x {spec.footprint.depth_m:g} m, "
                f"{sum(r.count for r in spec.rooms)} rooms, {spec.floors} floor"
                f"{'s' if spec.floors > 1 else ''})",
                "footprint": [spec.footprint.width_m, spec.footprint.depth_m],
                "floors": spec.floors,
                "occupancy": spec.occupancy.total,
                "rooms": [{"type": r.room_type, "count": r.count} for r in spec.rooms],
            }
        )
    return out


def layout_scene(layout: Layout, spec: BuildingSpec, bundle: ConfigBundle) -> dict[str, Any]:
    """Geometry (metres), navigation graph and rule-check results for one layout."""
    cs = layout.cell_size_m
    regions = region_index(layout)
    loads = room_loads(layout, spec)
    topo = build_topology(layout, loads)
    graph = build_navgraph(layout, loads)

    rooms, corridors, doors, exits = [], [], [], []
    for plan in layout.floors:
        for r in plan.rooms:
            req = spec.request_for(r.room_type)
            cat = bundle.catalogue.get(r.room_type)
            rooms.append(
                {
                    "id": r.room_id,
                    "type": r.room_type,
                    "label": cat.label if cat else r.room_type,
                    "floor": plan.floor,
                    "x": r.rect.x * cs,
                    "y": r.rect.y * cs,
                    "w": r.rect.w * cs,
                    "d": r.rect.d * cs,
                    "area": round(r.rect.area_cells * cs * cs, 2),
                    "load": loads.get(r.room_id, 0),
                    "needs_window": bool(req and req.needs_window),
                    "functional": bool(req and req.functional),
                }
            )
        for c in plan.corridors:
            corridors.append(
                {
                    "id": c.corridor_id,
                    "floor": plan.floor,
                    "x": c.rect.x * cs,
                    "y": c.rect.y * cs,
                    "w": c.rect.w * cs,
                    "d": c.rect.d * cs,
                }
            )
        for d in plan.doors:
            ra, rb = regions.get(d.a), regions.get(d.b)
            boundary = shared_boundary(ra.rect, rb.rect) if ra and rb else None
            if boundary is None:
                continue
            cx, cy = door_center_m(d, boundary, cs)
            doors.append(
                {
                    "id": d.door_id,
                    "floor": plan.floor,
                    "x": cx,
                    "y": cy,
                    "width": d.width_cells * cs,
                    "horizontal": boundary.orientation == "h",
                }
            )
        for e in plan.exits:
            side = exit_side(e, layout.grid_w, layout.grid_d)
            if side is None:
                continue
            cx, cy = exit_center_m(e, side, cs)
            node = topo.exit_node.get(e.exit_id)
            exits.append(
                {
                    "id": e.exit_id,
                    "floor": plan.floor,
                    "x": cx,
                    "y": cy,
                    "width": e.width_cells * cs,
                    "side": side.value,
                    "node": node,
                }
            )

    ver = check_layout(layout, spec, bundle.thresholds, bundle.catalogue)
    hard = [
        {
            "id": r.constraint_id,
            "ok": r.satisfied,
            "detail": r.detail,
            "entities": list(r.entity_ids),
            "value": r.value,
            "limit": r.limit,
        }
        for r in ver.hard
    ]
    return {
        "layout_id": layout.layout_id,
        "size": [layout.grid_w * cs, layout.grid_d * cs],
        "floors": len(layout.floors),
        "rooms": rooms,
        "corridors": corridors,
        "doors": doors,
        "exits": exits,
        "nodes": [
            {
                "id": n.node_id,
                "kind": n.kind.value,
                "x": n.x_m,
                "y": n.y_m,
                "floor": n.floor,
                "region": n.region_id,
            }
            for n in graph.nodes
        ],
        "edges": [
            {"u": e.u, "v": e.v, "len": round(e.length_m, 2), "door": e.door_id}
            for e in graph.edges
        ],
        "hard": hard,
        "soft": [{"name": s.name, "cost": round(s.cost, 3), "weight": s.weight} for s in ver.soft],
        "feasible": all(bool(h["ok"]) for h in hard),
        "csr": round(sum(bool(h["ok"]) for h in hard) / max(1, len(hard)), 3),
        "conflicts": [c.explanation for c in ver.conflicts],
    }


def solve_scenes(
    spec: BuildingSpec,
    bundle: ConfigBundle,
    seed: int,
    max_layouts: int,
    time_limit_s: float = 20.0,
) -> tuple[list[tuple[Layout, dict[str, Any]]], dict[str, Any]]:
    out = solve(
        spec,
        bundle.catalogue,
        bundle.thresholds,
        seed=seed,
        max_layouts=max_layouts,
        time_limit_s=time_limit_s,
    )
    scenes = [(r.layout, layout_scene(r.layout, spec, bundle)) for r in out.results]
    info = {
        "feasible": out.feasible_count,
        "infeasible": out.infeasible_count,
        "conflict": out.conflict_explanation,
        "time_s": round(out.solve_time_s, 2),
        "seed": out.seed,
    }
    return scenes, info


def construction_scene(layout: Layout) -> dict[str, Any]:
    plan = plan_construction(layout)
    sched = {s.task_id: s for s in plan.schedule}
    critical = set(plan.critical_path)
    tasks = []
    for t in plan.tasks:
        s = sched.get(t.task_id)
        tasks.append(
            {
                "id": t.task_id,
                "type": t.task_type.value,
                "target": t.target_id,
                "start": s.start_day if s else 0.0,
                "end": s.end_day if s else t.duration_days,
                "cost": t.cost_units,
                "critical": t.task_id in critical,
                "depends_on": list(t.depends_on),
            }
        )
    return {
        "tasks": tasks,
        "total_days": plan.total_duration_days,
        "total_cost": plan.total_cost_units,
    }


# ------------------------------------------------------------------ geometry for smoke cells


def _cells(scene: dict[str, Any]) -> dict[str, dict[str, float]]:
    """A rectangle (metres) for every NavGraph node: where its smoke is drawn."""
    rooms = {r["id"]: r for r in scene["rooms"]}
    corr = {c["id"]: c for c in scene["corridors"]}
    cells: dict[str, dict[str, float]] = {}
    by_corridor: dict[str, list[dict[str, Any]]] = {}
    for n in scene["nodes"]:
        if n["kind"] == "room" and n["region"] in rooms:
            r = rooms[n["region"]]
            cells[n["id"]] = {
                "x": r["x"],
                "y": r["y"],
                "w": r["w"],
                "d": r["d"],
                "floor": r["floor"],
            }
        elif n["kind"] == "corridor" and n["region"] in corr:
            by_corridor.setdefault(n["region"], []).append(n)
        elif n["kind"] == "exit":
            ex = next((e for e in scene["exits"] if e["id"] == n["region"]), None)
            half = 0.6
            cells[n["id"]] = (
                {
                    "x": n["x"] - half,
                    "y": n["y"] - half,
                    "w": 2 * half,
                    "d": 2 * half,
                    "floor": n["floor"],
                }
                if ex
                else {}
            )
    for cid, nodes in by_corridor.items():
        c = corr[cid]
        horizontal = c["w"] >= c["d"]
        key = "x" if horizontal else "y"
        nodes.sort(key=lambda n: n[key])
        lo, hi = (c["x"], c["x"] + c["w"]) if horizontal else (c["y"], c["y"] + c["d"])
        cuts = [lo] + [(a[key] + b[key]) / 2 for a, b in zip(nodes, nodes[1:], strict=False)] + [hi]
        for i, n in enumerate(nodes):
            a, b = cuts[i], cuts[i + 1]
            cells[n["id"]] = (
                {"x": a, "y": c["y"], "w": b - a, "d": c["d"], "floor": c["floor"]}
                if horizontal
                else {"x": c["x"], "y": a, "w": c["w"], "d": b - a, "floor": c["floor"]}
            )
    return cells


def dose_levels(bundle: ConfigBundle) -> dict[str, float]:
    d = bundle.run.dose
    return {"alert": d.alert_min, "critical": d.critical_min, "severe": d.severe_min}


# ------------------------------------------------------------------ simulation

_EVENT_TYPES = {"exit_blocked", "detector_fail", "second_fire", "door_stuck", "lift_out"}


def build_scenario(
    bundle: ConfigBundle,
    *,
    ignition_room: str,
    events: list[dict[str, Any]],
    period: str,
    hazard_variant: str,
    seed: int,
    horizon_s: float,
) -> Scenario:
    """A scenario from user choices, based on the 'normal fire' family so groups follow the profile."""
    base = next(
        s for s in bundle.scenarios().scenarios if s.scenario_id.startswith("SC_normal_fire_")
    )
    evs = []
    for e in events:
        if e["type"] not in _EVENT_TYPES:
            raise ValueError(f"unknown event type {e['type']!r}")
        evs.append(
            EventSpec(
                type=EventType(e["type"]),
                time_s=float(e.get("time", 0)),
                target=Selector(mode=SelectorMode.explicit_id, ref=str(e["target"])),
            )
        )
    return base.model_copy(
        update={
            "ignition": IgnitionSpec(
                where=Selector(mode=SelectorMode.explicit_id, ref=ignition_room), time_s=0.0
            ),
            "events": tuple(evs),
            "occupancy_period": period,
            "hazard_variant": HazardVariant(hazard_variant),
            "seed": seed,
            "horizon_s": horizon_s,
        }
    )


class _Recorder:
    """Wraps a controller to remember the routes it chose at each decision tick (for drawing)."""

    def __init__(self, inner: Any, index: dict[str, int]) -> None:
        self.inner, self.index = inner, index
        self.routes: dict[int, dict[str, list[int]]] = {}

    def observe(self, t_s: float, observations: Any) -> None:
        self.inner.observe(t_s, observations)

    def decide(self, public: Any) -> Any:
        plan = self.inner.decide(public)
        if plan is not None:
            self.routes[public.tick] = {
                d.group_id: [self.index[n] for n in d.route.nodes if n in self.index]
                for d in plan.decisions
            }
        return plan


def simulate_all(
    layout: Layout,
    spec: BuildingSpec,
    bundle: ConfigBundle,
    scene: dict[str, Any],
    *,
    ignition_room: str,
    events: list[dict[str, Any]],
    methods: tuple[str, ...] = METHODS,
    period: str = "typical",
    hazard_variant: str = "A",
    seed: int = 42,
    horizon_s: float = 300.0,
) -> dict[str, Any]:
    """Run every method on the same scenario and return per-tick frames for drawing."""
    graph = navgraph_for(layout, spec)
    index = {n.node_id: i for i, n in enumerate(graph.nodes)}
    sc = build_scenario(
        bundle,
        ignition_room=ignition_room,
        events=events,
        period=period,
        hazard_variant=hazard_variant,
        seed=seed,
        horizon_s=horizon_s,
    )
    resolved = resolve_for(bundle, sc, graph, spec)
    out: dict[str, Any] = {"methods": {}, "tick_s": bundle.run.sim.tick_s, "seed": seed}
    group_meta = None
    for m in methods:
        sim, inner = make_run(bundle, graph, resolved, spec, Baseline(m))
        rec = _Recorder(inner, index)
        tracker = getattr(inner, "tracker", None)
        has_belief = hasattr(tracker, "model") and hasattr(tracker, "belief")
        nodes = {n.node_id: n for n in graph.nodes}

        def frame(
            st: Any, trk: Any = tracker, hb: bool = has_belief, nd: Any = nodes, sm: Any = sim
        ) -> dict[str, Any]:
            g = []
            for x in st.groups:
                n0 = nd[x.node]
                px, py, fl = n0.x_m, n0.y_m, n0.floor
                if x.nxt is not None:
                    n1 = nd[x.nxt]
                    f = min(1.0, x.progress_m / max(1e-6, sm.edge[(x.node, x.nxt)].length_m))
                    px, py = px + (n1.x_m - px) * f, py + (n1.y_m - py) * f
                g.append(
                    [
                        round(px, 2),
                        round(py, 2),
                        fl,
                        round(x.dose, 2),
                        int(x.evacuated),
                        int(x.waiting),
                    ]
                )
            fr: dict[str, Any] = {
                "t": st.t_s,
                "g": g,
                "s": [round(st.smoke[n.node_id] * 100) for n in graph.nodes],
                "f": [index[n] for n in sorted(st.fire)],
                "x": sorted(index[n] for n in st.exits_blocked),
            }
            if hb:
                b = trk.belief()
                fr["b"] = [round((1 - b.node_belief[n.node_id][0]) * 100) for n in graph.nodes]
                fr["e"] = {k: round(v * 100) for k, v in b.exit_avail.items()}
            return fr

        frames = [frame(sim.reset())]
        pending: list[Any] = []

        def record(st: Any, _obs: Any, acc: list[Any] = pending, fr: Any = frame) -> None:
            acc.append(fr(st))

        res = sim.run(rec, record)
        for fr in pending:
            r = rec.routes.get(round(fr["t"] / sim.sim_config.tick_s) - 1)
            if r is not None:
                fr["r"] = r
            frames.append(fr)
        summary = asdict(summarise(res, bundle))
        out["methods"][m] = {"frames": frames, "summary": summary}
        if group_meta is None:
            group_meta = [
                {"id": g.group_id, "size": g.size, "template": g.template_id, "start": g.start_node}
                for g in resolved.groups
            ]
    out["groups"] = group_meta
    out["ignition"] = ignition_room
    out["events"] = events
    return out
