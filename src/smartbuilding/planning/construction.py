"""Construction planner: STRIPS-style task decomposition and critical-path schedule.

Given a Layout, this module decomposes it into ConstructionTask objects,
builds the dependency graph, and computes the critical-path schedule.

Task types (from ConstructionTaskType enum):
  structure            - one per layout (must come first)
  prepare_space        - one per floor
  build_walls          - one per room and corridor
  install_door         - one per door (depends on both adjacent wall tasks)
  place_utilities      - one per room
  install_safety_equipment - one per exit
  build_corridor       - one per corridor
  build_exit           - one per exit

Critical path is computed with a forward/backward pass.
"""

from __future__ import annotations

import re

from smartbuilding.core.construction import (
    ConstructionPlan,
    ConstructionTask,
    ScheduledTask,
)
from smartbuilding.core.layout import Layout
from smartbuilding.core.primitives import ConstructionTaskType

# Sourcing: team assumptions (placeholder values documented in specification)
_TASK_DURATIONS: dict[ConstructionTaskType, float] = {
    ConstructionTaskType.structure: 5.0,
    ConstructionTaskType.prepare_space: 2.0,
    ConstructionTaskType.build_walls: 3.0,
    ConstructionTaskType.build_corridor: 2.0,
    ConstructionTaskType.install_door: 1.0,
    ConstructionTaskType.build_exit: 2.0,
    ConstructionTaskType.install_safety_equipment: 1.0,
    ConstructionTaskType.place_utilities: 2.0,
}

_TASK_COSTS: dict[ConstructionTaskType, float] = {
    ConstructionTaskType.structure: 500.0,
    ConstructionTaskType.prepare_space: 100.0,
    ConstructionTaskType.build_walls: 200.0,
    ConstructionTaskType.build_corridor: 150.0,
    ConstructionTaskType.install_door: 50.0,
    ConstructionTaskType.build_exit: 150.0,
    ConstructionTaskType.install_safety_equipment: 80.0,
    ConstructionTaskType.place_utilities: 120.0,
}


def _clean_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]", "_", raw.lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def plan_construction(
    layout: Layout,
    *,
    seed: int = 0,
) -> ConstructionPlan:
    """Decompose layout into construction tasks, compute dependencies, and schedule critical path."""
    del seed  # Deterministic Kahn's sort handles reproducibility

    tasks: list[ConstructionTask] = []
    region_task_map: dict[str, str] = {}

    # 1. Structure task (one per layout, must come first)
    structure_task = ConstructionTask(
        task_id="T_structure",
        task_type=ConstructionTaskType.structure,
        target_id=layout.layout_id,
        duration_days=_TASK_DURATIONS[ConstructionTaskType.structure],
        cost_units=_TASK_COSTS[ConstructionTaskType.structure],
        preconditions=(),
        effects=("structure_built",),
        depends_on=(),
    )
    tasks.append(structure_task)

    # 2. Per-floor tasks
    for plan in sorted(layout.floors, key=lambda p: p.floor):
        f_id = plan.floor
        prep_task_id = f"T_prepare_f{f_id}"
        prep_task = ConstructionTask(
            task_id=prep_task_id,
            task_type=ConstructionTaskType.prepare_space,
            target_id=f"floor_{f_id}",
            duration_days=_TASK_DURATIONS[ConstructionTaskType.prepare_space],
            cost_units=_TASK_COSTS[ConstructionTaskType.prepare_space],
            preconditions=("structure_built",),
            effects=(f"space_prepared_f{f_id}",),
            depends_on=("T_structure",),
        )
        tasks.append(prep_task)

        # Rooms: build_walls and place_utilities
        for room in sorted(plan.rooms, key=lambda r: r.room_id):
            rid_clean = _clean_id(room.room_id)
            wall_task_id = f"T_build_walls_{rid_clean}"
            wall_task = ConstructionTask(
                task_id=wall_task_id,
                task_type=ConstructionTaskType.build_walls,
                target_id=room.room_id,
                duration_days=_TASK_DURATIONS[ConstructionTaskType.build_walls],
                cost_units=_TASK_COSTS[ConstructionTaskType.build_walls],
                preconditions=(f"space_prepared_f{f_id}",),
                effects=(f"walls_built_{room.room_id}",),
                depends_on=(prep_task_id,),
            )
            tasks.append(wall_task)
            region_task_map[room.room_id] = wall_task_id

            util_task_id = f"T_place_utilities_{rid_clean}"
            util_task = ConstructionTask(
                task_id=util_task_id,
                task_type=ConstructionTaskType.place_utilities,
                target_id=room.room_id,
                duration_days=_TASK_DURATIONS[ConstructionTaskType.place_utilities],
                cost_units=_TASK_COSTS[ConstructionTaskType.place_utilities],
                preconditions=(f"walls_built_{room.room_id}",),
                effects=(f"utilities_placed_{room.room_id}",),
                depends_on=(wall_task_id,),
            )
            tasks.append(util_task)

        # Corridors: build_corridor
        for corridor in sorted(plan.corridors, key=lambda c: c.corridor_id):
            cid_clean = _clean_id(corridor.corridor_id)
            corr_task_id = f"T_build_corridor_{cid_clean}"
            corr_task = ConstructionTask(
                task_id=corr_task_id,
                task_type=ConstructionTaskType.build_corridor,
                target_id=corridor.corridor_id,
                duration_days=_TASK_DURATIONS[ConstructionTaskType.build_corridor],
                cost_units=_TASK_COSTS[ConstructionTaskType.build_corridor],
                preconditions=(f"space_prepared_f{f_id}",),
                effects=(f"corridor_built_{corridor.corridor_id}",),
                depends_on=(prep_task_id,),
            )
            tasks.append(corr_task)
            region_task_map[corridor.corridor_id] = corr_task_id

        # Doors: install_door (depends on both adjacent wall / corridor tasks)
        for door in sorted(plan.doors, key=lambda d: d.door_id):
            did_clean = _clean_id(door.door_id)
            door_task_id = f"T_install_door_{did_clean}"
            deps: list[str] = []
            if door.a in region_task_map:
                deps.append(region_task_map[door.a])
            if door.b in region_task_map:
                deps.append(region_task_map[door.b])
            if not deps:
                deps.append(prep_task_id)
            door_deps = tuple(sorted(set(deps)))

            door_task = ConstructionTask(
                task_id=door_task_id,
                task_type=ConstructionTaskType.install_door,
                target_id=door.door_id,
                duration_days=_TASK_DURATIONS[ConstructionTaskType.install_door],
                cost_units=_TASK_COSTS[ConstructionTaskType.install_door],
                preconditions=tuple(f"ready_for_{dep}" for dep in door_deps),
                effects=(f"door_installed_{door.door_id}",),
                depends_on=door_deps,
            )
            tasks.append(door_task)

        # Exits: build_exit and install_safety_equipment
        for exit_obj in sorted(plan.exits, key=lambda e: e.exit_id):
            eid_clean = _clean_id(exit_obj.exit_id)
            exit_task_id = f"T_build_exit_{eid_clean}"
            exit_dep = region_task_map.get(exit_obj.connects_to, prep_task_id)
            exit_task = ConstructionTask(
                task_id=exit_task_id,
                task_type=ConstructionTaskType.build_exit,
                target_id=exit_obj.exit_id,
                duration_days=_TASK_DURATIONS[ConstructionTaskType.build_exit],
                cost_units=_TASK_COSTS[ConstructionTaskType.build_exit],
                preconditions=(f"ready_for_{exit_dep}",),
                effects=(f"exit_built_{exit_obj.exit_id}",),
                depends_on=(exit_dep,),
            )
            tasks.append(exit_task)

            safety_task_id = f"T_install_safety_equipment_{eid_clean}"
            safety_task = ConstructionTask(
                task_id=safety_task_id,
                task_type=ConstructionTaskType.install_safety_equipment,
                target_id=exit_obj.exit_id,
                duration_days=_TASK_DURATIONS[ConstructionTaskType.install_safety_equipment],
                cost_units=_TASK_COSTS[ConstructionTaskType.install_safety_equipment],
                preconditions=(f"exit_built_{exit_obj.exit_id}",),
                effects=(f"safety_equipment_installed_{exit_obj.exit_id}",),
                depends_on=(exit_task_id,),
            )
            tasks.append(safety_task)

    # 3. Topological sort (Kahn's algorithm)
    task_map = {t.task_id: t for t in tasks}
    children: dict[str, list[str]] = {t.task_id: [] for t in tasks}
    for t in tasks:
        for d in t.depends_on:
            children[d].append(t.task_id)

    indegree = {t.task_id: len(t.depends_on) for t in tasks}
    ready = sorted([tid for tid, n in indegree.items() if n == 0])
    topo_order: list[str] = []

    while ready:
        curr = ready.pop(0)
        topo_order.append(curr)
        for child in sorted(children[curr]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()

    if len(topo_order) != len(tasks):
        raise ValueError("Dependency cycle detected among construction tasks")

    # 4. Forward pass: earliest start and earliest finish
    earliest_start: dict[str, float] = {}
    earliest_end: dict[str, float] = {}
    for tid in topo_order:
        task = task_map[tid]
        if not task.depends_on:
            es = 0.0
        else:
            es = max(earliest_end[d] for d in task.depends_on)
        earliest_start[tid] = es
        earliest_end[tid] = es + task.duration_days

    project_duration = max(earliest_end.values()) if earliest_end else 0.0

    # 5. Backward pass: latest finish and latest start
    latest_start: dict[str, float] = {}
    latest_end: dict[str, float] = {}
    for tid in reversed(topo_order):
        task = task_map[tid]
        child_ids = children[tid]
        if not child_ids:
            lf = project_duration
        else:
            lf = min(latest_start[c] for c in child_ids)
        latest_end[tid] = lf
        latest_start[tid] = lf - task.duration_days

    # 6. Critical path: tasks with zero slack (earliest_start == latest_start)
    critical_path = tuple(
        tid for tid in topo_order if abs(earliest_start[tid] - latest_start[tid]) < 1e-6
    )

    # 7. Scheduled tasks
    schedule = tuple(
        ScheduledTask(
            task_id=tid,
            start_day=earliest_start[tid],
            end_day=earliest_end[tid],
        )
        for tid in topo_order
    )

    total_cost = sum(t.cost_units for t in tasks)

    # Valid Name plan_id (matches r"^[A-Za-z][A-Za-z0-9_\-]*$")
    plan_id = re.sub(r"[^A-Za-z0-9_\-]", "_", f"CP_{layout.layout_id}")

    return ConstructionPlan(
        plan_id=plan_id,
        layout_geometry_hash=layout.geometry_hash,
        tasks=tuple(tasks),
        schedule=schedule,
        critical_path=critical_path,
        total_duration_days=project_duration,
        total_cost_units=total_cost,
    )
