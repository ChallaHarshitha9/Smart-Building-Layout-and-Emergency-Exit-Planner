"""Corridor-template CSP solver.

Formulation
-----------
Variables:  The room requests from the BuildingSpec. Each request specifies a
            room_type and a count; each instance is a separate variable.
Domain:     width in cells, computed from the request's area bounds and the
            corridor depth.  Width ranges from ceil(min_area / (depth * cell)) to
            floor(max_area / (depth * cell)).  If the range is empty the
            instance is flagged as infeasible.
Constraints (hard): total width ≤ grid width; HC_required_rooms; HC_room_area;
            HC_travel_distance; HC_connectivity (all structural, not overlap
            since the template guarantees no overlap).
Soft:       adjacency, daylight, energy (from checker.analyse_layout).

Algorithm
---------
1. Enumerate all room type instances (e.g. 6 classrooms → 6 variables).
2. Split them into south and north rows (round-robin by type, sorted for
   determinism).
3. For each row, try widths in ascending order (most-constrained first).
4. Backtrack if the sum of assigned widths would exceed the grid.
5. On a complete assignment call assemble_corridor_layout and analyse_layout.
6. Keep at most max_layouts feasible layouts (sorted by total soft cost ASC).
7. If no feasible layout is found, call group_violations on the last attempt
   and format a conflict explanation.
"""

from __future__ import annotations

import math
import time

import numpy as np

from smartbuilding.core.config import RoomCatalogue, Thresholds
from smartbuilding.core.layout import LayoutProvenance
from smartbuilding.core.spec import BuildingSpec, RoomRequest
from smartbuilding.layout.checker import (
    CheckerOptions,
    LayoutAnalysis,
    analyse_layout,
    group_violations,
    layout_metrics,
)
from smartbuilding.layout.template import (
    SlotRoom,
    assemble_corridor_layout,
)
from smartbuilding.solver.result import SolverOutput, SolverResult


def solve(
    spec: BuildingSpec,
    catalogue: RoomCatalogue,
    thresholds: Thresholds,
    *,
    seed: int = 0,
    max_layouts: int = 5,
    time_limit_s: float = 60.0,
    checker_options: CheckerOptions | None = None,
) -> SolverOutput:
    """Solve the corridor-template CSP for the given spec and catalogue."""
    start_time = time.monotonic()
    _rng = np.random.default_rng(seed)

    cell_size_m = 0.5
    grid_w = round(spec.footprint.width_m / cell_size_m)
    grid_d = round(spec.footprint.depth_m / cell_size_m)
    corridor_width_cells = 4

    south_depth_cells = (grid_d - corridor_width_cells) // 2
    north_depth_cells = grid_d - corridor_width_cells - south_depth_cells
    south_depth_m = south_depth_cells * cell_size_m
    north_depth_m = north_depth_cells * cell_size_m

    spec_hash = spec.compute_hash() if hasattr(spec, "compute_hash") else "a" * 64

    # The corridor template is a single-floor layout generator.
    eff_spec = spec
    # A one-floor template can never satisfy a multi-floor request: report that after the first
    # candidates instead of searching until the time limit.
    cannot_satisfy = spec.floors > 1
    if spec.floors > 1:
        eff_spec = spec.model_copy(update={"floors": 1, "content_hash": None})

    # 1. Enumerate all room type instances, sorted by room_type then instance number
    instances: list[tuple[str, str, RoomRequest]] = []
    for req in sorted(eff_spec.rooms, key=lambda r: r.room_type):
        for inst_id in req.instance_ids():
            instances.append((inst_id, req.room_type, req))

    # 2. Split into south (even indices) and north (odd indices)
    south_items = [inst for i, inst in enumerate(instances) if i % 2 == 0]
    north_items = [inst for i, inst in enumerate(instances) if i % 2 == 1]

    # Calculate width domain for each instance: [ceil(min_area / depth_area_per_cell), floor(max_area / depth_area_per_cell)]
    # where depth_area_per_cell = depth_m * cell_size_m
    def compute_domain(
        items: list[tuple[str, str, RoomRequest]], depth_m: float
    ) -> tuple[list[tuple[str, str, RoomRequest, int, int]], bool]:
        vars_with_domain: list[tuple[str, str, RoomRequest, int, int]] = []
        depth_cell_area = depth_m * cell_size_m
        for rid, rtype, req in items:
            min_w = math.ceil(req.min_area_m2 / depth_cell_area)
            max_w = math.floor(req.max_area_m2 / depth_cell_area)
            if min_w > max_w:
                return [], True
            vars_with_domain.append((rid, rtype, req, min_w, max_w))
        return vars_with_domain, False

    south_vars, south_empty = compute_domain(south_items, south_depth_m)
    north_vars, north_empty = compute_domain(north_items, north_depth_m)

    if south_empty or north_empty:
        return SolverOutput(
            spec_hash=spec_hash,
            seed=seed,
            results=(),
            feasible_count=0,
            infeasible_count=0,
            conflict_explanation="One or more room instances have an empty width domain for the given corridor depth.",
            solve_time_s=round(time.monotonic() - start_time, 4),
        )

    min_door_m = max(
        thresholds.min_door_width_m.value, eff_spec.accessibility.min_door_width_m or 0.0
    )
    door_w_cells = max(2, math.ceil(min_door_m / cell_size_m))

    n_backtracks = 0
    feasible_results: list[SolverResult] = []
    infeasible_results: list[SolverResult] = []
    last_analysis: LayoutAnalysis | None = None
    max_candidates = max(max_layouts * 4, 20)

    # 3 & 4. Backtracking search over widths (ascending order)
    def search_north(
        south_assign: list[tuple[str, str, int]],
        north_idx: int,
        north_assign: list[tuple[str, str, int]],
        cur_north_w: int,
    ) -> bool:
        nonlocal n_backtracks, last_analysis
        if time.monotonic() - start_time >= time_limit_s:
            return True
        if len(feasible_results) >= max_candidates or (
            cannot_satisfy and len(infeasible_results) >= max_candidates
        ):
            return True

        if north_idx == len(north_vars):
            south_slots = tuple(
                SlotRoom(room_id=rid, room_type=rtype, width_cells=w)
                for rid, rtype, w in south_assign
            )
            north_slots = tuple(
                SlotRoom(room_id=rid, room_type=rtype, width_cells=w)
                for rid, rtype, w in north_assign
            )
            layout = assemble_corridor_layout(
                layout_id="LY_temp",
                spec=eff_spec,
                south=south_slots,
                north=north_slots,
                door_width_cells=door_w_cells,
                generator="csp_solver",
                seed=seed,
            )
            if spec_hash != eff_spec.compute_hash():
                layout = layout.model_copy(update={"spec_hash": spec_hash})

            # Judge against the REAL spec: a single-floor template must not be called feasible for a
            # multi-floor request (floor count and vertical access then show up as unmet).
            analysis = analyse_layout(layout, spec, thresholds, catalogue, checker_options)
            last_analysis = analysis

            verified_layout = layout.model_copy(
                update={"verification": analysis.verification, "content_hash": None}
            )
            metrics = layout_metrics(analysis, verified_layout)
            is_feasible = analysis.verification.all_hard_satisfied
            csr = analysis.quality.get("constraint_satisfaction_rate", 0.0)

            res = SolverResult(
                layout=verified_layout,
                metrics=metrics,
                feasible=is_feasible,
                solve_time_s=round(time.monotonic() - start_time, 4),
                n_backtracks=n_backtracks,
                constraint_satisfaction_rate=csr,
            )
            if is_feasible:
                feasible_results.append(res)
            else:
                infeasible_results.append(res)
            return len(feasible_results) >= max_candidates or (
                cannot_satisfy and len(infeasible_results) >= max_candidates
            )

        rid, rtype, req, min_w, max_w = north_vars[north_idx]
        for w in range(min_w, max_w + 1):
            if cur_north_w + w > grid_w:
                n_backtracks += 1
                break
            north_assign.append((rid, rtype, w))
            stop = search_north(south_assign, north_idx + 1, north_assign, cur_north_w + w)
            north_assign.pop()
            if stop:
                return True
        return False

    def search_south(
        south_idx: int,
        south_assign: list[tuple[str, str, int]],
        cur_south_w: int,
    ) -> bool:
        nonlocal n_backtracks
        if time.monotonic() - start_time >= time_limit_s:
            return True
        if len(feasible_results) >= max_candidates:
            return True

        if south_idx == len(south_vars):
            return search_north(south_assign, 0, [], 0)

        rid, rtype, req, min_w, max_w = south_vars[south_idx]
        for w in range(min_w, max_w + 1):
            if cur_south_w + w > grid_w:
                n_backtracks += 1
                break
            south_assign.append((rid, rtype, w))
            stop = search_south(south_idx + 1, south_assign, cur_south_w + w)
            south_assign.pop()
            if stop:
                return True
        return False

    search_south(0, [], 0)

    # 6. Keep at most max_layouts results, sorted by soft cost ASC (feasible first)
    if feasible_results:
        feasible_results.sort(key=lambda r: r.layout.verification.total_soft_cost)
        chosen = feasible_results[:max_layouts]
        conflict_explanation = ""
    else:
        infeasible_results.sort(
            key=lambda r: (
                -r.constraint_satisfaction_rate,
                r.layout.verification.total_soft_cost,
            )
        )
        chosen = infeasible_results[:max_layouts]
        if last_analysis is not None:
            conflicts = group_violations(last_analysis.verification.hard)
            conflict_explanation = " ".join(c.explanation for c in conflicts)
        else:
            conflict_explanation = "No room placement could fit within the building footprint."

    # Assign layout_id format: 'LY_csp_{seed}_{result_index:03d}'
    final_results: list[SolverResult] = []
    for idx, res in enumerate(chosen, start=1):
        lid = f"LY_csp_{seed}_{idx:03d}"
        updated_layout = res.layout.model_copy(
            update={
                "layout_id": lid,
                "provenance": LayoutProvenance(generator="csp_solver", seed=seed),
            }
        )
        final_results.append(
            SolverResult(
                layout=updated_layout,
                metrics=res.metrics,
                feasible=res.feasible,
                solve_time_s=res.solve_time_s,
                n_backtracks=res.n_backtracks,
                constraint_satisfaction_rate=res.constraint_satisfaction_rate,
            )
        )

    total_solve_time = round(time.monotonic() - start_time, 4)
    return SolverOutput(
        spec_hash=spec_hash,
        seed=seed,
        results=tuple(final_results),
        feasible_count=sum(1 for r in final_results if r.feasible),
        infeasible_count=sum(1 for r in final_results if not r.feasible),
        conflict_explanation=conflict_explanation,
        solve_time_s=total_solve_time,
    )
