"""ConstructionPlan: STRIPS-style tasks, partial order, schedule and critical path."""

from typing import Self

from pydantic import Field, model_validator

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    ConstructionTaskType,
    Hash,
    Name,
    NonNegFloat,
    TaskId,
)

_TOL = 1e-6


class ConstructionTask(Part):
    task_id: TaskId
    task_type: ConstructionTaskType
    target_id: str | None = None  # room, corridor, door or exit the task acts on
    duration_days: NonNegFloat
    cost_units: NonNegFloat
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    depends_on: tuple[TaskId, ...] = ()


class ScheduledTask(Part):
    task_id: TaskId
    start_day: NonNegFloat
    end_day: NonNegFloat

    @model_validator(mode="after")
    def _order(self) -> Self:
        if self.end_day < self.start_day:
            raise ValueError("end_day must not precede start_day")
        return self


def has_cycle(deps: dict[str, tuple[str, ...]]) -> bool:
    """Kahn's algorithm. ``deps[t]`` lists the tasks that must finish before ``t``."""
    indegree = {t: len(d) for t, d in deps.items()}
    children: dict[str, list[str]] = {t: [] for t in deps}
    for t, ds in deps.items():
        for d in ds:
            children[d].append(t)
    ready = [t for t, n in indegree.items() if n == 0]
    visited = 0
    while ready:
        t = ready.pop()
        visited += 1
        for c in children[t]:
            indegree[c] -= 1
            if indegree[c] == 0:
                ready.append(c)
    return visited != len(deps)


class ConstructionPlan(Contract):
    plan_id: Name
    layout_geometry_hash: Hash
    tasks: tuple[ConstructionTask, ...] = Field(min_length=1)
    schedule: tuple[ScheduledTask, ...] = ()
    critical_path: tuple[TaskId, ...] = ()
    total_duration_days: NonNegFloat = 0.0
    total_cost_units: NonNegFloat = 0.0

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [t.task_id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")
        known = set(ids)
        deps: dict[str, tuple[str, ...]] = {}
        for t in self.tasks:
            for d in t.depends_on:
                if d not in known:
                    raise ValueError(f"task {t.task_id} depends on unknown task {d}")
            if t.task_id in t.depends_on:
                raise ValueError(f"task {t.task_id} depends on itself")
            deps[t.task_id] = t.depends_on
        if has_cycle(deps):
            raise ValueError("task dependencies contain a cycle")

        if abs(self.total_cost_units - sum(t.cost_units for t in self.tasks)) > _TOL * max(
            1.0, self.total_cost_units
        ):
            raise ValueError("total_cost_units does not equal the sum of task costs")

        if self.schedule:
            by_id = {s.task_id: s for s in self.schedule}
            if set(by_id) != known or len(by_id) != len(self.schedule):
                raise ValueError("schedule must contain every task exactly once")
            for t in self.tasks:
                for d in t.depends_on:
                    if by_id[d].end_day > by_id[t.task_id].start_day + _TOL:
                        raise ValueError(f"task {t.task_id} starts before dependency {d} ends")
            latest = max(s.end_day for s in self.schedule)
            if abs(latest - self.total_duration_days) > _TOL * max(1.0, latest):
                raise ValueError("total_duration_days does not equal the latest end_day")
        for t_id in self.critical_path:
            if t_id not in known:
                raise ValueError(f"critical path contains unknown task {t_id}")
        return self
