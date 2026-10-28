"""SolverResult: the output of the CSP solver."""

from dataclasses import dataclass

from smartbuilding.core.layout import Layout
from smartbuilding.core.metrics import MetricSet


@dataclass(frozen=True)
class SolverResult:
    layout: Layout
    metrics: MetricSet | None
    feasible: bool  # passes all hard checks
    solve_time_s: float
    n_backtracks: int
    constraint_satisfaction_rate: float  # fraction of hard checks that pass


@dataclass(frozen=True)
class SolverOutput:
    spec_hash: str
    seed: int
    results: tuple[SolverResult, ...]  # at most max_layouts, sorted by soft cost
    feasible_count: int
    infeasible_count: int
    conflict_explanation: str  # empty if at least one feasible layout found
    solve_time_s: float
