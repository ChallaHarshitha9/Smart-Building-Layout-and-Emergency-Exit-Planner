"""Metric sets and the facts object that grounds the explainer."""

from typing import Self

from pydantic import Field, model_validator

from smartbuilding.core.base import Contract, Part
from smartbuilding.core.primitives import (
    FactId,
    Finite,
    Hash,
    MetricId,
    MetricScope,
    Name,
    NonNegInt,
    PosInt,
    ScenarioId,
    Split,
)

# Registry of metric ids. Typos fail validation instead of silently creating new metrics.
KNOWN_METRICS: frozenset[str] = frozenset(
    {
        # layout
        "constraint_satisfaction_rate",
        "feasibility_rate",
        "usable_area_ratio",
        "adjacency_quality",
        "avg_related_room_distance_m",
        "accessibility_coverage",
        "energy_proxy",
        "adaptability",
        "diversity_pairwise",
        "solve_time_s",
        "layout_revisions",
        # evacuation
        "evac_time_mean_s",
        "evac_time_max_s",
        "completion_rate",
        "dose_mean",
        "dose_level_alert_frac",
        "dose_level_critical_frac",
        "dose_level_severe_frac",
        "peak_congestion_ratio",
        "overload_integral_s",
        "expected_risk",
        "risk_cvar90",
        "robustness_completion",
        # construction
        "construction_cost_units",
        "construction_duration_days",
        "critical_path_days",
    }
)


class MetricValue(Part):
    metric_id: MetricId
    value: Finite
    unit: str = ""
    ci_low: Finite | None = None
    ci_high: Finite | None = None
    n: PosInt | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.metric_id not in KNOWN_METRICS:
            raise ValueError(f"unknown metric id {self.metric_id!r}")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("ci_low and ci_high must be given together")
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError("ci_low must not exceed ci_high")
        return self


class MetricSet(Contract):
    scope: MetricScope
    layout_geometry_hash: Hash
    method: str | None = None
    scenario_id: ScenarioId | None = None
    split: Split | None = None
    n_scenarios: PosInt | None = None
    seed: NonNegInt | None = None
    values: tuple[MetricValue, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [v.metric_id for v in self.values]
        if len(set(ids)) != len(ids):
            raise ValueError("metric ids must be unique within a metric set")
        if self.scope == MetricScope.run and (self.scenario_id is None or self.seed is None):
            raise ValueError("run-level metrics need scenario_id and seed")
        if self.scope == MetricScope.aggregate and (
            self.split is None or self.n_scenarios is None or self.method is None
        ):
            raise ValueError("aggregate metrics need method, split and n_scenarios")
        return self

    def get(self, metric_id: str) -> MetricValue | None:
        return next((v for v in self.values if v.metric_id == metric_id), None)


class Fact(Part):
    fact_id: FactId
    metric_id: MetricId
    value: Finite | str | bool
    unit: str = ""
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _known(self) -> Self:
        if self.metric_id not in KNOWN_METRICS:
            raise ValueError(f"unknown metric id {self.metric_id!r}")
        return self


class FactsObject(Contract):
    """The only input the explainer may see. Every claim must trace to a fact."""

    facts_id: Name
    layout_geometry_hash: Hash
    facts: tuple[Fact, ...] = ()
    allowed_claims: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [f.fact_id for f in self.facts]
        if len(set(ids)) != len(ids):
            raise ValueError("fact ids must be unique")
        return self
