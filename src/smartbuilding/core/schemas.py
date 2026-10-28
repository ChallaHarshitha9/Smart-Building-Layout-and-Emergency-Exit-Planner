"""JSON Schema export for every contract, plus a drift check used by the tests."""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from smartbuilding.core.cases import ReferenceCase
from smartbuilding.core.config import (
    HazardModel,
    OccupantProfiles,
    RoomCatalogue,
    RunConfig,
    SensorModel,
    Thresholds,
)
from smartbuilding.core.construction import ConstructionPlan
from smartbuilding.core.decision import Decision, Plan, Policy
from smartbuilding.core.layout import Layout
from smartbuilding.core.metrics import FactsObject, MetricSet
from smartbuilding.core.navgraph import NavGraph
from smartbuilding.core.repair import Finding, RepairMove
from smartbuilding.core.runtime import BeliefState, Observation
from smartbuilding.core.scenario import (
    ResolvedScenario,
    Scenario,
    ScenarioFamilies,
    ScenarioSet,
)
from smartbuilding.core.spec import BuildingSpec
from smartbuilding.sim.truth import GroundTruthState

_ALL: tuple[type[BaseModel], ...] = (
    BuildingSpec,
    Layout,
    NavGraph,
    ConstructionPlan,
    Scenario,
    ScenarioFamilies,
    ScenarioSet,
    ResolvedScenario,
    GroundTruthState,
    Observation,
    BeliefState,
    Decision,
    Plan,
    Policy,
    Finding,
    RepairMove,
    MetricSet,
    FactsObject,
    RoomCatalogue,
    Thresholds,
    HazardModel,
    SensorModel,
    OccupantProfiles,
    RunConfig,
    ReferenceCase,
)

CONTRACTS: dict[str, type[BaseModel]] = {cls.__name__: cls for cls in _ALL}


def schema_text(cls: type[BaseModel]) -> str:
    return json.dumps(cls.model_json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def export_schemas(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cls in CONTRACTS.items():
        target = out / f"{name}.schema.json"
        target.write_text(schema_text(cls), encoding="utf-8")
        written.append(target)
    return written


def check_schemas(directory: str | Path) -> list[str]:
    """Return a list of problems: missing, outdated or unexpected schema files."""
    d = Path(directory)
    problems: list[str] = []
    expected = {f"{name}.schema.json" for name in CONTRACTS}
    for name, cls in CONTRACTS.items():
        f = d / f"{name}.schema.json"
        if not f.exists():
            problems.append(f"missing: {f.name}")
        elif f.read_text(encoding="utf-8") != schema_text(cls):
            problems.append(f"outdated: {f.name}")
    if d.exists():
        for f in d.glob("*.schema.json"):
            if f.name not in expected:
                problems.append(f"unexpected: {f.name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export or check JSON Schemas of all contracts.")
    parser.add_argument("--out", default="schemas")
    parser.add_argument("--check", action="store_true", help="fail if committed schemas are stale")
    args = parser.parse_args(argv)
    if args.check:
        problems = check_schemas(args.out)
        for p in problems:
            print(p, file=sys.stderr)
        return 1 if problems else 0
    for path in export_schemas(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
