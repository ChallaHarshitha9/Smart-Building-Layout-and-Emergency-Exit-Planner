# Smart Building Layout and Emergency Exit Planner

AI course project (Group 6). A simulated system that turns natural-language building
requirements into constraint-checked layouts, stress-tests them in a stochastic fire
simulator with uncertain sensing, and feeds the results back into layout repair.
It is a research prototype and **not** a substitute for professional architectural,
accessibility or fire-safety approval.

## Status

**Modules 1, 2, and 3 Complete (Stage A Core Pipeline):**
- **Module 1 (Search & CSP):** Corridor-template CSP layout generator with domain pruning, forward checking, backtracking, and soft-constraint optimization (`solver/csp.py`).
- **Module 2 (Planning):** STRIPS-style construction task decomposition and Kahn's topological critical-path scheduler (`planning/construction.py`), plus static Dijkstra (B1) and SafeStep belief-aware threshold Dijkstra (B2) evacuation path planners (`planning/evacuation.py`).
- **Module 3 (Uncertainty):** Discrete-time stochastic fire propagation, smoke diffusion, and crowd evacuation simulator (`sim/engine.py`), paired with a factored Bayesian belief state filter (`sim/belief.py`).
- **E0 Benchmark:** SafeStep toy corridor sanity benchmark (`eval/e0.py`) confirming the B1 vs. B2 hypothesis under smoke and fire hazards.

## Installation

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

No private credentials, API keys or downloads are needed.

## Run the checks

```bash
python -m pytest                   # all 150 unit tests
make check                         # lint + type check + tests (needs make)
```

## Command-line tool

```bash
smartbuilding contracts                                    # list contract names
smartbuilding validate BuildingSpec data/examples/school_spec.json
smartbuilding hash BuildingSpec data/examples/school_spec.json
smartbuilding check-config configs/run_default.yaml --list-assumptions

# Module 1: Layout CSP Solver
smartbuilding solve --spec data/examples/school_spec.json --max-layouts 3 --out-dir layouts/

# Module 2a: Construction Planning & Critical Path
smartbuilding plan-construction layouts/LY_csp_0_001.json --out plan.json

# Module 2b & 3: Evacuation & Uncertainty Sanity Benchmark (E0)
smartbuilding run-e0
```

## Repository structure

```
src/smartbuilding/
  core/
    primitives.py   enums, id patterns, constrained numeric types
    base.py         Part (nested) and Contract (versioned, hashable) base classes
    spec.py         BuildingSpec            requirements as structured data
    layout.py       Layout + verification   rooms, corridors, doors, exits on a cell grid
    navgraph.py     NavGraph                graph derived from a layout
    scenario.py     ScenarioFamily/Scenario/ResolvedScenario, deterministic expansion
    runtime.py      Observation, BeliefState (what decision modules may know)
    decision.py     Route, Decision, Plan, Policy
    repair.py       Finding, RepairMove (typed, discriminated union of 8 move types)
    construction.py ConstructionPlan        tasks, schedule, critical path
    metrics.py      MetricSet, FactsObject  metric registry and explainer grounding
    config.py       catalogue, thresholds, hazard, sensors, occupant profiles, run config
    bundle.py       load all configs and cross-check them against each other
    loaders.py      YAML/JSON loading, path-aware errors, lenient try_validate
    devices.py      device ids DV_<kind>_<n> (lift, door, detector, equipment, alarm)
    diff.py         stable-id check and layout edit distance
    terminology.py  neutral wording and banned-word lint
    hashing.py      canonical JSON and SHA-256
    seeding.py      derive_seed / make_rng (no global random state)
    schemas.py      JSON Schema export and drift check
  layout/
    checker.py      17 hard constraint checks (HC_*) and soft scoring
    geometry.py     cell boundaries, overlap, exterior contacts
    graph.py        build_topology and build_navgraph derivation
    paths.py        multi-source Dijkstra shortest path
    reference.py    canonical office/school layouts and flawed test variants
    template.py     corridor-template assembly
  solver/           Module 1: Search & Constraint Satisfaction
    csp.py          forward checking + backtracking layout CSP solver
    result.py       SolverResult, SolverOutput dataclasses
  planning/         Module 2: Planning
    construction.py STRIPS-style task decomposition and critical-path scheduler
    evacuation.py   B1 static Dijkstra & B2 SafeStep belief-aware evacuation planners
  sim/              Module 3: Reasoning under Uncertainty (Ground Truth)
    engine.py       discrete-time fire spread, smoke diffusion & crowd flow simulator
    belief.py       factored Bayesian belief filter over discrete hazard states
    truth.py        GroundTruthState (information barrier protected)
  eval/             Evaluation & Benchmarks
    e0.py           SafeStep toy corridor sanity benchmark (B1 vs B2)
  cli.py            smartbuilding command-line entrypoint
configs/            default engineering inputs (YAML)
data/examples/      example BuildingSpec files (school wing, small office)
schemas/            generated JSON Schemas (kept in sync by a test)
scripts/            make_examples.py
tests/              pytest + Hypothesis
docs/decisions.md   design decisions and why
docs/conventions.md units, coordinates, ids, hashes, wording
```

## Key conventions

* **Immutable, strict objects.** Unknown fields are rejected; models are frozen.
* **Integer geometry.** Layout coordinates are integer cells; `cell_size_m` (default 0.5)
  converts to metres. This keeps hashes exact.
* **Units in names.** `_m`, `_m2`, `_s`, `_mps`.
* **Hashes.** `content_hash` is SHA-256 of canonical JSON. `Layout.geometry_hash` ignores
  ids, verification and provenance and is the cache key for evaluation results.
* **Truth is separate from belief.** `GroundTruthState` lives in `sim/truth.py`. A test scans
  the whole source tree: only `sim/`, `eval/`, the schema registry and oracle files may use it.
* **Stable room ids.** A conversion (classroom to lab) keeps the id and sets `converted_from`.
* **Contracts check structure, not design quality.** A layout with overlapping rooms is a
  valid `Layout`; the rule checker (Step 2) rejects it. This lets baselines be represented.
* **Every number has provenance.** Inputs carry `sourcing: {basis, source}`. Values with
  `basis: assumption` are allowed but listed by `check-config --list-assumptions`.

## Placeholder values

The numbers in `configs/` are **placeholders** (`basis: assumption`). Before reporting
results, replace the thresholds in `configs/thresholds.yaml` with values from a building
code or fire-engineering reference the team cites, and set `basis: code` or `reference`
with a `source` citation.

## Regenerating generated files

```bash
python scripts/make_examples.py                       # example specs
smartbuilding export-schemas --out schemas            # JSON Schemas
```

## Reproducibility

All randomness goes through `smartbuilding.core.seeding`. Scenario expansion is
deterministic: the same `base_seed` always produces the same scenarios and seeds, and seeds
are unique across the repair and test splits.

## Acknowledgements

Inspired by Lang2Plan (Pan et al., 2026), HouseTune (Zong et al., 2026), SafeStep
(Fang et al., 2026) and Fitkau and Hartmann (2025). Libraries: pydantic, PyYAML, NumPy,
pytest, Hypothesis, ruff, mypy.

## Evacuation simulation (v2)

`smartbuilding run-e0 --seed 42` runs the sanity benchmark. See `docs/module1-3_review_and_v2.md` for the
simulator, belief filter and planners, and for the limitations. Minimal use:

```python
from smartbuilding.eval.episode import navgraph_for, resolve_for, run_episode, summarise
res, sim = run_episode(bundle, graph, resolved_scenario, spec, Baseline.B5)
print(summarise(res, bundle))
```

## 3D interface

    smartbuilding ui        # then open http://127.0.0.1:8765

See `docs/ui_3d.md`.
