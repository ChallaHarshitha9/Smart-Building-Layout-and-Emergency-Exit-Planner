"""Tests for the corridor-template CSP layout solver."""

import pytest

from smartbuilding.core.bundle import load_bundle
from smartbuilding.core.layout import Layout
from smartbuilding.solver.csp import solve
from smartbuilding.solver.result import SolverOutput
from tests.factories import ROOT, load_example_spec


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ROOT / "configs" / "run_default.yaml")


@pytest.fixture(scope="module")
def school_spec():
    return load_example_spec("school_spec.json")


@pytest.fixture(scope="module")
def office_spec():
    return load_example_spec("office_spec.json")


def test_solver_returns_solver_output(bundle, school_spec):
    output = solve(school_spec, bundle.catalogue, bundle.thresholds)
    assert isinstance(output, SolverOutput)
    assert isinstance(output.results, tuple)
    assert output.spec_hash == school_spec.compute_hash()
    assert output.seed == 0
    assert output.solve_time_s >= 0.0


def test_solver_finds_feasible_layout_for_single_floor_school(bundle, school_spec):
    one_floor = school_spec.model_copy(update={"floors": 1, "content_hash": None})
    output = solve(one_floor, bundle.catalogue, bundle.thresholds)
    assert output.feasible_count >= 1
    assert any(r.feasible for r in output.results)


def test_multi_floor_spec_is_reported_infeasible_not_silently_relaxed(bundle, school_spec):
    """Stage A builds one-floor layouts; a 2-floor request must not be called feasible."""
    assert school_spec.floors > 1
    output = solve(school_spec, bundle.catalogue, bundle.thresholds)
    assert output.feasible_count == 0 and output.infeasible_count >= 1
    assert "HC_floor_count" in output.conflict_explanation
    assert output.solve_time_s < 30  # stops early instead of searching to the time limit


def test_solver_finds_feasible_layout_for_office(bundle, office_spec):
    output = solve(office_spec, bundle.catalogue, bundle.thresholds)
    assert output.feasible_count >= 1
    assert any(r.feasible for r in output.results)


def test_solver_is_deterministic(bundle, school_spec):
    out1 = solve(school_spec, bundle.catalogue, bundle.thresholds, seed=42)
    out2 = solve(school_spec, bundle.catalogue, bundle.thresholds, seed=42)
    assert len(out1.results) == len(out2.results)
    assert out1.results[0].layout.geometry_hash == out2.results[0].layout.geometry_hash
    assert out1.results[0].layout.layout_id == out2.results[0].layout.layout_id


def test_solver_result_passes_hard_checks(bundle, school_spec):
    one_floor = school_spec.model_copy(update={"floors": 1, "content_hash": None})
    output = solve(one_floor, bundle.catalogue, bundle.thresholds)
    feasible = [r for r in output.results if r.feasible]
    assert len(feasible) >= 1
    first = feasible[0]
    assert first.feasible is True
    assert first.constraint_satisfaction_rate == 1.0
    assert first.layout.verification.all_hard_satisfied is True


def test_solver_output_layout_is_valid_contract(bundle, school_spec):
    output = solve(school_spec, bundle.catalogue, bundle.thresholds)
    assert len(output.results) >= 1
    first_layout = output.results[0].layout
    assert isinstance(first_layout, Layout)
    validated = Layout.model_validate(first_layout.model_dump())
    assert validated.layout_id == first_layout.layout_id
    assert validated.grid_w == first_layout.grid_w
    assert validated.grid_d == first_layout.grid_d
