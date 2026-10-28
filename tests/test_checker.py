"""Tests for the layout rule checker (layout/checker.py).

Uses the canonical reference layouts (build_office, build_school) and the
config bundle to exercise the full check pipeline.  All tests are deterministic
and make no assumptions about RNG state.
"""

import pytest

from smartbuilding.core.bundle import load_bundle
from smartbuilding.layout.checker import (
    CheckerOptions,
    LayoutAnalysis,
    analyse_layout,
    check_layout,
    layout_metrics,
)
from smartbuilding.layout.graph import build_navgraph
from smartbuilding.layout.reference import build_office, build_school
from tests.factories import ROOT, load_example_spec

# Metric ids that analyse_layout is known to produce.
KNOWN_METRICS: frozenset[str] = frozenset(
    {
        "constraint_satisfaction_rate",
        "usable_area_ratio",
        "accessibility_coverage",
        "energy_proxy",
        "adjacency_quality",
        "avg_related_room_distance_m",
        "adaptability",
    }
)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def bundle():
    """Load the default config bundle once per test-module run."""
    return load_bundle(ROOT / "configs" / "run_default.yaml")


@pytest.fixture(scope="module")
def school_spec():
    """School BuildingSpec loaded from data/examples/school_spec.json."""
    return load_example_spec("school_spec.json")


@pytest.fixture(scope="module")
def office_spec():
    """Office BuildingSpec loaded from data/examples/office_spec.json."""
    return load_example_spec("office_spec.json")


# --------------------------------------------------------------------------- tests


def test_office_ok_passes_all_hard_rules(bundle, office_spec):
    """The canonical office layout must satisfy every configured hard constraint."""
    layout = build_office(office_spec)
    result = check_layout(
        layout,
        office_spec,
        bundle.thresholds,
        bundle.catalogue,
    )
    failing = [r.constraint_id for r in result.hard if not r.satisfied]
    assert failing == [], f"The reference office layout failed hard constraints: {failing}"


def test_school_ok_passes_geometry_hard_rules(bundle, school_spec):
    """The canonical school layout (single floor) passes all geometry hard constraints.

    The school spec requires 2 floors and a lift, but build_school() produces a
    single-floor reference layout intentionally (the second floor and lift are added
    by the solver in Step 5).  HC_floor_count and HC_vertical_access are therefore
    expected to fail.  All other constraints must pass.
    """
    # Known failures: spec says floors=2 + lift_required, reference is 1 floor
    KNOWN_EXPECTED_FAILURES = {"HC_floor_count", "HC_vertical_access"}

    layout = build_school(school_spec)
    result = check_layout(
        layout,
        school_spec,
        bundle.thresholds,
        bundle.catalogue,
    )
    failing = {r.constraint_id for r in result.hard if not r.satisfied}
    unexpected = failing - KNOWN_EXPECTED_FAILURES
    assert unexpected == set(), (
        f"School reference layout failed unexpected hard constraints: {unexpected}"
    )
    # At least some hard rules must pass (connectivity, area, no-overlap, ...)
    passing = [r for r in result.hard if r.satisfied]
    assert len(passing) > 0, "At least some hard rules must pass on the reference layout"


def test_layout_metrics_returns_known_metric_ids(bundle, office_spec):
    """Every metric_id in layout_metrics output must be a member of KNOWN_METRICS."""
    layout = build_office(office_spec)
    analysis: LayoutAnalysis = analyse_layout(
        layout,
        office_spec,
        bundle.thresholds,
        bundle.catalogue,
    )
    metric_set = layout_metrics(analysis, layout)
    for mv in metric_set.values:
        assert mv.metric_id in KNOWN_METRICS, (
            f"Unexpected metric_id returned by layout_metrics: {mv.metric_id!r}"
        )


def test_office_navgraph_has_correct_node_kinds(office_spec):
    """build_navgraph on the office layout must contain room, corridor and exit nodes."""
    layout = build_office(office_spec)
    navgraph = build_navgraph(layout)

    kinds = {n.kind for n in navgraph.nodes}
    assert "room" in kinds, "NavGraph must contain at least one room node"
    assert "corridor" in kinds, "NavGraph must contain at least one corridor node"
    assert "exit" in kinds, "NavGraph must contain at least one exit node"


def test_check_layout_returns_verification_contract(bundle, office_spec):
    """check_layout must return a Verification whose status is 'checked'."""
    layout = build_office(office_spec)
    verification = check_layout(
        layout,
        office_spec,
        bundle.thresholds,
        bundle.catalogue,
        CheckerOptions(),
    )
    assert verification.status == "checked", (
        f"Expected status 'checked', got {verification.status!r}"
    )
    # The hard tuple must be non-empty (at least some rules were evaluated).
    assert len(verification.hard) > 0, "Verification should contain at least one hard result"
