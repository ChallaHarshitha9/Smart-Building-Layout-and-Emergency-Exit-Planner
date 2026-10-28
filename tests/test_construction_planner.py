import pytest

from smartbuilding.core.bundle import load_bundle
from smartbuilding.core.construction import ConstructionPlan
from smartbuilding.layout.reference import build_office, build_school
from smartbuilding.planning.construction import plan_construction
from tests.factories import ROOT, load_example_spec


@pytest.fixture
def school_spec():
    return load_example_spec("school_spec.json")


@pytest.fixture
def office_spec():
    return load_example_spec("office_spec.json")


@pytest.fixture
def bundle():
    return load_bundle(ROOT / "configs" / "run_default.yaml")


def test_plan_is_valid_contract(office_spec):
    layout = build_office(office_spec)
    plan = plan_construction(layout)
    assert isinstance(plan, ConstructionPlan)
    assert plan.plan_id.startswith("CP_")
    assert plan.layout_geometry_hash == layout.geometry_hash
    assert len(plan.tasks) > 0


def test_cost_equals_sum_of_tasks(office_spec):
    layout = build_office(office_spec)
    plan = plan_construction(layout)
    expected_cost = sum(t.cost_units for t in plan.tasks)
    assert plan.total_cost_units == pytest.approx(expected_cost, abs=1e-6)


def test_schedule_covers_all_tasks(office_spec):
    layout = build_office(office_spec)
    plan = plan_construction(layout)
    assert len(plan.schedule) == len(plan.tasks)
    scheduled_ids = {s.task_id for s in plan.schedule}
    task_ids = {t.task_id for t in plan.tasks}
    assert scheduled_ids == task_ids


def test_critical_path_is_subset_of_tasks(office_spec):
    layout = build_office(office_spec)
    plan = plan_construction(layout)
    assert len(plan.critical_path) > 0
    task_ids = {t.task_id for t in plan.tasks}
    assert set(plan.critical_path).issubset(task_ids)


def test_dependencies_respected_in_schedule(office_spec):
    layout = build_office(office_spec)
    plan = plan_construction(layout)
    schedule_map = {s.task_id: s for s in plan.schedule}
    for task in plan.tasks:
        s_task = schedule_map[task.task_id]
        for dep in task.depends_on:
            s_dep = schedule_map[dep]
            assert s_dep.end_day <= s_task.start_day + 1e-6


def test_school_plan_is_valid_contract(school_spec):
    layout = build_school(school_spec)
    plan = plan_construction(layout)
    assert isinstance(plan, ConstructionPlan)
    assert len(plan.tasks) > 0
    assert len(plan.schedule) == len(plan.tasks)
    assert plan.total_duration_days > 0.0
