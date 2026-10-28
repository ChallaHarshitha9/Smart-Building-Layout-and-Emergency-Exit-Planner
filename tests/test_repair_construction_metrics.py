import pytest
from pydantic import ValidationError

from smartbuilding.core.construction import (
    ConstructionPlan,
    ConstructionTask,
    ScheduledTask,
    has_cycle,
)
from smartbuilding.core.metrics import KNOWN_METRICS, Fact, FactsObject, MetricSet, MetricValue
from smartbuilding.core.primitives import MoveType
from smartbuilding.core.repair import (
    AddDoor,
    AddExit,
    Finding,
    MoveExit,
    MoveRoom,
    ReassignFlexUse,
    RepairMove,
    ResizeRoom,
    SwapRooms,
    WidenCorridor,
)

H = "a" * 64


def _move(action):
    return RepairMove(move_id="MV_1", proposer="llm", action=action, finding_ids=("F_1",))


def test_every_move_type_has_exactly_one_action_class():
    classes = [
        WidenCorridor,
        AddExit,
        MoveExit,
        AddDoor,
        SwapRooms,
        MoveRoom,
        ResizeRoom,
        ReassignFlexUse,
    ]
    literals = {c.model_fields["type"].default for c in classes}
    assert literals == {m.value for m in MoveType}


def test_repair_move_roundtrip_from_json():
    raw = {
        "move_id": "MV_2",
        "proposer": "llm",
        "action": {"type": "widen_corridor", "corridor_id": "C_1", "new_width_cells": 5},
    }
    move = RepairMove.model_validate(raw)
    assert isinstance(move.action, WidenCorridor)
    assert RepairMove.model_validate_json(move.model_dump_json()) == move


def test_llm_cannot_invent_moves_or_parameters():
    with pytest.raises(ValidationError):
        RepairMove.model_validate(
            {"move_id": "MV_1", "proposer": "llm", "action": {"type": "demolish_building"}}
        )
    with pytest.raises(ValidationError):
        RepairMove.model_validate(
            {
                "move_id": "MV_1",
                "proposer": "llm",
                "action": {
                    "type": "widen_corridor",
                    "corridor_id": "C_1",
                    "new_width_cells": 5,
                    "also_paint_it": True,
                },
            }
        )
    with pytest.raises(ValidationError):
        RepairMove.model_validate(
            {
                "move_id": "MV_1",
                "proposer": "llm",
                "action": {"type": "widen_corridor", "corridor_id": "C_1", "new_width_cells": 0},
            }
        )
    with pytest.raises(ValidationError):
        _move(WidenCorridor(corridor_id="room_3", new_width_cells=2))


def test_finding_needs_entities():
    Finding(finding_id="F_1", kind="congested_edge", severity=2.0, entity_ids=("N_a",))
    with pytest.raises(ValidationError):
        Finding(finding_id="F_1", kind="congested_edge", severity=2.0, entity_ids=())


# ---- construction


def _tasks():
    return (
        ConstructionTask(
            task_id="T_walls", task_type="build_walls", duration_days=4, cost_units=10
        ),
        ConstructionTask(
            task_id="T_doors",
            task_type="install_door",
            duration_days=1,
            cost_units=2,
            depends_on=("T_walls",),
        ),
        ConstructionTask(
            task_id="T_safety",
            task_type="install_safety_equipment",
            duration_days=2,
            cost_units=3,
            depends_on=("T_doors",),
        ),
    )


def _plan(**kw):
    base = dict(
        plan_id="cp",
        layout_geometry_hash=H,
        tasks=_tasks(),
        schedule=(
            ScheduledTask(task_id="T_walls", start_day=0, end_day=4),
            ScheduledTask(task_id="T_doors", start_day=4, end_day=5),
            ScheduledTask(task_id="T_safety", start_day=5, end_day=7),
        ),
        critical_path=("T_walls", "T_doors", "T_safety"),
        total_duration_days=7,
        total_cost_units=15,
    )
    base.update(kw)
    return ConstructionPlan(**base)


def test_construction_plan_valid_roundtrip():
    plan = _plan()
    assert ConstructionPlan.model_validate_json(plan.model_dump_json()) == plan


def test_cycle_detection():
    assert has_cycle({"a": ("b",), "b": ("a",)})
    assert not has_cycle({"a": (), "b": ("a",)})
    t = list(_tasks())
    t[0] = t[0].model_copy(update={"depends_on": ("T_safety",)})
    with pytest.raises(ValidationError, match="cycle"):
        _plan(tasks=tuple(t), schedule=())


def test_construction_consistency_rules():
    with pytest.raises(ValidationError, match="unknown task"):
        t = _tasks()[0].model_copy(update={"depends_on": ("T_x",)})
        _plan(tasks=(t,), schedule=(), critical_path=(), total_duration_days=0, total_cost_units=10)
    with pytest.raises(ValidationError, match="sum of task costs"):
        _plan(total_cost_units=99)
    with pytest.raises(ValidationError, match="latest end_day"):
        _plan(total_duration_days=3)
    bad = (
        ScheduledTask(task_id="T_walls", start_day=0, end_day=4),
        ScheduledTask(task_id="T_doors", start_day=2, end_day=3),
        ScheduledTask(task_id="T_safety", start_day=5, end_day=7),
    )
    with pytest.raises(ValidationError, match="starts before dependency"):
        _plan(schedule=bad)
    with pytest.raises(ValidationError, match="every task"):
        _plan(schedule=bad[:2])
    with pytest.raises(ValidationError, match="end_day"):
        ScheduledTask(task_id="T_a", start_day=3, end_day=1)


# ---- metrics and facts


def test_metric_ids_are_registered():
    with pytest.raises(ValidationError, match="unknown metric"):
        MetricValue(metric_id="evacuation_time", value=1)
    assert "evac_time_mean_s" in KNOWN_METRICS


def test_metric_set_scope_rules():
    v = (MetricValue(metric_id="dose_mean", value=1.0),)
    MetricSet(scope="layout", layout_geometry_hash=H, values=v)
    with pytest.raises(ValidationError, match="run-level"):
        MetricSet(scope="run", layout_geometry_hash=H, values=v)
    MetricSet(scope="run", layout_geometry_hash=H, scenario_id="SC_a_1", seed=1, values=v)
    with pytest.raises(ValidationError, match="aggregate"):
        MetricSet(scope="aggregate", layout_geometry_hash=H, values=v)
    MetricSet(
        scope="aggregate",
        layout_geometry_hash=H,
        method="B5",
        split="test",
        n_scenarios=30,
        values=(MetricValue(metric_id="dose_mean", value=1, ci_low=0.5, ci_high=1.5, n=30),),
    )
    with pytest.raises(ValidationError, match="unique"):
        MetricSet(scope="layout", layout_geometry_hash=H, values=v + v)
    with pytest.raises(ValidationError, match="given together"):
        MetricValue(metric_id="dose_mean", value=1, ci_low=0.1)
    with pytest.raises(ValidationError, match="exceed"):
        MetricValue(metric_id="dose_mean", value=1, ci_low=2, ci_high=1)


def test_facts_object():
    f = Fact(
        fact_id="FA_1",
        metric_id="completion_rate",
        value=0.98,
        unit="fraction",
        text="{value:.0%} of occupants evacuated",
    )
    obj = FactsObject(
        facts_id="f", layout_geometry_hash=H, facts=(f,), allowed_claims=("unmet: lift",)
    )
    assert FactsObject.model_validate_json(obj.model_dump_json()) == obj
    with pytest.raises(ValidationError, match="unique"):
        FactsObject(facts_id="f", layout_geometry_hash=H, facts=(f, f))
    with pytest.raises(ValidationError, match="unknown metric"):
        Fact(fact_id="FA_2", metric_id="safe", value=True, text="the building is safe")


def test_non_finite_numbers_rejected():
    with pytest.raises(ValidationError):
        MetricValue(metric_id="dose_mean", value=float("nan"))
    with pytest.raises(ValidationError):
        MetricValue(metric_id="dose_mean", value=float("inf"))
