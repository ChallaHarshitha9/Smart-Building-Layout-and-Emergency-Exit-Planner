import pytest
from pydantic import ValidationError

from smartbuilding.core.decision import Decision, Plan, Policy, Route
from smartbuilding.core.runtime import BeliefState, Observation
from smartbuilding.core.scenario import (
    EventSpec,
    GroupSpec,
    IgnitionSpec,
    ResolvedScenario,
    Scenario,
    ScenarioFamily,
    ScenarioSet,
    Selector,
)
from smartbuilding.sim.truth import GroundTruthState

H = "a" * 64


def _family(**kw):
    base = dict(
        family_id="f1",
        ignition=IgnitionSpec(where=Selector(mode="random_room")),
        groups=(
            GroupSpec(
                group_id="G_all", template_id="all", placement=Selector(mode="distributed_by_load")
            ),
        ),
        sensor_model_ref="s",
    )
    base.update(kw)
    return ScenarioFamily(**base)


def _scenario(i=1, seed=1, split="test"):
    f = _family()
    return Scenario(
        scenario_id=f"SC_f1_{i}",
        family_id="f1",
        split=split,
        hazard_variant="A",
        ignition=f.ignition,
        groups=f.groups,
        sensor_model_ref="s",
        seed=seed,
        horizon_s=900,
    )


def test_selector_ref_rules():
    Selector(mode="explicit_id", ref="E_1")
    Selector(mode="room_type", ref="lab")
    with pytest.raises(ValidationError, match="needs a 'ref'"):
        Selector(mode="room_type")
    with pytest.raises(ValidationError, match="must not have"):
        Selector(mode="random_room", ref="x")


def test_near_exit_selector_needs_an_explicit_fraction():
    Selector(mode="random_room_near_exit", nearest_fraction=0.25)
    with pytest.raises(ValidationError, match="needs 'nearest_fraction'"):
        Selector(mode="random_room_near_exit")
    with pytest.raises(ValidationError, match="only allowed"):
        Selector(mode="random_room", nearest_fraction=0.5)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValidationError):
            Selector(mode="random_room_near_exit", nearest_fraction=bad)


def test_family_rules():
    with pytest.raises(ValidationError, match="sorted"):
        _family(
            events=(
                EventSpec(type="lift_out", time_s=10, target=Selector(mode="random_lift")),
                EventSpec(type="lift_out", time_s=5, target=Selector(mode="random_lift")),
            )
        )
    g = GroupSpec(group_id="G_a", template_id="adult", placement=Selector(mode="random_room"))
    with pytest.raises(ValidationError, match="unique"):
        _family(groups=(g, g))
    with pytest.raises(ValidationError, match="size_override"):
        GroupSpec(
            group_id="G_a",
            template_id="all",
            placement=Selector(mode="random_room"),
            size_override=3,
        )


def test_scenario_set_no_leakage():
    ScenarioSet(set_id="s", scenarios=(_scenario(1, 1, "repair"), _scenario(2, 2, "test")))
    with pytest.raises(ValidationError, match="seeds must be unique"):
        ScenarioSet(set_id="s", scenarios=(_scenario(1, 7, "repair"), _scenario(2, 7, "test")))
    with pytest.raises(ValidationError, match="ids must be unique"):
        ScenarioSet(set_id="s", scenarios=(_scenario(1, 1), _scenario(1, 2)))


def test_resolved_scenario_roundtrip():
    r = ResolvedScenario(
        scenario_id="SC_f1_1",
        graph_hash=H,
        hazard_variant="A",
        ignition_node="N_room_1",
        ignition_time_s=0,
        events=({"type": "exit_blocked", "time_s": 60, "target_id": "E_1"},),
        groups=({"group_id": "G_a", "template_id": "adult", "size": 5, "start_node": "N_room_2"},),
        sensor_model_ref="s",
        seed=3,
        horizon_s=900,
    )
    assert ResolvedScenario.model_validate_json(r.model_dump_json()) == r


def test_belief_must_be_normalised_and_lookup_works():
    b = BeliefState(
        graph_hash=H,
        t_s=0,
        node_belief={"N_room_1": (0.7, 0.2, 0.1)},
        exit_avail={"E_1": 0.9},
        occupancy={"R_office_1": {"mean": 3.0, "var": 1.0}},
    )
    assert b.p_hazard("N_room_1", "smoky") == pytest.approx(0.2)
    assert BeliefState.model_validate_json(b.model_dump_json()) == b
    with pytest.raises(ValidationError, match="sum to 1"):
        BeliefState(graph_hash=H, t_s=0, node_belief={"N_room_1": (0.5, 0.2, 0.1)})
    with pytest.raises(ValidationError):
        BeliefState(graph_hash=H, t_s=0, node_belief={"N_room_1": (1.2, -0.1, -0.1)})


def test_truth_and_belief_are_different_types():
    assert GroundTruthState is not BeliefState
    assert not issubclass(BeliefState, GroundTruthState)
    gt = GroundTruthState(graph_hash=H, t_s=2, hazard={"N_room_1": "fire"}, smoke={"N_room_1": 1.0})
    assert GroundTruthState.model_validate_json(gt.model_dump_json()) == gt


def test_observation_validation():
    Observation(
        t_s=1,
        channel="smoke",
        source_id="DV_smoke_1",
        target_id="N_room_1",
        value=1,
        reliability=0.9,
    )
    with pytest.raises(ValidationError):
        Observation(t_s=1, channel="smoke", source_id="x", value=1, reliability=1.5)


def test_decision_plan_policy():
    route = Route(nodes=("N_room_1", "N_corridor_1", "N_exit_1"), est_time_s=20)
    d = Decision(group_id="G_a", t_s=0, target_exit="E_1", route=route, alternatives=(route,))
    plan = Plan(plan_id="p1", graph_hash=H, method="B5", decisions=(d,))
    assert Plan.model_validate_json(plan.model_dump_json()) == plan
    Policy(
        policy_id="pol",
        algorithm="dqn",
        checkpoint_path="checkpoints/p.pt",
        obs_spec_version="1",
        layout_geometry_hash=H,
        training_seeds=(1, 2),
        uses_belief=True,
    )
    with pytest.raises(ValidationError, match="relative"):
        Policy(
            policy_id="pol",
            algorithm="dqn",
            checkpoint_path="/abs/p.pt",
            obs_spec_version="1",
            layout_geometry_hash=H,
            training_seeds=(1,),
            uses_belief=True,
        )
    with pytest.raises(ValidationError):
        Plan(plan_id="p1", graph_hash=H, method="B9")
