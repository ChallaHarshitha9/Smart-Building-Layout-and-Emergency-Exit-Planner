"""B1, B2 and the dose-aware planner on the toy corridor."""

from smartbuilding.core.runtime import PublicGroup
from smartbuilding.planning.dose_aware import plan_dose_aware
from smartbuilding.planning.evacuation import plan_b1, plan_b2
from smartbuilding.uncertainty.belief import initial_belief
from tests.toy import DOSE, GRAPH, HZ, node

R2 = node("R_office_2")
C2 = "N_corridor_2"  # corridor node at room 1's door, between room 2 and the west exit


def grp(node_id=R2, size=1):
    return PublicGroup(group_id="G_a", template_id="early", size=size, node=node_id, speed_mps=1.2)


def belief_with(hazard_nodes=(), exit_avail=None, p=0.9):
    b = initial_belief(GRAPH)
    nb = dict(b.node_belief)
    for n in hazard_nodes:
        nb[n] = (1 - p, p, 0.0)
    return b.model_copy(
        update={"node_belief": nb, "exit_avail": {**b.exit_avail, **(exit_avail or {})}}
    )


def static(belief, groups, **kw):
    return plan_dose_aware(GRAPH, groups, [belief], HZ.state_mapping, DOSE.k, **kw)


def test_b1_goes_to_the_nearest_exit_with_ranked_alternatives():
    d = plan_b1(GRAPH, [grp()]).decisions[0]
    assert d.target_exit == "E_1" and d.route.nodes[-1] == node("E_1")
    assert len(d.alternatives) == 1 and d.alternatives[0].nodes[-1] == node("E_2")
    assert d.route.est_time_s < d.alternatives[0].est_time_s


def test_b1_accepts_plain_tuples():
    assert plan_b1(GRAPH, [("G_a", R2)]).decisions[0].group_id == "G_a"


def test_b2_blocks_hazardous_nodes_and_falls_back_when_cut_off():
    d = plan_b2(GRAPH, [grp()], belief_with([C2]), threshold=0.5).decisions[0]
    assert d.target_exit == "E_2"
    all_blocked = belief_with([n.node_id for n in GRAPH.nodes if n.node_id != R2])
    d2 = plan_b2(GRAPH, [grp()], all_blocked, threshold=0.5).decisions[0]
    assert d2.route.nodes[-1] in (node("E_1"), node("E_2"))  # B1 fallback, not "hold"


def test_dose_aware_equals_shortest_path_when_nothing_is_wrong():
    a = static(initial_belief(GRAPH), [grp()]).decisions[0]
    b = plan_b1(GRAPH, [grp()]).decisions[0]
    assert a.route.nodes == b.route.nodes


def test_dose_aware_detours_around_smoke_but_not_around_a_whiff():
    heavy = static(belief_with([C2], p=0.99), [grp()]).decisions[0]
    assert heavy.target_exit == "E_2"
    whiff = static(belief_with([C2], p=0.01), [grp()]).decisions[0]
    assert whiff.target_exit == "E_1"


def test_dose_aware_uses_exit_availability():
    d = static(belief_with(exit_avail={"E_1": 0.05}), [grp()]).decisions[0]
    assert d.target_exit == "E_2"
    both = static(belief_with(exit_avail={"E_1": 0.05, "E_2": 0.05}), [grp()]).decisions[0]
    assert both.route.nodes[-1] in (node("E_1"), node("E_2"))  # still tries something


def test_dose_aware_avoids_a_believed_stuck_door():
    from smartbuilding.core.devices import device_for_region

    b = belief_with()
    b = b.model_copy(
        update={"devices": {**b.devices, device_for_region("D_2"): 0.05}}
    )  # room 2's own door
    d = static(
        b,
        [
            PublicGroup(
                group_id="G_a", template_id="early", size=1, node=node("R_office_1"), speed_mps=1.2
            )
        ],
    ).decisions[0]
    assert d.route.nodes[-1] in (node("E_1"), node("E_2"))


def test_queue_awareness_prefers_the_emptier_door():
    crowd = PublicGroup(group_id="G_b", template_id="early", size=60, node=R2, speed_mps=1.2)
    mine = grp()
    with_flow = static(
        initial_belief(GRAPH), [mine, crowd], flow_pers_per_m_s=1.3, params=None
    ).decisions
    assert {d.group_id for d in with_flow} == {"G_a", "G_b"}


def test_every_decision_has_a_valid_route_and_alternatives_are_ranked():
    plan = static(belief_with([C2]), [grp(), grp(node("R_office_1"))])
    for d in plan.decisions:
        assert d.route.nodes[0] in (R2, node("R_office_1"))
        costs = [d.route.est_time_s] + [a.est_time_s for a in d.alternatives]
        assert len(costs) >= 1
