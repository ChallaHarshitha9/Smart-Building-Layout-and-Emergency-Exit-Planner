"""Simulator v2: each test pins a defect found in the Module 1-3 review."""

import pytest

from smartbuilding.core.decision import Baseline, Decision, Plan, Route
from smartbuilding.core.runtime import PublicState
from smartbuilding.core.scenario import ResolvedGroup
from smartbuilding.planning.controllers import B1Controller, make_controller
from tests.toy import BP, DOSE, GRAPH, HZ, SENS, blocked_event, no_fire, node, scenario, sim

FAR = 1e6  # ignition so late it never happens


def _run(s, method=Baseline.B1):
    ctrl = (
        B1Controller(GRAPH)
        if method == Baseline.B1
        else make_controller(method, GRAPH, HZ, SENS, DOSE, s.sim_config.tick_s, 1.3, BP)
    )
    return s.run(ctrl)


def test_walking_time_is_distance_over_speed():
    # room 2 -> corridor -> west exit is 3.1 + 10 + 6.5 + 0.2 = 19.8 m... measured on the graph:
    s = sim(scenario(ignition_time_s=FAR), hz=no_fire(HZ))
    res = _run(s)
    adj = {n.node_id: [] for n in GRAPH.nodes}
    for e in GRAPH.edges:
        adj[e.u].append((e.v, e.length_m))
        adj[e.v].append((e.u, e.length_m))
    from smartbuilding.layout.paths import shortest_distances

    dist = shortest_distances(adj, [node("E_1"), node("E_2")])[node("R_office_2")]
    assert res.evac_times_s["G_a"] == pytest.approx(dist / 1.2, abs=1e-6)


def test_response_delay_comes_from_the_right_profile():
    s = sim(
        scenario(
            ignition_time_s=FAR,
            groups=(
                ResolvedGroup(
                    group_id="G_a", template_id="late", size=1, start_node=node("R_office_2")
                ),
            ),
        ),
        hz=no_fire(HZ),
    )
    early = _run(sim(scenario(ignition_time_s=FAR), hz=no_fire(HZ))).evac_times_s["G_a"]
    assert _run(s).evac_times_s["G_a"] == pytest.approx(early + 40.0, abs=1e-6)


def test_ambiguous_or_unknown_templates_fail_loudly():
    from smartbuilding.core.config import OccupantProfiles

    two = OccupantProfiles(
        profiles_id="p",
        profiles=(
            __import__("tests.toy", fromlist=["PROFS"]).PROFS.profiles[0],
            __import__("tests.toy", fromlist=["PROFS"])
            .PROFS.profiles[0]
            .model_copy(
                update={
                    "profile_id": "prof_other",
                    "groups": tuple(
                        g.model_copy(update={"speed_mps": 0.5})
                        for g in __import__("tests.toy", fromlist=["PROFS"])
                        .PROFS.profiles[0]
                        .groups
                    ),
                }
            ),
        ),
    )
    import numpy as np

    from smartbuilding.core.config import SimConfig
    from smartbuilding.sim.engine import Simulator

    with pytest.raises(ValueError, match="differ between occupant profiles"):
        Simulator(GRAPH, scenario(), HZ, SENS, two, SimConfig(), DOSE, np.random.default_rng(0))
    bad = scenario(
        groups=(
            ResolvedGroup(
                group_id="G_a", template_id="nope", size=1, start_node=node("R_office_2")
            ),
        )
    )
    with pytest.raises(ValueError, match="not defined"):
        sim(bad).reset()


def test_smoke_physics_does_not_depend_on_the_tick():
    results = []
    for tick in (0.5, 1.0, 2.0, 4.0):
        s = sim(scenario(), hz=no_fire(HZ), tick=tick)
        st = s.reset()
        st.fire.add(node("R_office_1"))  # burn room 1 without spread
        for _ in range(int(60 / tick)):
            st, _ = s.step(st, None)
        results.append(st.smoke)
    for nid in results[0]:
        vals = [r[nid] for r in results]
        assert max(vals) - min(vals) < 0.04, (nid, vals)


def test_queues_form_when_flow_is_limited_and_not_otherwise():
    big = (
        ResolvedGroup(group_id="G_a", template_id="early", size=40, start_node=node("R_office_2")),
        ResolvedGroup(group_id="G_b", template_id="early", size=40, start_node=node("R_office_2")),
    )
    limited = _run(sim(scenario(groups=big, ignition_time_s=FAR), hz=no_fire(HZ), flow=1.3))
    free = _run(sim(scenario(groups=big, ignition_time_s=FAR), hz=no_fire(HZ), flow=None))
    assert (
        limited.person_wait_s > 0
        and limited.peak_queue_persons > 0
        and limited.peak_queue_delay_s > 0
    )
    assert free.person_wait_s == 0
    # door D_2 is 2 cells = 1 m wide: 80 people cannot all pass in less than 80 / 1.3 s
    assert max(t for t in limited.evac_times_s.values() if t) >= 80 / 1.3 - 40 / 1.3 - 1e-6
    assert limited.completion_rate == 1.0


def test_same_seed_same_result_and_fire_is_independent_of_the_controller():
    def trace(method):
        s = sim(seed=7)
        fires = []
        ctrl = (
            B1Controller(GRAPH)
            if method == Baseline.B1
            else make_controller(method, GRAPH, HZ, SENS, DOSE, 2.0, 1.3, BP)
        )
        res = s.run(ctrl, lambda st, obs: fires.append((sorted(st.fire), dict(st.smoke))))
        return res, fires

    r1, f1 = trace(Baseline.B1)
    r1b, f1b = trace(Baseline.B1)
    assert r1.evac_times_s == r1b.evac_times_s and f1 == f1b
    n = 10  # compare the shared prefix: paired comparisons need identical hazard histories
    _, f5 = trace(Baseline.B5)
    assert f1[:n] == f5[:n]


def test_blocked_exit_B1_gets_stuck_B5_reroutes():
    sc = scenario(events=(blocked_event("E_1", 0.0),), ignition_time_s=FAR)
    assert _run(sim(sc, hz=no_fire(HZ)), Baseline.B1).completion_rate == 0.0
    r5 = _run(sim(sc, hz=no_fire(HZ)), Baseline.B5)
    assert r5.completion_rate == 1.0


class _Scripted:
    """Controller that issues a fixed list of routes at given times, ignoring observations."""

    def __init__(self, script):
        self.script, self.seen = script, []

    def observe(self, t_s, obs):
        pass

    def decide(self, public):
        assert isinstance(public, PublicState)
        g = public.groups[0]
        self.seen.append((public.t_s, g.node, g.in_transit, g.came_from))
        for t, nodes in self.script:
            if abs(public.t_s - t) < 1e-9:
                return Plan(
                    plan_id="p",
                    graph_hash=GRAPH.derived_from,
                    method=Baseline.B1,
                    decisions=(
                        Decision(
                            group_id=g.group_id,
                            t_s=public.t_s,
                            target_exit="E_1",
                            route=Route(nodes=tuple(nodes)),
                        ),
                    ),
                )
        return None


def test_invalid_routes_are_rejected_not_teleported():
    bad = [(0.0, [node("R_office_2"), node("E_1")])]  # not adjacent
    s = sim(scenario(ignition_time_s=FAR), hz=no_fire(HZ))
    res = s.run(_Scripted(bad))
    assert res.invalid_decisions >= 1 and res.completion_rate == 0.0


def test_a_group_in_transit_can_turn_around():
    corr = sorted((n for n in GRAPH.nodes if n.region_id == "C_1"), key=lambda n: n.x_m)
    c3 = min(corr, key=lambda n: abs(n.x_m - 16.5)).node_id  # corridor node at room 2's door
    c4 = corr[-1].node_id  # east end
    r2 = node("R_office_2")
    # t=0 walk east to c4; t=12 the group is mid-edge between c3 and c4 -> send it back to room 2
    script = [(0.0, [r2, c3, c4]), (12.0, [c3, r2])]
    s = sim(scenario(ignition_time_s=FAR, horizon_s=60.0), hz=no_fire(HZ))
    ctrl = _Scripted(script)
    states = []
    res = s.run(ctrl, lambda st, obs: states.append((st.t_s, st.groups[0].node, st.groups[0].nxt)))
    assert res.invalid_decisions == 0
    mid = [x for x in ctrl.seen if x[0] == 12.0][0]
    assert mid[2] is True and mid[3] == c3  # was in transit, last left c3
    assert states[-1][1] == r2 and states[-1][2] is None  # ended up back in room 2, standing
    assert res.completion_rate == 0.0


def test_decision_makers_never_receive_ground_truth():
    seen = []

    class Spy(_Scripted):
        def decide(self, public):
            seen.append(public)
            assert not hasattr(public, "smoke") and not hasattr(public, "hazard")
            return None

    sim().run(Spy([]))
    assert seen and all(isinstance(p, PublicState) for p in seen)
    assert not any(hasattr(g, "dose") for p in seen for g in p.groups)
