"""Belief filter v2: the defects found in review (ratchet to certainty, silence ignored, ...)."""

import pytest

from smartbuilding.core.primitives import Channel
from smartbuilding.core.runtime import Observation
from smartbuilding.uncertainty.belief import (
    BeliefFilter,
    expected_smoke,
    initial_belief,
    predict_series,
)
from smartbuilding.uncertainty.raw import RawTracker
from tests.toy import BP, GRAPH, HZ, SENS, node

C = "N_corridor_2"


def smoke_obs(nid, alarm, t=0.0):
    return Observation(
        t_s=t, channel=Channel.smoke, source_id=nid, value=1.0 if alarm else 0.0, reliability=0.95
    )


def filt():
    return BeliefFilter(GRAPH, HZ, SENS, 2.0, BP)


def test_initial_belief_is_clear_and_exits_probably_work():
    b = initial_belief(GRAPH)
    assert all(t == (1.0, 0.0, 0.0) for t in b.node_belief.values())
    assert set(b.exit_avail) == {"E_1", "E_2"} and all(0.9 < v < 1 for v in b.exit_avail.values())
    assert set(b.occupancy) == {"R_office_1", "R_office_2"}


def test_one_false_alarm_is_forgotten_not_ratcheted_to_fire():
    f = filt()
    f.observe(2.0, [smoke_obs(C, True)])
    peak = 1 - f.belief().node_belief[C][0]
    sensors = [n.node_id for n in GRAPH.nodes if n.kind.value in ("room", "corridor")]
    for k in range(30):  # every working detector reports every tick
        f.observe(4.0 + 2 * k, [smoke_obs(n, False) for n in sensors])
    after = f.belief().node_belief[C]
    assert peak > 0.0 and after[1] + after[2] < 0.02  # recovered
    # and it did not leak fire into the neighbours
    nbrs = [e.v if e.u == C else e.u for e in GRAPH.edges if C in (e.u, e.v)]
    assert all(1 - f.belief().node_belief[n][0] < 0.02 for n in nbrs)


def test_silence_is_evidence():
    quiet, blind = filt(), filt()
    for _ in range(3):
        quiet.observe(0.0, [smoke_obs(C, True)])
        blind.observe(0.0, [smoke_obs(C, True)])
    for _ in range(5):
        quiet.observe(0.0, [smoke_obs(C, False)])
        blind.observe(0.0, [])  # detector silent: no reading at all
    assert quiet.belief().node_belief[C][0] > blind.belief().node_belief[C][0]


def test_belief_converges_to_a_real_fire():
    f = filt()
    for k in range(6):
        f.observe(2.0 * k, [smoke_obs(C, True)])
    tri = f.belief().node_belief[C]
    assert tri[2] > 0.9 and tri[0] < 0.01


def test_belief_converges_to_truth_with_perfect_sensors():
    sens = SENS.model_copy(
        update={
            "smoke_detection_prob": 1.0,
            "smoke_false_alarm_prob": 0.0,
            "smoky_detection_factor": 1.0,
        }
    )
    f = BeliefFilter(GRAPH, HZ, sens, 2.0, BP)
    for k in range(4):
        f.observe(2.0 * k, [smoke_obs(C, False)])
    assert f.belief().node_belief[C][0] > 0.999
    f.observe(10.0, [smoke_obs(C, True)])
    assert f.belief().node_belief[C][0] < 1e-3


def test_belief_is_always_a_distribution():
    f = filt()
    for k in range(40):
        f.observe(2.0 * k, [smoke_obs(C, k % 3 == 0)])
        for tri in f.belief().node_belief.values():
            assert sum(tri) == pytest.approx(1.0, abs=1e-9) and min(tri) >= 0


def test_hazard_spreads_in_prediction_without_observations():
    f = filt()
    for k in range(6):
        f.observe(2.0 * k, [smoke_obs(C, True)])
    model = f.model
    series = predict_series(f.belief(), model, 60.0, 6.0)
    nbr = next(e.v if e.u == C else e.u for e in GRAPH.edges if C in (e.u, e.v))
    assert 1 - series[-1].node_belief[nbr][0] > 1 - series[0].node_belief[nbr][0]


def test_exit_reports_update_exit_belief_and_one_bad_report_does_not_flip_it():
    f = filt()
    e1 = node("E_1")

    def rep(ok):
        return Observation(
            t_s=0.0, channel=Channel.exit, source_id=e1, value=1.0 if ok else 0.0, reliability=0.95
        )

    f.observe(2.0, [rep(False)])
    assert 0.01 < f.belief().exit_avail["E_1"] < 0.99  # single report: not decisive...
    for k in range(3):
        f.observe(4.0 + 2 * k, [rep(False)])
    assert f.belief().exit_avail["E_1"] < 0.01  # ...several agreeing reports are
    assert f.belief().exit_avail["E_2"] > 0.9


def test_user_report_from_a_blocked_exit_lowers_its_availability():
    f = filt()
    ob = Observation(
        t_s=0.0,
        channel=Channel.report,
        source_id="G_x",
        target_id=node("E_2"),
        value=0.0,
        reliability=0.8,
    )
    for k in range(5):  # a 0.8-reliable report against a 0.99 prior needs several repeats
        f.observe(2.0 * (k + 1), [ob])
    assert f.belief().exit_avail["E_2"] < 0.2


def test_occupancy_is_tracked_with_a_kalman_update():
    f = filt()
    prior = f.belief().occupancy["R_office_1"]
    ob = Observation(
        t_s=0.0, channel=Channel.occupancy, source_id="R_office_1", value=20.0, reliability=0.9
    )
    f.observe(2.0, [ob])
    post = f.belief().occupancy["R_office_1"]
    assert post.mean > prior.mean and post.var < prior.var + 0.2


def test_expected_smoke_and_raw_tracker():
    f = filt()
    assert expected_smoke(f.belief(), C, HZ.state_mapping) == 0.0
    raw = RawTracker(GRAPH)
    raw.observe(2.0, [smoke_obs(C, True)])
    assert raw.belief().node_belief[C] == (0.0, 1.0, 0.0)
    raw.observe(4.0, [smoke_obs(C, False)])
    assert raw.belief().node_belief[C] == (1.0, 0.0, 0.0)  # no memory
