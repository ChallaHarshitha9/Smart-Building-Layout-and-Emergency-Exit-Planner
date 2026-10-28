"""Factored Bayes filter over hazards, exits, devices and occupancy.

This package never imports the simulator: it sees only ``Observation`` objects.

Hazard (per node, three states: clear / smoky / fire)
    predict:  mean-field transition. A node is fed by its neighbours with rates derived from the
              hazard model the filter is *given* (the tuned variant) plus small background rates
              (``BeliefParams``). Fire is absorbing; smoke can clear when nothing feeds it.
    correct:  every reading from a working detector updates its node, *including silence*. A reading
              without alarm is evidence for 'clear'. A detector that stops reporting simply gives no
              evidence (we do not assume silence means safe).
    The filter ignores correlations between nodes (mean-field), and its transition model is not the
    simulator's, so belief and truth do not coincide.

Exits, lifts and doors (Bernoulli)
    A known-reliability binary filter per device with a small failure rate. Reports from exit and
    lift status sensors, door monitors and occupants standing at a blocked exit all feed it.
    (The plan names Beta-Bernoulli; reliabilities are configured and known, so a plain Bernoulli
    filter is the exact Bayes update here.)

Occupancy (per room, Gaussian)
    Kalman filter with a random-walk process (people leave). Prior mean from the room load.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from smartbuilding.core.config import BeliefParams, HazardModel, SensorModel, StateMapping
from smartbuilding.core.devices import device_for_region
from smartbuilding.core.hazard_rates import (
    edge_smoke_factor,
    edge_spread_prob,
    prob_over,
    rate_per_s,
)
from smartbuilding.core.navgraph import NavGraph
from smartbuilding.core.primitives import Channel, HazardState, NodeKind
from smartbuilding.core.runtime import BeliefState, Observation, OccupancyBelief

Triple = tuple[float, float, float]
_ALARM = 0.5  # an observation value >= this is an alarm / an 'available' report


def _normalise(t: Triple, floor: float) -> Triple:
    a = max(t[0], floor)
    b = max(t[1], floor)
    c = max(t[2], floor)
    tot = a + b + c
    return (a / tot, b / tot, c / tot)


def expected_smoke(belief: BeliefState, node_id: str, mapping: StateMapping) -> float:
    """Expected smoke intensity at a node, with representative intensities per hazard state."""
    triple = belief.node_belief.get(node_id)
    if triple is None:
        return 0.0
    s_smoky = 0.5 * (mapping.smoky_min + mapping.fire_min)
    s_fire = 0.5 * (mapping.fire_min + 1.0)
    return triple[1] * s_smoky + triple[2] * s_fire


@dataclass(frozen=True)
class _Link:
    nbr: str
    smoke_rate: float  # per second
    fire_rate: float  # per second


class BeliefModel:
    """Everything about the filter that depends only on graph, hazard model and parameters."""

    def __init__(
        self,
        navgraph: NavGraph,
        hazard_model: HazardModel,
        sensor_model: SensorModel,
        params: BeliefParams | None = None,
    ) -> None:
        self.graph = navgraph
        self.hazard = hazard_model
        self.sensor = sensor_model
        self.params = params or BeliefParams()
        hp = hazard_model.params
        base_smoke = rate_per_s(hp.smoke_diffusion, hp.rate_interval_s)
        self.links: dict[str, list[_Link]] = {n.node_id: [] for n in navgraph.nodes}
        for e in navgraph.edges:
            smoke = base_smoke * edge_smoke_factor(hp, e)
            fire = rate_per_s(edge_spread_prob(hp, e), hp.rate_interval_s)
            self.links[e.u].append(_Link(e.v, smoke, fire))
            self.links[e.v].append(_Link(e.u, smoke, fire))
        self.recovery_rate = self.params.smoke_recovery_per_s
        s = sensor_model
        self.p_alarm: Triple = (
            s.smoke_false_alarm_prob,
            min(1.0, s.smoke_detection_prob * s.smoky_detection_factor),
            s.smoke_detection_prob,
        )
        self.exit_key = {
            n.node_id: n.region_id
            for n in navgraph.nodes
            if n.kind == NodeKind.exit and n.region_id
        }
        self.lift_key = {
            n.node_id: device_for_region(n.region_id)
            for n in navgraph.nodes
            if n.kind == NodeKind.lift and n.region_id
        }
        self.door_ids = sorted({e.door_id for e in navgraph.edges if e.door_id})


def initial_belief(
    navgraph: NavGraph,
    t_s: float = 0.0,
    params: BeliefParams | None = None,
    occupancy_prior: Mapping[str, float] | None = None,
) -> BeliefState:
    """Prior: every node clear; exits, lifts and doors probably working; occupancy from room load."""
    params = params or BeliefParams()
    node_belief: dict[str, Triple] = {n.node_id: (1.0, 0.0, 0.0) for n in navgraph.nodes}
    exits = {
        n.region_id: params.prior_available
        for n in navgraph.nodes
        if n.kind == NodeKind.exit and n.region_id
    }
    devices: dict[str, float] = {
        device_for_region(n.region_id): params.prior_available
        for n in navgraph.nodes
        if n.kind == NodeKind.lift and n.region_id
    }
    for e in navgraph.edges:
        if e.door_id:
            devices[device_for_region(e.door_id)] = params.prior_available
    occ: dict[str, OccupancyBelief] = {}
    for n in navgraph.nodes:
        if n.kind != NodeKind.room or not n.region_id:
            continue
        load = (
            occupancy_prior[n.region_id]
            if occupancy_prior and n.region_id in occupancy_prior
            else n.capacity_pers * params.occupancy_prior_fraction
        )
        occ[n.region_id] = OccupancyBelief(mean=load, var=max(load, 0.25))
    return BeliefState(
        graph_hash=navgraph.derived_from,
        t_s=t_s,
        node_belief=node_belief,
        exit_avail=exits,
        devices=devices,
        occupancy=occ,
    )


# ------------------------------------------------------------------------------ predict


def predict_belief(prior: BeliefState, model: BeliefModel, dt_s: float) -> BeliefState:
    """Advance the belief by dt_s with no new evidence."""
    p = model.params
    ign = prob_over(rate_per_s_abs(p.ignition_prob_per_s), dt_s)
    spont = prob_over(rate_per_s_abs(p.spontaneous_smoke_prob_per_s), dt_s)
    esc = prob_over(rate_per_s_abs(p.escalation_per_s), dt_s)
    recov = prob_over(rate_per_s_abs(model.recovery_rate), dt_s)
    new: dict[str, Triple] = {}
    for nid, (pc, ps, pf) in prior.node_belief.items():
        keep_smoke = 1.0  # prob that no neighbour delivers smoke
        keep_fire = 1.0
        for link in model.links.get(nid, ()):
            _, ns, nf = prior.node_belief.get(link.nbr, (1.0, 0.0, 0.0))
            keep_smoke *= 1.0 - (ns + nf) * prob_over(link.smoke_rate, dt_s)
            keep_fire *= 1.0 - nf * prob_over(link.fire_rate, dt_s)
        in_fire = 1.0 - keep_fire
        in_smoke = 1.0 - keep_smoke
        # from clear
        c_to_f = 1.0 - (1.0 - in_fire) * (1.0 - ign)
        c_to_s = (1.0 - c_to_f) * (1.0 - (1.0 - in_smoke) * (1.0 - spont))
        # from smoky
        s_to_f = 1.0 - (1.0 - in_fire) * (1.0 - esc)
        s_to_c = (1.0 - s_to_f) * (1.0 - in_smoke) * recov
        n_clear = pc * (1.0 - c_to_f - c_to_s) + ps * s_to_c
        n_smoky = pc * c_to_s + ps * (1.0 - s_to_f - s_to_c)
        n_fire = pc * c_to_f + ps * s_to_f + pf
        new[nid] = _normalise((n_clear, n_smoky, n_fire), p.min_prob)

    decay_exit = math.exp(-p.exit_fail_prob_per_s * dt_s)
    decay_dev = math.exp(-p.device_fail_prob_per_s * dt_s)
    exits = {k: max(p.min_prob, v * decay_exit) for k, v in prior.exit_avail.items()}
    devices = {k: max(p.min_prob, v * decay_dev) for k, v in prior.devices.items()}
    occ = {
        k: OccupancyBelief(mean=v.mean, var=v.var + p.occupancy_process_var_per_s * dt_s)
        for k, v in prior.occupancy.items()
    }
    return BeliefState(
        graph_hash=prior.graph_hash,
        t_s=round(prior.t_s + dt_s, 6),
        node_belief=new,
        exit_avail=exits,
        devices=devices,
        occupancy=occ,
    )


def rate_per_s_abs(p_per_s: float) -> float:
    """A background probability given per second, as a rate."""
    return rate_per_s(p_per_s, 1.0)


def predict_series(
    belief: BeliefState, model: BeliefModel, horizon_s: float, step_s: float
) -> list[BeliefState]:
    """[belief now, belief +step, ...] up to the horizon (no evidence). Used for time-dependent planning."""
    out = [belief]
    t = 0.0
    while t + 1e-9 < horizon_s:
        out.append(predict_belief(out[-1], model, step_s))
        t += step_s
    return out


# ------------------------------------------------------------------------------ correct


def _bernoulli_update(p: float, says_ok: bool, reliability: float, floor: float) -> float:
    r = min(max(reliability, 0.5), 1.0 - 1e-9)  # a report is never worse than a coin flip
    l_ok = r if says_ok else 1.0 - r
    l_bad = 1.0 - r if says_ok else r
    post = p * l_ok / (p * l_ok + (1.0 - p) * l_bad)
    return min(1.0 - floor, max(floor, post))


def correct_belief(
    prior: BeliefState, observations: Iterable[Observation], model: BeliefModel
) -> BeliefState:
    """Condition the belief on observations made at the belief's current time."""
    p = model.params
    node_belief = dict(prior.node_belief)
    exits = dict(prior.exit_avail)
    devices = dict(prior.devices)
    occ = dict(prior.occupancy)
    occ_var = max(model.sensor.occupancy_noise_sd, 0.25) ** 2

    for ob in observations:
        ch = ob.channel
        if ch == Channel.smoke and ob.source_id in node_belief:
            alarm = ob.value >= _ALARM
            cur = node_belief[ob.source_id]
            lik = tuple(pa if alarm else 1.0 - pa for pa in model.p_alarm)
            node_belief[ob.source_id] = _normalise(
                (cur[0] * lik[0], cur[1] * lik[1], cur[2] * lik[2]), p.min_prob
            )
        elif ch == Channel.exit:
            key = model.exit_key.get(ob.source_id)
            if key in exits and key is not None:
                exits[key] = _bernoulli_update(
                    exits[key], ob.value >= _ALARM, ob.reliability, p.min_prob
                )
        elif ch == Channel.report and ob.target_id is not None:
            key = model.exit_key.get(ob.target_id)
            if key in exits and key is not None:
                exits[key] = _bernoulli_update(
                    exits[key], ob.value >= _ALARM, ob.reliability, p.min_prob
                )
        elif ch == Channel.lift:
            key = model.lift_key.get(ob.source_id)
            if key in devices and key is not None:
                devices[key] = _bernoulli_update(
                    devices[key], ob.value >= _ALARM, ob.reliability, p.min_prob
                )
        elif ch == Channel.device and ob.source_id in devices:
            devices[ob.source_id] = _bernoulli_update(
                devices[ob.source_id], ob.value >= _ALARM, ob.reliability, p.min_prob
            )
        elif ch == Channel.occupancy and ob.source_id in occ:
            cur_o = occ[ob.source_id]
            gain = cur_o.var / (cur_o.var + occ_var)
            mean = max(0.0, cur_o.mean + gain * (ob.value - cur_o.mean))
            occ[ob.source_id] = OccupancyBelief(mean=mean, var=(1.0 - gain) * cur_o.var)

    return BeliefState(
        graph_hash=prior.graph_hash,
        t_s=prior.t_s,
        node_belief=node_belief,
        exit_avail=exits,
        devices=devices,
        occupancy=occ,
    )


def update_belief(
    prior: BeliefState,
    observations: Sequence[Observation],
    navgraph: NavGraph,
    hazard_model: HazardModel,
    sensor_model: SensorModel,
    tick_s: float,
    params: BeliefParams | None = None,
) -> BeliefState:
    """One filter step: predict by tick_s, then correct with that tick's observations."""
    model = BeliefModel(navgraph, hazard_model, sensor_model, params)
    return correct_belief(predict_belief(prior, model, tick_s), observations, model)


class BeliefFilter:
    """Stateful wrapper: feed it every tick's observations, read the belief when deciding."""

    def __init__(
        self,
        navgraph: NavGraph,
        hazard_model: HazardModel,
        sensor_model: SensorModel,
        tick_s: float,
        params: BeliefParams | None = None,
        occupancy_prior: Mapping[str, float] | None = None,
    ) -> None:
        self.model = BeliefModel(navgraph, hazard_model, sensor_model, params)
        self.tick_s = tick_s
        self._belief = initial_belief(navgraph, 0.0, self.model.params, occupancy_prior)

    def observe(self, t_s: float, observations: Sequence[Observation]) -> None:
        self._belief = correct_belief(
            predict_belief(self._belief, self.model, self.tick_s), observations, self.model
        )

    def belief(self) -> BeliefState:
        return self._belief

    def p_hazard(self, node_id: str, state: HazardState = HazardState.smoky) -> float:
        return self._belief.p_hazard(node_id, state)
