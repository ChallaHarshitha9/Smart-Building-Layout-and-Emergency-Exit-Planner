"""Discrete-time fire-and-evacuation simulator (ground truth). Version 2.

INFORMATION BARRIER: this module is in sim/. It may use ground truth and is the ONLY place where
ground truth is mutated. Controllers (planners, belief filters, RL, GenAI) receive only
``PublicState`` and ``Observation`` objects and must not import this module.

Model
-----
Time        tick = SimConfig.tick_s. All dynamics are expressed per second, so changing the tick
            changes the resolution, not the physics (smoke is sub-stepped; fire spread uses
            1 - exp(-rate * dt)).
Fire        burning nodes spread to neighbours with a probability that depends on the edge type
            (open space, open door, closed or stuck door). Fire does not go out. Hazard variant B
            adds a two-hop kernel. Fire uses its own random stream, so the fire history of a
            scenario is identical for every controller (paired comparisons).
Smoke       per-node intensity in [0, 1]: generated at burning nodes, diffusing along edges (scaled
            by edge type) and decaying elsewhere.
People      groups walk along edges at their own speed (position in metres along the edge). Each
            directed edge admits ``width * specific_flow`` persons per second; groups that cannot
            enter queue at the node, first come first served.
Dose        dose proxy (NOT FED): k * smoke at the group's position * time, integrated along the path.
Sensing     every working detector reports every tick (alarm or no alarm, with the configured
            detection and false-alarm probabilities). Exit, lift, door and occupancy reports are
            issued every ``report_period_s``. Occupants standing at a blocked exit report it.
            These draw on a separate random stream.
Decisions   a Controller is asked on every decision tick. Its route is validated edge by edge;
            invalid routes are rejected and counted.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from smartbuilding.core.config import (
    DoseConfig,
    GroupTemplate,
    HazardModel,
    OccupantProfiles,
    SensorModel,
    SimConfig,
)
from smartbuilding.core.decision import Controller, Plan
from smartbuilding.core.devices import device_for_region
from smartbuilding.core.hazard_rates import (
    edge_smoke_factor,
    edge_spread_prob,
    prob_over,
    rate_per_s,
)
from smartbuilding.core.navgraph import NavEdge, NavGraph, NavNode
from smartbuilding.core.primitives import (
    Channel,
    EventType,
    HazardState,
    NodeKind,
    SensorPlacement,
    SpreadKernel,
)
from smartbuilding.core.runtime import Observation, PublicGroup, PublicState
from smartbuilding.core.scenario import ResolvedScenario
from smartbuilding.sim.truth import GroundTruthState

_EPS = 1e-9


@dataclass
class GroupState:
    group_id: str
    template_id: str
    size: int
    speed_mps: float
    can_use_stairs: bool
    can_use_lift: bool
    response_delay_s: float
    node: str  # node the group is at, or the node it last left while in transit
    nxt: str | None = None  # node it is walking towards (None when standing at `node`)
    progress_m: float = 0.0  # metres walked along the current edge
    route: list[str] = field(default_factory=list)  # nodes still to visit after `nxt` / `node`
    dose: float = 0.0
    evacuated: bool = False
    evacuation_time_s: float | None = None
    wait_s: float = 0.0  # seconds spent queueing for an edge
    waiting: bool = False
    queue_since: float = 0.0  # first come, first served
    used_dt: float = 0.0  # seconds of this tick the group was still in the building

    @property
    def here(self) -> str:
        """Node the group is at, or heading to when in transit."""
        return self.nxt if self.nxt is not None else self.node


@dataclass
class SimState:
    t_s: float
    tick: int
    smoke: dict[str, float]  # node id -> intensity [0, 1]
    fire: set[str]  # burning nodes
    hazard: dict[str, HazardState]
    exits_blocked: set[str]  # exit node ids
    doors_stuck: set[str]  # door ids
    devices_ok: dict[str, bool]  # detector / lift node id -> False when failed
    groups: list[GroupState]
    covered_rooms: set[str]  # rooms with an occupancy sensor
    tokens: dict[tuple[str, str], float] = field(default_factory=dict)
    peak_queue_persons: int = 0
    peak_queue_delay_s: float = 0.0
    person_wait_s: float = 0.0
    invalid_decisions: int = 0


@dataclass(frozen=True)
class SimResult:
    evac_times_s: dict[str, float | None]  # group id -> time or None
    doses: dict[str, float]  # group id -> dose proxy per person
    sizes: dict[str, int]
    wait_s: dict[str, float]
    completion_rate: float  # persons evacuated / persons
    ticks_run: int
    final_t_s: float
    peak_queue_persons: int
    peak_queue_delay_s: float
    person_wait_s: float
    invalid_decisions: int
    observations: list[list[Observation]]  # one list per tick


class Simulator:
    def __init__(
        self,
        navgraph: NavGraph,
        scenario: ResolvedScenario,
        hazard_model: HazardModel,
        sensor_model: SensorModel,
        occupant_profiles: OccupantProfiles,
        sim_config: SimConfig,
        dose_config: DoseConfig,
        rng: np.random.Generator,
        *,
        profile_id: str | None = None,
        specific_flow_pers_per_m_s: float | None = None,
    ) -> None:
        self.navgraph = navgraph
        self.scenario = scenario
        self.hazard_model = hazard_model
        self.sensor_model = sensor_model
        self.sim_config = sim_config
        self.dose_config = dose_config
        self.flow = specific_flow_pers_per_m_s  # None = unlimited
        self._rng0 = copy.deepcopy(rng)

        self.nodes: dict[str, NavNode] = {n.node_id: n for n in navgraph.nodes}
        self.adj: dict[str, list[tuple[str, NavEdge]]] = {n: [] for n in self.nodes}
        self.edge: dict[tuple[str, str], NavEdge] = {}
        for e in navgraph.edges:
            self.adj[e.u].append((e.v, e))
            self.adj[e.v].append((e.u, e))
            self.edge[(e.u, e.v)] = e
            self.edge[(e.v, e.u)] = e
        for lst in self.adj.values():
            lst.sort(key=lambda x: x[0])

        self._templates = self._index_templates(occupant_profiles, profile_id)
        hp = hazard_model.params
        self._smoke_base = rate_per_s(hp.smoke_diffusion, hp.rate_interval_s)
        self._gen_rate = rate_per_s(hp.smoke_generation, hp.rate_interval_s)
        self._decay_rate = rate_per_s(hp.smoke_decay, hp.rate_interval_s)
        self._exit_nodes = sorted(n for n, nd in self.nodes.items() if nd.kind == NodeKind.exit)
        self._state: SimState | None = None
        self._triggered: set[int] = set()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _index_templates(
        profiles: OccupantProfiles, profile_id: str | None
    ) -> dict[str, GroupTemplate]:
        """Group templates by id. The same id (e.g. 'impaired') exists in several profiles with
        different values, so the profile must be unambiguous: never let one silently win."""
        if profile_id is not None:
            prof = profiles.get(profile_id)
            if prof is None:
                raise ValueError(f"unknown occupant profile {profile_id!r}")
            return {g.template_id: g for g in prof.groups}
        if len(profiles.profiles) == 1:
            return {g.template_id: g for g in profiles.profiles[0].groups}
        seen: dict[str, GroupTemplate] = {}
        clash: set[str] = set()
        for prof in profiles.profiles:
            for g in prof.groups:
                if g.template_id in seen and seen[g.template_id] != g:
                    clash.add(g.template_id)
                seen.setdefault(g.template_id, g)
        if clash:
            raise ValueError(
                f"group templates {sorted(clash)} differ between occupant profiles; "
                "pass profile_id to the Simulator"
            )
        return seen

    def _is_sensor_node(self, node: NavNode) -> bool:
        placement = self.sensor_model.smoke_placement
        if placement == SensorPlacement.all_rooms:
            return node.kind == NodeKind.room
        if placement == SensorPlacement.corridors_only:
            return node.kind == NodeKind.corridor
        if placement == SensorPlacement.rooms_and_corridors:
            return node.kind in (NodeKind.room, NodeKind.corridor)
        return False

    def reset(self) -> SimState:
        self.rng_fire, self.rng_sensor = copy.deepcopy(self._rng0).spawn(2)
        covered = {
            nid
            for nid in sorted(self.nodes)
            if self.nodes[nid].kind == NodeKind.room
            and self.rng_sensor.random() < self.sensor_model.occupancy_sensor_coverage
        }
        groups: list[GroupState] = []
        for rg in self.scenario.groups:
            tpl = self._templates.get(rg.template_id)
            if tpl is None:
                raise ValueError(
                    f"group {rg.group_id}: template {rg.template_id!r} is not defined by the "
                    "occupant profile"
                )
            if rg.start_node not in self.nodes:
                raise ValueError(f"group {rg.group_id}: unknown start node {rg.start_node!r}")
            is_exit = self.nodes[rg.start_node].kind == NodeKind.exit
            groups.append(
                GroupState(
                    group_id=rg.group_id,
                    template_id=rg.template_id,
                    size=rg.size,
                    speed_mps=tpl.speed_mps,
                    can_use_stairs=tpl.can_use_stairs,
                    can_use_lift=tpl.can_use_lift,
                    response_delay_s=tpl.response_delay_s,
                    node=rg.start_node,
                    evacuated=is_exit,
                    evacuation_time_s=0.0 if is_exit else None,
                )
            )
        state = SimState(
            t_s=0.0,
            tick=0,
            smoke={nid: 0.0 for nid in self.nodes},
            fire=set(),
            hazard={nid: HazardState.clear for nid in self.nodes},
            exits_blocked=set(),
            doors_stuck=set(),
            devices_ok={},
            groups=groups,
            covered_rooms=covered,
        )
        self._triggered = set()
        self._state = state
        return state

    # ------------------------------------------------------------------ views

    def public_state(self, state: SimState) -> PublicState:
        return PublicState(
            t_s=state.t_s,
            tick=state.tick,
            groups=tuple(
                PublicGroup(
                    group_id=g.group_id,
                    template_id=g.template_id,
                    size=g.size,
                    node=g.here,
                    in_transit=g.nxt is not None,
                    came_from=g.node if g.nxt is not None else None,
                    waiting=g.waiting,
                    evacuated=g.evacuated,
                    speed_mps=g.speed_mps,
                    can_use_stairs=g.can_use_stairs,
                    can_use_lift=g.can_use_lift,
                )
                for g in state.groups
            ),
        )

    def ground_truth(self, state: SimState | None = None) -> GroundTruthState:
        st = state or self._state
        if st is None:
            raise RuntimeError("call reset() first")
        return GroundTruthState(
            graph_hash=self.navgraph.derived_from,
            t_s=st.t_s,
            hazard=dict(st.hazard),
            smoke=dict(st.smoke),
            exits_blocked=tuple(sorted(self.nodes[n].region_id or n for n in st.exits_blocked)),
            devices_ok=dict(st.devices_ok),
        )

    def _flow_cap(self, edge: NavEdge) -> float | None:
        return None if self.flow is None else edge.width_m * self.flow

    # ------------------------------------------------------------------ hazard

    def _hazard_from_smoke(self, value: float) -> HazardState:
        m = self.hazard_model.state_mapping
        if value >= m.fire_min:
            return HazardState.fire
        if value >= m.smoky_min:
            return HazardState.smoky
        return HazardState.clear

    def _refresh_hazard(self, state: SimState) -> None:
        """Discrete hazard state: burning nodes are 'fire', others follow the smoke thresholds."""
        for nid, s in state.smoke.items():
            state.hazard[nid] = (
                HazardState.fire if nid in state.fire else self._hazard_from_smoke(s)
            )

    def _trigger_events(self, state: SimState) -> None:
        for i, ev in enumerate(self.scenario.events):
            if i in self._triggered or state.t_s < ev.time_s:
                continue
            self._triggered.add(i)
            if ev.type == EventType.exit_blocked:
                state.exits_blocked.add(ev.target_id)
            elif ev.type in (EventType.detector_fail, EventType.lift_out):
                state.devices_ok[ev.target_id] = False
            elif ev.type == EventType.door_stuck:
                state.doors_stuck.add(ev.target_id)
            elif ev.type == EventType.second_fire and ev.target_id in self.nodes:
                state.fire.add(ev.target_id)
        sc = self.scenario
        if sc.ignition_node in self.nodes and state.t_s >= sc.ignition_time_s:
            state.fire.add(sc.ignition_node)

    def _spread_fire(self, state: SimState, dt: float) -> None:
        hp = self.hazard_model.params
        newly: set[str] = set()
        for u in sorted(state.fire):
            for v, e in self.adj[u]:
                if v in state.fire:
                    continue
                closed = e.door_id is not None and e.door_id in state.doors_stuck
                p1 = edge_spread_prob(hp, e, closed)
                if self.rng_fire.random() < prob_over(rate_per_s(p1, hp.rate_interval_s), dt):
                    newly.add(v)
                if hp.spread_kernel == SpreadKernel.two_hop:
                    for w, e2 in self.adj[v]:
                        if w == u or w in state.fire:
                            continue
                        closed2 = e2.door_id is not None and e2.door_id in state.doors_stuck
                        p2 = p1 * edge_spread_prob(hp, e2, closed2)
                        if self.rng_fire.random() < prob_over(
                            rate_per_s(p2, hp.rate_interval_s), dt
                        ):
                            newly.add(w)
        state.fire |= newly

    def _diffuse_smoke(self, state: SimState, dt: float) -> None:
        hp = self.hazard_model.params
        n_sub = max(1, math.ceil(dt / self.sim_config.max_substep_s - _EPS))
        h = dt / n_sub
        kappa: dict[str, list[tuple[str, float]]] = {
            u: [
                (
                    v,
                    self._smoke_base
                    * edge_smoke_factor(
                        hp, e, e.door_id is not None and e.door_id in state.doors_stuck
                    ),
                )
                for v, e in lst
            ]
            for u, lst in self.adj.items()
        }
        smoke = state.smoke
        for _ in range(n_sub):
            nxt: dict[str, float] = {}
            for u, s_u in smoke.items():
                flux = sum(k * (smoke[v] - s_u) for v, k in kappa[u])
                if u in state.fire:
                    flux += self._gen_rate * (1.0 - s_u)
                else:
                    flux -= self._decay_rate * s_u
                nxt[u] = min(1.0, max(0.0, s_u + h * flux))
            smoke = nxt
        state.smoke = smoke

    # ------------------------------------------------------------------ sensing

    def _observe(self, state: SimState) -> list[Observation]:
        sm = self.sensor_model
        rng = self.rng_sensor
        p_alarm = {
            HazardState.clear: sm.smoke_false_alarm_prob,
            HazardState.smoky: min(1.0, sm.smoke_detection_prob * sm.smoky_detection_factor),
            HazardState.fire: sm.smoke_detection_prob,
        }
        out: list[Observation] = []
        for nid, node in self.nodes.items():
            if not self._is_sensor_node(node) or state.devices_ok.get(nid) is False:
                continue
            alarm = rng.random() < p_alarm[state.hazard[nid]]
            out.append(
                Observation(
                    t_s=state.t_s,
                    channel=Channel.smoke,
                    source_id=nid,
                    value=1.0 if alarm else 0.0,
                    reliability=sm.smoke_detection_prob,
                )
            )

        report_ticks = max(1, round(sm.report_period_s / self.sim_config.tick_s))
        if state.tick % report_ticks != 0:
            return out

        def noisy(truth: bool, reliability: float) -> float:
            return 1.0 if (truth if rng.random() < reliability else not truth) else 0.0

        for nid in self._exit_nodes:
            out.append(
                Observation(
                    t_s=state.t_s,
                    channel=Channel.exit,
                    source_id=nid,
                    value=noisy(nid not in state.exits_blocked, sm.exit_report_reliability),
                    reliability=sm.exit_report_reliability,
                )
            )
        for nid in sorted(n for n, nd in self.nodes.items() if nd.kind == NodeKind.lift):
            out.append(
                Observation(
                    t_s=state.t_s,
                    channel=Channel.lift,
                    source_id=nid,
                    value=noisy(state.devices_ok.get(nid) is not False, sm.lift_report_reliability),
                    reliability=sm.lift_report_reliability,
                )
            )
        for door in sorted({e.door_id for e in self.navgraph.edges if e.door_id}):
            out.append(
                Observation(
                    t_s=state.t_s,
                    channel=Channel.device,
                    source_id=device_for_region(door),
                    value=noisy(door not in state.doors_stuck, sm.device_report_reliability),
                    reliability=sm.device_report_reliability,
                )
            )
        count: dict[str, int] = {}
        for g in state.groups:
            if not g.evacuated and g.nxt is None:
                count[g.node] = count.get(g.node, 0) + g.size
        for nid in sorted(state.covered_rooms):
            region = self.nodes[nid].region_id
            if region is None:
                continue
            z = count.get(nid, 0) + float(rng.normal(0.0, sm.occupancy_noise_sd))
            out.append(
                Observation(
                    t_s=state.t_s,
                    channel=Channel.occupancy,
                    source_id=region,
                    value=max(0.0, z),
                    reliability=sm.occupancy_sensor_coverage,
                )
            )
        for g in sorted(state.groups, key=lambda x: x.group_id):
            if g.evacuated or g.nxt is not None or g.node not in state.exits_blocked:
                continue
            out.append(
                Observation(
                    t_s=state.t_s,
                    channel=Channel.report,
                    source_id=g.group_id,
                    target_id=g.node,
                    value=noisy(False, sm.user_report_reliability),
                    reliability=sm.user_report_reliability,
                )
            )
        return out

    # ------------------------------------------------------------------ people

    def _passable(self, g: GroupState, u: str, v: str, state: SimState) -> bool:
        e = self.edge.get((u, v))
        if e is None:
            return False
        if e.door_id is not None and e.door_id in state.doors_stuck:
            return False
        if e.stairs and not g.can_use_stairs:
            return False
        for n in (u, v):
            if self.nodes[n].kind == NodeKind.lift and (
                self.sim_config.lifts_disabled_in_fire or not g.can_use_lift
            ):
                return False
        return True

    def _apply_decision(self, g: GroupState, nodes: list[str], state: SimState) -> bool:
        """Validate a route edge by edge and adopt it. False (group unchanged) if invalid.

        A group at a node follows the route from that node. A group in transit may continue to the
        node it is walking towards or turn back to the node it left; anything else is rejected.
        """
        turn = False
        if g.nxt is None:
            anchor = g.node
            if nodes and nodes[0] == g.node:
                nodes = nodes[1:]
        elif nodes and nodes[0] == g.nxt:
            anchor, nodes = g.nxt, nodes[1:]
        elif nodes and nodes[0] == g.node:
            anchor, nodes, turn = g.node, nodes[1:], True
        else:
            return False
        prev = anchor
        for n in nodes:
            if n not in self.nodes or not self._passable(g, prev, n, state):
                return False
            prev = n
        if turn:
            assert g.nxt is not None
            length = self.edge[(g.node, g.nxt)].length_m
            g.node, g.nxt = g.nxt, g.node
            g.progress_m = max(0.0, length - g.progress_m)
        g.route = list(nodes)
        return True

    def _smoke_at(self, g: GroupState, state: SimState) -> float:
        if g.nxt is None:
            return state.smoke[g.node]
        f = min(1.0, g.progress_m / self.edge[(g.node, g.nxt)].length_m)
        return (1.0 - f) * state.smoke[g.node] + f * state.smoke[g.nxt]

    def _move(self, g: GroupState, state: SimState, dt: float) -> None:
        """Advance one group by one tick: walk, queue for edge capacity, leave through an exit."""
        t0 = state.t_s
        g.used_dt = dt
        if t0 + dt <= g.response_delay_s + _EPS:
            return
        time_left = dt if t0 >= g.response_delay_s else t0 + dt - g.response_delay_s
        while time_left > _EPS and not g.evacuated:
            if g.nxt is None:
                if not g.route:
                    g.waiting = False
                    return
                nxt = g.route[0]
                if not self._passable(g, g.node, nxt, state):
                    g.route.clear()  # the way is closed: stop and wait for a new decision
                    g.waiting = False
                    return
                cap = self._flow_cap(self.edge[(g.node, nxt)])
                if cap is not None:
                    tok = state.tokens.get((g.node, nxt), cap * dt)
                    if tok <= 0.0:
                        g.waiting = True
                        g.wait_s += time_left
                        return
                    state.tokens[(g.node, nxt)] = tok - g.size
                g.waiting = False
                g.route.pop(0)
                g.nxt, g.progress_m = nxt, 0.0
            assert g.nxt is not None
            need = (self.edge[(g.node, g.nxt)].length_m - g.progress_m) / g.speed_mps
            if need > time_left + _EPS:
                g.progress_m += time_left * g.speed_mps
                return
            time_left = max(0.0, time_left - need)
            g.node, g.nxt, g.progress_m = g.nxt, None, 0.0
            g.queue_since = t0 + dt - time_left
            if self.nodes[g.node].kind == NodeKind.exit and g.node not in state.exits_blocked:
                g.evacuated = True
                g.evacuation_time_s = t0 + dt - time_left
                g.used_dt = g.evacuation_time_s - t0
                return

    def step(self, state: SimState, decision: Plan | None) -> tuple[SimState, list[Observation]]:
        dt = self.sim_config.tick_s
        self._trigger_events(state)
        self._spread_fire(state, dt)
        self._diffuse_smoke(state, dt)
        self._refresh_hazard(state)
        observations = self._observe(state)

        if decision is not None:
            by_group = {d.group_id: d for d in decision.decisions}
            for g in state.groups:
                d = by_group.get(g.group_id)
                if (
                    d is not None
                    and not g.evacuated
                    and not self._apply_decision(g, list(d.route.nodes), state)
                ):
                    state.invalid_decisions += 1

        for key in state.tokens:
            cap = self._flow_cap(self.edge[key])
            assert cap is not None
            state.tokens[key] = min(cap * dt, state.tokens[key] + cap * dt)

        active = sorted(
            (g for g in state.groups if not g.evacuated),
            key=lambda g: (g.queue_since, g.group_id),
        )
        start_smoke = {g.group_id: self._smoke_at(g, state) for g in active}
        for g in active:
            self._move(g, state, dt)
        k = self.dose_config.k
        for g in active:
            g.dose += k * 0.5 * (start_smoke[g.group_id] + self._smoke_at(g, state)) * g.used_dt

        queued: dict[tuple[str, str], int] = {}
        for g in state.groups:
            if g.waiting and not g.evacuated and g.route:
                key = (g.node, g.route[0])
                queued[key] = queued.get(key, 0) + g.size
                state.person_wait_s += g.size * dt
        for key, persons in queued.items():
            state.peak_queue_persons = max(state.peak_queue_persons, persons)
            cap = self._flow_cap(self.edge[key])
            if cap:
                state.peak_queue_delay_s = max(state.peak_queue_delay_s, persons / cap)

        state.tick += 1
        state.t_s = round(state.t_s + dt, 6)
        return state, observations

    def run(
        self,
        controller: Controller,
        on_tick: Callable[[SimState, Sequence[Observation]], None] | None = None,
    ) -> SimResult:
        state = self.reset()
        all_obs: list[list[Observation]] = []
        max_ticks = math.ceil(self.scenario.horizon_s / self.sim_config.tick_s)
        ticks_run = 0
        for _ in range(max_ticks):
            if all(g.evacuated for g in state.groups):
                break
            decision = None
            if state.tick % self.sim_config.decision_period_ticks == 0:
                decision = controller.decide(self.public_state(state))
            state, obs = self.step(state, decision)
            controller.observe(state.t_s, obs)
            all_obs.append(obs)
            ticks_run += 1
            if on_tick is not None:
                on_tick(state, obs)
        return self.summarise(state, ticks_run, all_obs)

    def summarise(
        self, state: SimState, ticks_run: int, observations: list[list[Observation]]
    ) -> SimResult:
        people = sum(g.size for g in state.groups)
        done = sum(g.size for g in state.groups if g.evacuated)
        return SimResult(
            evac_times_s={g.group_id: g.evacuation_time_s for g in state.groups},
            doses={g.group_id: g.dose for g in state.groups},
            sizes={g.group_id: g.size for g in state.groups},
            wait_s={g.group_id: g.wait_s for g in state.groups},
            completion_rate=float(done / people) if people else 1.0,
            ticks_run=ticks_run,
            final_t_s=state.t_s,
            peak_queue_persons=state.peak_queue_persons,
            peak_queue_delay_s=state.peak_queue_delay_s,
            person_wait_s=state.person_wait_s,
            invalid_decisions=state.invalid_decisions,
            observations=observations,
        )
