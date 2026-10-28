"""'Raw observations' tracker: no filtering, no memory beyond the latest reading per sensor.

Used for the ablation 'decisions on raw observations instead of belief' (B4, B6). It presents the
latest readings as a degenerate belief so planners and policies receive the same data structure.
"""

from __future__ import annotations

from collections.abc import Sequence

from smartbuilding.core.devices import device_for_region
from smartbuilding.core.navgraph import NavGraph
from smartbuilding.core.primitives import Channel, NodeKind
from smartbuilding.core.runtime import BeliefState, Observation
from smartbuilding.uncertainty.belief import initial_belief

_ALARM = 0.5


class RawTracker:
    def __init__(self, navgraph: NavGraph) -> None:
        self._graph = navgraph
        self._base = initial_belief(navgraph)
        self._exit_key = {
            n.node_id: n.region_id
            for n in navgraph.nodes
            if n.kind == NodeKind.exit and n.region_id
        }
        self._lift_key = {
            n.node_id: device_for_region(n.region_id)
            for n in navgraph.nodes
            if n.kind == NodeKind.lift and n.region_id
        }
        self._alarm: dict[str, bool] = {}
        self._exit: dict[str, float] = {}
        self._dev: dict[str, float] = {}
        self._t = 0.0

    def observe(self, t_s: float, observations: Sequence[Observation]) -> None:
        self._t = t_s
        for ob in observations:
            if ob.channel == Channel.smoke:
                self._alarm[ob.source_id] = ob.value >= _ALARM
            elif ob.channel == Channel.exit and ob.source_id in self._exit_key:
                self._exit[self._exit_key[ob.source_id]] = 1.0 if ob.value >= _ALARM else 0.0
            elif ob.channel == Channel.report and ob.target_id in self._exit_key:
                self._exit[self._exit_key[ob.target_id]] = 1.0 if ob.value >= _ALARM else 0.0
            elif ob.channel == Channel.lift and ob.source_id in self._lift_key:
                self._dev[self._lift_key[ob.source_id]] = 1.0 if ob.value >= _ALARM else 0.0
            elif ob.channel == Channel.device:
                self._dev[ob.source_id] = 1.0 if ob.value >= _ALARM else 0.0

    def belief(self) -> BeliefState:
        node_belief = {
            nid: ((0.0, 1.0, 0.0) if self._alarm.get(nid) else (1.0, 0.0, 0.0))
            for nid in self._base.node_belief
        }
        exits = {k: self._exit.get(k, v) for k, v in self._base.exit_avail.items()}
        devices = {k: self._dev.get(k, v) for k, v in self._base.devices.items()}
        return BeliefState(
            graph_hash=self._base.graph_hash,
            t_s=self._t,
            node_belief=node_belief,
            exit_avail=exits,
            devices=devices,
            occupancy=dict(self._base.occupancy),
        )
