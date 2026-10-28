"""B3: the dose-aware planner fed with ground truth. An upper bound, never a deployable method.

This module is in eval/, the only place besides sim/ that may read GroundTruthState.
"""

from __future__ import annotations

from collections.abc import Sequence

from smartbuilding.core.runtime import BeliefState, Observation
from smartbuilding.sim.engine import Simulator
from smartbuilding.uncertainty.belief import initial_belief


class OracleTracker:
    def __init__(self, sim: Simulator) -> None:
        self._sim = sim
        self._base = initial_belief(sim.navgraph)

    def observe(self, t_s: float, observations: Sequence[Observation]) -> None:
        return None

    def belief(self) -> BeliefState:
        gt = self._sim.ground_truth()
        node_belief = {
            nid: (
                (1.0, 0.0, 0.0)
                if gt.hazard[nid].value == "clear"
                else (0.0, 1.0, 0.0)
                if gt.hazard[nid].value == "smoky"
                else (0.0, 0.0, 1.0)
            )
            for nid in self._base.node_belief
        }
        exits = {k: (0.0 if k in gt.exits_blocked else 1.0) for k in self._base.exit_avail}
        return BeliefState(
            graph_hash=self._base.graph_hash,
            t_s=gt.t_s,
            node_belief=node_belief,
            exit_avail=exits,
            devices=dict(self._base.devices),
            occupancy=dict(self._base.occupancy),
        )
