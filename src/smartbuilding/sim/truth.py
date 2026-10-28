"""Ground-truth simulator state.

INFORMATION BARRIER: only the simulator, the evaluation code and the oracle baseline may
import this module. Planners, belief, RL and GenAI code must not (enforced by a test).
The simulator hot path uses arrays; this class is for logging, tests and replays.
"""

from smartbuilding.core.base import Contract
from smartbuilding.core.primitives import (
    DeviceId,
    ExitId,
    Hash,
    HazardState,
    NodeId,
    NonNegFloat,
    Prob,
)


class GroundTruthState(Contract):
    """Simulator-only. Never visible to planners, belief or RL (except the oracle baseline)."""

    graph_hash: Hash
    t_s: NonNegFloat
    hazard: dict[NodeId, HazardState]
    smoke: dict[NodeId, Prob]
    exits_blocked: tuple[ExitId, ...] = ()
    devices_ok: dict[DeviceId, bool] = {}
