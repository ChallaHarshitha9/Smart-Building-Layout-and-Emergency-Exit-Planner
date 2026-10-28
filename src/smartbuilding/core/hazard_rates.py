"""Conversion of per-interval probabilities to per-second rates (tick-invariant dynamics)."""

import math

from smartbuilding.core.config import HazardParams
from smartbuilding.core.navgraph import NavEdge

_MAX_RATE_P = 0.999999


def rate_per_s(p_per_interval: float, interval_s: float) -> float:
    """Constant hazard rate lambda such that 1 - exp(-lambda * interval) = p."""
    p = min(max(p_per_interval, 0.0), _MAX_RATE_P)
    if p <= 0.0:
        return 0.0
    return -math.log1p(-p) / interval_s


def prob_over(rate: float, dt_s: float) -> float:
    """Probability that an event with the given rate happens within dt_s."""
    return -math.expm1(-rate * dt_s) if rate > 0.0 else 0.0


def edge_spread_prob(params: HazardParams, edge: NavEdge, door_closed: bool = False) -> float:
    """Per-interval fire spread probability across one edge, by edge type."""
    if edge.door_id is None:
        return params.spread_prob_open_space
    return params.spread_prob_closed_door if door_closed else params.spread_prob_open_door


def edge_smoke_factor(params: HazardParams, edge: NavEdge, door_closed: bool = False) -> float:
    """Smoke conductance of an edge relative to open space (1.0)."""
    base = params.spread_prob_open_space
    if base <= 0.0:
        return 0.0
    return edge_spread_prob(params, edge, door_closed) / base
