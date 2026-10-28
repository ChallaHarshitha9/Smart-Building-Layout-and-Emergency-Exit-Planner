"""Relations between rooms: merging catalogue defaults and scoring how well a layout meets them."""

from collections.abc import Mapping

from smartbuilding.core.config import RoomCatalogue, Thresholds
from smartbuilding.core.primitives import ROOM_INSTANCE_RE, RelationType
from smartbuilding.core.spec import BuildingSpec, Relation


def type_of_ref(ref: str) -> str:
    """The room type a reference stands for (an instance id like R_lab_1 gives 'lab')."""
    match = ROOM_INSTANCE_RE.match(ref)
    return match.group(1) if match else ref


def effective_relations(
    spec: BuildingSpec, catalogue: RoomCatalogue, thresholds: Thresholds
) -> tuple[Relation, ...]:
    """The spec's relations plus catalogue defaults for type pairs the spec does not mention."""
    out = list(spec.relations)
    spec_types = {r.room_type for r in spec.rooms}
    covered = {frozenset((type_of_ref(r.a), type_of_ref(r.b))) for r in spec.relations}
    for d in catalogue.default_relations:
        if d.a not in spec_types or d.b not in spec_types:
            continue
        if frozenset((d.a, d.b)) in covered:
            continue
        extra: dict[str, float] = {}
        if d.hard and d.type == RelationType.near:
            extra["max_distance_m"] = thresholds.near_reference_distance_m.value
        if d.hard and d.type == RelationType.far:
            extra["min_distance_m"] = thresholds.far_reference_distance_m.value
        out.append(Relation(a=d.a, b=d.b, type=d.type, hard=d.hard, weight=d.weight, **extra))
    return tuple(out)


def members(room_types: Mapping[str, str], ref: str) -> list[str]:
    """Placed rooms a reference points to, sorted by id."""
    if ROOM_INSTANCE_RE.match(ref):
        return [ref] if ref in room_types else []
    return sorted(rid for rid, t in room_types.items() if t == ref)


def closeness(distance: float, reference: float) -> float:
    """1 within the reference distance, falling linearly to 0 at twice the reference."""
    if distance <= reference:
        return 1.0
    return max(0.0, 1.0 - (distance - reference) / reference)


def pair_quality(
    rel: Relation,
    distance: float,
    contact_cells: int,
    adjacent_min_contact_cells: int,
    thresholds: Thresholds,
) -> float:
    """How well one room pair meets a relation, between 0 and 1."""
    near_ref = rel.max_distance_m or thresholds.near_reference_distance_m.value
    far_ref = rel.min_distance_m or (
        thresholds.incompatible_min_separation_m.value
        if rel.type == RelationType.incompatible
        else thresholds.far_reference_distance_m.value
    )
    if rel.type == RelationType.adjacent:
        if contact_cells >= adjacent_min_contact_cells:
            return 1.0
        return 0.5 * closeness(distance, near_ref)
    if rel.type == RelationType.near:
        return closeness(distance, near_ref)
    # far and incompatible: larger distance is better; unreachable counts as far apart
    if distance == float("inf") or distance >= far_ref:
        return 1.0
    return max(0.0, distance / far_ref)
