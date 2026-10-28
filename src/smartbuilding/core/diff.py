"""Comparing a layout with the layout it was repaired from.

Two jobs: (1) check that room ids stay stable across repair, which the adaptability metric
needs; (2) count how many entities changed (a simple edit distance).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from smartbuilding.core.layout import Layout


def check_stable_room_ids(
    parent: Layout,
    child: Layout,
    *,
    allow_removed: Iterable[str] = (),
    allow_added: Iterable[str] = (),
) -> list[str]:
    """Problems if room ids changed between ``parent`` and ``child`` without permission."""
    before, after = set(parent.room_ids()), set(child.room_ids())
    problems = [f"room {r} disappeared" for r in sorted(before - after - set(allow_removed))]
    problems += [f"room {r} appeared" for r in sorted(after - before - set(allow_added))]
    return problems


@dataclass(frozen=True)
class EditDistance:
    rooms: int
    corridors: int
    doors: int
    exits: int
    vertical_core: int

    @property
    def total(self) -> int:
        return self.rooms + self.corridors + self.doors + self.exits + self.vertical_core


def _changed(a: dict[str, object], b: dict[str, object]) -> int:
    """Entities added, removed or modified between two id-keyed maps."""
    keys = set(a) | set(b)
    return sum(1 for k in keys if a.get(k) != b.get(k))


def layout_edit_distance(a: Layout, b: Layout) -> EditDistance:
    """Number of entities that differ. One changed room counts once, however much it moved."""

    def rooms(lay: Layout) -> dict[str, object]:
        return {r.room_id: r for p in lay.floors for r in p.rooms}

    def corridors(lay: Layout) -> dict[str, object]:
        return {c.corridor_id: c for p in lay.floors for c in p.corridors}

    def doors(lay: Layout) -> dict[str, object]:
        return {d.door_id: d for p in lay.floors for d in p.doors}

    def exits(lay: Layout) -> dict[str, object]:
        return {e.exit_id: e for p in lay.floors for e in p.exits}

    def core(lay: Layout) -> dict[str, object]:
        out: dict[str, object] = {s.stair_id: s for s in lay.vertical_core.stairs}
        out.update({s.lift_id: s for s in lay.vertical_core.lifts})
        return out

    return EditDistance(
        rooms=_changed(rooms(a), rooms(b)),
        corridors=_changed(corridors(a), corridors(b)),
        doors=_changed(doors(a), doors(b)),
        exits=_changed(exits(a), exits(b)),
        vertical_core=_changed(core(a), core(b)),
    )
