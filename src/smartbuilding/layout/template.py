"""Corridor template: one central corridor with a row of rooms on each side.

Used for the hand-made reference layouts now and by the solver later (the solver chooses the
room order and widths; this module turns that choice into a Layout).

Geometry (cells): the corridor runs the full width. South rooms sit below it, north rooms above
it, packed left to right from x = 0. Each room gets one door, centred on the wall it shares
with the corridor. Exits are openings in the corridor's two end walls.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from smartbuilding.core.layout import (
    CorridorPlacement,
    Door,
    Exit,
    FloorPlan,
    Layout,
    LayoutProvenance,
    Rect,
    RoomPlacement,
)
from smartbuilding.core.primitives import Side
from smartbuilding.core.spec import BuildingSpec


@dataclass(frozen=True)
class SlotRoom:
    room_id: str
    room_type: str
    width_cells: int


@dataclass(frozen=True)
class ExitPlan:
    side: Side  # only west and east (the corridor ends) are supported
    width_cells: int = 3


def assemble_corridor_layout(
    *,
    layout_id: str,
    spec: BuildingSpec,
    south: Sequence[SlotRoom],
    north: Sequence[SlotRoom],
    corridor_width_cells: int = 4,
    south_depth_cells: int | None = None,
    north_depth_cells: int | None = None,
    door_width_cells: int = 2,
    exits: Sequence[ExitPlan] = (ExitPlan(Side.west), ExitPlan(Side.east)),
    cell_size_m: float = 0.5,
    generator: str = "corridor_template",
    seed: int = 0,
    parent_layout_id: str | None = None,
) -> Layout:
    grid_w = round(spec.footprint.width_m / cell_size_m)
    grid_d = round(spec.footprint.depth_m / cell_size_m)
    cw = corridor_width_cells
    sd = south_depth_cells if south_depth_cells is not None else (grid_d - cw) // 2
    nd = north_depth_cells if north_depth_cells is not None else grid_d - cw - sd
    corridor = Rect(x=0, y=sd, w=grid_w, d=cw)

    rooms: list[RoomPlacement] = []
    doors: list[Door] = []
    door_no = 0

    def place(slots: Sequence[SlotRoom], y: int, depth: int, wall_y: int) -> None:
        nonlocal door_no
        x = 0
        for slot in slots:
            rooms.append(
                RoomPlacement(
                    room_id=slot.room_id,
                    room_type=slot.room_type,
                    rect=Rect(x=x, y=y, w=slot.width_cells, d=depth),
                )
            )
            dw = min(door_width_cells, slot.width_cells)
            door_no += 1
            doors.append(
                Door(
                    door_id=f"D_{door_no}",
                    a=slot.room_id,
                    b="C_1",
                    x=x + (slot.width_cells - dw) // 2,
                    y=wall_y,
                    width_cells=dw,
                )
            )
            x += slot.width_cells

    place(south, 0, sd, sd)
    place(north, sd + cw, nd, sd + cw)

    exit_list: list[Exit] = []
    for i, plan in enumerate(exits, start=1):
        if plan.side not in (Side.west, Side.east):
            raise ValueError("corridor template exits must be on the west or east end")
        if plan.width_cells > cw:
            raise ValueError("an exit cannot be wider than the corridor")
        exit_list.append(
            Exit(
                exit_id=f"E_{i}",
                x=0 if plan.side == Side.west else grid_w,
                y=sd + (cw - plan.width_cells) // 2,
                width_cells=plan.width_cells,
                connects_to="C_1",
                side=plan.side,
            )
        )

    return Layout(
        layout_id=layout_id,
        spec_hash=spec.compute_hash(),
        cell_size_m=cell_size_m,
        grid_w=grid_w,
        grid_d=grid_d,
        floors=(
            FloorPlan(
                floor=0,
                rooms=tuple(rooms),
                corridors=(CorridorPlacement(corridor_id="C_1", rect=corridor),),
                doors=tuple(doors),
                exits=tuple(exit_list),
            ),
        ),
        provenance=LayoutProvenance(
            generator=generator, seed=seed, parent_layout_id=parent_layout_id
        ),
    )
