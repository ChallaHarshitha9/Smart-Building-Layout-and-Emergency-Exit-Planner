"""Small valid objects used across tests. Later steps reuse them."""

from pathlib import Path

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
from smartbuilding.core.navgraph import NavEdge, NavGraph, NavNode
from smartbuilding.core.spec import BuildingSpec

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
EXAMPLES = ROOT / "data" / "examples"
HASH_A = "a" * 64


def make_layout(layout_id: str = "LY_test", **overrides: object) -> Layout:
    """Two offices on either side of a corridor, two doors, two exits (0.5 m cells)."""
    plan = FloorPlan(
        floor=0,
        rooms=(
            RoomPlacement(
                room_id="R_office_1", room_type="office", rect=Rect(x=0, y=20, w=20, d=16)
            ),
            RoomPlacement(
                room_id="R_office_2", room_type="office", rect=Rect(x=0, y=0, w=20, d=16)
            ),
        ),
        corridors=(CorridorPlacement(corridor_id="C_1", rect=Rect(x=0, y=16, w=60, d=4)),),
        doors=(
            Door(door_id="D_1", a="R_office_1", b="C_1", x=10, y=20, width_cells=2),
            Door(door_id="D_2", a="R_office_2", b="C_1", x=10, y=16, width_cells=2),
        ),
        exits=(
            Exit(exit_id="E_1", x=0, y=17, width_cells=3, connects_to="C_1"),
            Exit(exit_id="E_2", x=60, y=17, width_cells=3, connects_to="C_1"),
        ),
    )
    data: dict[str, object] = {
        "layout_id": layout_id,
        "spec_hash": HASH_A,
        "grid_w": 60,
        "grid_d": 36,
        "floors": (plan,),
        "provenance": LayoutProvenance(generator="test", seed=1),
    }
    data.update(overrides)
    return Layout(**data)  # type: ignore[arg-type]


def make_navgraph(derived_from: str = HASH_A) -> NavGraph:
    nodes = (
        NavNode(node_id="N_room_1", kind="room", x_m=5, y_m=14, capacity_pers=10, area_m2=80),
        NavNode(node_id="N_room_2", kind="room", x_m=5, y_m=4, capacity_pers=10, area_m2=80),
        NavNode(node_id="N_corridor_1", kind="corridor", x_m=15, y_m=9, area_m2=60),
        NavNode(node_id="N_exit_1", kind="exit", x_m=0, y_m=9),
        NavNode(node_id="N_exit_2", kind="exit", x_m=30, y_m=9),
    )
    edges = (
        NavEdge(u="N_room_1", v="N_corridor_1", length_m=6, width_m=1.0, door_id="D_1"),
        NavEdge(u="N_room_2", v="N_corridor_1", length_m=6, width_m=1.0, door_id="D_2"),
        NavEdge(u="N_corridor_1", v="N_exit_1", length_m=15, width_m=2.0),
        NavEdge(u="N_corridor_1", v="N_exit_2", length_m=15, width_m=2.0),
    )
    return NavGraph(derived_from=derived_from, nodes=nodes, edges=edges)


def load_example_spec(name: str = "school_spec.json") -> BuildingSpec:
    return BuildingSpec.model_validate_json((EXAMPLES / name).read_text(encoding="utf-8"))
