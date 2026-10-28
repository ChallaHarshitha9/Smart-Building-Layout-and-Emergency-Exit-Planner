import pytest
from pydantic import ValidationError

from smartbuilding.core.layout import (
    ConstraintResult,
    FloorPlan,
    Layout,
    Lift,
    Rect,
    RoomPlacement,
    Stair,
    Verification,
    VerticalCore,
)
from smartbuilding.core.navgraph import NavEdge, NavGraph, NavNode
from tests.factories import make_layout, make_navgraph


def test_design_invalid_layouts_are_representable():
    """Overlapping, out-of-bounds rooms must load: the checker rejects them, not the contract."""
    plan = FloorPlan(
        floor=0,
        rooms=(
            RoomPlacement(
                room_id="R_office_1", room_type="office", rect=Rect(x=0, y=0, w=10, d=10)
            ),
            RoomPlacement(
                room_id="R_office_2", room_type="office", rect=Rect(x=5, y=5, w=500, d=10)
            ),
        ),
    )
    layout = make_layout(floors=(plan,))
    a, b = (r.rect for r in layout.floors[0].rooms)
    assert a.overlaps(b)


def test_rect_overlap_edges_touching_is_not_overlap():
    assert not Rect(x=0, y=0, w=2, d=2).overlaps(Rect(x=2, y=0, w=2, d=2))
    assert Rect(x=0, y=0, w=3, d=3).overlaps(Rect(x=2, y=2, w=3, d=3))


def test_conversion_keeps_the_id_and_must_be_recorded():
    rect = Rect(x=0, y=0, w=1, d=1)
    # converting a classroom to a lab keeps the stable id
    ok = RoomPlacement(
        room_id="R_classroom_1", room_type="lab", rect=rect, converted_from="classroom"
    )
    assert ok.room_id == "R_classroom_1"
    with pytest.raises(ValidationError, match="converted_from"):
        RoomPlacement(room_id="R_classroom_1", room_type="lab", rect=rect)
    with pytest.raises(ValidationError, match="converted_from"):
        RoomPlacement(room_id="R_classroom_1", room_type="lab", rect=rect, converted_from="office")
    with pytest.raises(ValidationError, match="must be empty"):
        RoomPlacement(room_id="R_lab_1", room_type="lab", rect=rect, converted_from="classroom")


def test_duplicate_ids_rejected(layout):
    plan = layout.floors[0]
    dup = plan.model_copy(update={"rooms": (plan.rooms[0], plan.rooms[0])})
    with pytest.raises(ValidationError, match="unique"):
        make_layout(floors=(dup,))


def test_door_and_exit_must_reference_existing_regions(layout):
    plan = layout.floors[0]
    bad_door = plan.doors[0].model_copy(update={"b": "C_9"})
    with pytest.raises(ValidationError, match="unknown region"):
        make_layout(floors=(plan.model_copy(update={"doors": (bad_door,)}),))
    bad_exit = plan.exits[0].model_copy(update={"connects_to": "C_9"})
    with pytest.raises(ValidationError, match="unknown region"):
        make_layout(floors=(plan.model_copy(update={"exits": (bad_exit,)}),))


def test_floor_consistency_and_shafts(layout):
    plan = layout.floors[0]
    wrong_floor = plan.rooms[0].model_copy(update={"floor": 1})
    with pytest.raises(ValidationError, match="floor"):
        make_layout(floors=(plan.model_copy(update={"rooms": (wrong_floor, plan.rooms[1])}),))
    core = VerticalCore(
        stairs=(
            Stair(stair_id="S_1", rect=Rect(x=0, y=0, w=4, d=4), floors=(0, 1), width_cells=2),
        ),
        lifts=(Lift(lift_id="L_1", rect=Rect(x=4, y=0, w=4, d=4), floors=(0, 1)),),
    )
    with pytest.raises(ValidationError, match="does not exist"):
        make_layout(vertical_core=core)  # only floor 0 exists


def test_geometry_hash_ignores_ids_verification_provenance(layout):
    other = make_layout(
        layout_id="LY_other",
        verification=Verification(
            status="checked",
            hard=(ConstraintResult(constraint_id="HC_x", kind="hard", satisfied=True),),
        ),
    )
    assert other.geometry_hash == layout.geometry_hash
    assert other.compute_hash() != layout.compute_hash()
    assert make_layout(grid_w=61).geometry_hash != layout.geometry_hash


def test_verification_rules():
    with pytest.raises(ValidationError, match="unchecked"):
        Verification(hard=(ConstraintResult(constraint_id="HC_x", kind="hard", satisfied=True),))
    v = Verification(
        status="checked",
        hard=(
            ConstraintResult(constraint_id="HC_a", kind="hard", satisfied=True),
            ConstraintResult(constraint_id="HC_b", kind="hard", satisfied=False),
        ),
    )
    assert not v.all_hard_satisfied
    assert not Verification().all_hard_satisfied


def test_layout_helpers(layout):
    assert layout.room_ids() == ("R_office_1", "R_office_2")
    assert {"C_1", "E_1", "R_office_1"} <= layout.region_ids()
    assert layout.floor_plan(0).floor == 0
    with pytest.raises(KeyError):
        layout.floor_plan(3)
    assert layout.floors[0].corridors[0].width_cells == 4


def test_navgraph_roundtrip_and_helpers(navgraph):
    assert NavGraph.model_validate_json(navgraph.model_dump_json()) == navgraph
    assert len(navgraph.nodes_of_kind("exit")) == 2
    adj = navgraph.adjacency()
    assert {n for n, _ in adj["N_corridor_1"]} == {"N_room_1", "N_room_2", "N_exit_1", "N_exit_2"}
    assert navgraph.node("N_exit_1").kind == "exit"


def test_edge_endpoints_are_sorted_so_hashes_match():
    a = NavEdge(u="N_room_2", v="N_room_1", length_m=1, width_m=1)
    b = NavEdge(u="N_room_1", v="N_room_2", length_m=1, width_m=1)
    assert a == b and (a.u, a.v) == ("N_room_1", "N_room_2")


def test_navgraph_structural_rules(navgraph):
    with pytest.raises(ValidationError, match="does not match kind"):
        NavNode(node_id="N_room_1", kind="exit", x_m=0, y_m=0)
    with pytest.raises(ValidationError, match="self-loops"):
        NavEdge(u="N_room_1", v="N_room_1", length_m=1, width_m=1)
    with pytest.raises(ValidationError, match="step-free"):
        NavEdge(u="N_room_1", v="N_room_2", length_m=1, width_m=1, stairs=True)
    with pytest.raises(ValidationError, match="unknown node"):
        NavGraph(
            derived_from="a" * 64,
            nodes=navgraph.nodes,
            edges=(NavEdge(u="N_room_1", v="N_room_9", length_m=1, width_m=1),),
        )
    dup = (navgraph.edges[0], navgraph.edges[0])
    with pytest.raises(ValidationError, match="duplicate"):
        NavGraph(derived_from="a" * 64, nodes=navgraph.nodes, edges=dup)
    with pytest.raises(ValidationError, match="unique"):
        NavGraph(derived_from="a" * 64, nodes=(navgraph.nodes[0], navgraph.nodes[0]))


def test_navgraph_links_to_layout_geometry(layout):
    graph = make_navgraph(derived_from=layout.geometry_hash)
    assert graph.derived_from == layout.geometry_hash


def test_layout_type_is_exported():
    assert Layout.__name__ == "Layout"
