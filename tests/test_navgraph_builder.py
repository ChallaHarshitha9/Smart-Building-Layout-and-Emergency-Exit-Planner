"""Tests for layout/graph.py — build_topology and build_navgraph.

Uses the factory helpers from tests/factories.py so these tests do not
depend on real specs or config bundles.  All tests are fully deterministic.
"""

from smartbuilding.core.navgraph import NavGraph
from smartbuilding.core.primitives import NodeKind
from smartbuilding.layout.graph import build_navgraph, build_topology
from smartbuilding.layout.paths import INF, shortest_distances
from tests.factories import HASH_A, make_layout, make_navgraph

# --------------------------------------------------------------------------- tests


def test_build_navgraph_from_factory_layout():
    """build_navgraph on the factory layout must produce room, corridor and exit nodes
    with at least one edge connecting them."""
    layout = make_layout()
    navgraph = build_navgraph(layout)

    kinds = {n.kind for n in navgraph.nodes}
    assert NodeKind.room in kinds, "NavGraph must contain room nodes"
    assert NodeKind.corridor in kinds, "NavGraph must contain corridor nodes"
    assert NodeKind.exit in kinds, "NavGraph must contain exit nodes"
    assert len(navgraph.edges) > 0, "NavGraph must contain at least one edge"


def test_all_rooms_have_nodes():
    """Every room_id in the layout must map to a node in the topology."""
    layout = make_layout()
    topo = build_topology(layout)

    room_ids = {r.room_id for plan in layout.floors for r in plan.rooms}
    assert room_ids == set(topo.room_node.keys()), (
        "room_node dict must contain an entry for every room in the layout"
    )
    node_ids = {n.node_id for n in topo.nodes}
    for room_id, node_id in topo.room_node.items():
        assert node_id in node_ids, (
            f"room_node[{room_id!r}] = {node_id!r} does not exist in topo.nodes"
        )


def test_all_exits_have_nodes():
    """Every geometrically valid exit must appear in topology.exit_node."""
    layout = make_layout()
    topo = build_topology(layout)

    exit_ids = {e.exit_id for plan in layout.floors for e in plan.exits}
    # The factory layout has no invalid exits, so all should be mapped.
    for exit_id in exit_ids:
        assert exit_id in topo.exit_node, (
            f"Exit {exit_id!r} should be in topo.exit_node but is missing"
        )
    node_ids = {n.node_id for n in topo.nodes}
    for exit_id, node_id in topo.exit_node.items():
        assert node_id in node_ids, (
            f"exit_node[{exit_id!r}] = {node_id!r} does not exist in topo.nodes"
        )


def test_shortest_path_room_to_exit_is_finite():
    """Multi-source Dijkstra from exits must reach every room (distance < INF)."""
    layout = make_layout()
    topo = build_topology(layout)

    adj = topo.adjacency()
    exit_nodes = list(topo.exit_node.values())
    distances = shortest_distances(adj, exit_nodes)

    for room_id, room_node_id in topo.room_node.items():
        d = distances.get(room_node_id, INF)
        assert d < INF, f"Room {room_id!r} (node {room_node_id!r}) is not reachable from any exit"


def test_invalid_door_not_on_shared_wall():
    """A door whose coordinates fall outside the shared wall must be reported as invalid."""
    from smartbuilding.core.layout import (
        Door,
        FloorPlan,
        Layout,
        LayoutProvenance,
        Rect,
        RoomPlacement,
        VerticalCore,
    )

    # Two horizontally adjacent rooms sharing the vertical wall at x=10.
    # Door connects them but its y-coord is pushed far above both rooms (y=20)
    # which is outside the shared wall segment.
    plan = FloorPlan(
        floor=0,
        rooms=(
            RoomPlacement(
                room_id="R_office_1", room_type="office", rect=Rect(x=0, y=0, w=10, d=10)
            ),
            RoomPlacement(
                room_id="R_office_2", room_type="office", rect=Rect(x=10, y=0, w=10, d=10)
            ),
        ),
        corridors=(),
        doors=(
            # Shared wall is at x=10, y in [0, 10). y=20 is well outside.
            Door(door_id="D_1", a="R_office_1", b="R_office_2", x=10, y=20, width_cells=2),
        ),
        exits=(),
    )
    bad_layout = Layout(
        layout_id="LY_bad_door",
        spec_hash=HASH_A,
        grid_w=20,
        grid_d=10,
        floors=(plan,),
        vertical_core=VerticalCore(),
        provenance=LayoutProvenance(generator="test", seed=0),
    )
    topo = build_topology(bad_layout)
    invalid_door_ids = [d for d, _ in topo.invalid_doors]
    assert "D_1" in invalid_door_ids, (
        "Door with coordinates outside the shared wall must appear in topo.invalid_doors"
    )


def test_navgraph_derived_from_equals_geometry_hash():
    """navgraph.derived_from must equal layout.geometry_hash (structural linkage)."""
    layout = make_layout()
    navgraph = build_navgraph(layout)
    assert navgraph.derived_from == layout.geometry_hash, (
        f"derived_from {navgraph.derived_from!r} != geometry_hash {layout.geometry_hash!r}"
    )


def test_self_contained_factory_navgraph():
    """make_navgraph() must construct a valid NavGraph without errors."""
    navgraph = make_navgraph()
    assert isinstance(navgraph, NavGraph)
    assert len(navgraph.nodes) > 0, "Factory NavGraph must have at least one node"
    assert len(navgraph.edges) > 0, "Factory NavGraph must have at least one edge"
    assert navgraph.derived_from == HASH_A
