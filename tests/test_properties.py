"""Property-based tests: random valid objects round-trip and hash deterministically."""

from hypothesis import given, settings
from hypothesis import strategies as st

from smartbuilding.core.hashing import hash_object
from smartbuilding.core.layout import FloorPlan, Layout, LayoutProvenance, Rect, RoomPlacement
from smartbuilding.core.metrics import KNOWN_METRICS, MetricSet, MetricValue
from smartbuilding.core.seeding import derive_seed

H = "b" * 64

rects = st.builds(
    Rect, x=st.integers(0, 200), y=st.integers(0, 200), w=st.integers(1, 100), d=st.integers(1, 100)
)
types = st.sampled_from(["office", "lab", "classroom", "staff_room"])


@st.composite
def layouts(draw):
    n = draw(st.integers(1, 6))
    rooms = []
    for i in range(1, n + 1):
        t = draw(types)
        rooms.append(RoomPlacement(room_id=f"R_{t}_{i}", room_type=t, rect=draw(rects)))
    return Layout(
        layout_id="LY_prop",
        spec_hash=H,
        grid_w=300,
        grid_d=300,
        floors=(FloorPlan(floor=0, rooms=tuple(rooms)),),
        provenance=LayoutProvenance(generator="prop", seed=draw(st.integers(0, 10**6))),
    )


@given(layouts())
@settings(max_examples=60, deadline=None)
def test_layout_roundtrip_and_hash_determinism(layout):
    again = Layout.model_validate_json(layout.model_dump_json())
    assert again == layout
    assert again.compute_hash() == layout.compute_hash()
    hashed = layout.with_hash()
    assert Layout.model_validate_json(hashed.model_dump_json()) == hashed


@given(layouts())
@settings(max_examples=40, deadline=None)
def test_geometry_hash_independent_of_layout_id(layout):
    other = layout.model_copy(update={"layout_id": "LY_other"})
    assert other.geometry_hash == layout.geometry_hash


@given(st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=1, max_size=6))
def test_hash_independent_of_insertion_order(d):
    reordered = dict(reversed(list(d.items())))
    assert hash_object(d) == hash_object(reordered)


@given(st.integers(0, 2**62), st.text(max_size=8), st.integers(0, 1000))
def test_derive_seed_stable(base, key, k):
    assert derive_seed(base, key, k) == derive_seed(base, key, k)


@given(
    st.sampled_from(sorted(KNOWN_METRICS)),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
)
def test_metric_set_roundtrip(metric_id, value):
    ms = MetricSet(
        scope="layout",
        layout_geometry_hash=H,
        values=(MetricValue(metric_id=metric_id, value=value),),
    )
    assert MetricSet.model_validate_json(ms.model_dump_json()) == ms
