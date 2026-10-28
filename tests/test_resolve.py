import pytest

from smartbuilding.core.bundle import load_bundle
from smartbuilding.eval.e0 import build_e0_toy_layout
from smartbuilding.layout.graph import build_navgraph
from smartbuilding.sim.resolve import ResolveError, resolve_scenario
from tests.factories import ROOT
from tests.toy import PROFS

BUNDLE = load_bundle(ROOT / "configs" / "run_default.yaml")
GRAPH = build_navgraph(build_e0_toy_layout(), {"R_office_1": 10, "R_office_2": 30})
PROFILE = PROFS.profiles[0]


def test_every_family_resolves_with_exact_head_counts_or_fails_loudly():
    ok = 0
    for sc in BUNDLE.scenarios().scenarios[:60]:
        try:
            rs = resolve_scenario(sc, GRAPH, BUNDLE.profiles.get("profile_office"), 40)
        except ResolveError:
            continue  # e.g. a lift or room type the toy layout does not have
        period = next(
            p
            for p in BUNDLE.profiles.get("profile_office").periods
            if p.name == sc.occupancy_period
        )
        assert sum(g.size for g in rs.groups) == round(40 * period.mean_fraction)
        assert all(g.size >= 1 for g in rs.groups)
        ok += 1
    assert ok > 0


def test_people_are_spread_over_rooms_by_load_and_resolution_is_deterministic():
    sc = next(s for s in BUNDLE.scenarios().scenarios if s.scenario_id.startswith("SC_normal_fire"))
    a = resolve_scenario(sc, GRAPH, BUNDLE.profiles.get("profile_office"), 40)
    b = resolve_scenario(sc, GRAPH, BUNDLE.profiles.get("profile_office"), 40)
    assert a == b
    by_room: dict[str, int] = {}
    for g in a.groups:
        by_room[g.start_node] = by_room.get(g.start_node, 0) + g.size
    assert (
        by_room[next(n.node_id for n in GRAPH.nodes if n.region_id == "R_office_2")]
        > by_room[next(n.node_id for n in GRAPH.nodes if n.region_id == "R_office_1")]
    )


def test_unknown_profile_period_is_an_error():
    sc = next(s for s in BUNDLE.scenarios().scenarios if s.occupancy_period)
    bad = sc.model_copy(update={"occupancy_period": "never"})
    with pytest.raises(ResolveError):
        resolve_scenario(bad, GRAPH, BUNDLE.profiles.get("profile_office"), 40)
