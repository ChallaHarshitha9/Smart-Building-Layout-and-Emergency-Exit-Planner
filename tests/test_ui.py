"""3D interface backend: payloads come from the real modules; the HTTP layer is safe and works."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from smartbuilding.core.bundle import load_bundle
from smartbuilding.core.loaders import load_model
from smartbuilding.core.spec import BuildingSpec
from smartbuilding.ui import scene
from smartbuilding.ui.server import Context, make_handler
from tests.factories import ROOT

BLOCK_E1 = [{"type": "exit_blocked", "time": 5, "target": "E_1"}]


@pytest.fixture(scope="module")
def office():
    bundle = load_bundle(ROOT / "configs" / "run_default.yaml")
    spec = load_model(ROOT / "data" / "examples" / "office_spec.json", BuildingSpec)
    scenes, _ = scene.solve_scenes(spec, bundle, 0, 1)
    layout, sc = scenes[0]
    sc["cells"] = scene._cells(sc)
    return bundle, spec, layout, sc


def test_layout_scene_is_complete_and_every_node_has_a_cell(office):
    _, _, _, sc = office
    assert sc["feasible"] and sc["hard"] and sc["soft"]
    assert {n["id"] for n in sc["nodes"] if n["kind"] != "exit"} <= set(sc["cells"])
    for c in sc["cells"].values():
        if c:
            assert c["w"] > 0 and c["d"] > 0


def test_simulation_payload_is_consistent_and_shows_the_blocked_exit_effect(office):
    bundle, spec, layout, sc = office
    out = scene.simulate_all(layout, spec, bundle, sc, ignition_room="R_toilet_1", events=BLOCK_E1)
    n_nodes, n_groups = len(sc["nodes"]), len(out["groups"])
    assert set(out["methods"]) == set(scene.METHODS)
    for m in out["methods"].values():
        f0, fl = m["frames"][0], m["frames"][-1]
        assert f0["t"] == 0 and len(f0["s"]) == n_nodes and len(f0["g"]) == n_groups
        assert fl["t"] >= m["frames"][1]["t"]
    assert out["methods"]["B5"]["frames"][5].get("b") is not None  # belief is exposed for the lens
    s = {k: v["summary"] for k, v in out["methods"].items()}
    assert s["B1"]["completion"] < 1.0 <= s["B5"]["completion"] and s["B3"]["completion"] == 1.0
    assert s["B5"]["mean_dose"] < s["B1"]["mean_dose"]
    assert any("r" in f for f in out["methods"]["B5"]["frames"])  # routes recorded


def test_simulation_is_deterministic_and_rejects_unknown_events(office):
    bundle, spec, layout, sc = office
    kw = {"ignition_room": "R_office_1", "events": [], "methods": ("B5",)}
    a = scene.simulate_all(layout, spec, bundle, sc, **kw)
    b = scene.simulate_all(layout, spec, bundle, sc, **kw)
    assert json.dumps(a) == json.dumps(b)
    with pytest.raises(ValueError):
        scene.simulate_all(
            layout,
            spec,
            bundle,
            sc,
            ignition_room="R_office_1",
            events=[{"type": "meteor", "target": "E_1"}],
        )


@pytest.fixture(scope="module")
def server():
    ctx = Context(Path(ROOT))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ctx))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(base, path, body):
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def test_http_round_trip(server):
    assert b"Smart Building Studio" in urllib.request.urlopen(server + "/").read()
    assert urllib.request.urlopen(server + "/app.js").status == 200
    specs = json.loads(urllib.request.urlopen(server + "/api/specs").read())
    assert {s["type"] for s in specs} >= {"office", "school"} and all(s["periods"] for s in specs)
    solved = _post(server, "/api/solve", {"spec": "office_spec.json", "max_layouts": 2})
    key = solved["layouts"][0]["key"]
    sim = _post(
        server,
        "/api/simulate",
        {"layout": key, "ignition": "R_toilet_1", "events": BLOCK_E1, "methods": ["B1", "B5"]},
    )
    assert set(sim["methods"]) == {"B1", "B5"}
    assert _post(server, "/api/construction", {"layout": key})["total_days"] > 0


def test_http_rejects_bad_requests_and_path_traversal(server):
    for bad in ("/../pyproject.toml", "/%2e%2e/pyproject.toml", "/nope.js"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(server + bad)
        assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server, "/api/simulate", {"layout": "nope/nope", "ignition": "R_x"})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server, "/api/solve", {"spec": "../../pyproject.toml"})
    assert e.value.code in (400, 500)
