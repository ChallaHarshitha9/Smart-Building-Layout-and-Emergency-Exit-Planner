"""Local web server for the 3D interface (standard library only).

    smartbuilding ui            # then open http://127.0.0.1:8765

The browser app is static; every solve, simulation and rule check is computed here by the real
modules, so clicking a room to start a fire or blocking an exit re-runs the actual simulator.
"""

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from smartbuilding.core.bundle import ConfigBundle, load_bundle
from smartbuilding.core.layout import Layout
from smartbuilding.core.loaders import load_model
from smartbuilding.core.spec import BuildingSpec
from smartbuilding.eval.e0 import run_e0
from smartbuilding.ui import scene

WEB_DIR = Path(__file__).resolve().parent / "web"
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
}


class Context:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.examples = root / "data" / "examples"
        self.bundle: ConfigBundle = load_bundle(root / "configs" / "run_default.yaml")
        self.layouts: dict[str, tuple[Layout, BuildingSpec]] = {}
        self.scenes: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def spec(self, filename: str) -> BuildingSpec:
        path = self.examples / Path(filename).name  # never follow a path out of examples/
        return load_model(path, BuildingSpec)

    def specs(self) -> list[dict[str, Any]]:
        out = scene.list_specs(self.examples)
        for item in out:
            spec = self.spec(item["file"])
            prof = self.bundle.profiles.get(spec.occupancy.profile_ref)
            item["periods"] = [p.name for p in prof.periods] if prof else ["typical"]
        return out

    def solve(self, body: dict[str, Any]) -> dict[str, Any]:
        spec = self.spec(str(body["spec"]))
        scenes, info = scene.solve_scenes(
            spec, self.bundle, int(body.get("seed", 0)), int(body.get("max_layouts", 4))
        )
        out = []
        with self.lock:
            for layout, sc in scenes:
                key = f"{spec.spec_id}/{layout.layout_id}"
                self.layouts[key] = (layout, spec)
                sc["cells"] = scene._cells(sc)
                self.scenes[key] = sc
                out.append({**sc, "key": key})
        return {"layouts": out, "info": info, "dose_levels": scene.dose_levels(self.bundle)}

    def _layout(self, key: str) -> tuple[Layout, BuildingSpec]:
        with self.lock:
            if key not in self.layouts:
                raise KeyError(f"unknown layout {key!r}; solve first")
            return self.layouts[key]

    def simulate(self, body: dict[str, Any]) -> dict[str, Any]:
        key = str(body["layout"])
        layout, spec = self._layout(key)
        methods = tuple(body.get("methods") or scene.METHODS)
        if not set(methods) <= set(scene.METHODS):
            raise ValueError(f"unknown methods {sorted(set(methods) - set(scene.METHODS))}")
        return scene.simulate_all(
            layout,
            spec,
            self.bundle,
            self.scenes[key],
            ignition_room=str(body["ignition"]),
            events=list(body.get("events", [])),
            methods=methods,
            period=str(body.get("period", "typical")),
            hazard_variant=str(body.get("hazard", "A")),
            seed=int(body.get("seed", 42)),
            horizon_s=float(body.get("horizon", 300)),
        )

    def construction(self, body: dict[str, Any]) -> dict[str, Any]:
        layout, _ = self._layout(str(body["layout"]))
        return scene.construction_scene(layout)

    def e0(self) -> dict[str, Any]:
        r = run_e0()
        return {
            "mean_dose": r.mean_dose,
            "exit_counts": r.exit_counts,
            "check_passed": r.check_passed,
            "note": r.note,
        }


def make_handler(ctx: Context) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            return

        def _send(self, code: int, payload: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, separators=(",", ":")).encode(), MIME[".json"])

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            try:
                if path == "/api/specs":
                    return self._json(ctx.specs())
                if path == "/api/e0":
                    return self._json(ctx.e0())
                rel = "index.html" if path in ("/", "") else path.lstrip("/")
                target = (WEB_DIR / rel).resolve()
                if WEB_DIR.resolve() not in target.parents or not target.is_file():
                    return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                self._send(
                    200, target.read_bytes(), MIME.get(target.suffix, "application/octet-stream")
                )
            except Exception as exc:  # report, never crash the server
                self._json({"error": str(exc)}, 500)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                route = {
                    "/api/solve": ctx.solve,
                    "/api/simulate": ctx.simulate,
                    "/api/construction": ctx.construction,
                }.get(self.path)
                if route is None:
                    return self._json({"error": "unknown endpoint"}, 404)
                self._json(route(body))
            except (KeyError, ValueError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    ctx = Context(root)
    httpd = ThreadingHTTPServer((host, port), make_handler(ctx))
    print(f"SmartBuilding 3D Studio running at http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        httpd.server_close()
