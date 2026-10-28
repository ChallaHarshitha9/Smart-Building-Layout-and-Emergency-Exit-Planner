# 3D interface

    smartbuilding ui                    # from the project folder; open http://127.0.0.1:8765
    smartbuilding ui --root <project> --port 9000

Everything shown is computed by the real modules on the server (CSP solver, rule checker, construction
planner, simulator v2, Bayes filter, planners B1-B5). The browser only draws it.

## What you can do
* **Building**: pick a spec, Generate K layouts (seed), switch candidates; 3D orbit/zoom; Iso/Top/Side/Follow-crowd cameras.
* **Rules**: every hard constraint with pass/fail; click one to highlight the rooms/doors/exits involved; soft-cost bars.
  Infeasible requests are shown honestly (e.g. the 2-floor school: the Stage A template is one floor).
* **Evacuation**: Place fire (click a room), Block exit, Second fire, Fail detector; set times, occupancy period,
  hazard variant A/B, seed. "Run all methods" simulates B1, B2, B3 (oracle), B4 (raw obs.), B5 (belief) on the
  *same* scenario. Timeline (play/scrub/speed), people coloured by dose proxy, smoke and flames, planned routes,
  blocked exits turn red, **Lens: truth / belief** (what the Bayes filter believes), **Compare** another method as
  rings, charts of mean dose and evacuated people over time, metric cards.
* **Construction**: Gantt (critical path in amber) and a day slider that grows the building.
* **E0**: the sanity benchmark.

## Rebuilding the front end (only if you edit `web_src/`)
    cd src/smartbuilding/ui/web_src && npm install && npm run build && cp index.html ../web/

`ui/web/app.js` (three.js bundled with esbuild, no CDN) is committed so no Node is needed to run the UI.

## API (JSON)
GET `/api/specs`, `/api/e0`; POST `/api/solve`, `/api/simulate`, `/api/construction`. Path traversal is rejected
and tested (`tests/test_ui.py`).

## Verified how
Rendered and clicked in a headless Chromium (WebGL via SwiftShader): layout, mouse placement of fire and blocked
exits, belief lens, compare rings, construction slider, E0 tab, school building. Not tested on mobile or in
Safari/Firefox. Static single-file export (`export-ui`) is not implemented.
