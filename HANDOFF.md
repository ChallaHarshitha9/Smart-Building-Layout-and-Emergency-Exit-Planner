# Handoff: Smart Building Layout and Emergency Exit Planner

Read `docs/module1-3_review_and_v2.md` first (what was wrong, what changed, limitations), then the
research plan (`Project_Plan_v2.pdf`).

## State (2026-10-22)
Modules 1-3 reviewed and hardened: simulator v2 (continuous movement, queues, per-second physics,
stochastic fire, full observation channels), belief v2 (`uncertainty/`), planners B1/B2/B3/B4/B5,
scenario resolver (`sim/resolve.py`), episode runner (`eval/episode.py`), E0 rewritten. 167 tests,
ruff and mypy clean. Legacy Streamlit app and static dashboard removed.

## Conventions (unchanged)
Information barrier (`GroundTruthState` only in `sim/` and `eval/`; controllers get `PublicState` and
`Observation`). Integer geometry, stable ids, typed repair moves, no "safe/compliant" wording, dose is
a proxy, config numbers are placeholders, no fabricated results, go step by step, short replies.

## Done since: 3D UI
`smartbuilding ui` (see `docs/ui_3d.md`): Three.js front end, real-module backend, tested in a headless
browser. Solver now judges layouts against the real spec (multi-floor requests are reported infeasible
until Stage C). `export-ui` (static HTML) not implemented.

## Next
1. Optional: static export; Stage C multi-floor layouts.
2. Calibrate hazard/scenario placeholders on variant A so scenarios discriminate (document it).
3. Module 4: DQN (shared policy over groups; state = belief + queues; reward = time + dose + congestion),
   B6/B7 vs B5, >= 5 training seeds, held-out evaluation.
4. Module 5: parser, typed-action agents, grounded explainer, cached LLM responses.
5. Repair loop, E1-E7, statistics, report.

## Resume
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest
