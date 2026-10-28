# Module 1-3 review and v2 hardening (2026-10)

A review of the first build found defects that would have invalidated later experiments. Each was
reproduced by running the code, fixed, and pinned by a test. Nothing below is a result.

## Defects found and fixed

| # | Defect | Fix | Test |
|---|--------|-----|------|
| 1 | Groups moved one graph hop per decision period; speed and edge length ignored (~0.4 m/s) | Continuous movement along edges at each group's speed | `test_walking_time_is_distance_over_speed` |
| 2 | No queues, capacity or congestion; groups had size 1 | Per-edge flow capacity (`width x specific_flow`), FIFO queues, head counts from occupancy | `test_queues_form_when_flow_is_limited...`, `test_resolve.py` |
| 3 | Blocked exit: nobody evacuated; `exit_avail` never updated or used | Exit/lift/door/occupancy observation channels; belief over them; planner uses them | `test_blocked_exit_B1_gets_stuck_B5_reroutes` |
| 4 | Belief ratcheted to P(fire)=1 after one false alarm; silence ignored; magic constants | Negative evidence, ignition prior, floors, all parameters in `BeliefParams` | `test_one_false_alarm_is_forgotten...`, `test_silence_is_evidence` |
| 5 | Smoke depended on tick size (`prob * tick_s`) | Per-second rates, sub-stepped smoke, exact `1-exp(-rate*dt)` fire spread | `test_smoke_physics_does_not_depend_on_the_tick` |
| 6 | Group template `impaired` resolved to the last profile (wrong delay/speed) | Templates looked up in the scenario's profile; ambiguity is an error | `test_response_delay_comes_from_the_right_profile`, `test_ambiguous_or_unknown_templates_fail_loudly` |
| 7 | Decision callback received full `SimState` (ground truth) | `PublicState` only; `Controller` protocol; belief moved to `uncertainty/` | `test_decision_makers_never_receive_ground_truth` + import barrier test |
| 8 | Only the last tick's observations reached the filter, and only every 3rd tick | `Controller.observe` after every tick | belief tests |
| 9 | Most hazard and sensor parameters unused | All used (see below); `user_report_rate_per_min` still unused | - |
| 10 | "B2" was belief-weighted, not SafeStep threshold Dijkstra | B2 = remove nodes with hazard probability >= threshold, shortest path, B1 fallback | `test_b2_blocks_hazardous_nodes...` |
| 11 | E0 "confirmed" because of the movement bug | E0 rewritten: early/late response, 20 seeds, B1/B2/B5 | `test_e0_pattern_and_determinism` |
| 12 | Scenario event times (60-120 s) assumed the slow movement | Moved to 5-20 s; they are placeholders | - |

## Model (v2)

* Per-second dynamics. `rate_interval_s` (10 s): `spread_prob_*`, `smoke_generation`, `smoke_diffusion`,
  `smoke_decay` are probabilities/fractions *per interval*. Changing `tick_s` changes resolution only.
* Fire: stochastic, own random stream (fire history identical for every controller: paired design).
  Variant B adds a two-hop kernel. Fire never goes out. Stuck doors count as closed doors.
* Sensors: own random stream. Each working detector reports every tick. Exit, lift, door and occupancy
  reports every `report_period_s`. Occupants at a blocked exit report it. A failed detector is silent
  (no reading), and the filter treats a missing reading as no evidence.
* Belief: mean-field Bayes filter over clear/smoky/fire per node; Bernoulli filters for exits, lifts,
  doors; Kalman filter for room occupancy. It uses the hazard model it is given (variant A) so on
  variant B belief and truth differ. Plan 9.5 says Beta-Bernoulli; with known configured reliabilities
  the exact update is a Bernoulli filter, so that is what is implemented.
* Planners: B1, B2 (re-run each decision tick on the same belief), dose-aware planner (B3 on truth,
  B4 on raw observations, B5 on belief with forward prediction). Queue term uses only publicly visible
  group positions; groups are planned independently (that is what RL is meant to improve on).
* A group in transit may turn back to the node it left; any other off-route decision is rejected and
  counted in `invalid_decisions`.

## Known limitations (state them in the report)

* Rooms are single graph nodes; the corridor graph is coarse (a node per door). Smoke and fire move
  node to node, not through space.
* Congestion is flow-limited entry to edges with FIFO queues, not a density model.
* Dose is a proxy, not FED. All hazard, sensor and belief numbers are placeholders (`basis: assumption`).
* Scenario families and hazard parameters are not calibrated; the office is easy (everyone is out in
  about 12 s). Hard cases come from blocked exits, delays (school/hostel/hospital) and high occupancy.
* `user_report_rate_per_min` is unused; user reports occur only at blocked exits.
* One-off runs (e.g. the sanity runs in this review) are not results: experiments need >= 30 paired
  scenarios, bootstrap CIs and held-out scenarios.

## Still open

3D UI port to the v2 API (`ui/scene.py`, `ui/server.py`, `ui/web_src`), Modules 4 and 5, repair loop,
experiments E1-E7.
