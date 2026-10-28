# Design decisions (Step 1)

| # | Decision | Why |
|---|---|---|
| 1 | `src/` layout, `pyproject.toml`, console script `smartbuilding` | Standard install on a clean machine; avoids import accidents |
| 2 | Pydantic v2, `extra="forbid"`, `frozen=True`, tuples not lists | Typos in YAML or LLM JSON fail loudly; objects are immutable and safely hashable |
| 3 | Integer cell geometry, `cell_size_m = 0.5` | Exact equality and stable hashes. A 1 m grid would force corridors and doors to whole metres, which is too coarse. The cell size is a field of `Layout`, so it can change |
| 4 | `content_hash` = SHA-256 of canonical JSON (sorted keys, floats rounded to 9 decimals, hash field excluded) | Same hash on every machine; verified on load when present |
| 5 | `Layout.geometry_hash` ignores ids, verification and provenance | Cache key for evaluation results; links NavGraph and ConstructionPlan to geometry |
| 6 | Layouts may be design-invalid (overlaps, out of bounds) | RQ1 compares LLM-only and random layouts against solver layouts. They must be representable so the checker can reject them |
| 7 | Scenarios are layout-agnostic (`ScenarioFamily` with selectors) and expand deterministically; `ResolvedScenario` binds one to a NavGraph | The repair loop evaluates different layouts on the same scenario recipes. Node ids differ between layouts, so scenarios cannot name nodes directly |
| 8 | Reserved template name `all` in scenario groups | One recipe works for every building type, whose occupant profiles differ |
| 9 | Truth and belief are separate classes; a test scans decision modules | Enforces the information barrier from the plan |
| 10 | Typed repair moves as a discriminated union | An LLM can only propose one of eight allowed move types with exactly the allowed parameters |
| 11 | Metric ids are a registry (`KNOWN_METRICS`) | A typo cannot create a new metric; the explainer cannot cite an unregistered one |
| 12 | `Sourcing` on every input value; `assumption` allowed but listed | Honest provenance; the final report can list all assumptions |
| 13 | Config paths must be relative and may not contain `..` | No hard-coded local paths (course execution rule) |
| 14 | `ScenarioSet` rejects duplicate seeds | Prevents leakage between repair and test splits |
| 15 | Modules are not named after stdlib modules (`types.py`, `io.py`) | A file shadowing a stdlib module breaks tools run from its folder; a test enforces it |
| 16 | YAML loader rejects duplicate keys | PyYAML silently keeps the last duplicate |
| 17 | `random_room_near_exit` takes an explicit relative `nearest_fraction` (closest fraction of rooms by graph distance) | "Near" must have one definition. A fraction behaves the same in small and large buildings, unlike a distance in metres. Selectors that a layout cannot satisfy (for example `random_lift` with no lift) are reported as not applicable by the Step 4 resolver |
| 18 | Room ids are stable identities. The type inside the id is the type the room was *created* as; `RoomPlacement.room_type` is the current type and `converted_from` records a conversion | The first version forced the id to match the current type, so converting a classroom to a lab would have required a new id and broken adaptability tracking. Found during the Step 1 review |
| 19 | `GroundTruthState` moved to `smartbuilding/sim/truth.py`; a test scans the whole source tree and only `sim/`, `eval/`, the schema registry and `oracle.py` files may reference it or import `smartbuilding.sim` | Core modules can no longer reach ground truth at all; the barrier is structural, not a convention |
| 20 | Device ids are `DV_<kind>_<n>` with kinds lift, door, detector, equipment, alarm; lift `L_n` and door `D_n` map to `DV_lift_n` and `DV_door_n` | `BeliefState.devices` tracks P(working) for all key devices, and the id itself says what the device is |
| 21 | `try_validate` returns an object or readable errors without raising | Two failure classes for RQ1: *structurally invalid* (cannot form a Layout, reported by the adapter) and *design-invalid* (loads fine, flagged by the rule checker) |
| 22 | `terminology.py`: `CHECKED_PHRASE`, banned-word lint (safe, unsafe, compliant, compliance, approved, certified, guaranteed) | Reports never claim safety or legal compliance. Whole-word matching, so "safety" is allowed |
| 23 | Schema version 1.1.0 (package 0.1.1). Files with the same major version still load | Changes were additive or stricter and nothing had consumed 1.0.0 yet |

## Boundary of Step 1

Contracts check **structure** (types, ranges, references that must exist, consistency within
an object). **Design validity** (corridor width, travel distance, accessibility, overlaps)
is checked by the rule checker in Step 2.
