# Conventions

* **Coordinates.** Plan view. x grows east, y grows north, origin at the south-west corner of
  the footprint. `floor` is an integer index, 0 = ground.
* **Units.** Lengths in metres, areas in square metres, times in seconds, speeds in metres per
  second, construction time in days. Every physical field name carries its unit
  (`width_m`, `area_m2`, `time_s`, `speed_mps`, `duration_days`). A test enforces this.
* **Layout geometry** is stored as integer cells of `cell_size_m` metres (default 0.5).
  `width_cells`, `x`, `y`, `w`, `d` on layout objects are cell counts.
* **Schema version.** Every top-level contract has `schema_version`. A different major
  version is rejected; the same major with a different minor loads.
* **Stable ids.** Room ids never change during repair. A conversion changes `room_type` and
  sets `converted_from`. Use `core/diff.py` to check stability and to count edits.
* **Hashes.** `content_hash` covers all content. `Layout.geometry_hash` covers geometry only.
* **Provenance.** Inputs carry `sourcing` (code, reference or assumption). Layouts carry
  generator, seed, parent layout and repair move.
* **Wording.** Reports say "constraint-checked against the configured rules". They never claim
  safety or legal compliance. `core/terminology.py` lints for banned words.
* **Information barrier.** Ground truth lives only in `smartbuilding/sim/truth.py`.
