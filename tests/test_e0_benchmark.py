from smartbuilding.eval.e0 import build_e0_toy_layout, run_e0


def test_toy_layout_has_two_exits_and_two_rooms():
    plan = build_e0_toy_layout().floors[0]
    assert len(plan.exits) == 2 and len(plan.rooms) == 2


def test_e0_pattern_and_determinism():
    a, b = run_e0(n_seeds=8), run_e0(n_seeds=8)
    assert a.rows == b.rows
    assert a.check_passed, a.note
    # early start: nobody has noticed the fire yet, every method takes the nearest exit
    assert all(set(c) == {"E_1"} for k, c in a.exit_counts.items() if k.endswith("early"))
    assert a.mean_dose["B1/late"] > a.mean_dose["B1/early"]
