from dual_system.eval.atomic import phase_done, plan_for, ran_ahead, summarise, wrong_lifts

SUBTASKS = {"t": [{"goal": "turnon", "act": "turn on the stove"}, {"goal": "moka", "pick": "pick up the moka pot", "place": "place the moka pot on the stove"}]}
GOALS = {"turnon flat_stove_1": False, "on moka_pot_1 flat_stove_1_cook_region": False}


def test_plan_keeps_list_order():
    assert [(p["kind"], p["command"]) for p in plan_for("t", SUBTASKS)] == [("act", "turn on the stove"), ("pick", "pick up the moka pot"), ("place", "place the moka pot on the stove")]


def test_pick_needs_grip_and_lift():
    pick = plan_for("t", SUBTASKS)[1]
    assert phase_done(pick, GOALS, lambda n: n == "moka_pot_1", {"moka_pot_1": 0.05})
    assert not phase_done(pick, GOALS, lambda n: True, {"moka_pot_1": 0.0})  # gripped on the table
    assert not phase_done(pick, GOALS, lambda n: False, {"moka_pot_1": 0.05})  # lifted but not held: bumped or thrown


def test_place_and_act_use_their_own_goal():
    act, _, place = plan_for("t", SUBTASKS)
    goals = dict(GOALS, **{"turnon flat_stove_1": True})
    assert phase_done(act, goals, None, {}) and not phase_done(place, goals, None, {})


def test_ran_ahead_only_counts_new_goals():
    assert ran_ahead({"a": True, "b": False}, {"a": True, "b": True})
    assert not ran_ahead({"a": True, "b": False}, {"a": True, "b": False})


def test_wrong_lift_ignores_target():
    assert wrong_lifts({"moka_pot_1": 0.1, "cream_cheese_1": 0.05, "butter_1": 0.01}, "moka_pot_1") == ["cream_cheese_1"]


def test_summary_counts_phases():
    rows = [dict(goal=True, reached=2, total=2, phases=[dict(kind="pick", done=True, steps=40, wrong_lift=[]), dict(kind="place", done=True, steps=60, wrong_lift=[])]),
            dict(goal=False, reached=0, total=2, phases=[dict(kind="pick", done=False, steps=150, wrong_lift=["x"])])]
    s = summarise(rows)
    assert (s["goal"], s["phases_reached"], s["pick"], s["place"], s["wrong_object"], s["median_steps"]) == ("1/2", "2/4", "1/2", "1/1", 1, 50)
