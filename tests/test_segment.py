import numpy as np

from dual_system.finetune.segment import find_phases, last_true_start, load_subtasks, match_episode


def flags(n, *ranges):
    out = np.zeros(n, bool)
    for a, b in ranges:
        out[a:b] = True
    return out


PICK_PLACE = [{"goal": "bowl", "pick": "pick up the bowl", "place": "place the bowl on the stove"}]


def test_last_true_start_skips_short_runs():
    assert last_true_start(flags(20, (2, 9), (15, 17)), min_len=5) == 2
    assert last_true_start(flags(20, (2, 9), (15, 17))) == 15
    assert last_true_start(flags(5)) is None


def test_pick_ends_at_grasp_and_place_at_goal():
    goals = {"on bowl_1 stove": flags(50, (40, 50))}
    phases, why = find_phases(PICK_PLACE, goals, {"bowl_1": flags(50, (20, 38))}, flags(50))
    assert why == ""
    assert [(p["kind"], p["start"], p["end"]) for p in phases] == [("pick", 0, 21), ("place", 21, 41)]


def test_pick_ends_at_first_grasp_when_contact_flickers_during_the_carry():
    goals = {"on bowl_1 stove": flags(100, (90, 100))}
    held = {"bowl_1": flags(100, (20, 50), (55, 60), (80, 88))}
    phases, _ = find_phases(PICK_PLACE, goals, held, flags(100))
    assert [(p["kind"], p["end"]) for p in phases] == [("pick", 21), ("place", 91)]


def test_place_runs_on_until_the_fingers_let_go():
    goals = {"on bowl_1 stove": flags(50, (40, 50))}
    touching = {"bowl_1": flags(50, (20, 45))}
    phases, _ = find_phases(PICK_PLACE, goals, {"bowl_1": flags(50, (20, 38))}, flags(50), touching)
    assert phases[-1]["end"] == 46
    never = {"bowl_1": flags(50, (20, 50))}
    assert find_phases(PICK_PLACE, goals, {"bowl_1": flags(50, (20, 38))}, flags(50), never)[0][-1]["end"] == 41


def test_second_object_ignores_grasps_before_the_first_is_placed():
    groups = [{"goal": "soup", "pick": "pick soup", "place": "place soup"}, {"goal": "sauce", "pick": "pick sauce", "place": "place sauce"}]
    goals = {"in soup_1 basket": flags(100, (40, 100)), "in sauce_1 basket": flags(100, (90, 100))}
    held = {"soup_1": flags(100, (10, 38)), "sauce_1": flags(100, (5, 12), (60, 88))}  # sauce nudged while reaching for the soup
    phases, _ = find_phases(groups, goals, held, flags(100))
    assert [p["end"] for p in phases] == [11, 41, 61, 91]


def test_groups_follow_demo_order_not_list_order():
    groups = [{"goal": "soup", "pick": "pick soup", "place": "place soup"}, {"goal": "sauce", "pick": "pick sauce", "place": "place sauce"}]
    goals = {"in soup_1 basket": flags(100, (80, 100)), "in sauce_1 basket": flags(100, (40, 100))}
    held = {"soup_1": flags(100, (60, 78)), "sauce_1": flags(100, (20, 38))}
    phases, _ = find_phases(groups, goals, held, flags(100))
    assert [p["command"] for p in phases] == ["pick sauce", "place sauce", "pick soup", "place soup"]
    assert all(a["end"] == b["start"] for a, b in zip(phases, phases[1:]))


def test_act_waits_for_release():
    groups = [{"goal": "turnon", "act": "turn on the stove"}]
    phases, _ = find_phases(groups, {"turnon stove": flags(50, (20, 50))}, {}, flags(50, (0, 25)))
    assert phases == [dict(command="turn on the stove", kind="act", start=0, end=26)]


def test_bad_demos_are_dropped_with_reason():
    goals = {"on bowl_1 stove": flags(50, (40, 45))}
    assert find_phases(PICK_PLACE, goals, {"bowl_1": flags(50, (20, 38))}, flags(50))[0] is None  # goal undone at the end
    goals = {"on bowl_1 stove": flags(50, (40, 50))}
    phases, why = find_phases(PICK_PLACE, goals, {"bowl_1": flags(50, (20, 23))}, flags(50))  # held too briefly
    assert phases is None and "never held" in why
    assert find_phases(PICK_PLACE, {"on bowl_1 stove": flags(50, (0, 50))}, {"bowl_1": flags(50)}, flags(50))[0] is None  # true from start


def test_match_episode_needs_nearly_equal_actions():
    a, b = np.random.default_rng(0).normal(size=(2, 30, 7))
    assert match_episode(a[:-1], {"x": b, "y": a}) == "y"
    assert match_episode(a, {"x": b}) is None


def test_every_task_has_commands_without_conjunctions():
    tasks = load_subtasks()
    assert len(tasks) == 40
    for groups in tasks.values():
        for g in groups:
            assert ("act" in g) != ("pick" in g and "place" in g)
            for c in [g.get("act"), g.get("pick"), g.get("place")]:
                assert c is None or " and " not in c.replace("between the plate and the ramekin", "").replace("yellow and white", "")
