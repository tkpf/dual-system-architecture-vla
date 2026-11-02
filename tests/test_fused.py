from dual_system.system2.fused import RECOVERY_COMMAND, FusedSwitcher, OracleSwitcher

PHASES = [dict(kind="pick", command="pick up the bowl", goal="b"), dict(kind="place", command="place the bowl on the stove", goal="b")]
ACT = [dict(kind="act", command="turn on the stove", goal="turnon"), dict(kind="pick", command="pick up the pot", goal="p")]


def drive(sw, trace, vlm=None):
    """trace: (width, z) per step, from t=1. Returns the events."""
    events = []
    for t, (w, z) in enumerate(trace, 1):
        e = sw.update(t, w, z, vlm)
        if e:
            events.append(e)
    return events


def test_pick_then_place_on_signals():
    sw = FusedSwitcher(PHASES)
    trace = [(0.078, 1.1)] * 20 + [(0.03, 0.9 + 0.01 * i) for i in range(20)] + [(0.03, 1.1)] * 5 + [(0.078, 0.95)] * 6
    ev = drive(sw, trace)
    assert [e["reason"] for e in ev] == ["signal", "signal"] and [e["phase"] for e in ev] == [0, 1]
    assert sw.command == "place the bowl on the stove" and ev[1].get("all_done")


def test_hold_without_lift_does_not_finish_pick():
    sw = FusedSwitcher(PHASES)
    assert drive(sw, [(0.078, 1.1)] * 5 + [(0.03, 0.9)] * 40) == []


def test_release_before_any_hold_does_not_finish_place():
    sw = FusedSwitcher(PHASES, timeout=1000)
    sw.idx, sw.phase_start = 1, 0
    sw._on_new_phase()
    assert drive(sw, [(0.078, 1.0)] * 50) == []


def test_empty_grasp_recovers_then_retries():
    sw = FusedSwitcher(PHASES)
    ev = drive(sw, [(0.078, 1.0)] * 5 + [(0.001, 0.9)] * 8)
    assert ev[0]["reason"] == "empty_grasp" and sw.command == RECOVERY_COMMAND
    t = 14
    out = []
    for _ in range(20):
        t += 1
        e = sw.update(t, 0.078, 1.0)
        if e:
            out.append(e)
    assert out[0]["reason"] == "retry" and sw.command == "pick up the bowl"


def test_no_recovery_when_disabled():
    sw = FusedSwitcher(PHASES, recover=False, timeout=1000)
    assert drive(sw, [(0.078, 1.0)] * 5 + [(0.001, 0.9)] * 30) == []


def test_timeout_advances_but_never_past_the_last_phase():
    sw = FusedSwitcher(PHASES, timeout=30)
    ev = drive(sw, [(0.078, 1.0)] * 100)
    assert [e["reason"] for e in ev] == ["timeout"] and sw.idx == 1


def test_act_needs_confirmed_vlm_after_minimum_time():
    sw = FusedSwitcher(ACT, vlm_min_steps=40, vlm_every=10, vlm_confirm=2, vlm_threshold=0.8)
    calls = []
    ev = drive(sw, [(0.078, 1.0)] * 100, vlm=lambda: calls.append(1) or 0.95)
    assert ev[0]["reason"] == "vlm" and ev[0]["t"] == 50 and len(calls) == 2  # checks at 40 and 50


def test_act_single_high_answer_is_not_enough():
    import itertools
    answers = itertools.cycle([0.9, 0.1])
    sw = FusedSwitcher(ACT, vlm_min_steps=40, timeout=1000)
    assert drive(sw, [(0.078, 1.0)] * 120, vlm=lambda: next(answers)) == []


def test_truth_is_only_logged():
    sw = FusedSwitcher(ACT, timeout=20)
    ev = drive(sw, [(0.078, 1.0)] * 30)
    assert ev[0]["truth_done"] is None
    ev = sw.log
    assert ev[0]["reason"] == "timeout"


def test_oracle_switches_on_truth_and_times_out():
    sw = OracleSwitcher(PHASES, timeout=10)
    assert sw.update(3, True)["reason"] == "oracle" and sw.idx == 1
    assert sw.update(12, False) is None
    assert sw.update(14, True).get("all_done")


def test_thin_bowl_grip_counts_as_held_not_empty():
    sw = FusedSwitcher(PHASES)
    ev = drive(sw, [(0.078, 1.0)] * 5 + [(0.0046, 0.9 + 0.01 * i) for i in range(20)])
    assert [e["reason"] for e in ev] == ["signal"]


def carry_then_release(descend: bool):
    """Pick with a lift, carry, then open: lowered first for a place, or in mid-air for a drop."""
    up = [(0.078, 1.0)] * 5 + [(0.03, 0.9 + 0.01 * i) for i in range(20)]
    down = [(0.03, 1.1 - 0.01 * i) for i in range(8)] if descend else [(0.03, 1.1)] * 8
    return up + down + [(0.078, down[-1][1])] * 5


def test_release_after_descent_is_a_place_with_drop_detection():
    sw = FusedSwitcher(PHASES, detect_drop=True)
    assert [e["reason"] for e in drive(sw, carry_then_release(True))] == ["signal", "signal"]


def test_release_in_mid_air_goes_back_to_the_pick():
    sw = FusedSwitcher(PHASES, detect_drop=True)
    ev = drive(sw, carry_then_release(False))
    assert [e["reason"] for e in ev] == ["signal", "dropped"] and sw.idx == 0 and sw.command == "pick up the bowl"


def test_mid_air_release_counts_as_place_without_drop_detection():
    sw = FusedSwitcher(PHASES)
    assert [e["reason"] for e in drive(sw, carry_then_release(False))] == ["signal", "signal"]


def test_oracle_undo_goes_back_when_object_leaves_the_gripper():
    sw = OracleSwitcher(PHASES, timeout=1000, undo=True, undo_steps=3)
    sw.reset()
    sw.update(1, True)
    for t in range(2, 4):
        assert sw.update(t, False, False) is None
    assert sw.update(4, False, False)["reason"] == "undone" and sw.idx == 0


def test_oracle_without_undo_stays():
    sw = OracleSwitcher(PHASES, timeout=1000)
    sw.update(1, True)
    assert all(sw.update(t, False, False) is None for t in range(2, 30))
