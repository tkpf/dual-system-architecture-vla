import numpy as np

from dual_system.triggers import EmptyGrasp, GripperToggle, Periodic, Quiescence, SubgoalTimeout, TriggerManager
from dual_system.types import TriggerContext


def ctx(t, grip=-1.0, move=0.5, pos=None, width=0.08, since=0):
    a = np.array([move, 0, 0, 0, 0, 0, grip])
    return TriggerContext(t, a, np.zeros(3) if pos is None else pos, width, since)


def test_periodic_every_second_of_sim_time():
    p = Periodic(20)
    assert [t for t in range(0, 61) if p.fired(ctx(t))] == [20, 40, 60]


def test_gripper_toggle_debounced():
    g = GripperToggle(debounce=3)
    grips = [-1, -1, 1, -1, 1, 1, 1, 1, 1]  # the single-step flicker at index 2 must not fire
    fired = [i for i, x in enumerate(grips) if g.fired(ctx(i, grip=x))]
    assert fired == [6]


def test_quiescence_rising_edge_only():
    q = Quiescence(window=5)
    fired = []
    for t in range(5):
        fired.append(q.fired(ctx(t, move=0.5, pos=np.array([t * 0.05, 0, 0]))))
    for t in range(5, 15):
        fired.append(q.fired(ctx(t, move=0.0, pos=np.array([0.25, 0, 0]))))
    assert sum(fired) == 1


def test_empty_grasp_needs_closed_fingers_and_close_command():
    e = EmptyGrasp(width_m=0.004)
    assert not e.fired(ctx(0, grip=-1, width=0.001))   # open command
    assert not e.fired(ctx(1, grip=1, width=0.03))     # closed on an object
    assert e.fired(ctx(2, grip=1, width=0.001))
    assert not e.fired(ctx(3, grip=1, width=0.001))    # only once


def test_timeout_fires_once_per_command():
    s = SubgoalTimeout(steps=10)
    assert [t for t in range(20) if s.fired(ctx(t, since=t))] == [10]
    assert not s.fired(ctx(21, since=0))
    assert s.fired(ctx(22, since=10))


def test_manager_merges_reasons_within_gap():
    class Always:
        def __init__(self, name, at):
            self.name, self.at = name, at

        def reset(self):
            pass

        def fired(self, c):
            return c.t in self.at

    m = TriggerManager([Always("a", {20}), Always("b", {22})], min_gap=5)
    events = [e for e in (m.poll(ctx(t)) for t in range(20, 30)) if e]
    assert [(e.t, e.reasons) for e in events] == [(20, ["a"]), (25, ["b"])]
