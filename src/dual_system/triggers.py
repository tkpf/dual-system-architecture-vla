"""When System 2 is asked. All timing is in control steps (20 Hz sim time), never wall clock.

SmolVLA reports no completion (it only outputs a 7-D action), so Quiescence stands in for that signal.
Each trigger reports a reason; TriggerManager merges reasons within min_gap steps into one System 2 call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .types import TriggerContext, TriggerEvent

STEPS_PER_SECOND = 20


class Trigger(ABC):
    name = ""

    def reset(self) -> None:
        pass

    @abstractmethod
    def fired(self, c: TriggerContext) -> bool:
        ...


class Periodic(Trigger):
    name = "periodic"

    def __init__(self, every_steps: int = STEPS_PER_SECOND):
        self.every = every_steps

    def fired(self, c):
        return c.t > 0 and c.t % self.every == 0


class GripperToggle(Trigger):
    """Sign change of action[6] (LIBERO: -1 open, +1 close), counted after `debounce` steps with the new sign."""
    name = "gripper_toggle"

    def __init__(self, debounce: int = 3):
        self.debounce = debounce
        self.reset()

    def reset(self):
        self.stable, self.candidate, self.count = None, None, 0

    def fired(self, c):
        if c.action is None:
            return False
        sign = bool(c.action[6] > 0)
        if self.stable is None:
            self.stable = sign
            return False
        if sign == self.stable:
            self.candidate, self.count = None, 0
            return False
        self.count = self.count + 1 if sign == self.candidate else 1
        self.candidate = sign
        if self.count >= self.debounce:
            self.stable, self.candidate, self.count = sign, None, 0
            return True
        return False


class Quiescence(Trigger):
    """Rising edge of: small commanded motion and small eef travel over the whole window."""
    name = "quiescence"

    def __init__(self, window: int = STEPS_PER_SECOND, action_norm: float = 0.05, travel_m: float = 0.005):
        self.window, self.action_norm, self.travel_m = window, action_norm, travel_m
        self.reset()

    def reset(self):
        self.actions, self.pos = deque(maxlen=self.window), deque(maxlen=self.window)
        self.quiet = False

    def fired(self, c):
        if c.action is not None:
            self.actions.append(float(np.linalg.norm(c.action[:6])))
        self.pos.append(np.asarray(c.eef_pos, dtype=float))
        if len(self.actions) < self.window or len(self.pos) < self.window:
            return False
        quiet = (np.mean(self.actions) < self.action_norm
                 and np.linalg.norm(self.pos[-1] - self.pos[0]) < self.travel_m)
        rising = quiet and not self.quiet
        self.quiet = quiet
        return rising


class EmptyGrasp(Trigger):
    """Rising edge of: gripper commanded closed and fingers closed to near zero, so nothing is between them."""
    name = "empty_grasp"

    def __init__(self, width_m: float = 0.004):
        self.width_m = width_m
        self.reset()

    def reset(self):
        self.empty = False

    def fired(self, c):
        empty = c.action is not None and c.action[6] > 0 and c.gripper_width < self.width_m
        rising = empty and not self.empty
        self.empty = empty
        return rising


class SubgoalTimeout(Trigger):
    """Fires once per command when it has been active for too long."""
    name = "subgoal_timeout"

    def __init__(self, steps: int = 300):
        self.steps = steps
        self.reset()

    def reset(self):
        self.done = False

    def fired(self, c):
        if c.steps_since_command < self.steps:
            self.done = False
            return False
        if self.done:
            return False
        self.done = True
        return True


TRIGGERS = {t.name: t for t in (Periodic, GripperToggle, Quiescence, EmptyGrasp, SubgoalTimeout)}


class TriggerManager:
    def __init__(self, triggers: list[Trigger], min_gap: int = 5):
        self.triggers, self.min_gap = triggers, min_gap
        self.reset()

    def reset(self):
        for t in self.triggers:
            t.reset()
        self.pending: list[str] = []
        self.last_call = -self.min_gap

    def poll(self, c: TriggerContext) -> TriggerEvent | None:
        for trig in self.triggers:
            if trig.fired(c) and trig.name not in self.pending:
                self.pending.append(trig.name)
        if self.pending and c.t - self.last_call >= self.min_gap:
            event = TriggerEvent(c.t, self.pending)
            self.pending, self.last_call = [], c.t
            return event
        return None


def build_triggers(cfg: dict | None) -> TriggerManager:
    """cfg: {min_gap: 5, enabled: {periodic: {every_steps: 20}, quiescence: {}, ...}}"""
    cfg = cfg or {}
    return TriggerManager([TRIGGERS[n](**(kw or {})) for n, kw in (cfg.get("enabled") or {}).items()],
                          cfg.get("min_gap", 5))
