"""The final System 2: it switches System 1's atomic commands on signals the robot has, and asks a VLM only where no robot
signal exists. It never ends an episode.

Runs 1 to 12 showed a VLM that cannot tell from images whether a subtask is finished (AUROC 0.71 at best, about a third of
unfinished frames called done) and that false verdicts ended healthy episodes. So:
  pick   done when the gripper holds something (width between the closed-empty and the open value) for `min_hold` steps and
         the hand has risen `lift` above where the hold began
  place  done when the gripper opens again after a hold
  act    (open, close, turn on, push) no robot signal shows these: the VLM's probability of "done" must stay above a
         threshold for `vlm_confirm` checks in a row, after a minimum time
  every phase also has a timeout, after which the next command is issued anyway
Recovery: the gripper closed on nothing during a pick -> "open the gripper" for a few steps, then the pick again. With
detect_drop, a release counts as a place only if the hand came down `drop_descent` while holding; a release in mid-air is a
drop and sends the switcher back to the pick (off by default, so earlier runs keep their behaviour).
Width is the sum of both finger positions: about 0.078 open, 0.003 to 0.005 when pi0.5 holds the thin bowl rim, about 0.014
for the moka pot handle, and 0.001 to 0.0017 closed on nothing (pi0.5 command-probe traces). The bowl margin is thin, so a
pick also needs the hand to rise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EMPTY_WIDTH = 0.002
OPEN_WIDTH = 0.07
HELD_MAX = 0.065
RECOVERY_COMMAND = "open the gripper"


@dataclass
class Switcher:
    """Shared bookkeeping: the phase list, the current command and the timeouts. Subclasses decide when a phase is done."""
    phases: list[dict]
    timeout: int = 150
    idx: int = 0
    phase_start: int = 0
    log: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.idx, self.phase_start, self.log = 0, 0, []
        self.recovering_until: int | None = None

    @property
    def phase(self) -> dict:
        return self.phases[self.idx]

    @property
    def command(self) -> str:
        return RECOVERY_COMMAND if self.recovering_until is not None else self.phase["command"]

    def _advance(self, t: int, reason: str, truth: bool | None) -> dict:
        event = dict(t=t, phase=self.idx, command=self.phase["command"], reason=reason, truth_done=truth, steps=t - self.phase_start)
        if self.idx + 1 < len(self.phases):
            self.idx += 1
            self.phase_start = t
            self._on_new_phase()
        else:
            event["all_done"] = True
        self.log.append(event)
        return event

    def _on_new_phase(self) -> None:
        pass

    def _back(self, t: int, reason: str) -> dict:
        """The previous phase was undone, e.g. the object was dropped: issue it again."""
        event = dict(t=t, phase=self.idx, command=self.phase["command"], reason=reason, steps=t - self.phase_start)
        self.idx -= 1
        self.phase_start = t
        self.loose = 0
        self._on_new_phase()
        self.log.append(event)
        return event


class OracleSwitcher(Switcher):
    """Simulator truth decides. The upper bound for any progress check. With undo, a place phase goes back to its pick when
    the simulator says the object has left the gripper for `undo_steps` steps without being placed."""

    def __init__(self, phases: list[dict], timeout: int = 150, undo: bool = False, undo_steps: int = 10):
        super().__init__(phases, timeout)
        self.undo, self.undo_steps = undo, undo_steps
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.loose = 0

    def update(self, t: int, truth: bool, gripped: bool | None = None) -> dict | None:
        if self.log and self.log[-1].get("all_done"):
            return None
        if self.undo and self.phase["kind"] == "place" and self.idx > 0 and self.phases[self.idx - 1]["kind"] == "pick" and not truth:
            self.loose = self.loose + 1 if gripped is False else 0
            if self.loose >= self.undo_steps:
                return self._back(t, "undone")
        if truth:
            return self._advance(t, "oracle", True)
        if t - self.phase_start >= self.timeout and self.idx + 1 < len(self.phases):
            return self._advance(t, "timeout", False)
        return None


class FusedSwitcher(Switcher):
    def __init__(self, phases: list[dict], timeout: int = 150, recover: bool = True, min_hold: int = 8, lift: float = 0.03,
                 min_empty: int = 6, recovery_steps: int = 15, max_retries: int = 2, vlm_every: int = 10,
                 vlm_min_steps: int = 40, vlm_threshold: float = 0.8, vlm_confirm: int = 2, detect_drop: bool = False,
                 drop_descent: float = 0.02, max_drops: int = 2):
        super().__init__(phases, timeout)
        self.detect_drop, self.drop_descent, self.max_drops = detect_drop, drop_descent, max_drops
        self.recover, self.min_hold, self.lift, self.min_empty = recover, min_hold, lift, min_empty
        self.recovery_steps, self.max_retries = recovery_steps, max_retries
        self.vlm_every, self.vlm_min_steps, self.vlm_threshold, self.vlm_confirm = vlm_every, vlm_min_steps, vlm_threshold, vlm_confirm
        self.vlm_probs: list[dict] = []
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.vlm_probs = []
        self.retries = self.drops = 0
        self._on_new_phase()

    def _on_new_phase(self) -> None:
        self.hold = self.empty = self.open_run = self.streak = 0
        self.hold_z = 0.0
        self.peak_z = float("-inf")
        self.was_held = False
        self.recovering_until = None
        if self.idx > 0 and self.phase["kind"] == "place" and self.phases[self.idx - 1]["kind"] == "pick" and self.log \
                and self.log[-1]["reason"] == "signal":
            self.was_held = True  # the pick was confirmed by a hold, so the gripper still holds the object

    def update(self, t: int, width: float, z: float, vlm=None, truth: bool | None = None) -> dict | None:
        """Called after every simulator step. vlm: a function returning the probability that the phase is done, called rarely.
        truth is the simulator's verdict on the phase. It is only written to the log, never used to decide."""
        if self.log and self.log[-1].get("all_done"):
            return None
        kind = self.phase["kind"]
        if self.recovering_until is not None:
            if t >= self.recovering_until:
                self.recovering_until = None
                self.phase_start = t
                self.hold = self.empty = 0
                return dict(t=t, phase=self.idx, command=self.command, reason="retry")
            return None

        held = EMPTY_WIDTH < width < HELD_MAX
        if held:
            self.peak_z = max(self.peak_z, z)
            self.hold += 1
            if self.hold == 1:
                self.hold_z = z
            self.was_held |= self.hold >= self.min_hold
        else:
            self.hold = 0
        self.empty = self.empty + 1 if width <= EMPTY_WIDTH else 0
        self.open_run = self.open_run + 1 if width >= OPEN_WIDTH else 0

        if kind == "pick" and self.hold >= self.min_hold and z - self.hold_z >= self.lift:
            return self._advance(t, "signal", truth)
        if kind == "place" and self.was_held and self.open_run >= 3:
            if self.detect_drop and self.peak_z - z < self.drop_descent and self.drops < self.max_drops and self.idx > 0:
                self.drops += 1
                return self._back(t, "dropped")
            return self._advance(t, "signal", truth)
        if kind == "act" and vlm is not None and t - self.phase_start >= self.vlm_min_steps and t % self.vlm_every == 0:
            p = vlm()
            self.vlm_probs.append(dict(t=t, phase=self.idx, p=round(p, 3)))
            self.streak = self.streak + 1 if p >= self.vlm_threshold else 0
            if self.streak >= self.vlm_confirm:
                return self._advance(t, "vlm", truth)
        if kind == "pick" and self.recover and self.empty >= self.min_empty and self.retries < self.max_retries:
            self.retries += 1
            self.recovering_until = t + self.recovery_steps
            return dict(t=t, phase=self.idx, command=RECOVERY_COMMAND, reason="empty_grasp")
        if t - self.phase_start >= self.timeout and self.idx + 1 < len(self.phases):
            return self._advance(t, "timeout", truth)
        return None
