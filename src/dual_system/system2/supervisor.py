"""The System 2 state machine. Deterministic: all perception goes through the Reasoner, so swapping the Reasoner
(Qwen 2B, 4B, later an oracle) changes the reasoning quality and nothing else.

Four toggles: gate, decompose, recover, judge. Every action it takes is a language command or a verdict.
Single answers do not change state: done / failed / success need `confirmations` agreeing calls in a row, because a
wrong "done" from a small VLM would otherwise skip a sub-goal or end a healthy episode. A sub-goal timeout counts as
failed at once.
"""
from __future__ import annotations

from ..knowledge import Retriever, snap_to_candidates
from ..types import Command, Decision, S2Input, Verdict
from .base import System2
from .reasoners import Answer, Reasoner


def _decision(a: Answer, **kw) -> Decision:
    return Decision(status=a.parsed, raw=a.raw, prompt_id=a.prompt_id, latency_s=a.latency_s, **kw)


class Supervisor(System2):
    def __init__(self, reasoner: Reasoner, retriever: Retriever, gate: bool = True, decompose: bool = True,
                 recover: bool = True, judge: bool = True, confirmations: int = 2, max_retries: int = 2,
                 recovery_steps: int = 40, min_verdict_step: int = 40):
        self.reasoner, self.retriever = reasoner, retriever
        self.gate, self.decompose, self.recover, self.judges = gate, decompose, recover, judge
        self.confirmations, self.max_retries = confirmations, max_retries
        self.recovery_steps, self.min_verdict_step = recovery_steps, min_verdict_step

    def reset(self, instruction):
        self.instruction = instruction
        self.started = False
        self.steps: list[str] = [instruction]
        self.idx = self.retries = self.replans = 0
        self.done_count = self.failed_count = self.success_count = self.failure_count = 0
        self.recovery_left: list[str] = []
        self.recovery_since: int | None = None
        self.reasoner.reset()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _images(s: S2Input):
        return [s.obs.agent, s.obs.wrist]

    def _command(self) -> Command:
        return Command(self.steps[self.idx], "task", self.idx, self.retries, self.replans)

    def _plan(self, s: S2Input, completed: list[str]) -> tuple[list[str], Answer]:
        """Falls back to the whole instruction when the answer is unusable, so the robot always has a command."""
        cands = self.retriever.candidates(self.instruction)
        a = self.reasoner.plan(self.instruction, self._images(s), self.retriever.rules(self.instruction), cands, completed)
        steps = (a.parsed or {}).get("steps")
        steps = snap_to_candidates([x for x in steps if isinstance(x, str)], cands) if isinstance(steps, list) else []
        return steps or [self.instruction], a

    def _reset_counts(self):
        self.done_count = self.failed_count = self.success_count = self.failure_count = 0

    # ---------------------------------------------------------------- decide
    def decide(self, s: S2Input) -> Decision:
        if not self.started:
            return self._start(s)
        if self.recovery_since is not None:
            return self._recovery_tick(s)
        return self._check(s)

    def _start(self, s: S2Input) -> Decision:
        self.started = True
        last = Answer(None)
        if self.gate:
            last = self.reasoner.validate(self.instruction, self._images(s))
            if last.parsed is not None and last.parsed.get("valid") is False:  # unusable output fails open
                return _decision(last, verdict=Verdict.REJECTED)
        if self.decompose:
            self.steps, last = self._plan(s, [])
        return _decision(last, command=self._command())

    def _check(self, s: S2Input) -> Decision:
        reasons = s.event.reasons
        a = self.reasoner.check(self.instruction, self.steps[self.idx], reasons, self._images(s))
        step = (a.parsed or {}).get("step")
        task = (a.parsed or {}).get("task")
        timeout = "subgoal_timeout" in reasons
        if a.parsed is None and not timeout:
            return _decision(a)

        self.done_count = self.done_count + 1 if step == "done" else 0
        self.failed_count = self.failed_count + 1 if step == "failed" else 0
        self.success_count = self.success_count + 1 if task == "success" and step == "done" else 0  # both must agree
        self.failure_count = self.failure_count + 1 if task == "failure" else 0
        can_end = self.judges and s.obs.t >= self.min_verdict_step

        if can_end and self.success_count >= self.confirmations:
            return _decision(a, verdict=Verdict.SUCCESS, closes="done")
        if timeout or self.failed_count >= self.confirmations or self.failure_count >= self.confirmations:
            return self._on_failure(s, a, can_end)
        if self.done_count >= self.confirmations:
            return self._on_step_done(a, can_end)
        return _decision(a)

    def _on_step_done(self, a: Answer, can_end: bool) -> Decision:
        self._reset_counts()
        if self.idx + 1 < len(self.steps):
            self.idx += 1
            self.retries = 0
            return _decision(a, command=self._command(), closes="done")
        if can_end:  # the last step is done and the judge is on
            return _decision(a, verdict=Verdict.SUCCESS, closes="done")
        return _decision(a)

    def _on_failure(self, s: S2Input, a: Answer, can_end: bool) -> Decision:
        self._reset_counts()
        if self.recover and self.retries < self.max_retries and self.retriever.recovery_commands():
            self.recovery_left = self.retriever.recovery_commands()
            return _decision(a, closes="failed", command=self._next_recovery(s))
        if can_end:
            return _decision(a, verdict=Verdict.FAILURE, closes="failed")
        return _decision(a)

    # ---------------------------------------------------------------- recovery: language only, System 1 executes it
    def _next_recovery(self, s: S2Input) -> Command:
        self.recovery_since = s.obs.t
        return Command(self.recovery_left.pop(0), "recovery", self.idx, self.retries, self.replans)

    def _recovery_tick(self, s: S2Input) -> Decision:
        """Each sentence gets recovery_steps of simulated time, then the sub-goal is retried."""
        if s.obs.t - self.recovery_since < self.recovery_steps:
            return Decision()
        if self.recovery_left:
            return Decision(command=self._next_recovery(s), closes="elapsed")
        self.recovery_since = None
        self.retries += 1
        last = Answer(None)
        if self.decompose:  # re-plan from the current scene, keeping what is already finished
            completed = self.steps[:self.idx]
            remaining, last = self._plan(s, completed)
            self.steps = completed + remaining
            self.replans += 1
        return _decision(last, command=self._command(), closes="elapsed")


class PhaseSupervisor(Supervisor):
    """The reasoner sees all phases, the current one and its finish condition, and names the current subtask. A later
    phase moves on by one, an earlier phase goes back to it; both need `confirmations` agreeing answers in a row."""

    def reset(self, instruction):
        super().reset(instruction)
        self.back_count = 0

    def _reset_counts(self):
        super()._reset_counts()
        self.back_count = 0

    def _check(self, s: S2Input) -> Decision:
        timeout = "subgoal_timeout" in s.event.reasons
        a = self.reasoner.phase_check(self.instruction, self.steps, self.idx, self.retriever.phase_definition(self.steps[self.idx]),
                                      s.event.reasons, self._images(s))
        step = (a.parsed or {}).get("step")
        can_end = self.judges and s.obs.t >= self.min_verdict_step
        self.done_count = self.done_count + 1 if step == "done" else 0
        self.back_count = self.back_count + 1 if step == "back" else 0
        if self.done_count >= self.confirmations:
            return self._on_step_done(a, can_end)
        if self.back_count >= self.confirmations:
            self._reset_counts()
            self.idx, self.retries = a.parsed["subtask_index"], 0
            return _decision(a, command=self._command(), closes="failed")
        if timeout:
            return self._on_failure(s, a, can_end)
        return _decision(a)
