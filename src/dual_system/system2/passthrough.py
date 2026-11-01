from __future__ import annotations

from ..types import Command, Decision, S2Input
from .base import System2


class Passthrough(System2):
    """No System 2: the instruction goes to System 1 unchanged. This is the single-system baseline."""

    def reset(self, instruction):
        self.started = False

    def decide(self, s: S2Input) -> Decision:
        if self.started:
            return Decision()
        self.started = True
        return Decision(command=Command(s.instruction))
