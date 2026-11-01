from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Decision, S2Input


class System2(ABC):
    judges = False  # True: only System 2's verdict or the time limit ends the episode, env success does not

    @abstractmethod
    def reset(self, instruction: str) -> None:
        ...

    @abstractmethod
    def decide(self, s: S2Input) -> Decision:
        """Called with the event 'start' at t=0, then whenever a trigger fires."""
