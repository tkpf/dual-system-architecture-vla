from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Observation


class System1(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def set_command(self, text: str) -> bool:
        """Change the language command. Returns True if it changed; only then queued actions are dropped."""

    @abstractmethod
    def act(self, obs: Observation) -> np.ndarray:
        """One 7-D action for the current observation."""
