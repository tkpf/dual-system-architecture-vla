"""Named start states. A: a stored LIBERO state. B: the same state plus one fixed joint offset, a real perturbation
(stored states only differ by about 1 degree per joint). Both are identical for all tasks and conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobotState:
    name: str
    init_state_id: int = 0
    joint_offset_deg: list[float] = field(default_factory=lambda: [0.0] * 7)

    @property
    def offset_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.joint_offset_deg, dtype=float))


def build_robot_states(cfg: dict) -> dict[str, RobotState]:
    return {name: RobotState(name, **(kw or {})) for name, kw in cfg.items()}
