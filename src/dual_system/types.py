"""Data passed between the pipeline parts. No lerobot / torch imports, so everything here is testable anywhere."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Verdict(str, Enum):
    """The only thing System 2 can do besides commanding: end the episode."""
    SUCCESS = "success"
    FAILURE = "failure"
    REJECTED = "rejected"


@dataclass
class Observation:
    t: int
    raw: dict                  # LiberoEnv observation, unbatched; what System 1 consumes
    agent: np.ndarray          # upright uint8 HxWx3
    wrist: np.ndarray
    eef_pos: np.ndarray
    gripper_qpos: np.ndarray

    @property
    def gripper_width(self) -> float:
        return float(np.sum(np.abs(self.gripper_qpos)))


@dataclass
class Command:
    """One language command to System 1. kind='task' works on the plan, kind='recovery' is a recovery sentence."""
    text: str
    kind: str = "task"
    subgoal_index: int = 0
    retry: int = 0
    replan: int = 0


@dataclass
class Decision:
    command: Command | None = None   # None keeps the current command
    verdict: Verdict | None = None
    closes: str | None = None        # outcome of the previous sub-goal when it ends here: done | failed | abandoned
    status: dict | None = None       # parsed reasoner answer of this call
    raw: str = ""
    prompt_id: str = ""
    latency_s: float = 0.0


@dataclass
class TriggerEvent:
    t: int
    reasons: list[str]


@dataclass
class S2Input:
    obs: Observation
    instruction: str
    event: TriggerEvent


@dataclass
class TriggerContext:
    t: int
    action: np.ndarray | None        # last executed action
    eef_pos: np.ndarray
    gripper_width: float
    steps_since_command: int


@dataclass
class RunSpec:
    """Everything that identifies one episode. Pairs differ only in condition."""
    suite: str
    task_id: int
    robot_state: str
    condition: str
    seed: int
    instruction: str
    max_steps: int
    run: str = ""
    git_sha: str = ""
    cell: str = ""


@dataclass
class EpisodeResult:
    success: bool                    # env predicate at the end of the episode; always the ground truth
    n_steps: int
    first_success_step: int | None
    verdict: str | None
    end_reason: str                  # env_success | s2_verdict | time_limit
    s2_calls: int = 0
    s2_time_s: float = 0.0
    wall_s: float = 0.0
    meta: dict = field(default_factory=dict)
