"""The control loop. Every robot action comes from System 1; System 2 only sends commands and verdicts.

Scoring: the env predicate is the ground truth, never System 2's verdict. If System 2 judges, env success does not end
the episode (the judge could not be evaluated otherwise) and the predicate at the final step is the score. Without a
judge the episode ends at env success, as in lerobot-eval.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .env import World
from .interface import InterfaceTransform
from .robot_states import RobotState
from .system1 import System1
from .system2 import System2
from .tracing import TraceWriter
from .triggers import TriggerManager
from .types import Decision, EpisodeResult, Observation, RunSpec, S2Input, TriggerContext, TriggerEvent


def seed_everything(seed: int) -> None:
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class RolloutRunner:
    def __init__(self, world: World, system1: System1, system2: System2, interface: InterfaceTransform,
                 triggers: TriggerManager, tracer: TraceWriter):
        self.world, self.system1, self.system2 = world, system1, system2
        self.interface, self.triggers, self.tracer = interface, triggers, tracer

    def _call_system2(self, event: TriggerEvent, obs: Observation, success: bool) -> Decision:
        d = self.system2.decide(S2Input(obs, self.instruction, event))
        self.s2_calls += 1
        self.s2_time += d.latency_s
        progress = self.world.goal_progress()
        self.tracer.decision(event, d, obs, progress, success)
        if d.closes:
            self.tracer.close_subgoal(d.closes, obs, progress)
        if d.command is not None:
            command = self.interface.apply(d.command)
            self.tracer.close_subgoal("abandoned", obs, progress)  # no-op unless a sub-goal is still open
            self.tracer.open_subgoal(command, obs, progress)
            self.system1.set_command(command.text)
            self.last_command_t = obs.t
        return d

    def run(self, spec: RunSpec, state: RobotState) -> EpisodeResult:
        t0 = time.perf_counter()
        seed_everything(spec.seed)
        self.instruction = spec.instruction
        self.s2_calls, self.s2_time, self.last_command_t = 0, 0.0, 0
        self.system1.reset()
        self.system2.reset(spec.instruction)
        self.interface.reset(spec.instruction)
        self.triggers.reset()

        obs = self.world.reset(state, spec.seed)
        self.tracer.start(obs, self.world.reset_info())
        success, first_success, verdict, end_reason = False, None, None, "time_limit"
        d = self._call_system2(TriggerEvent(0, ["start"]), obs, False)
        verdict = d.verdict

        action = None
        while verdict is None and obs.t < spec.max_steps:
            action = self.system1.act(obs)
            obs, success = self.world.step(action)
            self.tracer.step(obs, action, success)
            if success and first_success is None:
                first_success = obs.t
            if success and not self.system2.judges:
                end_reason = "env_success"
                break
            event = self.triggers.poll(TriggerContext(obs.t, action, obs.eef_pos, obs.gripper_width,
                                                      obs.t - self.last_command_t))
            if event is not None:
                verdict = self._call_system2(event, obs, success).verdict
        if verdict is not None:
            end_reason = "s2_verdict"

        self.tracer.close_subgoal("abandoned", obs, self.world.goal_progress())
        result = EpisodeResult(success=bool(success), n_steps=obs.t, first_success_step=first_success,
                               verdict=verdict.value if verdict else None, end_reason=end_reason, s2_calls=self.s2_calls,
                               s2_time_s=round(self.s2_time, 2), wall_s=round(time.perf_counter() - t0, 2))
        self.tracer.finish(result)
        return result
