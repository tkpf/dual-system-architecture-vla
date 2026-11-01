"""LIBERO behind a small World interface, so the rollout loop can be tested with a fake.

Uses LiberoEnv directly rather than a vector env: the vector env autoresets on termination, and with the judge on the
episode has to continue after env success.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .robot_states import RobotState
from .types import Observation

OBS_SIZE = 360  # lerobot's LiberoEnv config default, what lerobot-eval uses


def to_observation(t: int, raw: dict) -> Observation:
    # Raw robosuite frames are upside down; LiberoProcessorStep applies the same flip for System 1.
    return Observation(t=t, raw=raw, agent=raw["pixels"]["image"][::-1, ::-1].copy(),
                       wrist=raw["pixels"]["image2"][::-1, ::-1].copy(),
                       eef_pos=np.asarray(raw["robot_state"]["eef"]["pos"], dtype=float),
                       gripper_qpos=np.asarray(raw["robot_state"]["gripper"]["qpos"], dtype=float))


class World(ABC):
    instruction = ""
    max_steps = 0

    @abstractmethod
    def reset(self, state: RobotState, seed: int) -> Observation: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[Observation, bool]:
        """Returns the new observation and whether the env's success predicate holds."""

    def goal_progress(self) -> dict | None:
        """Per-predicate truth of the BDDL goal, or None if unavailable."""
        return None

    def reset_info(self) -> dict:
        return {}

    def close(self) -> None:
        pass


def _finger_side(geom: str | None) -> str | None:
    """Which finger a geom belongs to, from robosuite's names such as gripper0_finger1_pad_collision."""
    for side in ("finger1", "finger2"):
        if geom and side in geom:
            return side
    return None


class LiberoWorld(World):
    def __init__(self, suite: str, task_id: int, is_libero_plus: bool = False, max_steps: int | None = None):
        from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action

        self._dummy = get_libero_dummy_action
        self.env = LiberoEnv(task_suite=_get_suite(suite), task_id=task_id, task_suite_name=suite,
                             obs_type="pixels_agent_pos", observation_height=OBS_SIZE, observation_width=OBS_SIZE,
                             episode_length=max_steps, is_libero_plus=is_libero_plus)
        self.instruction = self.env.task_description
        self.max_steps = self.env._max_episode_steps
        self.t = 0
        self._reset_info: dict = {}

    def reset(self, state, seed):
        env = self.env
        env.init_state_id = state.init_state_id
        settle, env.num_steps_wait = env.num_steps_wait, 0
        env.reset(seed=seed)
        env.num_steps_wait = settle
        info = self._apply_offset(state)
        raw_obs = None
        for _ in range(settle):
            raw_obs, _, _, _ = env._env.step(self._dummy())
        robot = env._env.robots[0]
        joints = np.asarray(env._env.env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=float)
        info["eef_after_settle"] = np.asarray(raw_obs["robot0_eef_pos"]).round(4).tolist()
        info["joints_after_settle"] = joints.round(4).tolist()
        info["collision_contacts"] = self._robot_contacts()
        self._reset_info = info
        self.t = 0
        return to_observation(0, env._format_raw_obs(raw_obs))

    def _apply_offset(self, state: RobotState) -> dict:
        """Set the joints before the settle steps, and make the new pose the OSC nullspace target, so the
        controller does not slowly pull the arm back to the stored pose."""
        sim, robot = self.env._env.env.sim, self.env._env.robots[0]
        sim.forward()
        idx = robot._ref_joint_pos_indexes
        before = np.asarray(sim.data.get_site_xpos(robot.eef_site_name) if hasattr(robot, "eef_site_name")
                            else sim.data.site_xpos[robot.eef_site_id], dtype=float).copy()
        info = {"eef_before_offset": before.round(4).tolist()}
        if np.any(state.offset_rad):
            q = np.asarray(sim.data.qpos[idx], dtype=float) + state.offset_rad
            low, high = sim.model.jnt_range[robot._ref_joint_indexes].T
            clipped = np.clip(q, low, high)
            info["offset_clipped"] = bool(np.any(clipped != q))
            sim.data.qpos[idx] = clipped
            sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
            sim.forward()
            robot.controller.update_initial_joints(clipped)
            robot.controller.update(force=True)
            robot.controller.reset_goal()
        return info

    def _robot_contacts(self) -> int:
        """Contacts between robot geoms and anything that is not the robot, right after settling."""
        sim, robot = self.env._env.env.sim, self.env._env.robots[0]
        robot_geoms = set(robot.robot_model.contact_geoms)
        n = 0
        for c in sim.data.contact[:sim.data.ncon]:
            g1, g2 = sim.model.geom_id2name(c.geom1), sim.model.geom_id2name(c.geom2)
            if (g1 in robot_geoms) != (g2 in robot_geoms):
                n += 1
        return n

    def step(self, action):
        raw, _, _, _, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.t += 1
        return to_observation(self.t, raw), bool(info["is_success"])

    def goal_progress(self):
        inner = self.env._env.env
        try:
            return {" ".join(map(str, g)): bool(inner._eval_predicate(g)) for g in inner.parsed_problem["goal_state"]}
        except Exception:  # not all LIBERO forks expose this; ground truth then falls back to env success
            return None

    def gripped(self, name: str) -> bool:
        """Both gripper fingers touch the named object, or a body attached to it."""
        inner = self.env._env.env
        sim = inner.sim
        root = sim.model.body_rootid[inner.obj_body_id[name]]
        fingers = set()
        for c in sim.data.contact[:sim.data.ncon]:
            for mine, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                finger = _finger_side(sim.model.geom_id2name(mine))
                if finger and sim.model.body_rootid[sim.model.geom_bodyid[other]] == root:
                    fingers.add(finger)
        return len(fingers) == 2

    def reset_info(self):
        return self._reset_info

    def close(self):
        self.env.close()
