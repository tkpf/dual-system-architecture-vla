"""Frozen SmolVLA, through the same processor chain as lerobot_eval.rollout (checked against lerobot 0.6.1).

The command is written to observation["task"] before the preprocessor, so the tokenizer step always sees the current
text and no re-tokenising is needed inside the policy.
"""
from __future__ import annotations

import numpy as np
import torch

from ..types import Observation
from .base import System1

RENAME_MAP = {"observation.images.image": "observation.images.camera1",
              "observation.images.image2": "observation.images.camera2"}


def _batch(x):
    if isinstance(x, dict):
        return {k: _batch(v) for k, v in x.items()}
    return np.asarray(x)[None]


class SmolVLASystem1(System1):
    def __init__(self, path: str = "lerobot/smolvla_libero", n_action_steps: int = 10, device: str | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors, preprocess_observation
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.constants import ACTION

        self._preprocess_observation, self._action_key = preprocess_observation, ACTION
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        env_cfg = LiberoEnvConfig()
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.n_action_steps = n_action_steps
        cfg.device = device
        # Checkpoints differ in their camera names: rename only for those that expect camera1 / camera2.
        rename = RENAME_MAP if set(RENAME_MAP.values()) <= set(cfg.input_features) else {}
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device},
                                    "rename_observations_processor": {"rename_map": rename}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
        self.command = ""

    def reset(self):
        self.policy.reset()
        self.command = ""

    def set_command(self, text):
        if text == self.command:
            return False
        self.command = text
        self.policy._queues[self._action_key].clear()
        return True

    @torch.no_grad()
    def act(self, obs: Observation) -> np.ndarray:
        batch = self._preprocess_observation(_batch(obs.raw))
        batch["task"] = [self.command]
        batch = self.pre(self.env_pre(batch))
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.env_post({self._action_key: self.post(action)})[self._action_key]
        return action.detach().cpu().numpy().squeeze(0)
