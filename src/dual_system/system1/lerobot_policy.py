"""A frozen LeRobot policy (SmolVLA, pi0, pi0.5), through the same processor chain as lerobot_eval.rollout (checked
against lerobot 0.6.1).

The command is written to observation["task"] before the preprocessor, so the tokenizer step always sees the current
text and no re-tokenising is needed inside the policy.
"""
from __future__ import annotations

import functools

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


def clear_action_queue(policy, key: str) -> None:
    """SmolVLA reads _queues[action]; pi0 and pi0.5 read _action_queue instead."""
    queue = getattr(policy, "_action_queue", None)
    if queue is None:
        queue = policy._queues[key]
    queue.clear()


def _build_pi_backbone_on_gpu() -> None:
    """Stock loading builds the backbone in fp32 on the CPU and runs out of host RAM on a T4 (exit 137)."""
    from lerobot.policies.pi0 import modeling_pi0
    from lerobot.policies.pi05 import modeling_pi05

    for cls in (modeling_pi0.PaliGemmaWithExpertModel, modeling_pi05.PaliGemmaWithExpertModel):
        if getattr(cls, "_built_on_gpu", False):
            continue
        orig = cls.__init__

        @functools.wraps(orig)
        def init(self, *a, _orig=orig, **k):
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device("cuda"):
                    _orig(self, *a, **k)
            finally:
                torch.set_default_dtype(prev)

        cls.__init__, cls._built_on_gpu = init, True


class LeRobotSystem1(System1):
    def __init__(self, path: str = "lerobot/smolvla_libero", n_action_steps: int = 10, device: str | None = None,
                 pi_backbone: bool = False):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors, preprocess_observation
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.constants import ACTION

        self._preprocess_observation, self._action_key = preprocess_observation, ACTION
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        env_cfg = LiberoEnvConfig()
        if pi_backbone:
            _build_pi_backbone_on_gpu()
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.n_action_steps = n_action_steps
        cfg.device = device
        if hasattr(cfg, "compile_model"):  # the pi checkpoints enable max-autotune, which stalls the first episodes on a T4
            cfg.compile_model = False
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
        clear_action_queue(self.policy, self._action_key)
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
