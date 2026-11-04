"""System 1: the only part that produces robot actions."""
from __future__ import annotations

from .base import System1


def build_system1(cfg: dict) -> System1:
    if cfg["type"] in ("smolvla", "pi0", "pi05"):
        from .lerobot_policy import LeRobotSystem1  # imports lerobot

        return LeRobotSystem1(pi_backbone=cfg["type"] != "smolvla", **{k: v for k, v in cfg.items() if k != "type"})
    raise ValueError(f"unknown system1 type: {cfg['type']}")
