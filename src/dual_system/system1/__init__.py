"""System 1: the only part that produces robot actions."""
from __future__ import annotations

from .base import System1


def build_system1(cfg: dict) -> System1:
    if cfg["type"] == "smolvla":
        from .smolvla import SmolVLASystem1  # imports lerobot

        return SmolVLASystem1(**{k: v for k, v in cfg.items() if k != "type"})
    raise ValueError(f"unknown system1 type: {cfg['type']}")
