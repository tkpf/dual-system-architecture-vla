"""System 2: decides language commands for System 1 and episode verdicts. It has no access to robot control."""
from __future__ import annotations

from ..knowledge import build_retriever
from .base import System2
from .passthrough import Passthrough
from .reasoners import build_reasoner
from .supervisor import Supervisor


def build_system2(cfg: dict, shared: dict | None = None) -> System2:
    """shared caches heavy objects (the VLM) between conditions in one worker process."""
    if cfg["type"] == "passthrough":
        return Passthrough()
    if cfg["type"] == "supervisor":
        kw = {k: v for k, v in cfg.items() if k not in ("type", "reasoner", "knowledge")}
        return Supervisor(build_reasoner(cfg["reasoner"], shared), build_retriever(cfg["knowledge"]), **kw)
    raise ValueError(f"unknown system2 type: {cfg['type']}")
