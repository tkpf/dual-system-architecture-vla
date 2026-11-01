"""The System 2 -> System 1 text interface. Every command passes here, so this is the single ablation point."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Command


class InterfaceTransform(ABC):
    def reset(self, instruction: str) -> None:
        pass

    @abstractmethod
    def apply(self, command: Command) -> Command:
        ...


class Identity(InterfaceTransform):
    def apply(self, command):
        return command


INTERFACES = {"identity": Identity}


def build_interface(cfg: dict | None) -> InterfaceTransform:
    cfg = cfg or {"type": "identity"}
    return INTERFACES[cfg["type"]](**{k: v for k, v in cfg.items() if k != "type"})
