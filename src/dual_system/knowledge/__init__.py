"""Atomic output semantics. System 2's prompts hold no command vocabulary; it all comes from a profile here.

A profile (profiles/<name>.yaml) has: `rules` (how commands must look), `examples`, `recovery_commands`, and optional
`tasks` (instruction -> candidate commands) and `phase_definitions` (what a finished phase looks like). Swapping the file
changes the semantics without touching code.
"""
from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).parent / "profiles"


class Retriever(ABC):
    @abstractmethod
    def rules(self, instruction: str) -> list[str]:
        """Snippets telling System 2 how a command must look, for this instruction."""

    @abstractmethod
    def candidates(self, instruction: str) -> list[str] | None:
        """Allowed commands for this instruction, or None when the form is free."""

    @abstractmethod
    def recovery_commands(self) -> list[str]:
        """Sentences issued in order when a sub-goal failed."""

    def phase_definition(self, step: str) -> str:
        """What the scene looks like when this phase is finished."""
        return "the result of the command is clearly visible in the scene and the robot has finished."


class StaticRetriever(Retriever):
    """Returns the matching parts of one profile. The corpus is small, so no embedding search is needed yet."""

    def __init__(self, profile: str):
        self.name = profile
        self.doc = yaml.safe_load((PROFILE_DIR / f"{profile}.yaml").read_text(encoding="utf-8"))
        self.tasks = {t["instruction"]: t["candidates"] for t in self.doc.get("tasks", [])}

    def rules(self, instruction):
        return list(self.doc.get("rules", [])) + list(self.doc.get("examples", []))

    def candidates(self, instruction):
        return self.tasks.get(instruction)

    def recovery_commands(self):
        return list(self.doc.get("recovery_commands", []))

    def phase_definition(self, step):
        text = step.lower()
        for d in self.doc.get("phase_definitions", []):
            if any(m in text for m in d["match"]):
                return d["done"]
        return self.doc.get("default_definition") or super().phase_definition(step)


def snap_to_candidates(steps: list[str], candidates: list[str] | None, cutoff: float = 0.6) -> list[str]:
    """Keep the model's order; map each step to its closest allowed command and drop what matches none."""
    if candidates is None:
        return [s for s in steps if s.strip()]
    out = []
    for s in steps:
        m = difflib.get_close_matches(s.strip().lower(), candidates, n=1, cutoff=cutoff)
        if m and m[0] not in out:
            out.append(m[0])
    return out


RETRIEVERS = {"static": StaticRetriever}


def build_retriever(cfg: dict) -> Retriever:
    return RETRIEVERS[cfg.get("type", "static")](cfg["profile"])
