"""Reasoners answer the Supervisor's structured questions. The Supervisor logic is the same for all of them.

Answer.parsed is None when the output could not be used; the Supervisor then keeps the current command.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Answer:
    parsed: dict | None
    raw: str = ""
    prompt_id: str = ""
    latency_s: float = 0.0


def parse_json(text: str) -> dict | None:
    """First JSON object in the text (models like wrapping it in a code fence), or None."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {x}" for x in lines)


# The templates hold the question and answer format only. Command vocabulary comes from the knowledge profile.
VALIDATE_V1 = """You supervise a robot arm that manipulates objects on a table. The first image is the scene camera, the second the wrist camera.
Instruction: "{instruction}"
Is this a sensible manipulation instruction that can be carried out with the objects in the scene? Reject gibberish, questions and requests unrelated to the scene.
Answer with JSON only: {{"valid": true or false, "reason": "at most five words"}}"""

PLAN_V1 = """You supervise a robot arm that manipulates objects on a table. The first image is the scene camera, the second the wrist camera.
Instruction: "{instruction}"
Split the instruction into the commands the robot should carry out, in order.
Rules for commands:
{rules}
{candidates}{completed}Answer with JSON only: {{"steps": ["...", "..."]}}"""

# One letter: a JSON form with step and task fields answered task=success on every frame of a test episode, including
# the first. This form said "still running" until the result was visible (one episode, so not a general result).
CHECK_V2 = """You watch a robot arm. Command: "{step}".{signals}
Choose exactly one: A = the command is still being carried out or has not started. B = the result of the command is clearly visible and the robot has finished. C = it clearly failed, e.g. an object was dropped.
Answer with one letter."""

CHECK_LETTERS = {"A": "running", "B": "done", "C": "failed"}


def parse_check(text: str) -> dict | None:
    m = re.match(r"\W*([ABC])\b", text)
    return {"step": CHECK_LETTERS[m.group(1)]} if m else None


SIGNALS = {
    "gripper_toggle": "the gripper just opened or closed",
    "quiescence": "the robot stopped moving",
    "empty_grasp": "the gripper closed with nothing between the fingers",
    "subgoal_timeout": "the command has been running for a long time",
}


class Reasoner(ABC):
    def reset(self) -> None:
        """Called at the start of each episode."""

    @abstractmethod
    def validate(self, instruction: str, images: list[np.ndarray]) -> Answer: ...

    @abstractmethod
    def plan(self, instruction: str, images: list[np.ndarray], rules: list[str], candidates: list[str] | None,
             completed: list[str]) -> Answer: ...

    @abstractmethod
    def check(self, instruction: str, step: str, reasons: list[str], images: list[np.ndarray]) -> Answer: ...


class ScriptedReasoner(Reasoner):
    """Fixed answers, for tests and simulator smoke runs without a VLM. check_answers repeat their last entry."""

    def __init__(self, valid: bool = True, steps: list[str] | None = None, check_answers: list[dict] | None = None):
        self.valid, self.steps = valid, steps
        self.check_answers = check_answers or [{"step": "running", "task": "running"}]
        self.i = 0

    def reset(self):
        self.i = 0

    def validate(self, instruction, images):
        return Answer({"valid": self.valid, "reason": "scripted"}, prompt_id="scripted")

    def plan(self, instruction, images, rules, candidates, completed):
        return Answer({"steps": self.steps or [instruction]}, prompt_id="scripted")

    def check(self, instruction, step, reasons, images):
        a = self.check_answers[min(self.i, len(self.check_answers) - 1)]
        self.i += 1
        return Answer(a, json.dumps(a), "scripted")


class QwenReasoner(Reasoner):
    """Qwen3-VL through plain transformers, greedy. On a T4 use fp16 (no native bf16)."""

    def __init__(self, model: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = "cuda", dtype: str = "float16",
                 image_size: int = 256, max_tokens_check: int = 30, max_tokens_plan: int = 120):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device, self.image_size = device, image_size
        self.max_tokens_check, self.max_tokens_plan, self.max_tokens_letter = max_tokens_check, max_tokens_plan, 3
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=getattr(torch, dtype)).to(device).eval()

    def _generate(self, prompt: str, images: list[np.ndarray], max_new: int, prompt_id: str, parse=parse_json) -> Answer:
        from PIL import Image

        content = [{"type": "image", "image": Image.fromarray(i).resize((self.image_size, self.image_size))} for i in images]
        content.append({"type": "text", "text": prompt})
        t0 = time.perf_counter()
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return Answer(parse(raw), raw, prompt_id, time.perf_counter() - t0)

    def validate(self, instruction, images):
        return self._generate(VALIDATE_V1.format(instruction=instruction), images, self.max_tokens_check, "validate_v1")

    def plan(self, instruction, images, rules, candidates, completed):
        cand = f"Candidates: {json.dumps(candidates)}\n" if candidates else ""
        done = f"Already completed, do not repeat: {json.dumps(completed)}\n" if completed else ""
        prompt = PLAN_V1.format(instruction=instruction, rules=_bullets(rules), candidates=cand, completed=done)
        return self._generate(prompt, images, self.max_tokens_plan, "plan_v1")

    def check(self, instruction, step, reasons, images):
        signals = "; ".join(SIGNALS[r] for r in reasons if r in SIGNALS)
        prompt = CHECK_V2.format(step=step, signals=f"\nSignal: {signals}." if signals else "")
        return self._generate(prompt, images[:1], self.max_tokens_letter, "check_v2", parse_check)  # scene camera only, as tested


def build_reasoner(cfg: dict, shared: dict | None = None) -> Reasoner:
    kw = {k: v for k, v in cfg.items() if k != "type"}
    if cfg["type"] == "scripted":
        return ScriptedReasoner(**kw)
    if cfg["type"] == "qwen":
        shared = shared if shared is not None else {}
        key = ("qwen", json.dumps(kw, sort_keys=True))
        if key not in shared:
            shared[key] = QwenReasoner(**kw)
        return shared[key]
    raise ValueError(f"unknown reasoner type: {cfg['type']}")
