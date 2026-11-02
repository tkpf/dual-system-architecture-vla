"""Reasoners answer the Supervisor's structured questions. The Supervisor logic is the same for all of them.

Answer.parsed is None when the output could not be used; the Supervisor then keeps the current command.
"""
from __future__ import annotations

import difflib
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


PHASE_CHECK_V1 = """You watch a robot arm. The first image is the scene camera, the second the wrist camera.
Task: "{instruction}"
The task is split into {n} phases:
{phases}
The robot is in phase {k} of {n}: "{step}".
Phase {k} is finished only when {definition}{signals}
First reason in at most two short sentences: compare what the images show with that condition. Then end with one final line in exactly this form:
CURRENT SUBTASK: <number>. <text of that phase>
Give phase {k} while its condition is not met and phase {next_k} once it is. If an earlier phase was undone, e.g. an object was dropped, give that phase.{last}"""


def parse_phase(text: str, steps: list[str], idx: int) -> dict | None:
    """The last CURRENT SUBTASK line, as an index into steps; len(steps) means all phases are finished. None if unusable."""
    found = re.findall(r"CURRENT SUBTASK:\s*(.+)", text, re.I)
    if not found:
        return None
    answer = found[-1].strip()
    number = re.match(r"\W*(\d+)", answer)
    if re.match(r"\W*(DONE|ALL)\b", answer, re.I):
        i = len(steps)
    elif number:
        i = int(number.group(1)) - 1
    else:
        close = difflib.get_close_matches(answer.lower(), [s.lower() for s in steps], n=1, cutoff=0.6)
        i = [s.lower() for s in steps].index(close[0]) if close else -1
    if not 0 <= i <= len(steps):
        return None
    step = "done" if i > idx else "running" if i == idx else "back"
    return {"step": step, "subtask_index": i, "subtask": steps[i] if i < len(steps) else "DONE"}


# One word, so a finetuned judge can be trained on one token and thresholded on its probability.
PHASE_DONE_V1 = """You watch a robot arm. The first image is the scene camera, the second the wrist camera on the gripper.
Subtask: "{step}".
It is finished only when {definition}
Is this subtask finished? Answer with one word: yes or no."""

# SmolVLM otherwise cuts each image into 17 tiles, 2,909 tokens for two cameras.
IMAGES_KWARGS = {"do_image_splitting": False}


def phase_done_prompt(step: str, definition: str) -> str:
    return PHASE_DONE_V1.format(step=step, definition=definition)


def parse_done(text: str) -> dict | None:
    """Yes means the current phase is finished. No going back: the sub-goal timeout handles recovery with this judge."""
    m = re.match(r"\W*(yes|no)\b", text, re.I)
    return {"step": "done" if m.group(1).lower() == "yes" else "running"} if m else None


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

    def phase_check(self, instruction: str, steps: list[str], idx: int, definition: str, reasons: list[str],
                    images: list[np.ndarray]) -> Answer:
        raise NotImplementedError


class ScriptedReasoner(Reasoner):
    """Fixed answers, for tests and simulator smoke runs without a VLM. check_answers repeat their last entry."""

    def __init__(self, valid: bool = True, steps: list[str] | None = None, check_answers: list[dict] | None = None,
                 phase_answers: list[str] | None = None):
        self.valid, self.steps, self.phase_answers = valid, steps, phase_answers or ["CURRENT SUBTASK: 1"]
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

    def phase_check(self, instruction, steps, idx, definition, reasons, images):
        raw = self.phase_answers[min(self.i, len(self.phase_answers) - 1)]
        self.i += 1
        return Answer(parse_phase(raw, steps, idx), raw, "scripted")


class QwenReasoner(Reasoner):
    """Qwen3-VL (or another AutoModelForImageTextToText VLM) through plain transformers, greedy. On a T4 use fp16 (no native bf16)."""

    def __init__(self, model: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = "cuda", dtype: str = "float16",
                 image_size: int = 256, max_tokens_check: int = 30, max_tokens_plan: int = 120, max_tokens_phase: int = 100,
                 quantize: bool = False, images_kwargs: dict | None = None, adapter: str | None = None,
                 phase_format: str = "subtask", done_threshold: float = 0.5):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device, self.image_size = device, image_size
        self.images_kwargs, self.phase_format, self.done_threshold = images_kwargs, phase_format, done_threshold
        self.adapter = adapter
        self.max_tokens_check, self.max_tokens_plan, self.max_tokens_letter = max_tokens_check, max_tokens_plan, 3
        self.max_tokens_phase = max_tokens_phase
        self.processor = AutoProcessor.from_pretrained(model)
        if quantize:  # 4-bit, for models that do not fit on the GPU otherwise
            from transformers import BitsAndBytesConfig

            config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype))
            self.model = AutoModelForImageTextToText.from_pretrained(model, quantization_config=config, device_map={"": device}).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=getattr(torch, dtype)).to(device).eval()
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).eval()

    def _inputs(self, prompt: str, images: list[np.ndarray]):
        from PIL import Image

        content = [{"type": "image", "image": Image.fromarray(i).resize((self.image_size, self.image_size))} for i in images]
        content.append({"type": "text", "text": prompt})
        extra = {"images_kwargs": self.images_kwargs} if self.images_kwargs else {}
        return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors="pt", **extra).to(self.device)

    def _generate(self, prompt: str, images: list[np.ndarray], max_new: int, prompt_id: str, parse=parse_json) -> Answer:
        t0 = time.perf_counter()
        inputs = self._inputs(prompt, images)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return Answer(parse(raw), raw, prompt_id, time.perf_counter() - t0)

    def _without_adapter(self, prompt: str, images: list[np.ndarray], max_new: int, prompt_id: str) -> Answer:
        """The LoRA only learned to judge subtasks, so the gate and the plan use the untrained model."""
        if not self.adapter:
            return self._generate(prompt, images, max_new, prompt_id)
        with self.model.disable_adapter():
            return self._generate(prompt, images, max_new, prompt_id)

    def _done_probability(self, prompt: str, images: list[np.ndarray]) -> float:
        """Probability of yes against no on the first answer token, as the judge was trained."""
        inputs = self._inputs(prompt, images)
        ids = [self.processor.tokenizer(w, add_special_tokens=False).input_ids[0] for w in (" no", " yes")]
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.float16):
            logits = self.model(**inputs).logits[0, -1, ids].float()
        return float(self.torch.softmax(logits, dim=0)[1])

    def validate(self, instruction, images):
        return self._without_adapter(VALIDATE_V1.format(instruction=instruction), images, self.max_tokens_check, "validate_v1")

    def plan(self, instruction, images, rules, candidates, completed):
        cand = f"Candidates: {json.dumps(candidates)}\n" if candidates else ""
        done = f"Already completed, do not repeat: {json.dumps(completed)}\n" if completed else ""
        prompt = PLAN_V1.format(instruction=instruction, rules=_bullets(rules), candidates=cand, completed=done)
        return self._without_adapter(prompt, images, self.max_tokens_plan, "plan_v1")

    def check(self, instruction, step, reasons, images):
        signals = "; ".join(SIGNALS[r] for r in reasons if r in SIGNALS)
        prompt = CHECK_V2.format(step=step, signals=f"\nSignal: {signals}." if signals else "")
        return self._generate(prompt, images[:1], self.max_tokens_letter, "check_v2", parse_check)  # scene camera only, as tested

    def phase_check(self, instruction, steps, idx, definition, reasons, images):
        if self.phase_format == "yes_no":
            prompt = phase_done_prompt(steps[idx], definition)
            if self.adapter:
                t0 = time.perf_counter()
                p = self._done_probability(prompt, images)
                return Answer({"step": "done" if p >= self.done_threshold else "running"}, f"{p:.3f}", "phase_done_v1", time.perf_counter() - t0)
            return self._generate(prompt, images, 4, "phase_done_v1", parse_done)
        n = len(steps)
        state = lambda i: "finished" if i < idx else "current" if i == idx else "not started"
        phases = "\n".join(f"{i + 1}. {s} [{state(i)}]" for i, s in enumerate(steps))
        signals = "; ".join(SIGNALS[r] for r in reasons if r in SIGNALS)
        last = " If the last phase is finished, write: CURRENT SUBTASK: DONE" if idx == n - 1 else ""
        prompt = PHASE_CHECK_V1.format(instruction=instruction, n=n, phases=phases, k=idx + 1, step=steps[idx], definition=definition,
                                       signals=f"\nSignal: {signals}." if signals else "", next_k=idx + 2 if idx < n - 1 else "DONE", last=last)
        return self._generate(prompt, images, self.max_tokens_phase, "phase_check_v1", lambda raw: parse_phase(raw, steps, idx))


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
