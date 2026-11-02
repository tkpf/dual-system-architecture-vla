"""Which way of asking lets an untrained VLM see that a subtask is finished? Same demo frames and labels as eval.demo_judge.

    python -m dual_system.eval.judge_prompts run qwen4b --res <res>     # needs <res>/demo_labels.jsonl and frames from demo_judge sample
    python -m dual_system.eval.judge_prompts report --res <res>

Styles, each built from the subtask text:
  finished  "Is this subtask finished?" with its finish condition (the old prompt, for reference)
  state     a yes/no question about the visible result
  negated   the opposite question ("Is the bowl still on the table?"), so a model that leans to one word is wrong the other way
  camera    tells the model which camera shows the result and what to look for there
  choice    lettered descriptions of the possible states, one of them the finished one
  describe  one sentence on what is visible, then the answer
  one_camera  only the camera where the result shows: the wrist for a grasp, the scene for everything else
  before_after  the scene at the start of the subtask next to the scene and wrist now, asked to compare
Besides the generated answer, the probability of the finished answer is read from the first answer token (not for describe).
That gives AUROC, which ignores where a model puts its threshold, and a threshold chosen on half of the episodes and scored
on the other half. A model that always says "no" but ranks frames well shows a low accuracy and a high AUROC.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ..knowledge import StaticRetriever
from .demo_judge import SMOLVLM_KWARGS
from .perception import MODELS

STYLES = ("finished", "state", "negated", "camera", "choice", "describe", "one_camera", "before_after")
CAMS = "The first image is the scene camera, the second the wrist camera on the gripper."
NOW = [("now", "agent"), ("now", "wrist")]


def parse_step(step: str) -> dict:
    """Kind, object and target of a subtask command."""
    m = re.match(r"pick up (?:a |the )?(?:other )?(.+?)$", step)
    if m:
        return dict(kind="pick", obj=m.group(1))
    m = re.match(r"place (?:the )?(?:other )?(.+?) (on top of|on|in|to the right of) (.+)$", step)
    if m:
        return dict(kind="place", obj=m.group(1), rel=m.group(2), target=m.group(3))
    m = re.match(r"(open|close) (?:the )?(.+)$", step)
    if m:
        return dict(kind=m.group(1), obj=m.group(2))
    m = re.match(r"turn on (?:the )?(.+)$", step)
    if m:
        return dict(kind="turnon", obj=m.group(1))
    m = re.match(r"push (?:the )?(.+?) to (.+)$", step)
    if m:
        return dict(kind="push", obj=m.group(1), target=m.group(2))
    return dict(kind="other", obj=step)


def _state(p: dict) -> str:
    return {"pick": "Is the {obj} held between the gripper fingers and lifted above the table?",
            "place": "Is the {obj} resting {rel} {target}, with the gripper no longer holding it?",
            "open": "Is the {obj} open?", "close": "Is the {obj} fully closed?", "turnon": "Is the {obj} turned on?",
            "push": "Is the {obj} at {target}?"}.get(p["kind"], 'Is "{obj}" done?').format(**p)


def _negated(p: dict) -> str:
    return {"pick": "Is the {obj} still resting on the table or on what it stood on, not lifted by the gripper?",
            "place": "Is the {obj} still held by the gripper, or not yet {rel} {target}?",
            "open": "Is the {obj} still closed?", "close": "Is the {obj} still open, even a little?",
            "turnon": "Is the {obj} still off?", "push": "Is the {obj} still away from {target}?"}.get(p["kind"], 'Is "{obj}" still to do?').format(**p)


def _camera(p: dict) -> str:
    return {"pick": "Look at the second image, from the wrist camera. When an object is held, it sits between the two gripper "
                    "fingers at the bottom of that image, large and close. Is the {obj} held there, and does the first image show it lifted off the table?",
            "place": "Look at the first image, from the scene camera. Is the {obj} standing {rel} {target}, with the gripper "
                     "open or moved away from it?",
            "open": "Look at the first image, from the scene camera. Is the {obj} pulled out or swung open, with a visible gap?",
            "close": "Look at the first image, from the scene camera. Is the {obj} pushed fully shut, with no gap?",
            "turnon": "Look at the first image, from the scene camera. Does the burner ring on top of the {obj} glow red?",
            "push": "Look at the first image, from the scene camera. Is the {obj} at {target}?"}.get(p["kind"], _state(p)).format(**p)


def _choice(p: dict) -> tuple[str, str]:
    """Lettered states and the letter that means finished. The finished state is never the first letter."""
    options = {"pick": (["the {obj} stands on the table or its support, not touched", "the gripper touches the {obj} but it is not lifted",
                         "the {obj} is held by the gripper and lifted"], "C"),
               "place": (["the {obj} is still held by the gripper", "the {obj} rests {rel} {target} and the gripper has let go",
                          "the {obj} is somewhere else"], "B"),
               "open": (["the {obj} is closed", "the {obj} is open"], "B"),
               "close": (["the {obj} is open", "the {obj} is fully closed"], "B"),
               "turnon": (["the {obj} is off and its burner is dark", "the {obj} is on and its burner glows red"], "B"),
               "push": (["the {obj} is away from {target}", "the {obj} is at {target}"], "B")}
    texts, done = options.get(p["kind"], (["not done", "done"], "B"))
    lines = "\n".join(f"{chr(65 + i)}) {t.format(**p)}" for i, t in enumerate(texts))
    return f"Which describes the scene now?\n{lines}\nAnswer with one letter.", done


def build(step: str, style: str, retriever) -> dict:
    """prompt, answer type (yesno, choice, describe), the answer that means finished, and the images as (time, camera)."""
    spec = _build(step, style, retriever)
    spec.setdefault("images", NOW)
    return spec


def _build(step: str, style: str, retriever) -> dict:
    p = parse_step(step)
    if style == "one_camera":  # the camera where the result shows: the wrist for a grasp, the scene for everything else
        if p["kind"] == "pick":
            return dict(prompt="The image is from the wrist camera on the gripper. A held object sits between the two fingers, "
                               f"large and close.\nIs the {p['obj']} held between the gripper fingers?\nAnswer with one word: yes or no.",
                        type="yesno", done="yes", images=[("now", "wrist")])
        return dict(prompt=f"The image is from the scene camera.\n{_state(p)}\nAnswer with one word: yes or no.",
                    type="yesno", done="yes", images=[("now", "agent")])
    if style == "before_after":
        return dict(prompt="The first image is the scene camera when the robot started this step, the second the scene camera now, "
                           "the third the wrist camera now.\nCompare now with the start. " + _state(p) + "\nAnswer with one word: yes or no.",
                    type="yesno", done="yes", images=[("start", "agent"), ("now", "agent"), ("now", "wrist")])
    if style == "finished":
        return dict(prompt=f'{CAMS}\nSubtask: "{step}".\nIt is finished only when {retriever.phase_definition(step)}\n'
                           "Is this subtask finished? Answer with one word: yes or no.", type="yesno", done="yes")
    if style == "state":
        return dict(prompt=f"{CAMS}\n{_state(p)}\nAnswer with one word: yes or no.", type="yesno", done="yes")
    if style == "negated":
        return dict(prompt=f"{CAMS}\n{_negated(p)}\nAnswer with one word: yes or no.", type="yesno", done="no")
    if style == "camera":
        return dict(prompt=f"{CAMS}\n{_camera(p)}\nAnswer with one word: yes or no.", type="yesno", done="yes")
    if style == "choice":
        text, done = _choice(p)
        return dict(prompt=f"{CAMS}\n{text}", type="choice", done=done)
    if style == "describe":
        return dict(prompt=f"{CAMS}\nIn one sentence, say what the gripper is holding and where the {p['obj']} is. "
                           f"Then, on a new line, answer this question with ANSWER: yes or ANSWER: no.\n{_state(p)}", type="describe", done="yes")
    raise ValueError(style)


def parse_answer(text: str, kind: str) -> str | None:
    if kind == "describe":
        m = re.findall(r"ANSWER:\s*(yes|no)", text, re.I)
        return m[-1].lower() if m else None
    if kind == "choice":
        m = re.match(r"\W*([A-C])\b", text)
        return m.group(1) if m else None
    m = re.match(r"\W*(yes|no)\b", text, re.I)
    return m.group(1).lower() if m else None


def first_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids = set()
    for w in words:
        for form in (w, w.capitalize(), w.upper(), " " + w, " " + w.capitalize()):
            tokens = tokenizer(form, add_special_tokens=False).input_ids
            if tokens:
                ids.add(tokens[0])
    return sorted(ids)


def done_probability(reasoner, prompt: str, images: list, spec: dict) -> float:
    """Probability mass of the finished answer against the other answers, on the first generated token."""
    torch = reasoner.torch
    inputs = reasoner._inputs(prompt, images)
    with torch.inference_mode():
        probs = torch.softmax(reasoner.model(**inputs).logits[0, -1].float(), dim=0)
    tok = reasoner.processor.tokenizer
    if spec["type"] == "choice":
        letters = ["A", "B", "C"]
        mass = {l: float(probs[first_token_ids(tok, [l])].sum()) for l in letters}
        total = sum(mass.values())
        return mass[spec["done"]] / total if total else float("nan")
    yes, no = float(probs[first_token_ids(tok, ["yes"])].sum()), float(probs[first_token_ids(tok, ["no"])].sum())
    p_yes = yes / (yes + no) if yes + no else float("nan")
    return p_yes if spec["done"] == "yes" else 1 - p_yes


def run_model(res: Path, key: str, styles: list[str]) -> None:
    import time

    import torch
    from PIL import Image

    from ..system2.reasoners import QwenReasoner

    labels = [json.loads(line) for line in open(res / "demo_labels.jsonl", encoding="utf-8")]
    path = res / f"prompt_answers_{key}.jsonl"
    finished = {(r["frame"], r["step"], r["style"]) for r in map(json.loads, open(path, encoding="utf-8"))} if path.exists() else set()
    reasoner = QwenReasoner(**MODELS[key], images_kwargs=SMOLVLM_KWARGS if key == "smolvlm" else None)
    retriever = StaticRetriever("phase_atomic")
    out = open(path, "a", encoding="utf-8")
    for row in labels:
        for style in styles:
            if (row["frame"], row["step"], style) in finished:
                continue
            spec = build(row["step"], style, retriever)
            images = [np.asarray(Image.open(res / "frames" / f"{row['frame' if when == 'now' else 'start_frame']}_{cam}.jpg")) for when, cam in spec["images"]]
            t0 = time.perf_counter()
            a = reasoner._generate(spec["prompt"], images, 60 if spec["type"] == "describe" else 3, f"judge_{style}", lambda raw: raw)
            answer = parse_answer(a.raw, spec["type"])
            prob = None if spec["type"] == "describe" else done_probability(reasoner, spec["prompt"], images, spec)
            out.write(json.dumps(dict(model=key, frame=row["frame"], episode=row["episode"], step=row["step"], kind=row["kind"], style=style,
                                      truth=row["done"], answer=None if answer is None else answer == spec["done"], prob=prob, raw=a.raw,
                                      latency_s=round(time.perf_counter() - t0, 3))) + "\n")
            out.flush()
    print(key, "done, peak GPU", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GB", flush=True)


# ---------------------------------------------------------------- scoring
def auroc(scores: list[float], truths: list[bool]) -> float:
    """Probability that a finished frame scores above an unfinished one, ties counting half."""
    pos = [s for s, t in zip(scores, truths) if t]
    neg = [s for s, t in zip(scores, truths) if not t]
    if not pos or not neg:
        return float("nan")
    pos_a, neg_a = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(((pos_a > neg_a).sum() + 0.5 * (pos_a == neg_a).sum()) / (len(pos) * len(neg)))


def balanced(answers: list, truths: list[bool]) -> float:
    """Mean recall on finished and unfinished frames. A missing answer counts as wrong."""
    recalls = []
    for t in (True, False):
        group = [a for a, x in zip(answers, truths) if x is t]
        if not group:
            return float("nan")
        recalls.append(sum(a is not None and bool(a) == t for a in group) / len(group))  # NumPy booleans are not `True`
    return float(np.mean(recalls))


def cross_fit(rows: list[dict]) -> tuple[float, float]:
    """Choose the threshold with the best balanced accuracy on one half of the episodes, apply it to the other, both ways.
    Returns balanced accuracy and false-done rate over all frames. Halves alternate in sorted episode order (a CRC parity can
    put every name in one half, since CRC is linear)."""
    half = {e: i % 2 for i, e in enumerate(sorted({r["episode"] for r in rows}))}
    answers, truths = [], []
    for h in (0, 1):
        fit = [r for r in rows if half[r["episode"]] == h]
        test = [r for r in rows if half[r["episode"]] != h]
        best = max(np.linspace(0.01, 0.99, 99), key=lambda t: balanced([r["prob"] >= t for r in fit], [r["truth"] for r in fit]))
        answers += [bool(r["prob"] >= best) for r in test]
        truths += [r["truth"] for r in test]
    neg = [a for a, t in zip(answers, truths) if not t]
    return balanced(answers, truths), (sum(neg) / len(neg) if neg else float("nan"))


def interval(rows: list[dict], score, n: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    by_episode: dict[str, list] = {}
    for r in rows:
        by_episode.setdefault(r["episode"], []).append(r)
    names = sorted(by_episode)
    draws = [score([r for i in rng.integers(0, len(names), len(names)) for r in by_episode[names[i]]]) for _ in range(n)]
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def report(res: Path) -> str:
    lines = ["model        style      answer acc  says done  AUROC [95%]           threshold acc  false done  s/answer"]
    for f in sorted(res.glob("prompt_answers_*.jsonl")):
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        for style in STYLES:
            part = [r for r in rows if r["style"] == style]
            if not part:
                continue
            acc = balanced([r["answer"] for r in part], [r["truth"] for r in part])
            says = np.mean([r["answer"] is True for r in part])
            lat = np.median([r["latency_s"] for r in part])
            if part[0]["prob"] is None:
                lines.append(f"{part[0]['model']:<12} {style:<10} {acc:>10.2f}  {says:>9.2f}  {'-':<21} {'-':>13}  {'-':>10}  {lat:>8.2f}")
                continue
            scored = [r for r in part if r["prob"] is not None and not np.isnan(r["prob"])]
            auc = auroc([r["prob"] for r in scored], [r["truth"] for r in scored])
            lo, hi = interval(scored, lambda rs: auroc([r["prob"] for r in rs], [r["truth"] for r in rs]))
            fit_acc, fit_false = cross_fit(scored)
            lines.append(f"{part[0]['model']:<12} {style:<10} {acc:>10.2f}  {says:>9.2f}  {auc:.2f} [{lo:.2f}, {hi:.2f}]  {fit_acc:>13.2f}  {fit_false:>10.2f}  {lat:>8.2f}")
        lines.append("")
    text = "\n".join(lines)
    (res / "prompt_report.txt").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "report"])
    p.add_argument("model", nargs="?", choices=list(MODELS))
    p.add_argument("--res", required=True)
    p.add_argument("--styles", nargs="*", default=list(STYLES))
    a = p.parse_args()
    if a.command == "run":
        run_model(Path(a.res), a.model, a.styles)
    else:
        print(report(Path(a.res)))
