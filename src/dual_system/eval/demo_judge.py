"""Can an untrained VLM tell whether a subtask is finished? Scored on LIBERO demo frames, labelled by the subtask boundaries.

    python -m dual_system.eval.demo_judge sample --boundaries <file> --res <res>    # frames and labels, no simulator needed
    python -m dual_system.eval.demo_judge run qwen4b --res <res>                    # one model; finished questions are skipped
    python -m dual_system.eval.demo_judge report --res <res>

The boundaries come from simulator truth on the replayed demos, so every frame inside a subtask is "not finished" and every
frame after its end is "finished". Frames within a few steps of a boundary are left out, because the exact step is uncertain.
Demos contain no failures, so this measures seeing the result, not catching a drop; eval.perception covers rollouts.
Modes: plain = both cameras and the subtask, definition = plus its finish condition, scene = definition with the scene camera
only, state = both cameras and a question about the visible result ("Is the bowl held ... and lifted?") instead of "finished". Scores are balanced accuracy per subtask kind, averaged, with an interval that resamples whole episodes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ..knowledge import StaticRetriever
from .perception import MODELS, balanced_accuracy, episode_interval, parse_yes_no

MODES = ("plain", "definition", "scene", "state")
MARGIN = 4
BEFORE, AFTER = (0.3, 0.7), (MARGIN + 2, MARGIN + 12)
SMOLVLM_KWARGS = {"do_image_splitting": False}


def questions_for(row: dict, rng: np.random.Generator) -> list[dict]:
    """Per subtask: two frames well inside it (not finished), and up to two after its end (finished)."""
    out = []
    for p in row["phases"]:
        span = p["end"] - p["start"]
        if span < 2 * MARGIN + 2:
            continue
        for f in BEFORE:
            t = min(p["start"] + int(f * span), p["end"] - 1 - MARGIN)
            out.append(dict(t=t, step=p["command"], kind=p["kind"], done=False, start=p["start"]))
        for d in AFTER:
            t = p["end"] - 1 + d + int(rng.integers(0, 3))
            if t < row["length"]:
                out.append(dict(t=t, step=p["command"], kind=p["kind"], done=True, start=p["start"]))
    return out


def state_question(step: str) -> str:
    """The subtask's result as a question about what is visible, the form the rollout benchmark asked in."""
    m = re.match(r"pick up (?:a |the )?(.+?)$", step)
    if m:
        return f"Is the {re.sub(r'^(other )', '', m.group(1))} held between the gripper fingers and lifted above the table?"
    m = re.match(r"place (?:the )?(.+?) (on top of|on|in|to the right of) (.+)$", step)
    if m:
        return f"Is the {m.group(1)} resting {m.group(2)} {m.group(3)}, with the gripper no longer holding it?"
    m = re.match(r"(open|close) (?:the )?(.+)$", step)
    if m:
        return f"Is the {m.group(2)} {'open' if m.group(1) == 'open' else 'fully closed'}?"
    m = re.match(r"turn on (?:the )?(.+)$", step)
    if m:
        return f"Is the {m.group(1)} turned on?"
    m = re.match(r"push (?:the )?(.+?) to (.+)$", step)
    if m:
        return f"Is the {m.group(1)} at {m.group(2)}?"
    return f'Is "{step}" finished?'


def prompt_for(step: str, mode: str, retriever) -> str:
    if mode == "state":
        return ("The first image is the scene camera, the second the wrist camera on the gripper.\n"
                f"{state_question(step)}\nAnswer with one word: yes or no.")
    cams = "The image shows a robot arm from the scene camera." if mode == "scene" else \
        "The first image is the scene camera, the second the wrist camera on the gripper."
    condition = "" if mode == "plain" else f"\nIt is finished only when {retriever.phase_definition(step)}"
    return f'{cams}\nSubtask: "{step}".{condition}\nIs this subtask finished? Answer with one word: yes or no.'


def sample(boundaries: Path, res: Path, per_task: int = 1, seed: int = 0) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    rng = np.random.default_rng(seed)
    rows = [json.loads(line) for line in open(boundaries, encoding="utf-8")]
    by_task: dict[str, list] = {}
    for r in rows:
        if r["phases"]:
            by_task.setdefault(r["task"], []).append(r)
    ds = LeRobotDataset("lerobot/libero", video_backend="torchcodec")
    (res / "frames").mkdir(parents=True, exist_ok=True)
    out = open(res / "demo_labels.jsonl", "w", encoding="utf-8")
    for task, task_rows in sorted(by_task.items()):
        for row in [task_rows[i] for i in rng.choice(len(task_rows), min(per_task, len(task_rows)), replace=False)]:
            episode = f"ep{row['episode']}"
            for q in questions_for(row, rng):
                frame = f"{episode}__t{q['t']:04d}"
                for t in (q["t"], q["start"]):  # the subtask's first frame too, for before/after prompts
                    name = f"{episode}__t{t:04d}"
                    if not (res / "frames" / f"{name}_agent.jpg").exists():
                        item = ds[row["first_index"] + t]
                        for key, cam in (("observation.images.image", "agent"), ("observation.images.image2", "wrist")):
                            img = (item[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            Image.fromarray(img).save(res / "frames" / f"{name}_{cam}.jpg", quality=92)
                out.write(json.dumps(dict(frame=frame, episode=episode, task=task, start_frame=f"{episode}__t{q['start']:04d}", **q)) + "\n")
    out.close()
    labels = [json.loads(line) for line in open(res / "demo_labels.jsonl", encoding="utf-8")]
    print(len(labels), "questions,", sum(r["done"] for r in labels), "finished,", len({r["episode"] for r in labels}), "episodes", flush=True)


def run_model(res: Path, key: str) -> None:
    import time

    import torch
    from PIL import Image

    from ..system2.reasoners import QwenReasoner

    labels = [json.loads(line) for line in open(res / "demo_labels.jsonl", encoding="utf-8")]
    path = res / f"demo_answers_{key}.jsonl"
    finished = {(r["frame"], r["step"], r["mode"]) for r in map(json.loads, open(path, encoding="utf-8"))} if path.exists() else set()
    t0 = time.time()
    reasoner = QwenReasoner(**MODELS[key], images_kwargs=SMOLVLM_KWARGS if key == "smolvlm" else None)
    load_s = time.time() - t0
    retriever = StaticRetriever("phase_atomic")
    out = open(path, "a", encoding="utf-8")
    for row in labels:
        agent = np.asarray(Image.open(res / "frames" / f"{row['frame']}_agent.jpg"))
        wrist = np.asarray(Image.open(res / "frames" / f"{row['frame']}_wrist.jpg"))
        for mode in MODES:
            if (row["frame"], row["step"], mode) in finished:
                continue
            a = reasoner._generate(prompt_for(row["step"], mode, retriever), [agent] if mode == "scene" else [agent, wrist], 4, f"demo_{mode}", parse_yes_no)
            out.write(json.dumps(dict(model=key, frame=row["frame"], episode=row["episode"], step=row["step"], question=row["kind"], mode=mode,
                                      truth=row["done"], answer=a.parsed["yes"] if a.parsed else None, raw=a.raw, latency_s=round(a.latency_s, 3))) + "\n")
            out.flush()
    (res / f"demo_model_{key}.json").write_text(json.dumps(dict(model=MODELS[key]["model"], quantized=MODELS[key].get("quantize", False), load_s=round(load_s),
                                                                peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))))


def false_done(rows: list[dict]) -> float:
    neg = [r for r in rows if not r["truth"]]
    return sum(r["answer"] is True for r in neg) / len(neg) if neg else float("nan")


def report(res: Path) -> str:
    from .perception import macro_score

    lines = []
    for f in sorted(res.glob("demo_answers_*.jsonl")):
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        key = rows[0]["model"]
        meta = res / f"demo_model_{key}.json"
        info = json.loads(meta.read_text()) if meta.exists() else {}
        lines.append(f"\n== {key}  {info.get('model', '')}  quantized={info.get('quantized', '?')}  peak GPU {info.get('peak_gpu_gb', '?')} GB  "
                     f"median {np.median([r['latency_s'] for r in rows]):.2f} s  {len(rows)} answers  unusable {sum(r['answer'] is None for r in rows)}")
        for mode in MODES:
            part = [r for r in rows if r["mode"] == mode]
            if not part:
                continue
            lo, hi = episode_interval(part)
            kinds = "  ".join(f"{k} {balanced_accuracy([r for r in part if r['question'] == k]):.2f}" for k in ("pick", "place", "act"))
            lines.append(f"{mode:<11} score {macro_score(part):.2f} [{lo:.2f}, {hi:.2f}]  false done {false_done(part):.2f}  "
                         f"says yes {np.mean([r['answer'] is True for r in part]):.2f}  {kinds}")
    text = "\n".join(lines).strip() + "\n"
    (res / "demo_judge_report.txt").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["sample", "run", "report"])
    p.add_argument("model", nargs="?", choices=list(MODELS))
    p.add_argument("--res", required=True)
    p.add_argument("--boundaries")
    p.add_argument("--per-task", type=int, default=1)
    a = p.parse_args()
    if a.command == "sample":
        sample(Path(a.boundaries), Path(a.res), a.per_task)
    elif a.command == "run":
        run_model(Path(a.res), a.model)
    else:
        print(report(Path(a.res)))
