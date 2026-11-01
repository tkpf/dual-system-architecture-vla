"""Can a VLM see what System 2 has to judge? Yes/no questions on simulator frames, scored against simulator truth.

    python -m dual_system.eval.perception collect --res <res>      # pi0.5 rollouts, frames and labels; needs the simulator
    python -m dual_system.eval.perception run qwen2b --res <res>   # one model; finished questions are skipped on restart
    python -m dual_system.eval.perception report --res <res>       # tables, also written to <res>/perception_report.txt

Three prompt modes: both = scene and wrist camera, scene = scene camera only, definition = both cameras plus the finish
condition from the phase_atomic profile. Scores are balanced accuracy, so a model that always says the same word gets 0.5.
Frames of one episode are correlated, so the interval resamples whole episodes.
"""
from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np

from ..knowledge import StaticRetriever

MODELS = {
    "qwen2b": dict(model="Qwen/Qwen3-VL-2B-Instruct"),
    "smolvlm": dict(model="HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    "qwen4b": dict(model="Qwen/Qwen3-VL-4B-Instruct"),
    "qwen8b": dict(model="Qwen/Qwen3-VL-8B-Instruct", quantize=True),  # about 18 GB in fp16, more than a T4 has
}
MODES = ("both", "scene", "definition")
BOWL, MOKA = "akita_black_bowl_1", "moka_pot_1"
# name: (question, phase whose finish condition applies, label kind, label key)
QUESTIONS = {
    "stove_on": ("Is the stove turned on?", "turn on the stove", "goal", "turnon flat_stove_1"),
    "bowl_held": ("Is the bowl lifted above the table and held by the gripper?", "pick up the bowl", "held", BOWL),
    "moka_held": ("Is the moka pot lifted above the table and held by the gripper?", "pick up the moka pot", "held", MOKA),
    "bowl_on_stove": ("Is the bowl resting on the stove and released by the gripper?", "place the bowl on the stove", "goal",
                      f"on {BOWL} flat_stove_1_cook_region"),
    "bowl_on_cabinet": ("Is the bowl resting on top of the cabinet and released by the gripper?", "place the bowl on top of the cabinet",
                        "goal", f"on {BOWL} wooden_cabinet_1_top_side"),
    "moka_on_stove": ("Is the moka pot resting on the stove and released by the gripper?", "place the moka pot on the stove", "goal",
                      f"on {MOKA} flat_stove_1_cook_region"),
}
SCENES = {
    ("libero_goal", 1): ["bowl_held", "bowl_on_stove"],
    ("libero_goal", 4): ["bowl_held", "bowl_on_cabinet"],
    ("libero_10", 2): ["stove_on", "moka_held", "moka_on_stove"],
}
EVERY_STEPS, MAX_STEPS, LIFT_M, GRIP_MAX_M, NEAR_M = 20, 300, 0.03, 0.07, 0.15


def is_held(rise: float, gripper_width: float, distance: float) -> bool:
    """Above its start, gripper not fully open and the gripper at the object."""
    return bool(rise > LIFT_M and gripper_width < GRIP_MAX_M and distance < NEAR_M)


def parse_yes_no(text: str) -> dict | None:
    m = re.match(r"\W*(yes|no)\b", text, re.I)
    return {"yes": m.group(1).lower() == "yes"} if m else None


def prompt_for(question: str, phase: str, mode: str, retriever) -> str:
    context = {"both": "The first image is the scene camera, the second the wrist camera on the gripper.",
               "scene": "The image shows a robot arm from the scene camera.",
               "definition": "The first image is the scene camera, the second the wrist camera on the gripper. "
                             f"The step \"{phase}\" is finished only when {retriever.phase_definition(phase)}"}[mode]
    return f"{context}\n{question}\nAnswer with one word: yes or no."


# ---------------------------------------------------------------- collect
def collect(res: Path, states: list[str], seeds: int, path: str) -> None:
    from PIL import Image

    from ..env import LiberoWorld
    from ..robot_states import RobotState
    from ..rollout import seed_everything
    from ..system1 import build_system1
    from .command_probe import ROBOT_STATES, object_heights

    (res / "frames").mkdir(parents=True, exist_ok=True)
    labels = res / "labels.jsonl"
    finished = {json.loads(line)["episode"] for line in open(labels, encoding="utf-8")} if labels.exists() else set()
    out = open(labels, "a", encoding="utf-8")
    system1 = build_system1(dict(type="pi05", path=path, n_action_steps=10))
    for (suite, task) in SCENES:
        for state_name in states:
            for s in range(seeds):
                episode = f"{suite}__t{task}__{state_name}__s{s}"
                if episode in finished:
                    continue
                seed = zlib.crc32(f"{suite}-{task}-{s}".encode()) % 100000
                seed_everything(seed)
                world = LiberoWorld(suite, task, max_steps=MAX_STEPS)
                try:
                    system1.reset()
                    system1.set_command(world.instruction)
                    obs = world.reset(RobotState(state_name, joint_offset_deg=ROBOT_STATES[state_name]), seed)
                    start_z = object_heights(world)
                    inner = world.env._env.env
                    rows = []
                    while True:
                        if obs.t % EVERY_STEPS == 0:
                            frame = f"{episode}__t{obs.t:04d}"
                            Image.fromarray(obs.agent).save(res / "frames" / f"{frame}_agent.png")
                            Image.fromarray(obs.wrist).save(res / "frames" / f"{frame}_wrist.png")
                            held = {}
                            for name, z in object_heights(world).items():
                                gap = inner.sim.data.body_xpos[inner.obj_body_id[name]] - obs.eef_pos
                                held[name] = is_held(z - start_z[name], obs.gripper_width, float(np.linalg.norm(gap)))
                            rows.append(dict(frame=frame, episode=episode, suite=suite, task=task, robot_state=state_name, seed=seed,
                                             t=obs.t, goal=world.goal_progress(), held=held, gripper_width=round(obs.gripper_width, 4)))
                        if obs.t >= MAX_STEPS:
                            break
                        obs, _ = world.step(system1.act(obs))
                finally:
                    world.close()
                for row in rows:
                    out.write(json.dumps(row) + "\n")
                out.flush()
                print(episode, len(rows), "frames", flush=True)


# ---------------------------------------------------------------- run
def run_model(res: Path, key: str) -> None:
    import time

    import torch
    from PIL import Image

    from ..system2.reasoners import QwenReasoner

    labels = [json.loads(line) for line in open(res / "labels.jsonl", encoding="utf-8")]
    path = res / f"answers_{key}.jsonl"
    finished = set()
    if path.exists():
        finished = {(r["frame"], r["question"], r["mode"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    t0 = time.time()
    reasoner = QwenReasoner(**MODELS[key])
    load_s = time.time() - t0
    retriever = StaticRetriever("phase_atomic")
    out = open(path, "a", encoding="utf-8")
    for row in labels:
        agent = np.asarray(Image.open(res / "frames" / f"{row['frame']}_agent.png"))
        wrist = np.asarray(Image.open(res / "frames" / f"{row['frame']}_wrist.png"))
        for name in SCENES[(row["suite"], row["task"])]:
            question, phase, kind, label_key = QUESTIONS[name]
            for mode in MODES:
                if (row["frame"], name, mode) in finished:
                    continue
                images = [agent] if mode == "scene" else [agent, wrist]
                a = reasoner._generate(prompt_for(question, phase, mode, retriever), images, 4, f"perception_{mode}", parse_yes_no)
                out.write(json.dumps(dict(model=key, frame=row["frame"], episode=row["episode"], question=name, mode=mode, truth=bool(row[kind][label_key]),
                                          answer=a.parsed["yes"] if a.parsed else None, raw=a.raw, latency_s=round(a.latency_s, 3))) + "\n")
                out.flush()
    (res / f"model_{key}.json").write_text(json.dumps(dict(model=MODELS[key]["model"], quantized=MODELS[key].get("quantize", False), load_s=round(load_s),
                                                           peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))))


# ---------------------------------------------------------------- report
def balanced_accuracy(rows: list[dict]) -> float:
    """Mean of the recall on true and on false frames. An unusable answer counts as wrong. NaN if a class is missing."""
    recalls = []
    for truth in (True, False):
        group = [r for r in rows if r["truth"] is truth]
        if not group:
            return float("nan")
        recalls.append(sum(r["answer"] is truth for r in group) / len(group))
    return float(np.mean(recalls))


def macro_score(rows: list[dict]) -> float:
    """Each question counts the same, whatever its number of frames."""
    scores = [balanced_accuracy([r for r in rows if r["question"] == q]) for q in sorted({r["question"] for r in rows})]
    return float(np.nanmean(scores))


def episode_interval(rows: list[dict], n: int = 500, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    by_episode: dict[str, list] = {}
    for r in rows:
        by_episode.setdefault(r["episode"], []).append(r)
    names = sorted(by_episode)
    draws = []
    for _ in range(n):
        sample = [r for i in rng.integers(0, len(names), len(names)) for r in by_episode[names[i]]]
        draws.append(macro_score(sample))
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def report(res: Path) -> str:
    rows = [json.loads(line) for f in sorted(res.glob("answers_*.jsonl")) for line in open(f, encoding="utf-8")]
    lines = []
    for key in MODELS:
        meta = res / f"model_{key}.json"
        model_rows = [r for r in rows if r["model"] == key]
        if not model_rows:
            continue
        info = json.loads(meta.read_text()) if meta.exists() else {}
        lat = np.median([r["latency_s"] for r in model_rows])
        lines.append(f"\n== {key}  {info.get('model', '')}  quantized={info.get('quantized', '?')}  peak GPU {info.get('peak_gpu_gb', '?')} GB  "
                     f"median {lat:.2f} s per question  {len(model_rows)} answers  unusable {sum(r['answer'] is None for r in model_rows)}")
        for mode in MODES:
            part = [r for r in model_rows if r["mode"] == mode]
            if not part:
                continue
            lo, hi = episode_interval(part)
            yes = np.mean([r["answer"] is True for r in part])
            lines.append(f"{mode:<11} score {macro_score(part):.2f} [{lo:.2f}, {hi:.2f}]  says yes {yes:.2f}")
            for q in sorted({r["question"] for r in part}):
                sub = [r for r in part if r["question"] == q]
                pos = [r for r in sub if r["truth"]]
                neg = [r for r in sub if not r["truth"]]
                tpr = sum(r["answer"] is True for r in pos) / len(pos) if pos else float("nan")
                tnr = sum(r["answer"] is False for r in neg) / len(neg) if neg else float("nan")
                lines.append(f"    {q:<16} {balanced_accuracy(sub):.2f}  true frames {len(pos):>3} found {tpr:.2f}  false frames {len(neg):>3} rejected {tnr:.2f}")
    text = "\n".join(lines).strip() + "\n"
    (res / "perception_report.txt").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["collect", "run", "report"])
    p.add_argument("model", nargs="?", choices=list(MODELS))
    p.add_argument("--res", required=True)
    p.add_argument("--states", nargs="*", default=["A", "C"])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--path", default="lerobot/pi05_libero_finetuned_v044")
    a = p.parse_args()
    if a.command == "collect":
        collect(Path(a.res), a.states, a.seeds, a.path)
    elif a.command == "run":
        run_model(Path(a.res), a.model)
    else:
        print(report(Path(a.res)))
