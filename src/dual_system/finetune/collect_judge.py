"""Frames for the subtask judge, from rollouts of the finetuned System 1, labelled by the simulator.

    python -m dual_system.finetune.collect_judge --path <checkpoint> --res <res>     # finished episodes are skipped on restart

Demos show no failures, so the frames come from the model the judge will watch. Commands switch on simulator truth, after a
short dwell so that finished states get frames too. Noise runs open the gripper at random moments and gibberish runs fail on
their own, which gives the hard negatives: dropped objects, empty grasps, a gripper near but not holding. Splits are by episode
and scene, never by frame, because frames of one episode are nearly the same.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..eval.atomic import PHASE_BUDGET, SEEN, hash_text, phase_done, plan_for
from ..eval.command_probe import GIBBERISH, ROBOT_STATES

EVERY_STEPS, DWELL_STEPS, BURST_CHANCE, BURST_STEPS = 10, 30, 0.02, 5
TRAIN_TASKS = SEEN + [
    "put the wine bottle on the rack",
    "put both moka pots on the stove",
    "pick up the alphabet soup and place it in the basket",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
]


def episodes(unseen_tasks: list[str]) -> list[dict]:
    """About 100 episodes. The test split holds new init states of the ablation scenes and scenes the judge never trains on."""
    out = []
    add = lambda split, task, init, mode, state="A": out.append(dict(split=split, task=task, init_state=init, mode=mode, robot_state=state))
    for task in TRAIN_TASKS:
        for init in range(1, 9):
            add("train", task, init, ("real", "noise", "real", "gibberish")[init % 4], "AC"[init % 2])
        add("val", task, 9, "real")
        add("val", task, 9, "noise", "C")
    for task in SEEN + unseen_tasks:
        for init, mode in ((10, "real"), (11, "noise"), (12, "gibberish")):
            add("test", task, init, mode)
    return out


def episode_name(e: dict) -> str:
    return f"{hash_text(e['task']):08x}__i{e['init_state']}__{e['robot_state']}__{e['mode']}"


def truth(phase: dict, goals: dict[str, bool], gripped, rises: dict[str, float]) -> bool:
    """A placed object also has to be let go, as the finish condition says. The goal predicate alone passes while still held."""
    if not phase_done(phase, goals, gripped, rises):
        return False
    if phase["kind"] != "place":
        return True
    return not gripped(next(k for k in goals if phase["goal"] in k).split()[1])


def frame_rows(phases: list[dict], idx: int, flags: list[bool]) -> list[dict]:
    """The active phase, plus the next one as a negative that shares the scene: what the judge must not call finished too early."""
    return [dict(step=phases[i]["command"], kind=phases[i]["kind"], done=flags[i], active=i == idx) for i in range(idx, min(idx + 2, len(phases)))]


def perturb(action: np.ndarray, burst: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Random bursts with the gripper forced open and the arm pushed, to cause drops. Gripper action: above 0 closes."""
    if burst == 0 and rng.random() < BURST_CHANCE:
        burst = BURST_STEPS
    if burst == 0:
        return action, 0
    out = np.array(action, dtype=float)
    out[:6] += rng.normal(0, 0.3, 6)
    out[6] = -1.0
    return out, burst - 1


def run_episode(world, system1, state, seed: int, phases: list[dict], mode: str, save) -> tuple[list[dict], int]:
    """save(obs) stores the frame and returns its name. Returns the label rows and how many phases finished."""
    from ..rollout import seed_everything
    from ..eval.command_probe import object_heights

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    system1.reset()
    obs = world.reset(state, seed)
    start_z = object_heights(world)
    rows, burst, reached = [], 0, 0
    for idx, phase in enumerate(phases):
        system1.set_command(GIBBERISH if mode == "gibberish" else phase["command"])
        start, dwell = obs.t, 0
        while obs.t - start < PHASE_BUDGET + DWELL_STEPS and dwell < DWELL_STEPS:
            goals = world.goal_progress()
            rises = {k: v - start_z[k] for k, v in object_heights(world).items()}
            flags = [truth(p, goals, world.gripped, rises) for p in phases]
            if obs.t % EVERY_STEPS == 0:
                frame = save(obs)
                for r in frame_rows(phases, idx, flags):
                    rows.append(dict(frame=frame, t=obs.t, gripper_width=round(obs.gripper_width, 4), **r))
            dwell = dwell + 1 if flags[idx] else 0
            if dwell == 0 and obs.t - start >= PHASE_BUDGET:
                break
            action = system1.act(obs)
            if mode == "noise":
                action, burst = perturb(action, burst, rng)
            obs, _ = world.step(action)
        if dwell < DWELL_STEPS:
            break
        reached += 1
    return rows, reached


def main(a) -> None:
    from PIL import Image

    from ..env import LiberoWorld
    from ..robot_states import RobotState
    from ..system1 import build_system1
    from .dataset import load_splits
    from .segment import load_subtasks, tasks_to_suites

    res = Path(a.res)
    (res / "frames").mkdir(parents=True, exist_ok=True)
    labels = res / "judge_labels.jsonl"
    finished = {json.loads(line)["episode"] for line in open(labels, encoding="utf-8")} if labels.exists() else set()
    out = open(labels, "a", encoding="utf-8")
    subtasks, suites = load_subtasks(), tasks_to_suites()
    system1 = build_system1(dict(type="smolvla", path=a.path, n_action_steps=10))
    for e in episodes(load_splits()["unseen_tasks"]):
        name = episode_name(e)
        if name in finished:
            continue
        suite, task_id, _ = suites[e["task"]]
        phases = plan_for(e["task"], subtasks)
        seed = (hash_text(e["task"]) + e["init_state"] * 100) % 100000

        def save(obs):
            frame = f"{name}__t{obs.t:04d}"
            Image.fromarray(obs.agent).save(res / "frames" / f"{frame}_agent.jpg", quality=90)
            Image.fromarray(obs.wrist).save(res / "frames" / f"{frame}_wrist.jpg", quality=90)
            return frame

        world = LiberoWorld(suite, task_id, max_steps=(PHASE_BUDGET + DWELL_STEPS) * len(phases) + 1)
        state = RobotState(e["robot_state"], init_state_id=e["init_state"], joint_offset_deg=ROBOT_STATES[e["robot_state"]])
        try:
            rows, reached = run_episode(world, system1, state, seed, phases, e["mode"], save)
        finally:
            world.close()
        for r in rows:
            out.write(json.dumps(dict(episode=name, split=e["split"], task=e["task"], mode=e["mode"], **r)) + "\n")
        out.flush()
        print(name, e["split"], e["task"][:40], e["mode"], f"phases {reached}/{len(phases)}", "rows", len(rows), "done", sum(r["done"] for r in rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--res", required=True)
    main(p.parse_args())
