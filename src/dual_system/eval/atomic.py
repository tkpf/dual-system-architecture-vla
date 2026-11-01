"""How well does a System 1 carry out subtasks? No VLM: the next command is issued when the simulator says the phase is done.

    python -m dual_system.eval.atomic --path <checkpoint> --name stock --res <res> --split select [--seeds 1] [--states A]
    python -m dual_system.eval.atomic --report <res>

Each phase can succeed on its own, so a later subtask is not judged on an earlier failure: a phase that runs out of its step
budget ends the episode, and the phases after it count as not reached. On the first init state, each scene also runs with
gibberish in place of every command, because a model that does not follow language can pass on the scene alone, and a stop
run that keeps the first command after its phase is done, to see whether the policy waits. --split select is for choosing a
checkpoint, --split test for the final numbers, once.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from ..finetune.dataset import load_splits
from ..finetune.segment import load_subtasks, tasks_to_suites
from .command_probe import GIBBERISH, ROBOT_STATES

PHASE_BUDGET = 150
OVERRUN_WINDOW = 60
# Scenes in the ablation, then scenes never trained on.
SEEN = ["put the bowl on the stove", "put the bowl on top of the cabinet", "turn on the stove and put the moka pot on it"]


def plan_for(task: str, subtasks: dict) -> list[dict]:
    """Phases in list order, each with its kind, command and the text that finds its goal predicate."""
    phases = []
    for group in subtasks[task]:
        if "act" in group:
            phases.append(dict(kind="act", command=group["act"], goal=group["goal"]))
        else:
            phases.append(dict(kind="pick", command=group["pick"], goal=group["goal"]))
            phases.append(dict(kind="place", command=group["place"], goal=group["goal"]))
    return phases


def phase_done(phase: dict, goals: dict[str, bool], world_gripped, rises: dict[str, float]) -> bool:
    """Simulator truth for one phase. Pick: both fingers on the object and lifted. Others: the goal predicate holds."""
    key = next(k for k in goals if phase["goal"] in k)
    if phase["kind"] != "pick":
        return goals[key]
    name = key.split()[1]
    return rises.get(name, 0.0) > 0.02 and world_gripped(name)


def wrong_lifts(rises: dict[str, float], target: str | None) -> list[str]:
    return [n for n, r in rises.items() if r > 0.03 and n != target and not n.startswith(("flat_stove", "wooden_cabinet"))]


def ran_ahead(goals_before: dict[str, bool], goals_after: dict[str, bool]) -> bool:
    """A goal predicate became true that the finished phase did not ask for."""
    return any(v and not goals_before[k] for k, v in goals_after.items())


def run_scene(world, system1, state, seed: int, phases: list[dict], mode: str = "real") -> dict:
    """mode real: oracle switching. gibberish: the same, with gibberish as every command. stop: stay on the first command for
    OVERRUN_WINDOW steps after its phase is done, then end, to see whether the policy waits or runs ahead into the next one."""
    from ..rollout import seed_everything
    from .command_probe import object_heights

    seed_everything(seed)
    system1.reset()
    obs = world.reset(state, seed)
    start_z = object_heights(world)
    rows, overrun = [], None
    for phase in phases[:1] if mode == "stop" else phases:
        system1.set_command(GIBBERISH if mode == "gibberish" else phase["command"])
        start, done, picked_wrong = obs.t, False, set()
        while obs.t - start < PHASE_BUDGET and not done:
            obs, _ = world.step(system1.act(obs))
            goals = world.goal_progress()
            rises = {k: v - start_z[k] for k, v in object_heights(world).items()}
            done = phase_done(phase, goals, world.gripped, rises)
            key = next(k for k in goals if phase["goal"] in k)
            picked_wrong |= set(wrong_lifts(rises, key.split()[1] if len(key.split()) > 2 else None))
        rows.append(dict(command=phase["command"], kind=phase["kind"], done=bool(done), steps=obs.t - start, wrong_lift=sorted(picked_wrong)))
        if not done:
            break
    if mode == "stop" and rows[0]["done"]:
        before = world.goal_progress()
        for _ in range(OVERRUN_WINDOW):
            obs, _ = world.step(system1.act(obs))
        overrun = ran_ahead(before, world.goal_progress())
    return dict(phases=rows, reached=sum(r["done"] for r in rows), total=len(phases), goal=all(world.goal_progress().values()), end_step=obs.t, overrun=overrun)


def hash_text(text: str) -> int:
    return zlib.crc32(text.encode())


def main(a) -> None:
    from ..env import LiberoWorld
    from ..robot_states import RobotState
    from ..system1 import build_system1

    res = Path(a.res)
    res.mkdir(parents=True, exist_ok=True)
    splits, subtasks, suites = load_splits(), load_subtasks(), tasks_to_suites()
    path = res / f"atomic_{a.name}.jsonl"
    key = lambda r: (r["task"], r["init_state"], r["robot_state"], r["seed"], r["control"])
    finished = {key(json.loads(line)) for line in open(path, encoding="utf-8")} if path.exists() else set()
    out = open(path, "a", encoding="utf-8")
    system1 = build_system1(dict(type="smolvla", path=a.path, n_action_steps=10))
    init_states = splits["init_states"][a.split][:a.init_states]
    for group, tasks in (("seen", SEEN), ("unseen", splits["unseen_tasks"])):
        for task in tasks:
            suite, task_id, _ = suites[task]
            phases = plan_for(task, subtasks)
            for init in init_states:
                for state_name in a.states:
                    for s in range(a.seeds):
                        # the same seed for every checkpoint and control, so results pair up
                        seed = (hash_text(task) + init * 100 + s) % 100000
                        for control in ("real", "gibberish", "stop") if init == init_states[0] and not a.no_control else ("real",):
                            row = dict(model=a.name, group=group, task=task, init_state=init, robot_state=state_name, seed=seed, control=control)
                            if key(row) in finished:
                                continue
                            world = LiberoWorld(suite, task_id, max_steps=PHASE_BUDGET * len(phases) + OVERRUN_WINDOW + 1)
                            state = RobotState(state_name, init_state_id=init, joint_offset_deg=ROBOT_STATES[state_name])
                            try:
                                row.update(run_scene(world, system1, state, seed, phases, control))
                            finally:
                                world.close()
                            out.write(json.dumps(row) + "\n")
                            out.flush()
                            print(a.name, group, task[:40], init, state_name, control, f"{row['reached']}/{row['total']}", "goal", row["goal"], "ran ahead", row["overrun"], flush=True)


def summarise(rows: list[dict]) -> dict:
    """Per-phase-kind success, full-goal rate and wrong-object lifts."""
    phases = [p for r in rows for p in r["phases"]]
    by_kind = {}
    for kind in ("pick", "place", "act"):
        part = [p for p in phases if p["kind"] == kind]
        by_kind[kind] = f"{sum(p['done'] for p in part)}/{len(part)}"
    return dict(episodes=len(rows), goal=f"{sum(r['goal'] for r in rows)}/{len(rows)}", phases_reached=f"{sum(r['reached'] for r in rows)}/{sum(r['total'] for r in rows)}",
                wrong_object=sum(any(p["wrong_lift"] for p in r["phases"]) for r in rows), median_steps=int(np.median([p["steps"] for p in phases if p["done"]])) if any(p["done"] for p in phases) else None, **by_kind)


def report(res: Path) -> str:
    lines = []
    for f in sorted(res.glob("atomic_*.jsonl")):
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        lines.append(f"== {f.stem}")
        for group in ("seen", "unseen"):
            for mode in ("real", "gibberish"):
                part = [r for r in rows if r["group"] == group and r["control"] == mode]
                if part:
                    lines.append(f"{group} {mode:<12} {summarise(part)}")
        stop = [r for r in rows if r["control"] == "stop" and r["overrun"] is not None]
        lines.append(f"ran ahead after the first phase: {sum(r['overrun'] for r in stop)}/{len(stop)}")
        for state in sorted({r["robot_state"] for r in rows}):
            lines.append(f"state {state:<11} {summarise([r for r in rows if r['robot_state'] == state and r['control'] == 'real'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path")
    p.add_argument("--name")
    p.add_argument("--res")
    p.add_argument("--report")
    p.add_argument("--split", choices=["select", "test"], default="select")
    p.add_argument("--states", nargs="*", default=["A"], choices=list(ROBOT_STATES))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--init-states", type=int, default=3, help="how many of the split's init states")
    p.add_argument("--no-control", action="store_true")
    a = p.parse_args()
    if a.report:
        print(report(Path(a.report)))
    else:
        main(a)
