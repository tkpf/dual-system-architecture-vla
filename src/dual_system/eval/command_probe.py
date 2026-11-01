"""Does System 1 follow a language command? No System 2: fixed commands, compared with a gibberish control.

    python -m dual_system.eval.command_probe --res /content/probe [--type pi05 --path lerobot/pi05_libero_finetuned_v044]
        [--seeds 2] [--states A] [--kinds task switch schedule]

Three kinds of case. Task commands run from the start and report goal predicates and lifted objects. Switch commands first
run the instruction for pre_steps, then change to the command, and report what changes after the switch. Schedule cases
give atomic commands at fixed steps with no VLM, next to the whole instruction and gibberish for the same number of steps.
Finished runs are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

GIBBERISH = "qxv lorp zindle wug flarn"
LIFT_M = 0.03

TASK_CASES = [
    ("libero_goal", 1, ["put the bowl on the stove", "pick up the bowl", GIBBERISH]),
    ("libero_10", 2, ["turn on the stove and put the moka pot on it", "turn on the stove", "put the moka pot on the stove",
                      "pick up the moka pot", GIBBERISH]),
    ("libero_10", 0, ["put the alphabet soup in the basket", "put the tomato sauce in the basket", "pick up the tomato sauce",
                      GIBBERISH]),
]
SWITCH_CASES = [
    ("libero_goal", 1, ["move back to the home position", "open the gripper", "put the bowl on the stove", GIBBERISH]),
]
# (suite, task, atomic commands, steps per command, whole instruction)
SCHEDULE_CASES = [
    ("libero_goal", 1, ["pick up the bowl", "place the bowl on the stove"], 120, "put the bowl on the stove"),
    ("libero_goal", 4, ["pick up the bowl", "place the bowl on top of the cabinet"], 120, "put the bowl on top of the cabinet"),
    ("libero_10", 2, ["turn on the stove", "pick up the moka pot", "place the moka pot on the stove"], 130,
     "turn on the stove and put the moka pot on it"),
]
# Robot starts, as in free_atomic_test. C and B move the gripper away from the stored start; each run records the shift.
ROBOT_STATES = {
    "A": [0, 0, 0, 0, 0, 0, 0],
    "C": [12, -10, 0, -12, 0, 12, 0],
    "B": [25, -20, 0, -25, 0, 25, 0],
}


def object_heights(world) -> dict[str, float]:
    inner = world.env._env.env
    return {name: float(inner.sim.data.body_xpos[i][2]) for name, i in inner.obj_body_id.items()}


def run_case(world, system1, state, seed: int, commands: list[tuple[int, str]], max_steps: int) -> dict:
    """commands: (step, text), in time order. Returns per-segment measurements."""
    from ..rollout import seed_everything

    seed_everything(seed)
    system1.reset()
    obs = world.reset(state, seed)
    start_pos, start_z = obs.eef_pos.copy(), object_heights(world)
    first_true: dict[str, int] = {}
    max_rise = {k: 0.0 for k in start_z}
    trace, segments = [], []
    switch_at = {t: c for t, c in commands}
    seg = None

    def close_segment():
        if seg is not None:
            seg.update(end_step=obs.t, end_dist_to_start=round(float(np.linalg.norm(obs.eef_pos - start_pos)), 4),
                       end_gripper_width=round(obs.gripper_width, 4), steps=obs.t - seg["start_step"],
                       travel=round(seg.pop("_travel"), 4),
                       mean_grip_action=round(float(np.mean(seg.pop("_grip"))), 3) if seg["_grip"] else None)
            segments.append(seg)

    success = False
    while obs.t < max_steps:
        if obs.t in switch_at:
            close_segment()
            system1.set_command(switch_at[obs.t])
            seg = dict(command=switch_at[obs.t], start_step=obs.t, _travel=0.0, _grip=[],
                       start_dist_to_start=round(float(np.linalg.norm(obs.eef_pos - start_pos)), 4),
                       start_gripper_width=round(obs.gripper_width, 4))
        action = system1.act(obs)
        prev = obs.eef_pos.copy()
        obs, success = world.step(action)
        seg["_travel"] += float(np.linalg.norm(obs.eef_pos - prev))
        seg["_grip"].append(float(action[6]))
        for k, v in object_heights(world).items():
            max_rise[k] = max(max_rise[k], v - start_z[k])
        for k, v in (world.goal_progress() or {}).items():
            if v and k not in first_true:
                first_true[k] = obs.t
        trace.append(dict(t=obs.t, p=[round(float(x), 4) for x in obs.eef_pos], g=round(obs.gripper_width, 4)))
    close_segment()
    info = world.reset_info()
    shift = np.linalg.norm(np.subtract(info["eef_before_offset"], info["eef_after_settle"]))
    return dict(start_shift_cm=round(float(shift) * 100, 1), start_contacts=info["collision_contacts"],
                segments=segments, predicates_first_true=first_true, env_success_at_end=bool(success),
                lifted={k: round(v, 3) for k, v in max_rise.items() if v > LIFT_M}, trace=trace)


def main() -> None:
    from ..env import LiberoWorld
    from ..robot_states import RobotState
    from ..system1 import build_system1

    p = argparse.ArgumentParser()
    p.add_argument("--res", default="/content/probe")
    p.add_argument("--type", default="smolvla")
    p.add_argument("--path", default="lerobot/smolvla_libero_plus")
    p.add_argument("--states", nargs="*", default=["A"], choices=list(ROBOT_STATES))
    p.add_argument("--kinds", nargs="*", default=["task", "switch", "schedule"])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--task-steps", type=int, default=200)
    p.add_argument("--pre-steps", type=int, default=60)
    p.add_argument("--switch-steps", type=int, default=80)
    p.add_argument("--only-first", action="store_true", help="one command per case, for a quick check")
    a = p.parse_args()
    res = Path(a.res)
    res.mkdir(parents=True, exist_ok=True)
    system1 = build_system1(dict(type=a.type, path=a.path, n_action_steps=10))
    path = res / "probe.jsonl"
    key = lambda r: (r["kind"], r["suite"], r["task"], r["command"], r["seed"], r.get("robot_state", "A"))
    finished = {key(json.loads(line)) for line in open(path, encoding="utf-8")} if path.exists() else set()
    out = open(path, "a", encoding="utf-8")

    def record(kind, suite, task, command, schedule, max_steps):
        for state_name in a.states:
            state = RobotState(state_name, joint_offset_deg=ROBOT_STATES[state_name])
            for s in range(a.seeds):
                seed = zlib.crc32(f"{suite}-{task}-{s}".encode()) % 100000  # the same for every command of a scene
                row = dict(kind=kind, suite=suite, task=task, command=command, seed=seed, robot_state=state_name, checkpoint=a.path)
                if key(row) in finished:
                    continue
                world = LiberoWorld(suite, task, max_steps=max_steps)
                try:
                    r = run_case(world, system1, state, seed, schedule, max_steps)
                finally:
                    world.close()
                out.write(json.dumps(dict(row, **r)) + "\n")
                out.flush()
                print(kind, suite, task, state_name, f"{r['start_shift_cm']} cm", repr(command), "success", r["env_success_at_end"],
                      "pred", r["predicates_first_true"], "lift", r["lifted"], flush=True)

    for suite, task, commands in TASK_CASES if "task" in a.kinds else []:
        for c in commands[:1] if a.only_first else commands:
            record("task", suite, task, c, [(0, c)], a.task_steps)
    for suite, task, commands in SWITCH_CASES if "switch" in a.kinds else []:
        pre = next(c[2][0] for c in TASK_CASES if c[0] == suite and c[1] == task)
        for c in commands[:1] if a.only_first else commands:
            record("switch", suite, task, c, [(0, pre), (a.pre_steps, c)], a.pre_steps + a.switch_steps)
    for suite, task, atomic, steps, whole in SCHEDULE_CASES if "schedule" in a.kinds else []:
        total = steps * len(atomic)
        record("schedule", suite, task, " > ".join(atomic), [(i * steps, c) for i, c in enumerate(atomic)], total)
        if not a.only_first:
            record("schedule", suite, task, whole, [(0, whole)], total)
            record("schedule", suite, task, GIBBERISH, [(0, GIBBERISH)], total)


if __name__ == "__main__":
    main()
