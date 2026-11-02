"""The final comparison: the same scene, start and seed under four ways of commanding pi0.5.

    python -m dual_system.eval.final run --res <res> [--seeds 1] [--states C B] [--conditions passthrough frozen fused oracle]
        [--suite-tasks libero_10:3 libero_10:9]     # any LIBERO task instead of the three default scenes
    python -m dual_system.eval.final report <res>
    python -m dual_system.eval.final video <res>            # 2x2 side-by-side videos, one per scene

passthrough  no System 2: the whole instruction
frozen       System 2 frozen after its first command: the first atomic command for the whole episode
fused        System 2 switches atomic commands on gripper signals, a VLM for open and turn-on, and timeouts (system2/fused.py)
fused_norec  the same without the empty-grasp recovery
With --perturb drop, the gripper is forced open for PERTURB_STEPS steps once it has held something for PERTURB_AFTER steps
in a row (the same robot-side rule in every condition), so the object falls. fused then also treats a release without a
descent as a drop and goes back to the pick, and oracle goes back to the pick when the object has left the gripper.
oracle       the simulator says when a phase is done: the ceiling for any progress check
All conditions get the same step budget (150 per phase) and end at the environment's success. Every step is also scored
against simulator truth, so each switch of `fused` is logged as right or early, and its lag behind the truth is measured.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from .atomic import PHASE_BUDGET, SEEN, phase_done, plan_for
from .command_probe import ROBOT_STATES, object_heights

CONDITIONS = ["passthrough", "frozen", "fused", "oracle"]
PERTURB_AFTER, PERTURB_STEPS = 20, 10
ALL_CONDITIONS = CONDITIONS + ["fused_norec"]


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    h = 12 * len(lines) + 6
    d.rectangle([0, img.height - h, img.width, img.height], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((4, img.height - h + 3 + 12 * i), line, fill=(255, 255, 255))
    return np.asarray(img)


class VlmJudge:
    """Qwen 2B's probability that a subtask is done, with the `camera` prompt of run 12 (best of the 2B styles)."""

    def __init__(self):
        from ..knowledge import StaticRetriever
        from ..system2.reasoners import QwenReasoner
        from .perception import MODELS

        self.reasoner = QwenReasoner(**MODELS["qwen2b"])
        self.retriever = StaticRetriever("phase_atomic")
        self.calls, self.seconds = 0, 0.0

    def prob_done(self, step: str, obs) -> float:
        from .judge_prompts import build, done_probability

        t0 = time.perf_counter()
        spec = build(step, "camera", self.retriever)
        p = done_probability(self.reasoner, spec["prompt"], [obs.agent, obs.wrist], spec)
        self.calls, self.seconds = self.calls + 1, self.seconds + time.perf_counter() - t0
        return p

    def plan(self, instruction: str, obs) -> list[str]:
        """What Qwen would decompose the instruction into. Logged next to the reference plan, not used to control."""
        from ..knowledge import StaticRetriever, snap_to_candidates

        r = StaticRetriever("free_atomic")
        cands = r.candidates(instruction)
        a = self.reasoner.plan(instruction, [obs.agent, obs.wrist], r.rules(instruction), cands, [])
        steps = (a.parsed or {}).get("steps")
        return snap_to_candidates([x for x in steps if isinstance(x, str)], cands) if isinstance(steps, list) else []


def run_episode(world, system1, state, seed: int, phases: list[dict], condition: str, judge: VlmJudge | None, instruction: str,
                frames: list | None, perturb: str | None = None) -> dict:
    from ..rollout import seed_everything
    from ..system2.fused import EMPTY_WIDTH, HELD_MAX, FusedSwitcher, OracleSwitcher

    seed_everything(seed)
    system1.reset()
    obs = world.reset(state, seed)
    start_z = object_heights(world)
    budget = PHASE_BUDGET * len(phases)

    switcher = None
    if condition == "oracle":
        switcher = OracleSwitcher(phases, timeout=PHASE_BUDGET, undo=perturb == "drop")
    elif condition in ("fused", "fused_norec"):
        switcher = FusedSwitcher(phases, timeout=PHASE_BUDGET, recover=condition == "fused", detect_drop=perturb == "drop" and condition == "fused")
    switcher and switcher.reset()
    command = instruction if condition == "passthrough" else phases[0]["command"] if condition == "frozen" else switcher.command
    system1.set_command(command)
    plan = judge.plan(instruction, obs) if condition == "fused" and judge else None

    commands, events, truth_at = [(0, command)], [], {}
    success, first_success = False, None
    held_run, perturbed_at = 0, None
    while obs.t < budget:
        action = system1.act(obs)
        if perturb == "drop" and perturbed_at is None and held_run >= PERTURB_AFTER:
            perturbed_at = obs.t
        forced = perturbed_at is not None and obs.t < perturbed_at + PERTURB_STEPS
        if forced:
            action = np.array(action, dtype=float, copy=True)
            action[6] = -1.0  # open: positive closes the gripper
        obs, success = world.step(action)
        held_run = held_run + 1 if EMPTY_WIDTH < obs.gripper_width < HELD_MAX else 0
        if success:
            first_success = obs.t
            if frames is not None:
                frames.append(annotate(obs.agent, [f"{condition}  t={obs.t}  SUCCESS", f"S1 command: {command}"]))
            break
        if switcher is not None:
            goals = world.goal_progress()
            rises = {k: v - start_z[k] for k, v in object_heights(world).items()}
            phase = switcher.phase
            truth = bool(phase_done(phase, goals, world.gripped, rises))
            if truth:
                truth_at.setdefault(switcher.idx, obs.t)
            if condition == "oracle":
                held = world.gripped(next(k for k in goals if phase["goal"] in k).split()[1]) if phase["kind"] == "place" else None
                event = switcher.update(obs.t, truth, held)
            else:
                event = switcher.update(obs.t, obs.gripper_width, float(obs.eef_pos[2]),
                                        (lambda s=phase["command"], o=obs: judge.prob_done(s, o)) if judge else None, truth)
            if event is not None:
                events.append(event)
                if switcher.command != command:
                    command = switcher.command
                    system1.set_command(command)
                    commands.append((obs.t, command))
        if frames is not None:
            head = f"{condition}  t={obs.t}" + ("  PERTURBATION: gripper forced open" if forced else "")
            frames.append(annotate(obs.agent, [head, f"S1 command: {command}"]))
    return dict(perturbed_at=perturbed_at, success=bool(success), steps=obs.t, first_success=first_success, budget=budget, commands=commands, events=events,
                truth_done_at=truth_at, vlm_probs=getattr(switcher, "vlm_probs", None), vlm_plan=plan,
                reference_plan=[p["command"] for p in phases], vlm_calls=judge.calls if judge else 0)


def run(a) -> None:
    from ..env import LiberoWorld
    from ..finetune.dataset import load_splits
    from ..finetune.segment import load_subtasks, tasks_to_suites
    from ..robot_states import RobotState
    from ..system1 import build_system1

    res = Path(a.res)
    (res / "videos").mkdir(parents=True, exist_ok=True)
    splits, subtasks, suites = load_splits(), load_subtasks(), tasks_to_suites()
    path = res / "final.jsonl"
    key = lambda r: (r["task"], r["robot_state"], r["seed"], r["condition"])
    finished = {key(json.loads(line)) for line in open(path, encoding="utf-8")} if path.exists() else set()
    out = open(path, "a", encoding="utf-8")
    system1 = build_system1(dict(type="pi05", path=a.path, n_action_steps=10))
    judge = VlmJudge() if any(c.startswith("fused") for c in a.conditions) else None
    init = splits["init_states"]["select"][0]
    by_id = {(suite, i): task for task, (suite, i, _) in suites.items()}
    tasks = [by_id[(x.split(":")[0], int(x.split(":")[1]))] for x in a.suite_tasks] if a.suite_tasks else [SEEN[i] for i in a.tasks]
    t_start = time.time()
    for rep in range(a.seeds):  # seed-major, so stopping early leaves balanced pairs
        for task in tasks:
            suite, task_id, _ = suites[task]
            phases = plan_for(task, subtasks)
            for state_name in a.states:
                seed = (zlib.crc32(task.encode()) + init * 100 + rep) % 100000
                for condition in a.conditions:
                    row = dict(task=task, robot_state=state_name, seed=seed, condition=condition)
                    if key(row) in finished:
                        continue
                    if a.stop_after_min and (time.time() - t_start) / 60 > a.stop_after_min:
                        print("time box reached", flush=True)
                        return
                    world = LiberoWorld(suite, task_id, max_steps=PHASE_BUDGET * len(phases) + 1)
                    frames = []
                    t0 = time.time()
                    try:
                        r = run_episode(world, system1, RobotState(state_name, init_state_id=init, joint_offset_deg=ROBOT_STATES[state_name]),
                                        seed, phases, condition, judge, world.instruction, frames, a.perturb)
                    finally:
                        world.close()
                    video = f"{zlib.crc32(task.encode()):08x}__{state_name}__{seed}__{condition}.mp4"
                    import imageio.v2 as imageio

                    imageio.mimwrite(res / "videos" / video, frames, fps=20, codec="libx264", macro_block_size=1)
                    out.write(json.dumps(dict(row, **r, video=video, wall_s=round(time.time() - t0, 1))) + "\n")
                    out.flush()
                    print(task[:36], state_name, condition, "SUCCESS" if r["success"] else "fail", r["steps"], "steps",
                          [(e["t"], e["reason"]) for e in r["events"]], flush=True)


# ---------------------------------------------------------------- report
def report(res: Path) -> str:
    rows = [json.loads(line) for line in open(res / "final.jsonl", encoding="utf-8")]
    lines = []
    conds = [c for c in ALL_CONDITIONS if any(r["condition"] == c for r in rows)]
    lines.append(f"{'condition':<12} {'success':>8} {'median steps (successes)':>26}   per task (success)")
    tasks = sorted({r["task"] for r in rows})
    for c in conds:
        part = [r for r in rows if r["condition"] == c]
        wins = [r["steps"] for r in part if r["success"]]
        per = "  ".join(f"{sum(r['success'] for r in part if r['task'] == t)}/{sum(r['task'] == t for r in part)}" for t in tasks)
        lines.append(f"{c:<12} {sum(r['success'] for r in part):>3}/{len(part):<4} {int(np.median(wins)) if wins else '-':>26}   {per}")
    lines.append("tasks: " + " | ".join(t[:40] for t in tasks))
    lines.append("")
    lines.append("failure attribution per scene (task, start, seed):")
    scenes: dict[tuple, dict] = {}
    for r in rows:
        scenes.setdefault((r["task"], r["robot_state"], r["seed"]), {})[r["condition"]] = r["success"]
    full = [s for s in scenes.values() if {"fused", "oracle"} <= s.keys()]
    if full:
        fused_fail = [s for s in full if not s["fused"]]
        lines.append(f"  fused failed {len(fused_fail)}/{len(full)}: oracle also failed (System 1) {sum(not s['oracle'] for s in fused_fail)}, "
                     f"oracle succeeded (System 2) {sum(s['oracle'] for s in fused_fail)}")
        with_pass = [s for s in full if "passthrough" in s]
        lines.append(f"  passthrough failed, fused succeeded (rescue) {sum(not s['passthrough'] and s['fused'] for s in with_pass)}; "
                     f"passthrough succeeded, fused failed (harm) {sum(s['passthrough'] and not s['fused'] for s in with_pass)}; of {len(with_pass)}")
        with_frozen = [s for s in full if "frozen" in s]
        lines.append(f"  fused vs frozen: fused only {sum(s['fused'] and not s['frozen'] for s in with_frozen)}, "
                     f"frozen only {sum(s['frozen'] and not s['fused'] for s in with_frozen)}, of {len(with_frozen)}")
    lines.append("")
    lines.append("by start state (success)")
    for s in sorted({r["robot_state"] for r in rows}):
        lines.append(f"  {s}: " + "  ".join(f"{c} {sum(r['success'] for r in rows if r['condition'] == c and r['robot_state'] == s)}/"
                                              f"{sum(r['condition'] == c and r['robot_state'] == s for r in rows)}" for c in conds))
    lines.append("")
    for c in ("fused", "fused_norec", "oracle"):
        evs = [(r, e) for r in rows if r["condition"] == c for e in r["events"] if e["reason"] in ("signal", "vlm", "oracle", "timeout")]
        if not evs:
            continue
        by = {}
        for r, e in evs:
            by.setdefault(e["reason"], []).append(e)
        lines.append(f"{c}: {len(evs)} phase switches " + ", ".join(f"{k} {len(v)}" for k, v in by.items()))
        for k, v in by.items():
            early = [e for e in v if e["truth_done"] is False]
            lines.append(f"    {k:<8} early (simulator says not done) {len(early)}/{len(v)}")
        rec = sum(e["reason"] == "empty_grasp" for r in rows if r["condition"] == c for e in r["events"])
        lines.append(f"    empty-grasp recoveries: {rec}")
        # lag: switch step minus the first step the simulator said the phase was done
        lags = []
        for r in rows:
            if r["condition"] != c:
                continue
            for e in r["events"]:
                if e["reason"] in ("signal", "vlm") and str(e["phase"]) in r["truth_done_at"]:
                    lags.append(e["t"] - r["truth_done_at"][str(e["phase"])])
        if lags:
            lines.append(f"    switch minus first simulator-done step: median {int(np.median(lags))}, range {min(lags)} to {max(lags)}")
    if any(r.get("perturbed_at") is not None for r in rows):
        lines.append("drop perturbation: success among perturbed episodes (episodes never perturbed in brackets)")
        for c in conds:
            part = [r for r in rows if r["condition"] == c]
            hit = [r for r in part if r.get("perturbed_at") is not None]
            backs = sum(e["reason"] in ("dropped", "undone") for r in hit for e in r["events"])
            lines.append(f"  {c:<12} {sum(r['success'] for r in hit)}/{len(hit)}  ({len(part) - len(hit)} not perturbed)  "
                         f"back to the pick: {backs}  median perturbation step {int(np.median([r['perturbed_at'] for r in hit])) if hit else '-'}")
    probs = [(p, r) for r in rows if r["condition"] == "fused" for p in (r["vlm_probs"] or [])]
    if probs:
        lines.append(f"VLM probability of 'done' on act phases: {len(probs)} checks, median {np.median([p['p'] for p, _ in probs]):.2f}")
    plans = [(r["vlm_plan"], r["reference_plan"]) for r in rows if r["condition"] == "fused" and r["vlm_plan"] is not None]
    if plans:
        same = sum([x.lower() for x in a] == [x.lower() for x in b] for a, b in plans)
        lines.append(f"Qwen's own decomposition equals the reference plan in {same}/{len(plans)} episodes")
    text = "\n".join(lines)
    (res / "final_report.txt").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- video
def video(res: Path) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    rows = [json.loads(line) for line in open(res / "final.jsonl", encoding="utf-8")]
    scenes: dict[tuple, dict] = {}
    for r in rows:
        scenes.setdefault((r["task"], r["robot_state"], r["seed"]), {})[r["condition"]] = r
    (res / "side_by_side").mkdir(exist_ok=True)
    order = ["passthrough", "frozen", "fused", "oracle"]
    for (task, state, seed), conds in scenes.items():
        readers = {c: [f for f in imageio.get_reader(res / "videos" / conds[c]["video"])] for c in order if c in conds}
        if not readers:
            continue
        size = next(iter(readers.values()))[0].shape[0]
        n = max(len(f) for f in readers.values())
        out = []
        for i in range(n):
            tiles = []
            for c in order:
                tile = np.zeros((size + 22, size, 3), np.uint8)
                if c in readers:
                    r = conds[c]
                    f = readers[c][min(i, len(readers[c]) - 1)]
                    tile[22:] = f
                    img = Image.fromarray(tile)
                    d = ImageDraw.Draw(img)
                    done = i >= len(readers[c]) - 1
                    label = f"{c}: " + (("SUCCESS" if r["success"] else "failed") if done else "running") + f" ({r['steps']} steps)"
                    d.text((4, 5), label, fill=(120, 230, 140) if done and r["success"] else (240, 110, 110) if done else (230, 230, 230))
                    tile = np.asarray(img)
                tiles.append(tile)
            top, bottom = np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)
            out.append(np.concatenate([top, bottom], 0))
        name = f"{zlib.crc32(task.encode()):08x}__{state}__{seed}.mp4"
        imageio.mimwrite(res / "side_by_side" / name, out, fps=20, codec="libx264", macro_block_size=1)
        print(task, state, "->", name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "report", "video"])
    p.add_argument("res", nargs="?")
    p.add_argument("--res", dest="res_opt")
    p.add_argument("--path", default="lerobot/pi05_libero_finetuned_v044")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--states", nargs="*", default=["C", "B"], choices=list(ROBOT_STATES))
    p.add_argument("--tasks", nargs="*", type=int, default=[0, 1, 2], help="indices into atomic.SEEN")
    p.add_argument("--suite-tasks", nargs="*", help="suite:task_id, e.g. libero_10:3; replaces --tasks")
    p.add_argument("--conditions", nargs="*", default=CONDITIONS, choices=ALL_CONDITIONS)
    p.add_argument("--stop-after-min", type=float)
    p.add_argument("--perturb", choices=["drop"], help="force the gripper open once something has been held")
    a = p.parse_args()
    a.res = a.res_opt or a.res
    if a.command == "run":
        run(a)
    elif a.command == "report":
        print(report(Path(a.res)))
    else:
        video(Path(a.res))
