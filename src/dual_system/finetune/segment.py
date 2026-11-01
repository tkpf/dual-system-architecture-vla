"""Cut LIBERO demos into subtasks using simulator truth.

    python -m dual_system.finetune.segment run --out <res>/boundaries.jsonl     # replays all demos, resumable by task
    python -m dual_system.finetune.segment sheets --boundaries <file> --out <dir>  # contact sheets of boundary frames

lerobot/libero has no sim state, but its episodes match the raw LIBERO demos, whose states can be replayed without rendering
(about 0.15 s per demo). So each lerobot episode is matched to a raw demo by its actions, and the boundaries found on the
demo apply to the episode frame by frame. A demo whose events do not fit its subtask list is dropped and counted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

SUBTASKS = Path(__file__).with_name("subtasks.yaml")
MIN_HELD_FRAMES, MAX_RELEASE_WAIT, MATCH_TOLERANCE, RISE_M = 5, 20, 1e-4, 0.02


def load_subtasks() -> dict:
    return yaml.safe_load(open(SUBTASKS, encoding="utf-8"))


def last_true_start(flags: np.ndarray, min_len: int = 1) -> int | None:
    """Start of the last run of True at least min_len long."""
    end = len(flags)
    while end > 0:
        while end > 0 and not flags[end - 1]:
            end -= 1
        start = end
        while start > 0 and flags[start - 1]:
            start -= 1
        if end - start >= min_len:
            return start
        end = start
    return None


def find_phases(groups: list[dict], goals: dict[str, np.ndarray], held: dict[str, np.ndarray], gripper_closed: np.ndarray) -> tuple[list[dict] | None, str]:
    """Contiguous phases [{command, kind, start, end}], or None and the reason. end is exclusive. gripper_closed is per step."""
    timed = []
    for group in groups:
        key = next((k for k in goals if group["goal"] in k), None)
        if key is None:
            return None, f"no goal for {group['goal']}"
        done = last_true_start(goals[key])
        if done is None or done == 0 or not goals[key][-1]:
            return None, f"goal not reached: {key}"
        timed.append((done, key, group))
    phases, t = [], 0
    for done, key, group in sorted(timed, key=lambda x: x[0]):
        if "act" in group:
            release = next((u for u in range(done, min(done + MAX_RELEASE_WAIT, len(gripper_closed))) if not gripper_closed[u]), done)
            ends = [("act", group["act"], max(release, done) + 1)]
        else:
            flags = held[key.split()[1]][:done]
            grasp = last_true_start(flags, MIN_HELD_FRAMES)
            if grasp is None:
                return None, f"never held: {key}"
            ends = [("pick", group["pick"], grasp + 1), ("place", group["place"], done + 1)]
        for kind, command, end in ends:
            if end <= t:
                return None, f"phase out of order: {command}"
            phases.append(dict(command=command, kind=kind, start=t, end=end))
            t = end
    return phases, ""


def match_episode(actions: np.ndarray, demos: dict[str, np.ndarray]) -> str | None:
    """The raw demo with the same actions as a lerobot episode. Lengths may differ by up to 2 steps at the end."""
    best, best_dist = None, MATCH_TOLERANCE
    for name, raw in demos.items():
        if abs(len(raw) - len(actions)) > 2:
            continue
        m = min(len(raw), len(actions))
        dist = float(np.abs(raw[:m, :6] - actions[:m, :6]).mean())
        if dist < best_dist:
            best, best_dist = name, dist
    return best


def replay(world, states: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Goal predicates and held flags per step, from the stored sim states."""
    inner = world.env._env.env
    sim = inner.sim
    goals, held, z0 = {}, {}, {}
    for t, s in enumerate(states):
        sim.set_state_from_flattened(s)
        sim.forward()
        for k, v in world.goal_progress().items():
            goals.setdefault(k, np.zeros(len(states), bool))[t] = v
        for name, i in inner.obj_body_id.items():
            z = float(sim.data.body_xpos[i][2])
            z0.setdefault(name, z)
            flag = z - z0[name] > RISE_M and world.gripped(name)
            held.setdefault(name, np.zeros(len(states), bool))[t] = flag
    return goals, held


def tasks_to_suites() -> dict[str, tuple[str, int, str]]:
    from libero.libero import benchmark

    out = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        b = benchmark.get_benchmark_dict()[suite]()
        for i in range(b.n_tasks):
            task = b.get_task(i)
            out[task.language] = (suite, i, task.bddl_file.removesuffix(".bddl"))
    return out


def run(out_path: Path) -> None:
    import glob

    import h5py
    import pandas as pd
    from huggingface_hub import hf_hub_download, snapshot_download

    from ..env import LiberoWorld
    from ..robot_states import RobotState

    root = snapshot_download("lerobot/libero", repo_type="dataset", allow_patterns=["data/*", "meta/*"])
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index", "action", "index"]) for f in sorted(glob.glob(root + "/data/*/*.parquet"))])
    tasks = pd.read_parquet(root + "/meta/tasks.parquet")
    suites, subtasks = tasks_to_suites(), load_subtasks()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finished = {json.loads(line)["task"] for line in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    out = open(out_path, "a", encoding="utf-8")
    for task in tasks.index:
        if task in finished:
            continue
        suite, task_id, stem = suites[task]
        raw = hf_hub_download("yifengzhu-hf/LIBERO-datasets", f"{suite}/{stem}_demo.hdf5", repo_type="dataset")
        world = LiberoWorld(suite, task_id, max_steps=1000)
        world.reset(RobotState("A"), 0)
        counts: dict[str, int] = {}
        with h5py.File(raw) as f:
            data = f["data"]
            demos = {n: data[n]["actions"][()] for n in data}
            rows = []
            for episode, g in df[df.task_index == int(tasks.loc[task, "task_index"])].groupby("episode_index"):
                row = dict(task=task, suite=suite, task_id=task_id, episode=int(episode), first_index=int(g["index"].iloc[0]), length=len(g))
                actions = np.stack(g.action.values)
                demo = match_episode(actions, demos)
                if demo is None:
                    phases, why = None, "no matching demo"
                else:
                    goals, held = replay(world, data[demo]["states"][()])
                    phases, why = find_phases(subtasks[task], goals, held, demos[demo][:, 6] > 0)
                    if phases:
                        phases = [p for p in phases if p["start"] < len(g)]
                        phases[-1]["end"] = min(phases[-1]["end"], len(g))
                row.update(demo=demo, phases=phases, dropped=why or None)
                counts[why.split(":")[0] or "ok"] = counts.get(why.split(":")[0] or "ok", 0) + 1
                rows.append(row)
        world.close()
        for row in rows:
            out.write(json.dumps(row) + "\n")
        out.flush()
        print(task, counts, flush=True)


def sheets(boundaries: Path, out_dir: Path, per_task: int = 2) -> None:
    """One image per sampled demo: for each phase, the first and last frame of the scene and wrist camera."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image, ImageDraw

    ds = LeRobotDataset("lerobot/libero", video_backend="torchcodec")
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    for row in map(json.loads, open(boundaries, encoding="utf-8")):
        if not row["phases"] or seen.get(row["task"], 0) >= per_task:
            continue
        seen[row["task"]] = seen.get(row["task"], 0) + 1
        phases = row["phases"]
        sheet = Image.new("RGB", (4 * 256, len(phases) * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for y, p in enumerate(phases):
            draw.text((4, y * 280 + 4), f"{p['command']}  [{p['start']}, {p['end']})", fill="black")
            last = max(p["start"], p["end"] - 1)
            for x, (t, cam) in enumerate([(p["start"], "image"), (p["start"], "image2"), (last, "image"), (last, "image2")]):
                frame = ds[row["first_index"] + t][f"observation.images.{cam}"]
                img = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                sheet.paste(Image.fromarray(img), (x * 256, y * 280 + 24))
        sheet.save(out_dir / f"{row['suite']}_t{row['task_id']}_ep{row['episode']}.jpg", quality=85)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "sheets"])
    p.add_argument("--out", required=True)
    p.add_argument("--boundaries")
    a = p.parse_args()
    if a.command == "run":
        run(Path(a.out))
    else:
        sheets(Path(a.boundaries), Path(a.out))
