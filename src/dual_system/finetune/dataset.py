"""lerobot/libero with subtask commands, without re-encoding: each frame gets the command of its phase.

Action chunks stop at the phase end. Past it the target is a hold (no motion, gripper as it was), so "pick up" does not learn
to go on and place, and the policy learns to wait for System 2. Splits are by whole demo and whole task.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml

SPLITS = Path(__file__).with_name("splits.yaml")


def load_splits() -> dict:
    return yaml.safe_load(open(SPLITS, encoding="utf-8"))


def split_of(task: str, episode: int, splits: dict) -> str:
    """unseen_task, val or train. Depends only on the names, so it is the same in every session."""
    if task in splits["unseen_tasks"]:
        return "unseen_task"
    return "val" if zlib.crc32(f"{task}-{episode}".encode()) % 100 < splits["val_percent"] else "train"


def frame_index(boundaries: list[dict], splits: dict, split: str) -> list[tuple[int, int, str]]:
    """(dataset index, frames left in the phase after this one, command) for all frames of a split."""
    out = []
    for row in boundaries:
        if not row["phases"] or split_of(row["task"], row["episode"], splits) != split:
            continue
        for p in row["phases"]:
            for t in range(p["start"], p["end"]):
                out.append((row["first_index"] + t, p["end"] - 1 - t, p["command"]))
    return out


def hold_past_end(action: torch.Tensor, is_pad: torch.Tensor, frames_left: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace everything after the phase end with zero motion and the gripper value at the end. Holds are real targets."""
    keep = frames_left + 1
    if keep >= len(action):
        return action, is_pad
    action, is_pad = action.clone(), is_pad.clone()
    last = action[keep - 1]
    action[keep:, :6] = 0.0
    action[keep:, 6] = last[6]
    is_pad[keep:] = False
    return action, is_pad


class SubtaskDataset(torch.utils.data.Dataset):
    def __init__(self, base, frames: list[tuple[int, int, str]]):
        self.base, self.frames = base, frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        index, frames_left, command = self.frames[i]
        item = self.base[index]
        item["action"], item["action_is_pad"] = hold_past_end(item["action"], item["action_is_pad"], frames_left)
        item["task"] = command
        return item


def load_boundaries(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def summary(boundaries: list[dict], splits: dict) -> dict:
    """Episodes and frames per split, and why demos were dropped."""
    out = {"dropped": {}}
    for s in ("train", "val", "unseen_task"):
        out[s] = dict(frames=len(frame_index(boundaries, splits, s)),
                      episodes=sum(1 for r in boundaries if r["phases"] and split_of(r["task"], r["episode"], splits) == s))
    for r in boundaries:
        if not r["phases"]:
            why = (r["dropped"] or "?").split(":")[0]
            out["dropped"][why] = out["dropped"].get(why, 0) + 1
    out["commands"] = len({p["command"] for r in boundaries for p in r["phases"] or []})
    out["phase_length_median"] = int(np.median([p["end"] - p["start"] for r in boundaries for p in r["phases"] or []]))
    return out
