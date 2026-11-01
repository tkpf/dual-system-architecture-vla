"""Everything System 2 does, written as it happens, so a recycled runtime only loses the event in flight.

<cell>/trace.jsonl     one line per System 2 call and per sub-goal open/close
<cell>/subgoals.jsonl  one line per finished sub-goal (the lookup table)
<cell>/steps.jsonl     compact per-step log
<cell>/keyframes/      both cameras at each System 2 decision
<cell>/episode.mp4     scene camera at 20 fps
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .types import Command, Decision, Observation, RunSpec, TriggerEvent


def _vec(x, digits=4):
    return [round(float(v), digits) for v in np.asarray(x).ravel()]


class TraceWriter:
    def __init__(self, cell_dir: str | Path, spec: RunSpec, video: bool = True, keyframes: bool = True):
        self.dir = Path(cell_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spec, self.save_keyframes = spec, keyframes
        self.identity = dict(run=spec.run, git_sha=spec.git_sha, cell=spec.cell, suite=spec.suite, task_id=spec.task_id,
                             robot_state=spec.robot_state, condition=spec.condition, seed=spec.seed)
        self.trace = open(self.dir / "trace.jsonl", "a", encoding="utf-8")
        self.subgoals = open(self.dir / "subgoals.jsonl", "a", encoding="utf-8")
        self.steps = open(self.dir / "steps.jsonl", "a", encoding="utf-8")
        self.video = None
        if video:
            import imageio.v2 as imageio

            self.video = imageio.get_writer(self.dir / "episode.mp4", fps=20, codec="libx264", macro_block_size=1)
        if keyframes:
            (self.dir / "keyframes").mkdir(exist_ok=True)
        self.current: dict | None = None
        self.travel = 0.0
        self.last_pos = None
        self.start_pos = None

    def _write(self, fh, row: dict) -> None:
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def start(self, obs: Observation, reset_info: dict) -> None:
        self.start_pos = obs.eef_pos.copy()
        self.last_pos = obs.eef_pos.copy()
        self._write(self.trace, dict(type="episode_start", t=0, instruction=self.spec.instruction, reset_info=reset_info,
                                     eef_start=_vec(obs.eef_pos), **self.identity))
        self.frame(obs)

    def frame(self, obs: Observation) -> None:
        if self.video is not None:
            self.video.append_data(obs.agent)

    def step(self, obs: Observation, action: np.ndarray, success: bool) -> None:
        self.travel += float(np.linalg.norm(obs.eef_pos - self.last_pos))
        self.last_pos = obs.eef_pos.copy()
        self.steps.write(json.dumps(dict(t=obs.t, a=_vec(action), p=_vec(obs.eef_pos, 5), g=_vec(obs.gripper_qpos),
                                         s=bool(success))) + "\n")
        self.frame(obs)

    def _keyframes(self, obs: Observation) -> dict:
        if not self.save_keyframes:
            return {}
        from PIL import Image

        out = {}
        for cam, img in (("agent", obs.agent), ("wrist", obs.wrist)):
            rel = f"keyframes/t{obs.t:04d}_{cam}.jpg"
            Image.fromarray(img).save(self.dir / rel, quality=85)
            out[cam] = rel
        return out

    def decision(self, event: TriggerEvent, d: Decision, obs: Observation, progress: dict | None, success: bool) -> None:
        self.steps.flush()
        status = d.status
        self._write(self.trace, dict(
            type="s2_call", t=obs.t, reasons=event.reasons, prompt_id=d.prompt_id, raw=d.raw, parsed=status,
            latency_s=round(d.latency_s, 3), verdict=d.verdict.value if d.verdict else None,
            command=d.command.text if d.command else None, closes=d.closes,
            active=self.current["text"] if self.current else None, eef=_vec(obs.eef_pos),
            gripper_width=round(obs.gripper_width, 4), dist_to_start=round(float(np.linalg.norm(obs.eef_pos - self.start_pos)), 4),
            env_success=bool(success), goal_progress=progress, keyframes=self._keyframes(obs), **self.identity))
        if self.current is not None and status:
            self.current["status_history"].append(dict(t=obs.t, **{k: status.get(k) for k in ("step", "task") if k in status}))

    def open_subgoal(self, c: Command, obs: Observation, progress: dict | None) -> None:
        self.current = dict(subgoal_index=c.subgoal_index, text=c.text, kind=c.kind, retry=c.retry, replan=c.replan,
                            issued_step=obs.t, eef_start=_vec(obs.eef_pos), gripper_start=round(obs.gripper_width, 4),
                            travel_start=self.travel, progress_start=progress, status_history=[])
        self._write(self.trace, dict(type="subgoal_open", t=obs.t, text=c.text, kind=c.kind,
                                     subgoal_index=c.subgoal_index, retry=c.retry, replan=c.replan, **self.identity))

    def close_subgoal(self, outcome: str, obs: Observation, progress: dict | None) -> None:
        if self.current is None:
            return
        row = {**self.identity, **self.current}
        travel_start = row.pop("travel_start")
        row.update(outcome=outcome, finish_step=obs.t, steps_taken=obs.t - row["issued_step"], eef_end=_vec(obs.eef_pos),
                   gripper_end=round(obs.gripper_width, 4), eef_travel=round(self.travel - travel_start, 4),
                   dist_to_start=round(float(np.linalg.norm(obs.eef_pos - self.start_pos)), 4), progress_end=progress)
        self._write(self.subgoals, row)
        self._write(self.trace, dict(type="subgoal_close", t=obs.t, text=row["text"], outcome=outcome, **self.identity))
        self.current = None

    def finish(self, result) -> None:
        self.steps.flush()
        row = {**self.identity, "instruction": self.spec.instruction, **asdict(result)}
        self._write(self.trace, dict(type="episode_end", t=result.n_steps, **row))
        with open(self.dir / "episodes.jsonl", "a", encoding="utf-8") as f:
            self._write(f, row)
        self.close()

    def close(self) -> None:
        for fh in (self.trace, self.subgoals, self.steps):
            if not fh.closed:
                fh.close()
        if self.video is not None:
            self.video.close()
            self.video = None
