"""Comparison videos: left single system (System 2 passthrough), right dual system, same task, state and seed.

    python -m dual_system.eval.side_by_side <res> [--out <res>/videos]

Plays at the true 20 fps. Whichever side ends first holds its last frame with the outcome label.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .lookup import find_cells

HEADER = 70
FOOTER = 120
GAP = 8
BG = (18, 18, 22)
FG = (235, 235, 235)
DIM = (160, 160, 170)
GOOD = (90, 210, 120)
BAD = (235, 95, 95)


@dataclass
class Episode:
    instruction: str
    success: bool
    verdict: str | None
    n_steps: int
    events: list[dict]  # subgoal_open and s2_call rows, in time order


def load_episode(cell: Path) -> Episode:
    rows = [json.loads(line) for line in open(cell / "trace.jsonl", encoding="utf-8")]
    end = next(r for r in rows if r["type"] == "episode_end")
    start = next(r for r in rows if r["type"] == "episode_start")
    return Episode(start["instruction"], end["success"], end["verdict"], end["n_steps"],
                   [r for r in rows if r["type"] in ("subgoal_open", "s2_call")])


def state_at(ep: Episode, t: int) -> tuple[str, str, str]:
    """Current command, its label and the latest trigger reason at step t."""
    command, label, reason = "", "", ""
    for e in ep.events:
        if e["t"] > t:
            break
        if e["type"] == "subgoal_open":
            command = e["text"]
            label = f"{e['kind']} #{e['subgoal_index']}" + (f" retry {e['retry']}" if e["retry"] else "")
        else:
            reason = ",".join(e["reasons"])
    return command, label, reason


def outcome_text(ep: Episode) -> tuple[str, tuple]:
    verdict = f" | System 2 said: {ep.verdict}" if ep.verdict else ""
    return ("SUCCESS" if ep.success else "FAILED") + verdict, GOOD if ep.success else BAD


def _font(size: int):
    return ImageFont.load_default(size=size)


def _wrapped(draw, xy, text, width_chars, font, fill):
    y = xy[1]
    for line in textwrap.wrap(text, width_chars)[:3]:
        draw.text((xy[0], y), line, font=font, fill=fill)
        y += font.size + 3
    return y


def compose(left: np.ndarray, right: np.ndarray, single: Episode, dual: Episode, t: int, task_label: str) -> np.ndarray:
    h, w = left.shape[:2]
    canvas = Image.new("RGB", (2 * w + GAP, HEADER + h + FOOTER), BG)
    canvas.paste(Image.fromarray(left), (0, HEADER))
    canvas.paste(Image.fromarray(right), (w + GAP, HEADER))
    d = ImageDraw.Draw(canvas)
    title, small = _font(17), _font(14)
    d.text((10, 6), task_label, font=small, fill=DIM)
    _wrapped(d, (10, 26), f'"{dual.instruction}"', (2 * w) // 9, title, FG)
    x_r = w + GAP + 10
    y = HEADER + h + 6

    d.text((10, y), f"Single system   step {min(t, single.n_steps)}", font=title, fill=FG)
    d.text((10, y + 24), "System 2: passthrough", font=small, fill=DIM)
    if t >= single.n_steps:
        text, col = outcome_text(single)
        d.text((10, y + 46), text, font=title, fill=col)

    d.text((x_r, y), f"Dual system   step {min(t, dual.n_steps)}", font=title, fill=FG)
    command, label, reason = state_at(dual, t)
    d.text((x_r, y + 24), f"[{label}] {reason}", font=small, fill=DIM)
    y2 = _wrapped(d, (x_r, y + 44), f'"{command}"', w // 8, small, FG)
    if t >= dual.n_steps:
        text, col = outcome_text(dual)
        d.text((x_r, max(y2, y + 88)), text, font=title, fill=col)
    return np.asarray(canvas)


def render_pair(single_cell: Path, dual_cell: Path, out: Path) -> Path:
    import imageio.v2 as imageio

    single, dual = load_episode(single_cell), load_episode(dual_cell)
    frames_s = imageio.get_reader(single_cell / "episode.mp4")
    frames_d = imageio.get_reader(dual_cell / "episode.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=20, codec="libx264", macro_block_size=1)
    label = dual_cell.name.rsplit("__", 1)[0].replace("__", "  ")
    it_s, it_d = iter(frames_s), iter(frames_d)
    last_s, last_d, t = None, None, 0
    while True:
        cur_s, cur_d = next(it_s, None), next(it_d, None)
        if cur_s is None and cur_d is None:
            break
        last_s = cur_s if cur_s is not None else last_s
        last_d = cur_d if cur_d is not None else last_d
        writer.append_data(compose(last_s, last_d, single, dual, t, label))
        t += 1
    for _ in range(40):  # 2 s on the final frame
        writer.append_data(compose(last_s, last_d, single, dual, t, label))
    writer.close()
    frames_s.close()
    frames_d.close()
    return out


def render_all(res: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    res = Path(res)
    out_dir = Path(out_dir) if out_dir else res / "videos"
    made = []
    for dual_cell in find_cells(res, condition="full"):
        single_cell = dual_cell.with_name(dual_cell.name.rsplit("__", 1)[0] + "__passthrough")
        ready = all((c / "episode.mp4").exists() and (c / "DONE").exists() for c in (single_cell, dual_cell))
        if ready:
            made.append(render_pair(single_cell, dual_cell, out_dir / (dual_cell.name.rsplit("__", 1)[0] + ".mp4")))
    return made


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("res")
    p.add_argument("--out")
    a = p.parse_args()
    for v in render_all(a.res, a.out):
        print(v)
