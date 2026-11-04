"""Side-by-side videos of two command_probe runs: same scene, start state and seed, one model on each side.

    python -m dual_system.eval.compare_probe <left_res> <right_res> --out <dir>

Writes one video per episode pair plus all_episodes.mp4 with every pair in a row. Both sides play at the true 20 fps.
The footer shows the active command and turns green at the step the LIBERO goal is first reached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .side_by_side import BAD, BG, DIM, FG, FOOTER, GAP, GOOD, HEADER, _font, _wrapped


def load_runs(res: str | Path) -> dict[str, dict]:
    """Probe rows with a video, by video name."""
    rows = [json.loads(line) for line in open(Path(res) / "probe.jsonl", encoding="utf-8")]
    return {r["video"]: r for r in rows if r.get("video")}


def command_at(row: dict, t: int) -> str:
    text = ""
    for seg in row["segments"]:
        if seg["start_step"] <= t:
            text = seg["command"]
    return text


def _side(d, x: int, y: int, width: int, row: dict, t: int) -> None:
    title, small = _font(17), _font(14)
    steps = row["trace"][-1]["t"] if row["trace"] else 0
    d.text((x, y), f"{row['checkpoint'].split('/')[-1]}   step {min(t, steps)}", font=title, fill=FG)
    y2 = _wrapped(d, (x, y + 24), f'"{command_at(row, t)}"', width // 8, small, DIM)
    done = row.get("first_success_step")
    if done is not None and t >= done:
        d.text((x, max(y2, y + 66)), f"GOAL REACHED at step {done}", font=title, fill=GOOD)
    elif t >= steps:
        d.text((x, max(y2, y + 66)), "GOAL NOT REACHED", font=title, fill=BAD)


def compose(left: np.ndarray, right: np.ndarray, left_row: dict, right_row: dict, t: int, label: str) -> np.ndarray:
    h, w = left.shape[:2]
    canvas = Image.new("RGB", (2 * w + GAP, HEADER + h + FOOTER), BG)
    canvas.paste(Image.fromarray(left), (0, HEADER))
    canvas.paste(Image.fromarray(right), (w + GAP, HEADER))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), label, font=_font(14), fill=DIM)
    _wrapped(d, (10, 26), f'"{left_row["command"]}"', (2 * w) // 9, _font(15), FG)
    _side(d, 10, HEADER + h + 6, w, left_row, t)
    _side(d, w + GAP + 10, HEADER + h + 6, w, right_row, t)
    return np.asarray(canvas)


def _pair_frames(left_video: Path, right_video: Path, left_row: dict, right_row: dict, label: str):
    import imageio.v2 as imageio

    readers = imageio.get_reader(left_video), imageio.get_reader(right_video)
    it_l, it_r = iter(readers[0]), iter(readers[1])
    last_l, last_r, t = None, None, 0
    while True:
        cur_l, cur_r = next(it_l, None), next(it_r, None)
        if cur_l is None and cur_r is None:
            break
        last_l = cur_l if cur_l is not None else last_l
        last_r = cur_r if cur_r is not None else last_r
        yield compose(last_l, last_r, left_row, right_row, t, label)
        t += 1
    for _ in range(40):  # 2 s on the final frame
        yield compose(last_l, last_r, left_row, right_row, t, label)
    for r in readers:
        r.close()


def render_all(left_res: str | Path, right_res: str | Path, out: str | Path) -> list[Path]:
    import imageio.v2 as imageio

    left_res, right_res, out = Path(left_res), Path(right_res), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    left, right = load_runs(left_res), load_runs(right_res)
    made = []
    combined = imageio.get_writer(out / "all_episodes.mp4", fps=20, codec="libx264", macro_block_size=1)
    for name in sorted(left.keys() & right.keys()):
        l, r = left[name], right[name]
        label = f"{l['suite']} task {l['task']}  start {l['robot_state']} ({l['start_shift_cm']} cm)  seed {l['seed']}"
        writer = imageio.get_writer(out / name, fps=20, codec="libx264", macro_block_size=1)
        for frame in _pair_frames(left_res / "videos" / name, right_res / "videos" / name, l, r, label):
            writer.append_data(frame)
            combined.append_data(frame)
        writer.close()
        made.append(out / name)
    combined.close()
    return made


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("left_res")
    p.add_argument("right_res")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for v in render_all(a.left_res, a.right_res, a.out):
        print(v)
