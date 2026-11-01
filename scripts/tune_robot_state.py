"""Check a joint offset across tasks: gripper shift, contacts, joint limits, drift, and start-frame images.

    python scripts/tune_robot_state.py <out_dir> "<7 joint offsets in degrees, comma separated>" [suite:task ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from dual_system.env import LiberoWorld
from dual_system.robot_states import RobotState


def main(out: str, offset: str, tasks: list[str]) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_a = RobotState("A")
    state_b = RobotState("B", joint_offset_deg=[float(x) for x in offset.split(",")])
    panels = []
    for item in tasks:
        suite, task = item.split(":")
        w = LiberoWorld(suite, int(task))
        a = w.reset(state_a, 1)
        b = w.reset(state_b, 1)
        info = w.reset_info()
        hold = b.eef_pos.copy()
        for _ in range(40):  # zero-delta actions: does the arm stay where it was put?
            obs, _ = w.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        print(json.dumps(dict(
            task=item, instruction=w.instruction, shift_cm=round(float(np.linalg.norm(b.eef_pos - a.eef_pos)) * 100, 1),
            drift_40_steps_cm=round(float(np.linalg.norm(obs.eef_pos - hold)) * 100, 2), contacts=info["collision_contacts"],
            clipped=info.get("offset_clipped"), goal=w.goal_progress())), flush=True)
        panels.append(np.hstack([a.agent, b.agent, b.wrist]))
        w.close()
    Image.fromarray(np.vstack(panels)).resize((810, 270 * len(panels))).save(out_dir / "start_frames.jpg", quality=85)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
