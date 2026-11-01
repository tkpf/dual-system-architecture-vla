"""Print what System 2 did in an episode, in time order.

    python -m dual_system.eval.lookup <res> [--suite libero_10] [--task 0] [--state B] [--condition full] [--calls]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_cells(res: str | Path, suite=None, task=None, state=None, condition=None) -> list[Path]:
    out = []
    for d in sorted(p for p in Path(res).iterdir() if (p / "trace.jsonl").exists()):
        parts = d.name.split("__")
        if len(parts) == 4 and (suite in (None, parts[0])) and (task in (None, int(parts[1][1:]))) \
                and (state in (None, parts[2])) and (condition in (None, parts[3])):
            out.append(d)
    return out


def timeline(cell: Path, calls: bool = False) -> list[str]:
    """calls=True also lists routine System 2 checks that changed nothing."""
    lines = [f"== {cell.name}"]
    for raw in open(cell / "trace.jsonl", encoding="utf-8"):
        e = json.loads(raw)
        t, kind = e["t"], e["type"]
        if kind == "episode_start":
            lines.append(f"{t:4d} start   \"{e['instruction']}\" reset={e['reset_info']}")
        elif kind == "subgoal_open":
            lines.append(f"{t:4d} COMMAND [{e['kind']} #{e['subgoal_index']} retry {e['retry']} replan {e['replan']}] \"{e['text']}\"")
        elif kind == "subgoal_close":
            lines.append(f"{t:4d} closed  {e['outcome']}: \"{e['text']}\"")
        elif kind == "s2_call" and (calls or e["verdict"] or e["command"]):
            lines.append(f"{t:4d} s2      {','.join(e['reasons'])} -> {e['parsed']} ({e['latency_s']}s) goal={e['goal_progress']} "
                         f"env_success={e['env_success']} dist_to_start={e['dist_to_start']}"
                         + (f" VERDICT={e['verdict']}" if e["verdict"] else ""))
        elif kind == "episode_end":
            lines.append(f"{t:4d} end     success={e['success']} steps={e['n_steps']} first_success={e['first_success_step']} "
                         f"verdict={e['verdict']} reason={e['end_reason']} s2_calls={e['s2_calls']} wall={e['wall_s']}s")
    return lines


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("res")
    p.add_argument("--suite")
    p.add_argument("--task", type=int)
    p.add_argument("--state")
    p.add_argument("--condition")
    p.add_argument("--calls", action="store_true")
    a = p.parse_args()
    for c in find_cells(a.res, a.suite, a.task, a.state, a.condition):
        print("\n".join(timeline(c, a.calls)))
