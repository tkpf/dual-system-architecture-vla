"""Result tables. Runs on Colab or on the local mirror; incomplete pairs are reported and left out of the comparison.

    python -m dual_system.eval.report <res>
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_episodes(res: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for f in sorted(Path(res).glob("*/episodes.jsonl")) for line in open(f, encoding="utf-8")]
    return pd.DataFrame(rows)


def load_subgoals(res: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for f in sorted(Path(res).glob("*/subgoals.jsonl")) for line in open(f, encoding="utf-8")]
    return pd.DataFrame(rows)


def write_subgoals_csv(res: str | Path) -> Path | None:
    df = load_subgoals(res)
    if df.empty:
        return None
    out = Path(res) / "subgoals.csv"
    df["video"] = df["cell"] + "/episode.mp4"
    df.to_csv(out, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return out


def success_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["suite", "robot_state", "condition"]).agg(
        n=("success", "size"), k=("success", "sum"), mean_steps=("n_steps", "mean"),
        mean_s2_calls=("s2_calls", "mean"), mean_wall_s=("wall_s", "mean")).reset_index()
    g["rate"] = g["k"] / g["n"]
    g[["ci_lo", "ci_hi"]] = [wilson(int(k), int(n)) for k, n in zip(g["k"], g["n"])]
    return g.round(2)


def pairs_table(df: pd.DataFrame, a: str = "passthrough", b: str = "full") -> pd.DataFrame:
    """One row per complete pair; dual_only / single_only are the discordant pairs a sign test uses."""
    keys = ["suite", "task_id", "robot_state"]
    pa = df[df.condition == a].set_index(keys)
    pb = df[df.condition == b].set_index(keys)
    idx = pa.index.intersection(pb.index)
    out = pd.DataFrame({"single_success": pa.loc[idx, "success"], "dual_success": pb.loc[idx, "success"],
                        "single_steps": pa.loc[idx, "n_steps"], "dual_steps": pb.loc[idx, "n_steps"],
                        "dual_verdict": pb.loc[idx, "verdict"], "dual_first_success": pb.loc[idx, "first_success_step"]})
    return out.reset_index()


def verdict_quality(df: pd.DataFrame) -> dict:
    """How often System 2's verdict matches the env predicate."""
    d = df[df.verdict.notna()]
    false_done = int(((d.verdict == "success") & ~d.success).sum())
    false_failed = int(((d.verdict.isin(["failure", "rejected"])) & d.success).sum())
    correct = int((((d.verdict == "success") & d.success) | ((d.verdict != "success") & ~d.success)).sum())
    return dict(with_verdict=len(d), correct=correct, false_done=false_done, false_failed=false_failed,
                no_verdict=int(df[df.condition != "passthrough"].verdict.isna().sum()))


def false_successes(res: str | Path) -> pd.DataFrame:
    """Episodes where System 2 said success while the LIBERO condition was false: the last call's state and frames."""
    rows = []
    for f in sorted(Path(res).glob("*/trace.jsonl")):
        events = [json.loads(line) for line in open(f, encoding="utf-8")]
        end = next((e for e in events if e["type"] == "episode_end"), None)
        if end is None or end["verdict"] != "success" or end["success"]:
            continue
        call = [e for e in events if e["type"] == "s2_call"][-1]
        rows.append(dict(cell=end["cell"], instruction=end["instruction"], step=end["n_steps"], command=call["active"],
                         goal_progress=json.dumps(call["goal_progress"]), raw=call["raw"],
                         keyframe=f"{end['cell']}/{call['keyframes'].get('agent', '')}",
                         wrist_keyframe=f"{end['cell']}/{call['keyframes'].get('wrist', '')}",
                         video=f"{end['cell']}/episode.mp4"))
    return pd.DataFrame(rows)


def main(res: str) -> None:
    df = load_episodes(res)
    if df.empty:
        print("no finished episodes under", res)
        return
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(success_table(df).to_string(index=False), "\n")
    pairs = pairs_table(df)
    print(pairs.to_string(index=False), "\n")
    s, d = pairs.single_success.astype(bool), pairs.dual_success.astype(bool)
    print(f"complete pairs: {len(pairs)}; dual only: {int((d & ~s).sum())}; single only: {int((s & ~d).sum())}; "
          f"both: {int((s & d).sum())}; neither: {int((~s & ~d).sum())}")
    print("verdicts:", verdict_quality(df))
    print("subgoals:", write_subgoals_csv(res))
    fs = false_successes(res)
    print(f"System 2 said success, LIBERO false: {len(fs)}")
    if not fs.empty:
        fs.to_csv(Path(res) / "false_success.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(fs[["cell", "step", "command", "goal_progress", "keyframe"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
