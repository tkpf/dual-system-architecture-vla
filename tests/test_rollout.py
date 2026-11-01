"""The control loop and the video/report tools on a fake world, no simulator needed."""
import json
from collections import deque

import numpy as np
import pytest

from dual_system.env import World
from dual_system.eval import compare_probe, lookup, report, side_by_side
from dual_system.eval.grid import build_cells
from dual_system.interface import Identity
from dual_system.knowledge import StaticRetriever
from dual_system.robot_states import RobotState
from dual_system.rollout import RolloutRunner
from dual_system.system1.base import System1
from dual_system.system1.lerobot_policy import clear_action_queue
from dual_system.system2 import Passthrough, Supervisor
from dual_system.system2.reasoners import ScriptedReasoner
from dual_system.tracing import TraceWriter
from dual_system.triggers import GripperToggle, Periodic, TriggerManager
from dual_system.types import Observation, RunSpec

INSTR = "turn on the stove and put the moka pot on it"


class FakeWorld(World):
    def __init__(self, success_at=None):
        self.success_at, self.t = success_at, 0

    def _obs(self):
        img = np.full((64, 64, 3), self.t % 255, np.uint8)
        return Observation(self.t, {}, img, img, np.array([self.t * 0.001, 0, 0]), np.zeros(2) + 0.02)

    def reset(self, state, seed):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        return self._obs(), self.success_at is not None and self.t >= self.success_at


class FakeSystem1(System1):
    def __init__(self):
        self.log = []
        self.command = ""

    def reset(self):
        self.log, self.command = [], ""

    def set_command(self, text):
        changed = text != self.command
        self.command = text
        self.log.append(text)
        return changed

    def act(self, obs):
        return np.array([0.5, 0, 0, 0, 0, 0, -1.0])


def spec(condition, res, task=0, state="A", max_steps=100):
    cell = f"libero_10__t{task}__{state}__{condition}"
    return cell, RunSpec("libero_10", task, state, condition, 7, INSTR, max_steps, run="test", cell=cell)


def run(tmp_path, condition, system2, world, triggers=None, max_steps=100, video=True):
    cell, s = spec(condition, tmp_path, max_steps=max_steps)
    tracer = TraceWriter(tmp_path / cell, s, video=video)
    s1 = FakeSystem1()
    r = RolloutRunner(world, s1, system2, Identity(), triggers or TriggerManager([Periodic(20)]), tracer).run(s, RobotState("A"))
    (tmp_path / cell / "DONE").write_text("ok")
    return r, s1


def full(checks, **kw):
    return Supervisor(ScriptedReasoner(steps=["turn on the stove", "put the moka pot on the stove"], check_answers=checks),
                      StaticRetriever("closed_set"), **kw)


def test_passthrough_ends_at_env_success(tmp_path):
    r, s1 = run(tmp_path, "passthrough", Passthrough(), FakeWorld(success_at=30), TriggerManager([]))
    assert r.success and r.n_steps == 30 and r.end_reason == "env_success" and r.verdict is None
    assert s1.log == [INSTR] and r.s2_calls == 1


def test_judge_keeps_running_after_env_success(tmp_path):
    win = {"step": "done", "task": "success"}
    run_ = {"step": "running", "task": "running"}
    r, _ = run(tmp_path, "full", full([run_, run_, run_, win, win]), FakeWorld(success_at=30))
    assert r.first_success_step == 30 and r.n_steps == 100 and r.verdict == "success" and r.success
    assert r.end_reason == "s2_verdict"


def test_false_done_scores_as_failure(tmp_path):
    win = {"step": "done", "task": "success"}
    r, _ = run(tmp_path, "full", full([win]), FakeWorld(success_at=None))
    assert r.verdict == "success" and not r.success
    fs = report.false_successes(tmp_path)
    assert list(fs.cell) == ["libero_10__t0__A__full"] and fs.step[0] == r.n_steps  # System 2 still ended the episode
    assert (tmp_path / fs.keyframe[0]).exists()


def test_time_limit_scores_the_predicate(tmp_path):
    r, _ = run(tmp_path, "full", full([{"step": "running", "task": "running"}]), FakeWorld(success_at=50), max_steps=70)
    assert r.end_reason == "time_limit" and r.success and r.verdict is None


def test_rejection_runs_no_steps(tmp_path):
    s2 = Supervisor(ScriptedReasoner(valid=False), StaticRetriever("closed_set"))
    r, s1 = run(tmp_path, "full", s2, FakeWorld())
    assert r.verdict == "rejected" and r.n_steps == 0 and s1.log == []


def test_trace_and_subgoals_recorded(tmp_path):
    done = {"step": "done", "task": "running"}
    run(tmp_path, "full", full([done, done, {"step": "running", "task": "running"}]), FakeWorld(), max_steps=120)
    cell = tmp_path / "libero_10__t0__A__full"
    rows = [json.loads(l) for l in open(cell / "subgoals.jsonl")]
    assert [(r["text"], r["outcome"]) for r in rows] == [("turn on the stove", "done"),
                                                         ("put the moka pot on the stove", "abandoned")]
    assert rows[0]["finish_step"] == 40 and rows[0]["eef_travel"] > 0 and rows[0]["status_history"]
    calls = [json.loads(l) for l in open(cell / "trace.jsonl")]
    call = next(c for c in calls if c["type"] == "s2_call" and c["t"] == 20)
    assert call["reasons"] == ["periodic"] and (cell / call["keyframes"]["agent"]).exists()
    text = "\n".join(lookup.timeline(cell))
    assert "turn on the stove" in text and "end" in text


def test_trace_is_on_disk_before_the_episode_ends(tmp_path):
    cell, s = spec("full", tmp_path)
    tracer = TraceWriter(tmp_path / cell, s, video=False)

    class Crash(FakeSystem1):
        def act(self, obs):
            if obs.t == 25:
                raise RuntimeError("runtime died")
            return super().act(obs)

    with pytest.raises(RuntimeError):
        RolloutRunner(FakeWorld(), Crash(), full([{"step": "running", "task": "running"}]), Identity(),
                      TriggerManager([Periodic(20)]), tracer).run(s, RobotState("A"))
    rows = [json.loads(l) for l in open(tmp_path / cell / "trace.jsonl")]
    assert any(r["type"] == "subgoal_open" for r in rows) and any(r["t"] == 20 for r in rows)


def test_gripper_trigger_reaches_system2(tmp_path):
    class Flip(FakeSystem1):
        def act(self, obs):
            return np.array([0.5, 0, 0, 0, 0, 0, 1.0 if obs.t >= 10 else -1.0])

    cell, s = spec("full", tmp_path, max_steps=30)
    tracer = TraceWriter(tmp_path / cell, s, video=False)
    RolloutRunner(FakeWorld(), Flip(), full([{"step": "running", "task": "running"}]), Identity(),
                  TriggerManager([GripperToggle(3)]), tracer).run(s, RobotState("A"))
    rows = [json.loads(l) for l in open(tmp_path / cell / "trace.jsonl")]
    assert any(r["type"] == "s2_call" and r["reasons"] == ["gripper_toggle"] for r in rows)


def test_report_pairs_and_video(tmp_path):
    run(tmp_path, "passthrough", Passthrough(), FakeWorld(success_at=30), TriggerManager([]))
    done = {"step": "done", "task": "running"}
    run(tmp_path, "full", full([done, done, done, done]), FakeWorld(success_at=70), max_steps=120)
    df = report.load_episodes(tmp_path)
    pairs = report.pairs_table(df)
    assert len(pairs) == 1 and bool(pairs.single_success[0]) and bool(pairs.dual_success[0])
    assert report.verdict_quality(df)["correct"] == 1
    assert report.write_subgoals_csv(tmp_path).exists()
    videos = side_by_side.render_all(tmp_path)
    assert len(videos) == 1 and videos[0].stat().st_size > 0
    import imageio.v2 as imageio

    reader = imageio.get_reader(videos[0])
    frames = [f for f in reader]
    reader.close()
    assert frames[0].shape[1] == 2 * 64 + side_by_side.GAP
    assert len(frames) >= 41  # runs to the longer episode, then holds the final frame


def test_seeds_per_cell_keeps_pairs_and_adds_a_name_part(tmp_path):
    cfg = dict(suites=[dict(name="libero_goal", tasks=[0])], robot_states=dict(A={}), conditions=["passthrough", "full"],
               base_seed=1000, seeds_per_cell=2)
    cells = build_cells(cfg)
    assert [c.name for c in cells] == ["libero_goal__t0__A__s0__passthrough", "libero_goal__t0__A__s0__full",
                                       "libero_goal__t0__A__s1__passthrough", "libero_goal__t0__A__s1__full"]
    assert cells[0].seed == cells[1].seed != cells[2].seed == cells[3].seed
    for c in cells:
        (tmp_path / c.name).mkdir()
        (tmp_path / c.name / "trace.jsonl").write_text("")
    assert len(lookup.find_cells(tmp_path, task=0, state="A", condition="full")) == 2


def test_pairs_share_seed_and_order():
    cfg = dict(suites=[dict(name="libero_goal", tasks=[0, 1])], robot_states=dict(A={}, B={}),
               conditions=["passthrough", "full"], base_seed=1000)
    cells = build_cells(cfg)
    assert len(cells) == 8
    assert cells[0].condition == "passthrough" and cells[1].condition == "full" and cells[0].seed == cells[1].seed
    assert cells[0].seed != cells[2].seed
    assert [c.task_id for c in cells] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_clear_action_queue_empties_the_queue_the_policy_reads():
    class Smol:
        _queues = {"action": deque([1, 2])}

    class Pi:
        _queues = {"action": deque([1])}
        _action_queue = deque([1, 2])

    clear_action_queue(Smol, "action")
    clear_action_queue(Pi, "action")
    assert not Smol._queues["action"]
    assert not Pi._action_queue


def test_compare_probe_stitches_episodes_present_in_both_runs(tmp_path):
    import imageio.v2 as imageio

    def run(name, steps, first_success, extra=None):
        res = tmp_path / name
        (res / "videos").mkdir(parents=True)
        rows = []
        for video in ["a.mp4"] + ([extra] if extra else []):
            imageio.mimwrite(res / "videos" / video, [np.zeros((64, 64, 3), np.uint8)] * steps, fps=20, codec="libx264", macro_block_size=1)
            rows.append(dict(video=video, suite="libero_goal", task=1, robot_state="C", start_shift_cm=9.4, seed=1, checkpoint=f"lerobot/{name}",
                             command="pick > place", first_success_step=first_success, trace=[dict(t=steps)],
                             segments=[dict(command="pick", start_step=0), dict(command="place", start_step=2)]))
        (res / "probe.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return res

    left, right = run("pi05", 4, 3, "only_left.mp4"), run("smolvla", 6, None)
    assert compare_probe.command_at(compare_probe.load_runs(left)["a.mp4"], 2) == "place"
    made = compare_probe.render_all(left, right, tmp_path / "out")
    assert [m.name for m in made] == ["a.mp4"]
    assert imageio.get_reader(made[0]).count_frames() == 6 + 40  # the longer side, then the held last frame
    assert (tmp_path / "out" / "all_episodes.mp4").exists()
