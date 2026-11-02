import numpy as np

from dual_system.eval.atomic import phase_done as truth
from dual_system.finetune.collect_judge import TRAIN_TASKS, episodes, frame_rows, perturb
from dual_system.finetune.dataset import load_splits
from dual_system.finetune.segment import load_subtasks
from dual_system.finetune.system2 import balanced_order, choose_threshold, false_done_rate
from dual_system.system2.reasoners import Answer, QwenReasoner, parse_done, phase_done_prompt


def test_episode_splits_keep_scenes_and_init_states_apart():
    unseen = load_splits()["unseen_tasks"]
    all_episodes = episodes(unseen)
    assert 90 <= len(all_episodes) <= 110
    assert set(TRAIN_TASKS) <= set(load_subtasks())
    assert not set(TRAIN_TASKS) & set(unseen)
    by_split = {s: [e for e in all_episodes if e["split"] == s] for s in ("train", "val", "test")}
    assert not {e["init_state"] for e in by_split["train"]} & {e["init_state"] for e in by_split["test"]}
    assert not {e["init_state"] for e in by_split["train"]} & {e["init_state"] for e in by_split["val"]}
    assert set(unseen) <= {e["task"] for e in by_split["test"]}
    assert not set(unseen) & {e["task"] for e in by_split["train"] + by_split["val"]}
    assert all(e["mode"] != "gibberish" for e in by_split["val"])
    assert {e["mode"] for e in by_split["train"]} == {"real", "noise", "gibberish"}


PLACE = dict(kind="place", command="place the bowl", goal="bowl_1")
PICK = dict(kind="pick", command="pick up the bowl", goal="bowl_1")


def test_placed_object_must_be_released():
    goals = {"on bowl_1 stove": True}
    assert not truth(PLACE, goals, lambda name: True, {})
    assert truth(PLACE, goals, lambda name: False, {})
    assert not truth(PLACE, {"on bowl_1 stove": False}, lambda name: False, {})


def test_pick_needs_grip_and_lift():
    goals = {"on bowl_1 stove": False}
    assert truth(PICK, goals, lambda name: True, {"bowl_1": 0.05})
    assert not truth(PICK, goals, lambda name: True, {"bowl_1": 0.0})
    assert not truth(PICK, goals, lambda name: False, {"bowl_1": 0.05})


def test_rows_cover_active_and_next_phase():
    rows = frame_rows([PICK, PLACE], 0, [True, False])
    assert [(r["step"], r["done"], r["active"]) for r in rows] == [("pick up the bowl", True, True), ("place the bowl", False, False)]
    assert len(frame_rows([PICK, PLACE], 1, [True, False])) == 1


def test_perturb_opens_the_gripper_during_a_burst():
    rng = np.random.default_rng(0)
    action = np.array([0, 0, 0, 0, 0, 0, 1.0])
    out, left = perturb(action, 3, rng)
    assert out[6] == -1.0 and left == 2 and action[6] == 1.0


def test_balanced_order_repeats_the_small_class():
    rows = [dict(done=True)] * 2 + [dict(done=False)] * 10
    order = balanced_order(rows, np.random.default_rng(0))
    assert len(order) == 20 and sum(i < 2 for i in order) == 10


def test_threshold_respects_the_false_done_limit():
    probs = [0.9, 0.8, 0.7] + [0.75] + [0.1] * 16
    truths = [True, True, True] + [False] * 17
    t = choose_threshold(probs, truths, max_false_done=0.0)
    assert 0.75 < t <= 0.8
    rows = [dict(truth=x, answer=p >= t) for p, x in zip(probs, truths)]
    assert false_done_rate(rows) == 0


def test_parse_done():
    assert parse_done("Yes.") == {"step": "done"}
    assert parse_done(" no") == {"step": "running"}
    assert parse_done("maybe") is None
    assert "pick up the bowl" in phase_done_prompt("pick up the bowl", "held.")


class _Disable:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append("off")

    def __exit__(self, *a):
        self.log.append("on")


class _Model:
    def __init__(self, log):
        self.log = log

    def disable_adapter(self):
        return _Disable(self.log)


def _reasoner(adapter):
    r = QwenReasoner.__new__(QwenReasoner)
    r.log, r.adapter, r.phase_format, r.done_threshold = [], adapter, "yes_no", 0.7
    r.model = _Model(r.log)
    r.max_tokens_check, r.max_tokens_plan = 30, 120
    r._generate = lambda prompt, images, max_new, pid, parse=None: r.log.append(pid) or Answer({"valid": True}, "", pid)
    return r


def test_gate_and_plan_run_with_the_adapter_off():
    r = _reasoner("adapter")
    r.validate("x", [])
    r.plan("x", [], ["rule"], None, [])
    assert r.log == ["off", "validate_v1", "on", "off", "plan_v1", "on"]


def test_judge_uses_the_threshold_with_an_adapter():
    r = _reasoner("adapter")
    r._done_probability = lambda prompt, images: 0.6
    assert r.phase_check("x", ["a"], 0, "d", [], []).parsed == {"step": "running"}
    r._done_probability = lambda prompt, images: 0.8
    assert r.phase_check("x", ["a"], 0, "d", [], []).parsed == {"step": "done"}
