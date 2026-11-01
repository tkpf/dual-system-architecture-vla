"""The perception benchmark's labels and scores, no model or simulator needed."""
import json

from dual_system.eval import perception as pc
from dual_system.knowledge import StaticRetriever


def row(truth, answer, question="stove_on", episode="e0"):
    return dict(model="m", truth=truth, answer=answer, question=question, episode=episode, mode="both", latency_s=0.1)


def test_held_needs_lift_grip_and_nearness():
    assert pc.is_held(0.10, 0.005, 0.05)
    assert not pc.is_held(0.01, 0.005, 0.05)  # still on the table
    assert not pc.is_held(0.10, 0.08, 0.05)  # resting high up, gripper open
    assert not pc.is_held(0.10, 0.005, 0.30)  # gripper elsewhere


def test_parse_yes_no():
    assert pc.parse_yes_no("Yes.") == {"yes": True}
    assert pc.parse_yes_no(" no") == {"yes": False}
    assert pc.parse_yes_no("The stove is on") is None


def test_always_yes_scores_chance_and_unusable_counts_as_wrong():
    rows = [row(True, True), row(False, True)]
    assert pc.balanced_accuracy(rows) == 0.5
    assert pc.balanced_accuracy([row(True, True), row(False, False)]) == 1.0
    assert pc.balanced_accuracy([row(True, None), row(False, False)]) == 0.5


def test_macro_score_weighs_questions_equally():
    rows = [row(True, True), row(False, False)] * 5 + [row(True, False, "bowl_held"), row(False, True, "bowl_held")]
    assert pc.macro_score(rows) == 0.5


def test_every_scene_question_exists_and_prompts_differ_by_mode():
    r = StaticRetriever("phase_atomic")
    for names in pc.SCENES.values():
        for name in names:
            question, phase, kind, _ = pc.QUESTIONS[name]
            assert kind in ("goal", "held")
            assert "wrist" not in pc.prompt_for(question, phase, "scene", r)
            assert "finished only when" in pc.prompt_for(question, phase, "definition", r)


def test_report_runs_on_answers(tmp_path):
    rows = [dict(row(True, True, episode=f"e{i}"), model="qwen2b") for i in range(3)] + [dict(row(False, False, episode=f"e{i}"), model="qwen2b") for i in range(3)]
    (tmp_path / "answers_qwen2b.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    text = pc.report(tmp_path)
    assert "score 1.00" in text and (tmp_path / "perception_report.txt").exists()
