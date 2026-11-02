from dual_system.eval.judge_prompts import STYLES, auroc, balanced, build, cross_fit, parse_answer, parse_step
from dual_system.finetune.segment import load_subtasks


class _Retriever:
    def phase_definition(self, step):
        return "it is done."


def test_every_subtask_parses_and_builds_every_style():
    commands = {c for groups in load_subtasks().values() for g in groups for c in (g.get("act"), g.get("pick"), g.get("place")) if c}
    for c in commands:
        assert parse_step(c)["kind"] != "other", c
        for style in STYLES:
            spec = build(c, style, _Retriever())
            assert "{" not in spec["prompt"], (c, style)
    assert parse_step("place the other moka pot on the stove") == dict(kind="place", obj="moka pot", rel="on", target="the stove")


def test_negated_question_counts_no_as_finished():
    assert build("open the top drawer", "negated", _Retriever())["done"] == "no"
    assert build("pick up the bowl", "choice", _Retriever())["done"] == "C"


def test_answers_parse():
    assert parse_answer("Yes.", "yesno") == "yes"
    assert parse_answer("B) the bowl", "choice") == "B"
    assert parse_answer("The gripper holds the bowl.\nANSWER: no", "describe") == "no"
    assert parse_answer("unclear", "yesno") is None


def test_auroc_and_cross_fit():
    assert auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert balanced([True, None, False, False], [True, True, False, False]) == 0.75
    rows = [dict(episode=f"ep{i}", prob=0.02 + 0.01 * (i % 5) + (0.05 if t else 0), truth=t) for i in range(40) for t in (True, False)]
    acc, false_done = cross_fit(rows)
    assert acc > 0.7 and false_done < 0.3  # a model that always leans to "no" but ranks well is recovered by the threshold
