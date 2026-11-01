import numpy as np

from dual_system.knowledge import StaticRetriever, snap_to_candidates
from dual_system.system2.reasoners import Answer, ScriptedReasoner, parse_json
from dual_system.system2.supervisor import Supervisor
from dual_system.types import Observation, S2Input, TriggerEvent, Verdict

INSTR = "turn on the stove and put the moka pot on it"
RUN = {"step": "running", "task": "running"}
DONE = {"step": "done", "task": "running"}
FAIL = {"step": "failed", "task": "running"}
WIN = {"step": "done", "task": "success"}


def sup(checks, steps=("turn on the stove", "put the moka pot on the stove"), **kw):
    r = ScriptedReasoner(steps=list(steps), check_answers=checks)
    s = Supervisor(r, StaticRetriever("closed_set"), **kw)
    s.reset(INSTR)
    return s


def inp(t, reasons=("periodic",)):
    img = np.zeros((4, 4, 3), np.uint8)
    obs = Observation(t, {}, img, img, np.zeros(3), np.zeros(2))
    return S2Input(obs, INSTR, TriggerEvent(t, list(reasons)))


def test_start_issues_first_planned_step():
    d = sup([RUN]).decide(inp(0, ["start"]))
    assert d.command.text == "turn on the stove" and d.command.kind == "task"


def test_gate_rejects_and_issues_no_command():
    s = Supervisor(ScriptedReasoner(valid=False), StaticRetriever("closed_set"))
    s.reset("qxv lorp zindle")
    d = s.decide(inp(0, ["start"]))
    assert d.verdict == Verdict.REJECTED and d.command is None


def test_gate_off_passes_nonsense():
    s = Supervisor(ScriptedReasoner(valid=False), StaticRetriever("closed_set"), gate=False, decompose=False)
    s.reset("qxv lorp zindle")
    assert s.decide(inp(0, ["start"])).command.text == "qxv lorp zindle"


def test_decompose_off_uses_instruction():
    assert sup([RUN], decompose=False).decide(inp(0, ["start"])).command.text == INSTR


def test_one_done_is_not_enough_two_advance():
    s = sup([DONE, DONE])
    s.decide(inp(0, ["start"]))
    assert s.decide(inp(20)).command is None
    d = s.decide(inp(40))
    assert d.command.text == "put the moka pot on the stove" and d.closes == "done"


def test_flicker_resets_confirmation():
    s = sup([DONE, RUN, DONE])
    s.decide(inp(0, ["start"]))
    assert all(s.decide(inp(t)).command is None for t in (20, 40, 60))


def test_last_step_done_ends_with_success_when_judging():
    s = sup([DONE, DONE, DONE, DONE])
    s.decide(inp(0, ["start"]))
    results = [s.decide(inp(t)) for t in (60, 80, 100, 120)]
    assert results[-1].verdict == Verdict.SUCCESS


def test_judge_off_never_ends_episode():
    s = sup([WIN], judge=False)
    s.decide(inp(0, ["start"]))
    assert all(s.decide(inp(t)).verdict is None for t in range(60, 400, 20))


def test_no_verdict_before_min_step():
    s = sup([WIN], min_verdict_step=100)
    s.decide(inp(0, ["start"]))
    assert s.decide(inp(20)).verdict is None and s.decide(inp(40)).verdict is None
    s.decide(inp(80))
    assert s.decide(inp(100)).verdict == Verdict.SUCCESS


def test_recovery_is_language_then_retry():
    s = sup([FAIL, FAIL, RUN], recovery_steps=40)
    s.decide(inp(0, ["start"]))
    s.decide(inp(20))
    d = s.decide(inp(40))
    assert (d.command.kind, d.command.text, d.closes) == ("recovery", "open the gripper", "failed")
    assert s.decide(inp(60)).command is None        # still within recovery_steps
    d = s.decide(inp(80))
    assert (d.command.kind, d.command.text) == ("recovery", "move back to the home position")
    d = s.decide(inp(120))
    assert d.command.kind == "task" and d.command.retry == 1 and d.command.text == "turn on the stove"


def test_retry_budget_then_failure_verdict():
    s = sup([FAIL], max_retries=1, recovery_steps=20)
    s.decide(inp(0, ["start"]))
    verdict, t = None, 20
    while verdict is None and t < 2000:
        verdict = s.decide(inp(t)).verdict
        t += 20
    assert verdict == Verdict.FAILURE and s.retries == 1


def test_recover_off_goes_straight_to_failure():
    s = sup([FAIL], recover=False)
    s.decide(inp(0, ["start"]))
    s.decide(inp(40))
    assert s.decide(inp(60)).verdict == Verdict.FAILURE


def test_timeout_counts_as_failure_at_once():
    s = sup([RUN])
    s.decide(inp(0, ["start"]))
    assert s.decide(inp(300, ["subgoal_timeout"])).command.kind == "recovery"


def test_malformed_answer_keeps_current_command():
    class Broken(ScriptedReasoner):
        def check(self, *a):
            return Answer(parse_json("sure! the robot looks fine"), "sure! the robot looks fine", "x")

    s = Supervisor(Broken(steps=["turn on the stove"]), StaticRetriever("closed_set"))
    s.reset(INSTR)
    s.decide(inp(0, ["start"]))
    d = s.decide(inp(20))
    assert d.command is None and d.verdict is None and d.raw


def test_unusable_plan_falls_back_to_instruction():
    class BadPlan(ScriptedReasoner):
        def plan(self, *a):
            return Answer(None, "no json")

    s = Supervisor(BadPlan(), StaticRetriever("closed_set"))
    s.reset(INSTR)
    assert s.decide(inp(0, ["start"])).command.text == INSTR


def test_parse_json_handles_code_fence():
    assert parse_json('```json\n{"step": "done"}\n```') == {"step": "done"}
    assert parse_json("{broken") is None


def test_snap_to_candidates():
    c = ["turn on the stove", "put the moka pot on the stove"]
    assert snap_to_candidates(["Turn on the stove.", "fly to the moon", "put moka pot on the stove"], c) == c
    assert snap_to_candidates(["anything"], None) == ["anything"]


def test_prompts_hold_no_vocabulary():
    """Command words must come from the profile, so semantics can be swapped without code changes."""
    from dual_system.system2 import reasoners

    for tpl in (reasoners.VALIDATE_V1, reasoners.PLAN_V1, reasoners.CHECK_V1):
        for word in ("stove", "bowl", "drawer", "move back", "open the gripper"):
            assert word not in tpl
