"""Regression tests for the horror-story failure (job 836d56d4):
1. A correct holder/location with extra state qualifiers must pass continuity.
2. A genuine holder/location mismatch must still fail.
3. A mandatory setup clue missing from its designated scene must be detected.
4. A correctly named clue and its planned payoff must pass.
"""
import json
from workers.prompts import index as P


def horror_plan():
    return {
        "locations": ["manor study"],
        "bible": {"characters": {"Elliot Voss": "gaunt man in a black coat"}, "style": "gothic horror", "negative": "no gore"},
        "clues": [{"name": "entity learns from speech", "setup": 1, "payoff": 5}],
        "outline": [
            {"index": 1, "moment": "Elliot finds the notebook", "end_positions": {"Elliot Voss": "manor study: desk"},
             "start_props": {"notebook": "Elliot Voss"}, "end_props": {"notebook": "Elliot Voss"}},
            {"index": 2, "moment": "The whispers begin", "end_positions": {"Elliot Voss": "manor study: desk"},
             "start_props": {"notebook": "Elliot Voss"}, "end_props": {"notebook": "Elliot Voss"}},
        ],
    }


def test_correct_holder_with_extra_state_qualifiers_passes(monkeypatch):
    got = {"end_positions": {"Elliot Voss": "manor study: desk"},
           "start_props": {"notebook": "Elliot Voss (left pocket, closed)"},
           "end_props": {"notebook": "Elliot Voss (left pocket, closed)"}, "evidence": {}}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *a: json.dumps(got))
    assert P.fact_issues("Elliot slips the closed notebook into his left pocket.", 1, horror_plan()) == []


def test_genuine_holder_mismatch_still_fails(monkeypatch):
    got = {"end_positions": {"Elliot Voss": "manor study: desk"},
           "start_props": {"notebook": "Mara Chen"},
           "end_props": {"notebook": "Mara Chen"}, "evidence": {}}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *a: json.dumps(got))
    issues = P.fact_issues("Mara keeps the notebook.", 1, horror_plan())
    assert any("notebook must begin with/at Elliot Voss" in x for x in issues)
    assert any("notebook must end with/at Elliot Voss" in x for x in issues)


def test_undeclared_location_still_fails(monkeypatch):
    got = {"end_positions": {"Elliot Voss": "garden: gate"},
           "start_props": {"notebook": "Elliot Voss"}, "end_props": {"notebook": "Elliot Voss"}, "evidence": {}}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *a: json.dumps(got))
    issues = P.fact_issues("Elliot walks to the garden gate.", 1, horror_plan())
    assert any("must end at manor study: desk" in x for x in issues)


def test_required_qualifier_missing_still_fails():
    assert not P._same_fact("Elliot Voss", "Elliot Voss (closed)")
    assert P._same_fact("Elliot Voss (left pocket, closed)", "Elliot Voss (closed)")


def _spec():
    return {"multi": True, "noun": "scene", "count": 5, "sections": ["Visual Prompt", "Dialogue"],
            "dialogue_range": None, "devanagari": False, "fixed_background": False, "duration": None}


def _plan_for_clues():
    return {"bible": {"characters": {}, "style": "s", "negative": "n"},
            "clues": [{"name": "entity learns from speech", "setup": 1, "payoff": 5}],
            "outline": [{"index": n, "moment": f"m{n}", "characters": []} for n in range(1, 6)]}


def _long_scene(body: str) -> str:
    return "# Scene 1: Arrival\n\n## Visual Prompt\n" + body + " " + ("The study is silent. " * 40) + "\n\n## Dialogue\nElliot: Hello?\n"


def test_missing_setup_clue_is_detected():
    text = _long_scene("Elliot enters the manor study and opens the notebook.")
    problems = P.check_unit(text, 1, _plan_for_clues(), _spec())
    assert any("clue 'entity learns from speech' must appear by name" in p for p in problems)


def test_named_clue_and_payoff_pass():
    plant = _long_scene("Elliot realises the entity learns from speech, and dreads every word he speaks.")
    assert not any("clue" in p for p in P.check_unit(plant, 1, _plan_for_clues(), _spec()))
    payoff = _long_scene("The entity learns from speech, so Elliot stays silent and survives.")
    assert not any("clue" in p for p in P.check_unit(payoff, 5, _plan_for_clues(), _spec()))
    middle = _long_scene("Elliot explores the cellar alone.")
    assert not any("clue" in p for p in P.check_unit(middle, 3, _plan_for_clues(), _spec()))


def test_writer_receives_due_clues_in_context():
    plan = _plan_for_clues()
    plan["blueprint"] = "b"
    msgs = P._unit_messages(1, plan, _spec(), "a horror story in 5 scenes")
    sys_msg, user_msg = msgs[0]["content"], msgs[1]["content"]
    assert "mandatory planned clue(s)" in sys_msg
    assert "'entity learns from speech'" in sys_msg
    ctx = json.loads(user_msg)
    assert ctx["clues_due"] == [{"name": "entity learns from speech", "role": "plant", "setup": 1, "payoff": 5}]
    # A unit with no clue due gets no extra instruction and an empty list.
    msgs3 = P._unit_messages(3, plan, _spec(), "a horror story in 5 scenes")
    assert "mandatory planned clue(s)" not in msgs3[0]["content"]
    assert json.loads(msgs3[1]["content"])["clues_due"] == []
