import json
from workers.prompts import index as P


def spatial_plan():
    return {"locations": ["classroom"], "bible": {"characters": {"Monkey": "brown monkey in red shirt"}, "style": "animation", "negative": "no text"}, "outline": [{"index": 1, "moment": "Share homework", "end_positions": {"Monkey": "classroom: center-right"}, "start_props": {"sheet": "Rabbit"}, "end_props": {"sheet": "Monkey"}}]}


def test_extractor_schema_preserves_position_and_state():
    assert "full room ID AND within-room spot" in P.EXTRACT_SYS
    assert "full holder/location AND state" in P.EXTRACT_SYS
    assert "Never copy a contract value just because it is expected" in P.EXTRACT_SYS


def test_room_only_result_does_not_pass_detailed_position(monkeypatch):
    got = {"end_positions": {"Monkey": "classroom"}, "start_props": {"sheet": "Rabbit"}, "end_props": {"sheet": "Monkey"}, "evidence": {"sheet": "Rabbit hands the sheet to Monkey."}}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *args: json.dumps(got))
    issues = P.fact_issues("Rabbit hands the sheet to Monkey.", 1, spatial_plan())
    assert any("must end at classroom: center-right" in x for x in issues)
    got["end_positions"]["Monkey"] = "classroom: center-right"
    assert P.fact_issues("Rabbit hands the sheet to Monkey.", 1, spatial_plan()) == []


def test_metadata_cannot_supply_transfer_evidence(monkeypatch):
    quote = "Rabbit hands the sheet to Monkey."
    got = {"end_positions": {"Monkey": "classroom: center-right"}, "start_props": {"sheet": "Rabbit"}, "end_props": {"sheet": "Monkey"}, "evidence": {"sheet": quote}}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *args: json.dumps(got))
    text = "## Visual Prompt\nMonkey waits.\n## Continuity\n**Canonical state at scene end:**\n" + quote
    assert any("Scene Actions must show sheet" in x for x in P.fact_issues(text, 1, spatial_plan()))


def test_missing_quote_or_wrong_holder_still_fails(monkeypatch):
    got = {"end_positions": {"Monkey": "classroom: center-right"}, "start_props": {"sheet": "Rabbit"}, "end_props": {"sheet": "Rabbit"}, "evidence": {"sheet": ""}}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *args: json.dumps(got))
    issues = P.fact_issues("Monkey looks at Rabbit's sheet.", 1, spatial_plan())
    assert any("sheet must end with/at Monkey" in x for x in issues)
    assert any("Scene Actions must show sheet" in x for x in issues)


def test_wrong_spot_and_missing_prop_state_still_fail(monkeypatch):
    p = spatial_plan(); p["outline"][0]["end_props"]["sheet"] = "Monkey (open)"
    got = {"end_positions": {"Monkey": "classroom: center-left"}, "start_props": {"sheet": "Rabbit"}, "end_props": {"sheet": "Monkey"}, "evidence": {"sheet": "Rabbit hands the open sheet to Monkey."}}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *args: json.dumps(got))
    issues = P.fact_issues("Rabbit hands the open sheet to Monkey.", 1, p)
    assert any("must end at classroom: center-right" in x for x in issues)
    assert any("sheet must end with/at Monkey (open)" in x for x in issues)
    got["end_positions"]["Monkey"] = "classroom: center-right"
    got["end_props"]["sheet"] = "Monkey (open)"
    assert P.fact_issues("Rabbit hands the open sheet to Monkey.", 1, p) == []


def test_invalid_extraction_is_unverified_not_crash(monkeypatch):
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *args: '{"end_positions":[]}')
    assert P.fact_issues("Monkey waits.", 1, spatial_plan()) == ["Independent fact extraction returned invalid data"]


def test_spatial_splice_preserves_dialogue_exactly():
    dialogue = "## Dialogue\nMonkey: " + "homework " * 27
    draft = "# Scene 1: Homework\n\n## Visual Prompt\nOLD\n\n" + dialogue
    repaired = P._splice_visual_repair(draft, "## Visual Prompt\nRabbit hands the sheet to Monkey.")
    assert repaired.endswith(dialogue)
    assert "OLD" not in repaired
    assert P.dialogue_words(repaired) == 27
    assert P._splice_visual_repair(draft, "## Visual Prompt\nNEW\n## Dialogue\nChanged") == draft


def test_spatial_prompt_requests_only_visual_repair():
    p = spatial_plan(); p["blueprint"] = "Schoolwork"; p["clues"] = []
    spec = {"noun": "scene", "count": 1, "multi": True, "sections": ["Visual Prompt", "Dialogue"], "devanagari": False}
    messages = P._unit_messages(1, p, spec, "School", ["Scene Actions must show sheet reaching Monkey on screen (no sentence shows it)"], "## Visual Prompt\nOLD\n## Dialogue\nMonkey: Hello.")
    assert messages[-1]["content"].startswith("Fix ONLY the Visual Prompt section")
    assert "Do not alter dialogue or the plan" in messages[-1]["content"]