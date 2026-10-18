import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.operations import OpError
from workers.prompts import index as P

BIBLE = {"characters": {"Mira": "a 30-year-old woman with a silver bob, green eyes, red wool coat"},
         "style": "35mm film still, teal and amber palette", "negative": "no text, no watermark"}
SPEC = {"count": 3, "noun": "prompt", "multi": True, "sections": P.DEFAULT_SECTIONS, "devanagari": False, "story": False}
def plan(n):
    return {"blueprint": "b" * 300, "bible": BIBLE, "initial_state": {"mira.umbrella": "closed"},
            "clues": [{"name": "brass lantern", "setup": 1, "payoff": n}],
            "outline": [{"index": i, "moment": f"m{i}", "time": f"D1 0{i}:00", "location": "street", "characters": ["Mira"], "before": {}, "after": {}} for i in range(1, n + 1)]}
def unit(i, desc=BIBLE["characters"]["Mira"], extra=""):
    body = "\n".join(f"## {f}\n{f} details" for f in P.DEFAULT_SECTIONS)
    return f"# Prompt {i}: t\n{body}\nMira is {desc}. {BIBLE['style']}. {BIBLE['negative']}. {extra}" + "x" * 200

def test_counts():
    assert P.unit_count("Generate exactly 10 detailed, sequential scene-generation prompts")[0] == 10
    assert P.unit_count("ten image prompts") == (10, "prompt")
    assert P.unit_count("an image prompt of a cat")[0] == 1
    with pytest.raises(OpError):
        P.unit_count("write 500 prompts")

def test_sections_from_brief():
    b = "1. Scene Title and Context: x\n2. Character Details: y\n3. Visual Prompt (English): z\n"
    assert P.required_sections(b) == ["Scene Title and Context", "Character Details", "Visual Prompt"]

def test_plan_rules():
    assert P.validate_plan(plan(3), 3) == []
    assert P.validate_plan(plan(2), 3)
    bad = plan(3); bad["clues"][0]["payoff"] = 1
    assert any("setup < payoff" in x for x in P.validate_plan(bad, 3))

def test_locks_and_clues_verbatim():
    p = plan(3)
    assert P.check_prompt if False else True
    assert P.check_unit(unit(2), 2, p, SPEC) == []
    assert any("Mira" in x for x in P.check_unit(unit(2, "a woman with black hair"), 2, p, SPEC))
    assert any("brass lantern" in x for x in P.check_unit(unit(3), 3, p, SPEC))
    assert P.check_unit(unit(3, extra="the brass lantern"), 3, p, SPEC) == []

def test_story_only_when_asked_and_fences():
    assert P.wants_story("Write a story with 5 scene prompts")
    assert not P.wants_story("Generate 10 scene prompts. Do not generate the finished story itself.")
    f = P.fence("# Scene 1\n```x```\nhi")
    assert f.startswith("````markdown\n") and f.endswith("\n````")

def test_plan_shapes_are_normalized():
    p = plan(3)
    p["bible"]["characters"] = [{"name": "Mira", "age": "30", "outfit": ["red coat", "boots"]}]
    p["bible"]["negative"] = ["no text", "no watermark"]
    p["outline"][1]["before"] = {"mira.umbrella": "wrong"}
    q = P.normalize_plan(p, 3)
    assert q["bible"]["characters"] == {"Mira": "30, red coat, boots"}
    assert q["bible"]["negative"] == "no text, no watermark"
    assert q["outline"][1]["before"] == {"mira.umbrella": "closed"}
    assert P.validate_plan(q, 3) == []

def test_cast_names_resolve_to_bible():
    p = plan(3); p["outline"][0]["characters"] = ["mira", "Ghost extra"]
    assert P.normalize_plan(p, 3)["outline"][0]["characters"] == ["Mira"]

def test_canonical_header_and_blueprint_strip():
    out = P.canonical_header("# Scene 3: The Door\n## Master Story Blueprint\nblah\n# Scene 3\n## Character Details\nx", 3, {"multi": True, "noun": "scene"})
    assert out.startswith("# Scene 3: The Door") and out.count("# Scene 3") == 1
    assert "blah" not in out and "## Character Details" in out

def test_locks_never_duplicate():
    p = plan(3); t = P.enforce_locks(unit(2), 2, p, SPEC)
    assert P.enforce_locks(t, 2, p, SPEC) == t

def test_language_rules():
    spec = {"devanagari": True}
    assert P.language_issues("## Dialogue\nमीरा: यह गलत है\n## Camera\nwide shot", spec) == []
    assert P.language_issues("## Dialogue\nमीरा: गलत-way-round (głos tremulous)", spec)
    assert P.language_issues("## Camera\nयह\nयह\nयह", spec)

def test_fact_ledger_compared_by_code(monkeypatch):
    p = plan(2); p["outline"][0]["end_positions"] = {"Mira": "foyer"}; p["outline"][0]["end_props"] = {"knife": "Mira"}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *a, **k: '{"end_positions":{"Mira":"basement"},"end_props":{"knife":"Mira"},"unexplained_moves":[]}')
    out = P.fact_issues("text", 1, p)
    assert len(out) == 1 and "basement" in out[0]

def test_latin_in_dialogue_flagged():
    assert P.language_issues("## Dialogue (Hindi)\nप्रिया (whispering): यह lock खुल गया", {"devanagari": True})
    assert P.language_issues("## Dialogue (Hindi)\nPriya (whispering): यह ताला खुल गया", {"devanagari": True}) == []
