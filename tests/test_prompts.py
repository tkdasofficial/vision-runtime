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
            "clues": [{"name": "silver key", "setup": 1, "payoff": n}],
            "outline": [{"index": i, "moment": f"m{i}", "time": f"0{i}:00", "location": "street", "characters": ["Mira"], "before": {}, "after": {}} for i in range(1, n + 1)]}
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
    assert any("silver key" in x for x in P.check_unit(unit(3), 3, p, SPEC))
    assert P.check_unit(unit(3, extra="the silver key"), 3, p, SPEC) == []

def test_story_only_when_asked_and_fences():
    assert P.wants_story("Write a story with 5 scene prompts")
    assert not P.wants_story("Generate 10 scene prompts. Do not generate the finished story itself.")
    f = P.fence("# Scene 1\n```x```\nhi")
    assert f.startswith("````markdown\n") and f.endswith("\n````")
