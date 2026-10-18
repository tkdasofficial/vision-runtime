import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.operations import OpError
from workers.prompts import index as P

BIBLE = {"characters": {"Mira": "a 30-year-old woman with a silver bob, green eyes, red wool coat"},
         "style": "35mm film still, teal and amber palette", "setting": "rainy neon city", "negative": "no text, no watermark"}
def plan(n):
    return {"bible": BIBLE, "initial_state": {"mira.umbrella": "closed"},
            "outline": [{"index": i, "moment": f"m{i}", "characters": ["Mira"], "before": {}, "after": {}} for i in range(1, n + 1)]}
def prompt(i, desc=BIBLE["characters"]["Mira"]):
    return (f"## Prompt {i}\nSubject: Mira walks\nCharacters: Mira, {desc}\nSetting: street\nAction: walking slowly in the rain\n"
            f"Camera: low tracking shot\nLighting: neon rim light\nStyle: {BIBLE['style']}\nNegative: {BIBLE['negative']}\n" + "x" * 50)

def test_counts():
    assert P.prompt_count("write 20 linked scene prompts") == 20
    assert P.prompt_count("ten image prompts") == 10
    assert P.prompt_count("an image prompt of a cat") == 1
    with pytest.raises(OpError):
        P.prompt_count("write 500 prompts")

def test_plan_must_match_count_and_bible():
    assert P.validate_plan(plan(5), 5) == []
    assert P.validate_plan(plan(4), 5)
    bad = plan(2); bad["outline"][1]["characters"] = ["Ghost"]
    assert any("missing from the bible" in x for x in P.validate_plan(bad, 2))

def test_locked_descriptor_must_be_verbatim():
    assert P.check_prompt(prompt(3), 3, plan(5), True) == []
    drift = prompt(3, "a woman with long black hair")
    assert any("Mira" in x for x in P.check_prompt(drift, 3, plan(5), True))
    assert any("header" in x for x in P.check_prompt(prompt(2), 3, plan(5), True))

def test_review_requires_all_checks_and_evidence():
    text = prompt(1) + " she opens the umbrella"
    ok = {"checks": {c: True for c in P.CATEGORIES}, "issues": [],
          "changes": [{"key": "mira.umbrella", "from": "closed", "to": "open", "evidence": "she opens the umbrella"}]}
    issues, st = P.validate_review(ok, text, {"mira.umbrella": "closed"})
    assert issues == [] and st["mira.umbrella"] == "open"
    ok["changes"][0]["evidence"] = "invented"
    assert P.validate_review(ok, text, {"mira.umbrella": "closed"})[0]
    assert P.validate_review({"checks": {}, "issues": [], "changes": []}, text, {})[0]
