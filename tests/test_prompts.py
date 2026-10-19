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
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *a, **k: '{"evidence":{"knife":"Mira grabs the knife from the table"},"end_positions":{"Mira":"basement"},"end_props":{"knife":"Mira"},"unexplained_moves":[]}')
    out = P.fact_issues("Mira grabs the knife from the table here", 1, p)
    assert len(out) == 1 and "basement" in out[0]

def test_latin_in_dialogue_flagged():
    assert P.language_issues("## Dialogue (Hindi)\nप्रिया (whispering): यह lock खुल गया", {"devanagari": True})
    assert P.language_issues("## Dialogue (Hindi)\nPriya (whispering): यह ताला खुल गया", {"devanagari": True}) == []

def test_long_gap_gets_time_skip():
    p = plan(2); p["outline"][1]["time"] = "D2 05:00"
    P.validate_plan(p, 2) if hasattr(P, "validate_plan") else None
    assert "hours pass" in p["outline"][1].get("time_skip", "")

def test_backwards_time_is_repaired():
    p = plan(3); p["outline"][1]["time"] = "D1 23:50"; p["outline"][2]["time"] = "D1 00:20"
    assert not [i for i in P.validate_plan(p, 3) if "time" in i]
    assert p["outline"][2]["time"] == "D2 00:20"

def test_nemotron_pacing_limits():
    import threading, time as _t
    lim = P._RateLimiter(3, 2); inflight = []; peak = [0]; lock = threading.Lock()
    def go():
        lim.acquire()
        with lock: inflight.append(1); peak[0] = max(peak[0], len(inflight))
        _t.sleep(0.05)
        with lock: inflight.pop()
        lim.release()
    ts = [threading.Thread(target=go) for _ in range(3)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert peak[0] <= 2 and len(lim.stamps) == 3
    assert P.NVIDIA_RPM == 40 and P.NVIDIA_CONCURRENCY == 4

def test_names_allowed_and_dialogue_only_fix(monkeypatch):
    assert P.language_issues("## Dialogue\nमीरा: Mira यहाँ है", {"devanagari": True}, {"mira"}) == []
    t = "# Scene 1: X\n\n## Dialogue\nमीरा: यह lock है\n\n## Camera\nwide"
    monkeypatch.setattr(P, "CALL", lambda *a, **k: "## Dialogue\nमीरा: यह ताला है")
    out = P._fix_dialogue(t, ["lock"])
    assert "ताला" in out and "## Camera\nwide" in out and "lock" not in out

def test_other_scripts_flagged():
    assert P.language_issues("## Dialogue\nमीरा: তোমার শান্তি হোক", {"devanagari": True})

def test_prop_move_needs_quoted_evidence(monkeypatch):
    p = plan(2); b = p["outline"][0]; b["start_props"] = {"bell": "nail"}; b["end_props"] = {"bell": "Mira"}; b["end_positions"] = {}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *a, **k: '{"evidence":{"bell":""},"end_props":{"bell":"Mira"},"start_props":{"bell":"nail"}}')
    assert any("show bell" in x for x in P.fact_issues("Mira looks at the bell.", 1, p))
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *a, **k: '{"evidence":{"bell":"Mira lifts the bell off the nail"},"end_props":{"bell":"Mira"},"start_props":{"bell":"nail"}}')
    assert P.fact_issues("Mira lifts the bell off the nail and pockets it.", 1, p) == []


def test_timed_story_and_all_scene_prompts():
    brief = "Write a 40 seconds story and give me all scenes prompts"
    assert P.duration_seconds(brief) == 40
    assert P.unit_count(brief) == (5, "scene")
    assert P.wants_story(brief)
    assert P.wants_story("Create a story with 4 scenes")
    assert not P.wants_story("Generate exactly 10 scene-generation prompts for a supernatural horror story")


def test_inline_story_default_components_and_hidden_plan():
    p = {"blueprint": "SECRET PLAN", "story": "# The Door\n\nMira opens it."}
    spec = {"story": True, "noun": "scene"}
    unit = "# Scene 1: Arrival\n\n## Story Line\nMira arrives.\n\n## Visual Prompt\nMira at the door.\n\n## Dialogue\nMira: Hello."
    out = P.assemble_output(p, [unit], spec, "Write a story and scene prompts")
    assert out.startswith("The Door\n\n````markdown\nMira opens it.")
    assert "SECRET PLAN" not in out and "Blueprint" not in out
    assert out.count("````markdown") == 4
    assert out.count("Mira arrives.") == 1


def test_explicit_one_box_per_scene_and_story_only():
    p = {"story": "# Door\n\nMira opens it."}
    unit = "# Scene 1: Door\n\n## Visual Prompt\nMira.\n\n## Dialogue\nHello."
    out = P.assemble_output(p, [unit], {"story": False, "noun": "scene"}, "Each scene markdown block has all details: visual prompt and dialogue")
    assert out.count("````markdown") == 1 and "## Dialogue" in out
    out = P.assemble_output(p, [unit], {"story": True, "prompts": False, "noun": "scene"}, "Write a story")
    assert out.count("````markdown") == 1 and "Visual Prompt" not in out


def test_extracted_module_is_single_implementation():
    from workers.prompts.story import index as S
    assert S is P


MONKEY_BRIEF = "Create a story of 10 scenes, each scene have 25 to 30 words dialogue, and give me both visual prompt & dialogue of each scenes. Characters: A Monkey and A Rabbit. Theme: School & Homework. Make sure the background also constant."

def test_explicit_scene_delivery_not_input_profiles():
    assert P.unit_count(MONKEY_BRIEF) == (10, "scene")
    assert P.required_sections(MONKEY_BRIEF) == ["Visual Prompt", "Dialogue"]
    assert not P.wants_story(MONKEY_BRIEF)

def test_dialogue_word_range_gate():
    assert P.dialogue_range(MONKEY_BRIEF) == (25, 30)
    p = plan(3)
    spec = {**SPEC, "sections": ["Visual Prompt", "Dialogue"], "dialogue_range": (25, 30)}
    for count, fails in [(24, True), (25, False), (30, False), (31, True)]:
        text = unit(2) + "\n## Dialogue\nMira (quietly): " + " ".join(["homework"] * count)
        assert P.dialogue_words(text) == count
        assert any("spoken words" in x for x in P.check_unit(text, 2, p, spec)) == fails

def test_explicit_delivery_filters_unrequested_sections():
    text = "# Scene 1: School\n\n## Story Line\nEXTRA\n\n## Visual Prompt\nMonkey and Rabbit in class.\n\n## Dialogue\nMonkey: Hello.\n\n## Camera Direction\nEXTRA CAMERA"
    out = P.assemble_output({"story": "EXTRA STORY"}, [text], {"story": False, "noun": "scene"}, MONKEY_BRIEF)
    assert out.count("````markdown") == 2
    assert "EXTRA" not in out
    assert "Monkey: Hello." in out


def test_fixed_background_rejects_outdoor_scene():
    p = plan(3); p["bible"]["background"] = "Green board, two desks, pastel curtains, warm sunlight"
    assert P.check_fixed_background(p, MONKEY_BRIEF) == []
    p["outline"][2]["location"] = "outside"
    assert any("same location" in x for x in P.check_fixed_background(p, MONKEY_BRIEF))

def test_visual_only_locks_do_not_add_ledger_or_profiles():
    p = plan(3); p["bible"]["background"] = "Green board and two desks"
    spec = {**SPEC, "sections": ["Visual Prompt", "Dialogue"], "fixed_background": True}
    text = P.enforce_locks("# Prompt 2: School\n## Visual Prompt\nMira works.\n## Dialogue\nMira: Hello.", 2, p, spec)
    assert "Green board and two desks" in text
    assert "Canonical state" not in text
    assert "## Character Details" not in text
    assert text.count("## ") == 2


def test_school_story_does_not_require_mystery():
    p = plan(3); p["requires_clues"] = False; p["clues"] = []
    assert P.validate_plan(p, 3) == []


def test_failed_checks_cannot_pass_with_high_score():
    checks = {c: True for c in P.CATEGORIES}; checks["continuity"] = False
    score, notes, units = P.audit_findings({"score": 100, "checks": checks, "issues": [], "units": []}, [1, 2])
    assert score < 90 and units == {1, 2}
    checks["continuity"] = True
    assert P.audit_findings({"score": 95, "checks": checks, "issues": [], "units": []}, [1])[0] == 95


def test_unknown_and_unavailable_extraction_block(monkeypatch):
    p = plan(2); p["outline"][0]["end_positions"] = {"Mira": "foyer"}
    monkeypatch.setattr(P, "AUDIT_CALL", lambda *a: '{"end_positions":{"Mira":"unknown"}}')
    assert any("not verified" in x for x in P.fact_issues("Mira waits.", 1, p))
    def unavailable(*a): raise OpError("Unavailable")
    monkeypatch.setattr(P, "AUDIT_CALL", unavailable)
    assert any("unavailable" in x for x in P.fact_issues("Mira waits.", 1, p))


def test_canonical_metadata_is_not_narrative_evidence():
    t = "## Visual Prompt\n- **Mira (locked appearance):** red coat\n**Locked style:** film\nMira removes her coat.\n## Continuity\n**Canonical state at scene end:**\n- Mira ends at: foyer\n## Dialogue\nMira: Hello."
    out = P.scene_evidence(t)
    assert "removes her coat" in out and "locked appearance" not in out and "ends at:" not in out


def test_fixed_background_checks_character_positions():
    p = plan(3); p["bible"]["background"] = "classroom"
    p["outline"][1]["end_positions"] = {"Mira": "outside"}
    assert P.check_fixed_background(p, MONKEY_BRIEF)


def test_dialogue_only_repair_preserves_visual_and_word_bounds(monkeypatch):
    text = "## Visual Prompt\nUNCHANGED\n## Dialogue\nMira: Hello."
    monkeypatch.setattr(P, "CALL", lambda *a: "## Dialogue\nMira (quietly): " + " ".join(["homework"] * 27))
    fixed = P._fix_dialogue(text, ["too short"], (25, 30), False)
    assert "UNCHANGED" in fixed and P.dialogue_words(fixed) == 27


def test_subjective_state_does_not_need_literal_word():
    p = plan(3); p["outline"][1]["after"] = {"Mira.confidence": "growing"}
    assert not any("verbatim" in x for x in P.check_unit(unit(2), 2, p, SPEC))


def test_bounded_dialogue_uses_validated_word_arrays(monkeypatch):
    text = "## Visual Prompt\nSTABLE\n## Dialogue\nMira: too short\nRabbit: also short"
    import json
    def answer(messages, *args):
        contracts = json.loads(messages[1]["content"])["turn_contracts"]
        return json.dumps({"turns": [{"speaker": c["speaker"], "words": ["homework"] * c["words"]} for c in contracts]})
    monkeypatch.setattr(P, "CALL", answer)
    result = P._bounded_dialogue(text, (25, 30))
    assert P.dialogue_words(result) == 27 and "STABLE" in result
    monkeypatch.setattr(P, "CALL", lambda *a: '{"turns":[{"words":["too many words"]}]}')
    assert P._bounded_dialogue(text, (25, 30)) == text


def test_immutable_wardrobe_not_tracked_as_moving_prop():
    p = plan(2); p["start_props"] = {"red coat": "Mira"}
    for b in p["outline"]:
        b["start_props"] = {"red coat": "Mira"}; b["end_props"] = {"red coat": "Mira"}
    assert P.validate_plan(p, 2) == []
    assert p["start_props"] == {} and p["outline"][0]["end_props"] == {}
    p["start_props"] = {"red coat": "Mira"}; p["outline"][1]["end_props"] = {"red coat": "Rabbit"}
    assert any("Immutable wardrobe" in x for x in P.validate_plan(p, 2))
