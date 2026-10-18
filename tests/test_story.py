import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import story  # noqa: E402
from runtime.operations import OpError, get  # noqa: E402

PROMPT = "Generate exactly 3 detailed, sequential scene-generation prompts for a horror story. Include Scene Title, Character Details, Visual Prompt, Dialogue, Sound Design, Camera Direction, Scene Actions, Continuity Details, Negative Constraints, Ending and Transition in every scene."

SCENE = "\n".join("### " + section.title() + "\n" + ("Ari carries the lantern down the road. " * 5) for section in story.REQUIRED_SECTIONS)


def plan():
    return {"blueprint": "Master blueprint: Ari brings light home.", "initial_state": {"Ari.possession": "lantern"},
            "outline": [{"index": i, "goal": "Reach home", "conflict": "Storm", "turn": "Choose shelter", "ending": "Continue onward", "before": {"Ari.possession": "lantern"}, "after": {"Ari.possession": "lantern"}} for i in range(1, 4)]}


def review():
    return {"checks": {c: True for c in CATEGORIES}, "issues": [], "changes": []}


# Mirror the public review categories without relying on model prose.
from runtime.story_quality import CATEGORIES


def fake_call(messages, max_tokens):
    import json
    system = messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    if "Plan before writing" in system:
        return json.dumps(plan())
    if "Independently critique" in system:
        return json.dumps(review())
    return "## Scene %s\n" % payload['contract']['index'] + SCENE


def test_story_writes_all_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(story, "_call", fake_call)
    stages = []
    res = story.run_story({"prompt": PROMPT}, str(tmp_path), stages.append)
    assert res["result"]["scenes"] == 3
    assert res["result"]["audited"] is True
    doc = open(res["output"]).read()
    assert doc.count("Scene Title") == 3
    assert "planning" in stages and "assembling" in stages and "validating_output" in stages
    assert stages.count("writing_scene_2") == 1


def test_story_checkpoint_skips_accepted_scenes(tmp_path, monkeypatch):
    calls = {"n": 0}
    def counting(messages, max_tokens):
        calls["n"] += 1
        return fake_call(messages, max_tokens)
    monkeypatch.setattr(story, "_call", counting)
    story.run_story({"prompt": PROMPT}, str(tmp_path))
    first = calls["n"]
    story.run_story({"prompt": PROMPT}, str(tmp_path))
    assert calls["n"] == first
    assert first == 8  # plan + generation/review per scene + whole-story audit


def test_story_retries_failed_scene(tmp_path, monkeypatch):
    state = {"n": 0}
    def flaky(messages, max_tokens):
        state["n"] += 1
        return "too short" if state["n"] == 2 else fake_call(messages, max_tokens)
    monkeypatch.setattr(story, "_call", flaky)
    assert story.run_story({"prompt": PROMPT}, str(tmp_path))["result"]["scenes"] == 3


def test_story_rejects_empty_prompt(tmp_path):
    with pytest.raises(OpError):
        story.run_story({"prompt": ""}, str(tmp_path))


def test_story_registered_without_input():
    op = get("story_generate")
    assert op.needs_input is False and op.max_input == 0


def test_requested_nemotron_3_super_primary(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    calls = []
    def post(url, key, payload):
        calls.append((url, payload))
        return "generated"
    monkeypatch.setattr(story, "_post", post)
    assert story._call([{"role": "user", "content": "hello"}], 6000) == "generated"
    assert calls[0][0] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert calls[0][1]["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert "max_tokens" not in calls[0][1]
    assert len(calls) == 1


def test_requested_groq_120b_fallback(monkeypatch):
    import urllib.error
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    calls = []
    def post(url, key, payload):
        calls.append((url, payload))
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 413, "Too large", {}, None)
        return "fallback generated"
    monkeypatch.setattr(story, "_post", post)
    assert story._call([{"role": "user", "content": "hello"}], 6000) == "fallback generated"
    assert calls[1][0] == "https://api.groq.com/openai/v1/chat/completions"
    assert calls[1][1]["model"] == "openai/gpt-oss-120b"
    assert calls[1][1]["max_tokens"] == 6000


def test_rejects_shortened_counts():
    for count in (0, 51, 100):
        with pytest.raises(OpError):
            story._scene_count("Write %s scenes" % count)


def test_section_words_inside_prose_do_not_pass():
    assert story._valid_scene("## Scene 1\n" + " ".join(story.REQUIRED_SECTIONS) * 100, 1)


def test_wrong_scene_number_does_not_pass():
    assert story._valid_scene("## Scene 2\n" + SCENE, 1)


def test_plan_boundary_contradiction():
    from runtime.story_quality import validate_plan
    p = plan()
    p['outline'][1]['before']['Ari.possession'] = 'no lantern'
    assert any('precondition' in i for i in validate_plan(p, 3))


def test_empty_evidence_never_commits_state():
    from runtime.story_quality import validate_review
    r = review()
    r['changes'] = [{'key': 'Ari.possession', 'from': 'lantern', 'to': 'lost', 'evidence': ''}]
    problems, candidate = validate_review(r, 'Ari walks.', {'Ari.possession': 'lantern'})
    assert problems and candidate['Ari.possession'] == 'lantern'


def test_old_facts_reach_every_scene(tmp_path, monkeypatch):
    import json
    states = []
    def capture(messages, max_tokens):
        if 'Write ONLY the contracted' in messages[0]['content']:
            payload = json.loads(messages[1]['content'])
            states.append(payload['validated_state'])
            assert len(payload['full_outline']) == 3
        return fake_call(messages, max_tokens)
    monkeypatch.setattr(story, '_call', capture)
    story.run_story({'prompt': PROMPT}, str(tmp_path))
    assert states == [{'Ari.possession': 'lantern'}] * 3


def test_unchecked_review_stops_dependents(tmp_path, monkeypatch):
    import json
    generations = []
    def rejected(messages, max_tokens):
        payload = json.loads(messages[1]['content'])
        if 'Write ONLY the contracted' in messages[0]['content']:
            generations.append(payload['contract']['index'])
        if 'Independently critique' in messages[0]['content']:
            r = review(); r['checks']['continuity'] = False
            return json.dumps(r)
        return fake_call(messages, max_tokens)
    monkeypatch.setattr(story, '_call', rejected)
    with pytest.raises(OpError, match='continuity'):
        story.run_story({'prompt': PROMPT}, str(tmp_path))
    assert generations == [1, 1, 1]
    assert not (tmp_path / 'story.md').exists()


def test_different_brief_cannot_reuse_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(story, '_call', fake_call)
    story.run_story({'prompt': PROMPT}, str(tmp_path))
    with pytest.raises(OpError, match='different brief'):
        story.run_story({'prompt': PROMPT + ' Change ending.'}, str(tmp_path))


def test_failed_final_audit_never_delivers(tmp_path, monkeypatch):
    import json
    def bad_audit(messages, max_tokens):
        if 'Final whole-story audit' in messages[0]['content']:
            r = review(); r['issues'] = ['Scene 3 payoff was never seeded']
            return json.dumps(r)
        return fake_call(messages, max_tokens)
    monkeypatch.setattr(story, '_call', bad_audit)
    with pytest.raises(OpError, match='Final story audit remains unresolved'):
        story.run_story({'prompt': PROMPT}, str(tmp_path))
    assert not (tmp_path / 'story.md').exists()
