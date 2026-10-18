import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import story  # noqa: E402
from runtime.operations import OpError, get  # noqa: E402

PROMPT = "Generate exactly 3 detailed, sequential scene-generation prompts for a horror story. Include Scene Title, Character Details, Visual Prompt, Dialogue, Sound Design, Camera Direction, Scene Actions, Continuity Details, Negative Constraints, Ending and Transition in every scene."

SCENE = "## Scene Title and Context\n" + ("Character details, visual prompt, dialogue, sound, camera, scene actions, continuity, negative constraints, ending. " * 30)


def fake_call(messages, max_tokens):
    return SCENE


def test_story_writes_all_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(story, "_call", fake_call)
    stages = []
    res = story.run_story({"prompt": PROMPT}, str(tmp_path), stages.append)
    assert res["result"]["scenes"] == 3
    doc = open(res["output"]).read()
    assert doc.count("Scene Title") == 4  # blueprint mock + 3 scenes
    assert "planning" in stages and "assembling" in stages
    assert stages.count("writing_scene_2") == 1


def test_story_checkpoint_skips_accepted_scenes(tmp_path, monkeypatch):
    calls = {"n": 0}

    def counting(messages, max_tokens):
        calls["n"] += 1
        return SCENE

    monkeypatch.setattr(story, "_call", counting)
    story.run_story({"prompt": PROMPT}, str(tmp_path))
    first = calls["n"]
    story.run_story({"prompt": PROMPT}, str(tmp_path))  # resume: nothing regenerated
    assert calls["n"] == first
    assert first == 4  # blueprint + 3 scenes


def test_story_retries_failed_scene(tmp_path, monkeypatch):
    state = {"n": 0}

    def flaky(messages, max_tokens):
        state["n"] += 1
        return "too short" if state["n"] == 2 else SCENE

    monkeypatch.setattr(story, "_call", flaky)
    res = story.run_story({"prompt": PROMPT}, str(tmp_path))
    assert res["result"]["scenes"] == 3


def test_story_rejects_empty_prompt(tmp_path):
    with pytest.raises(OpError):
        story.run_story({"prompt": ""}, str(tmp_path))


def test_story_registered_without_input():
    op = get("story_generate")
    assert op.needs_input is False and op.max_input == 0
