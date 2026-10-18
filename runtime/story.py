"""Story generation operation: blueprint -> sequential scenes -> validation -> assembly.

Runs entirely on the GitHub runner (no Worker CPU limit). AI calls go to NVIDIA
(Nemotron) first, then Groq (GPT-OSS 120B), using repository secrets. Progress is
reported through the relay; a checkpoint file lets a timed-out run resume without
regenerating accepted scenes. Logs never contain prompt or story text.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable

from .operations import OpError

MAX_SCENES = 50
MAX_PROMPT_CHARS = 60_000
SCENE_RETRIES = 2
CALL_TIMEOUT_S = 240

REQUIRED_SECTIONS = [
    "scene title", "character", "visual prompt", "dialogue", "sound",
    "camera", "scene actions", "continuity", "negative", "ending",
]

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM = (
    "You are VISION's long-form story writer. Follow the user's specification exactly: "
    "structure, section headings, counts, languages and continuity rules. Output only the "
    "requested Markdown content — no commentary, no apologies."
)


def _post(url: str, key: str, payload: dict) -> str:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "vision-runtime",
    })
    with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
        data = json.loads(r.read())
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _call(messages: list[dict], max_tokens: int) -> str:
    """NVIDIA first, Groq fallback; one retry each on transient errors."""
    providers = []
    if os.environ.get("NVIDIA_API_KEY"):
        providers.append((NVIDIA_URL, os.environ["NVIDIA_API_KEY"], NVIDIA_MODEL))
    if os.environ.get("GROQ_API_KEY"):
        providers.append((GROQ_URL, os.environ["GROQ_API_KEY"], GROQ_MODEL))
    if not providers:
        raise OpError("No AI provider key is configured on the runner.")
    last = "no provider succeeded"
    for url, key, model in providers:
        for attempt in range(2):
            try:
                text = _post(url, key, {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7})
                if text:
                    return text
                last = "empty response"
            except urllib.error.HTTPError as e:
                last = f"provider HTTP {e.code}"
                if e.code in (400, 401, 403, 404, 413, 422):
                    break  # terminal for this provider
            except Exception:
                last = "provider unreachable"
            time.sleep(3 * (attempt + 1))
    raise OpError(f"The AI providers could not complete this step ({last}).")


def _scene_count(prompt: str) -> int:
    m = re.search(r"\b(\d{1,2})[\s,]+(?:[\w-]+[\s,]+){0,4}(?:scene|chapter|episode|part)s?\b", prompt, re.I)
    n = int(m.group(1)) if m else 10
    return max(1, min(n, MAX_SCENES))


def _valid_scene(text: str, index: int) -> list[str]:
    low = text.lower()
    missing = [s for s in REQUIRED_SECTIONS if s not in low]
    problems = []
    if len(text) < 800:
        problems.append("too short")
    if missing:
        problems.append("missing sections: " + ", ".join(missing[:3]))
    return problems


def run_story(params: dict, workdir: str, progress: Callable[[str], None] | None = None) -> dict:
    prompt = str(params.get("prompt", "")).strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        raise OpError("The story brief is missing or too large.")
    scenes = _scene_count(prompt)
    say = progress or (lambda s: None)

    ckpt_path = os.path.join(workdir, "checkpoint.json")
    ckpt = {"blueprint": None, "scenes": {}}
    if os.path.exists(ckpt_path):
        try:
            ckpt = json.load(open(ckpt_path))
        except Exception:
            pass

    if not ckpt["blueprint"]:
        say("planning")
        ckpt["blueprint"] = _call([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt + "\n\nFirst produce ONLY the Master Story Blueprint (no scenes yet)."},
        ], 6000)
        json.dump(ckpt, open(ckpt_path, "w"))
    blueprint = ckpt["blueprint"]

    written: list[str] = []
    for i in range(1, scenes + 1):
        key = str(i)
        if key in ckpt["scenes"]:
            written.append(ckpt["scenes"][key])
            continue
        say(f"writing_scene_{i}")
        base = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                f"{prompt}\n\nMaster Story Blueprint (canonical source of truth):\n{blueprint}\n\n"
                f"Previously written scenes (for continuity only, do not repeat):\n"
                + "\n\n".join(written[-2:])
                + f"\n\nNow write ONLY scene {i} of {scenes} as a complete, independently copyable Markdown document."
            )},
        ]
        text, problems = "", ["not attempted"]
        for attempt in range(1 + SCENE_RETRIES):
            text = _call(base, 6000)
            problems = _valid_scene(text, i)
            if not problems:
                break
            base.append({"role": "assistant", "content": text})
            base.append({"role": "user", "content": f"Scene {i} failed checks ({'; '.join(problems)}). Rewrite scene {i} completely, fixing every issue."})
        if problems:
            raise OpError(f"Scene {i} could not be completed after retries ({'; '.join(problems)}).")
        ckpt["scenes"][key] = text
        written.append(text)
        json.dump(ckpt, open(ckpt_path, "w"))

    say("assembling")
    document = blueprint.rstrip() + "\n\n---\n\n" + "\n\n---\n\n".join(written) + "\n"
    out = os.path.join(workdir, "story.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(document)
    return {
        "output": out,
        "mime": "text/markdown",
        "result": {"scenes": scenes, "words": len(document.split())},
    }
