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

from runtime.operations import OpError
import hashlib

# ---- quality checks ----
"""Pure story contracts and evidence-gated continuity; no provider dependencies."""

VERSION = 2
CATEGORIES = ('narrative', 'visual', 'dialogue', 'continuity', 'instructions', 'structure')


def fingerprint(prompt):
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def parse_json(raw):
    clean = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', clean)
    return json.loads(clean)


def validate_plan(plan, count):
    if not isinstance(plan, dict):
        return ['Plan must be an object']
    issues = []
    if not isinstance(plan.get('blueprint'), str) or not plan['blueprint'].strip():
        issues.append('Missing master blueprint')
    state = plan.get('initial_state')
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip() for k, v in state.items()):
        issues.append('Initial state must contain string facts keyed by entity.field')
        state = {}
    state = dict(state)
    outline = plan.get('outline')
    if not isinstance(outline, list) or len(outline) != count:
        return issues + ['Outline must contain exactly %s scenes' % count]
    for index, beat in enumerate(outline, 1):
        if not isinstance(beat, dict):
            issues.append('Invalid scene contract %s' % index)
            continue
        if beat.get('index') != index:
            issues.append('Scene order mismatch at %s' % index)
        for key in ('goal', 'conflict', 'turn', 'ending'):
            if not isinstance(beat.get(key), str) or not beat[key].strip():
                issues.append('Scene %s needs %s' % (index, key))
        for key in ('before', 'after'):
            facts = beat.get(key)
            if not isinstance(facts, dict) or not all(isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip() for k, v in facts.items()):
                issues.append('Scene %s has invalid %s state' % (index, key))
                continue
            if key == 'before':
                for field, value in facts.items():
                    if state.get(field) != value:
                        issues.append('Scene %s precondition does not follow prior state: %s' % (index, field))
            else:
                state.update(facts)
    return issues


def validate_review(review, text, state, contract=None):
    issues = []
    candidate = dict(state)
    if not isinstance(review, dict):
        return ['Review must be an object'], candidate
    checks = review.get('checks', {})
    if not isinstance(checks, dict):
        checks = {}
    issues.extend('Failed or unchecked: ' + c for c in CATEGORIES if checks.get(c) is not True)
    findings = review.get('issues')
    if not isinstance(findings, list) or not all(isinstance(i, str) for i in findings):
        issues.append('Review issues must be a list of strings')
    else:
        issues.extend(findings)
    changes = review.get('changes')
    if not isinstance(changes, list):
        return issues + ['Missing evidence-linked changes'], candidate
    seen = set()
    for change in changes:
        if not isinstance(change, dict):
            issues.append('Invalid state transition')
            continue
        key, before, after, evidence = (change.get(k) for k in ('key', 'from', 'to', 'evidence'))
        if (not isinstance(key, str) or not key.strip() or key in seen
                or candidate.get(key) != before or not isinstance(after, str) or not after.strip()
                or not isinstance(evidence, str) or not evidence.strip() or evidence not in text):
            issues.append('State transition lacks valid prior state or exact evidence')
            continue
        candidate[key] = after
        seen.add(key)
    if contract:
        for key, value in contract['before'].items():
            if state.get(key) != value:
                issues.append('Unmet scene precondition: ' + key)
        for key, value in contract['after'].items():
            if candidate.get(key) != value:
                issues.append('Required state change not shown: ' + key)
    return issues, candidate

# ---- story generation ----

MAX_SCENES = 50
MAX_PROMPT_CHARS = 60_000
SCENE_RETRIES = 2
CALL_TIMEOUT_S = 240

REQUIRED_SECTIONS = [
    "scene title", "character", "visual prompt", "dialogue", "sound",
    "camera", "scene actions", "continuity", "negative", "ending",
]

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
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
                payload = {"model": model, "messages": messages, "temperature": 0.7}
                # The upgraded primary uses its native output budget; keep the
                # existing fallback budget unchanged.
                if model == GROQ_MODEL:
                    payload["max_tokens"] = max_tokens
                text = _post(url, key, payload)
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
    m = re.search(r"\b(\d+)[\s,]+(?:[\w-]+[\s,]+){0,4}(?:scene|chapter|episode|part)s?\b", prompt, re.I)
    n = int(m.group(1)) if m else 10
    if not 1 <= n <= MAX_SCENES:
        raise OpError(f"Story requests must contain 1–{MAX_SCENES} scenes; the requested count will not be shortened.")
    return n


def _valid_scene(text: str, index: int) -> list[str]:
    headings = [re.sub(r"^\s*(?:#{1,6}\s+|\*\*)", "", line).lower().strip(" *:")
                for line in text.splitlines() if re.match(r"^\s*(?:#{1,6}\s+|\*\*)", line)]
    missing = [section for section in REQUIRED_SECTIONS if not any(section in h for h in headings)]
    problems = []
    if len(text) < 800:
        problems.append("too short")
    if missing:
        problems.append("missing section headings: " + ", ".join(missing))
    numbered = [int(n) for n in re.findall(r"^\s*#{1,2}\s+(?:scene|chapter|episode|part)\s+(\d+)\b", text, re.I | re.M)]
    if numbered != [index]:
        problems.append(f"Expected exactly one Scene {index} document header")
    return problems


def run_story(params: dict, workdir: str, progress: Callable[[str], None] | None = None) -> dict:
    prompt = str(params.get("prompt", "")).strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        raise OpError("The story brief is missing or too large.")
    scenes = _scene_count(prompt)
    say = progress or (lambda s: None)

    ckpt_path = os.path.join(workdir, "checkpoint.json")
    identity = fingerprint(prompt)
    ckpt = {"version": VERSION, "request": identity, "plan": None, "scenes": {}, "reviews": {}, "audit": None}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("request") != identity or saved.get("version") != VERSION:
                raise OpError("Story checkpoint belongs to a different brief or pipeline version.")
            ckpt = saved
        except (ValueError, TypeError, KeyError):
            raise OpError("Story checkpoint is invalid; no saved scenes were accepted.")

    def save():
        temporary = ckpt_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False)
        os.replace(temporary, ckpt_path)

    review_system = SYSTEM + """ Independently critique the draft, not its intentions. JSON only:
{"checks":{"narrative":true,"visual":true,"dialogue":true,"continuity":true,"instructions":true,"structure":true},"issues":[],"changes":[{"key":"entity.field","from":null,"to":"value","evidence":"exact quote from draft"}]}.
Every check must be explicitly performed; false means blocked. Issues are concrete strings naming scene, conflicting evidence and smallest fix.
Check goal/conflict/turn, causal escalation, setup/payoff, character motivation, subtext rather than exposition, visual/action/camera agreement,
original format/count/word/language constraints, prop/clothing/injury/location/knowledge consistency, and the ending bridge.
Changes include ONLY facts demonstrably changed in the scene; no invented transitions. An omitted recovery or knowledge acquisition fails continuity."""

    if not ckpt["plan"]:
        say("planning")
        problems = []
        for attempt in range(1 + SCENE_RETRIES):
            raw = _call([
                {"role": "system", "content": SYSTEM + """ Plan before writing. Return JSON only with
blueprint (complete Master Story Blueprint as Markdown), initial_state (string values keyed by entity.field),
outline (exactly N objects with index, goal, conflict, turn, ending, before and after state maps).
Build a causal arc with distinct dramatic turns, earned climax, seeded payoffs, consistent world rules and character motivations.
Lock appearance, clothing, visual style, voice and immutable facts in the blueprint. Track props, injuries, locations, knowledge and unresolved threads.
Every before value must equal the preceding state; after changes must be physically possible in that scene. Do not write scene scripts yet."""},
                {"role": "user", "content": json.dumps({"brief": prompt, "N": scenes, "repair": problems})},
            ], 6000)
            try:
                plan = parse_json(raw)
                problems = validate_plan(plan, scenes)
            except (ValueError, TypeError):
                problems = ["Invalid planning JSON"]
            if not problems:
                ckpt["plan"] = plan
                save()
                break
        if not ckpt["plan"]:
            raise OpError("Story planning remains unresolved: " + "; ".join(problems[:4]))
    plan = ckpt["plan"]
    problems = validate_plan(plan, scenes)
    if problems:
        raise OpError("Saved story plan is invalid: " + "; ".join(problems[:4]))
    blueprint = plan["blueprint"]
    state = dict(plan["initial_state"])
    written: list[str] = []
    for i in range(1, scenes + 1):
        key = str(i)
        contract = plan["outline"][i - 1]
        if key in ckpt["scenes"]:
            text = ckpt["scenes"][key]
            problems = _valid_scene(text, i)
            checked, candidate = validate_review(ckpt["reviews"].get(key), text, state, contract)
            if problems or checked:
                raise OpError(f"Saved scene {i} failed revalidation; no dependent scene was written.")
            state = candidate
            written.append(text)
            continue
        say(f"writing_scene_{i}")
        context = {"brief": prompt, "blueprint": blueprint, "full_outline": plan["outline"],
                   "contract": contract, "validated_state": state, "previous_scenes": written[-2:]}
        base = [
            {"role": "system", "content": SYSTEM + " Write ONLY the contracted scene as independently copyable Markdown. Show goal, obstacle and a meaningful turn; dramatize motivation and subtext. Start from validated state and end at the contracted boundary. Seed future payoffs but never reveal future events early. Match visuals, action, dialogue, sound and camera. Do not invent possessions, recoveries or character knowledge."},
            {"role": "user", "content": json.dumps(context)},
        ]
        text, problems = "", ["not attempted"]
        for attempt in range(1 + SCENE_RETRIES):
            text = _call(base, 6000)
            problems = _valid_scene(text, i)
            if not problems:
                say("validating_output")
                try:
                    review = parse_json(_call([
                        {"role": "system", "content": review_system},
                        {"role": "user", "content": json.dumps({**context, "draft": text})},
                    ], 6000))
                    problems, candidate = validate_review(review, text, state, contract)
                except (ValueError, TypeError):
                    problems = ["Reviewer did not return valid JSON"]
            if not problems:
                break
            base = base[:2] + [{"role": "assistant", "content": text}, {"role": "user", "content": f"Scene {i} failed checks ({'; '.join(problems[:8])}). Repair ONLY this scene, preserving approved facts and fixing every issue."}]
        if problems:
            raise OpError(f"Scene {i} could not be completed after retries ({'; '.join(problems[:4])}).")
        ckpt["scenes"][key], ckpt["reviews"][key] = text, review
        state = candidate
        written.append(text)
        save()

    document = blueprint.rstrip() + "\n\n---\n\n" + "\n\n---\n\n".join(written) + "\n"
    if not ckpt.get("audit"):
        say("validating_output")
        try:
            audit = parse_json(_call([
                {"role": "system", "content": review_system + " Final whole-story audit: examine all scenes, not summaries. Confirm escalation, seeded/earned payoffs, ending, continuity and exact original requirements. Return changes:[]; do not change final state."},
                {"role": "user", "content": json.dumps({"brief": prompt, "plan": plan, "story": document, "final_state": state})},
            ], 6000))
        except (ValueError, TypeError):
            raise OpError("Final story audit did not return valid JSON; story not approved.")
        problems, _ = validate_review(audit, document, state)
        if audit.get("changes"):
            problems.append("Final audit cannot mutate validated state")
        if problems:
            raise OpError("Final story audit remains unresolved: " + "; ".join(problems[:4]))
        ckpt["audit"] = audit
        save()
    else:
        problems, _ = validate_review(ckpt["audit"], document, state)
        if problems or ckpt["audit"].get("changes"):
            raise OpError("Saved final story audit is invalid.")
    say("assembling")
    out = os.path.join(workdir, "story.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(document)
    return {"output": out, "mime": "text/markdown",
            "result": {"scenes": scenes, "words": len(document.split()), "pipeline_version": VERSION, "audited": True}}
