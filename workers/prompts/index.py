"""VISION runtime worker: prompts — one prompt or a linked series (5, 10, 20 … up to 100) with locked consistency.

Pipeline (same contract as the story pipeline in workers/documents/story.py, which it reuses for provider calls):
  plan  -> a locked "bible" (characters with fixed visual descriptors, style, setting, negatives) + one contract per prompt
  write -> each prompt in order, carrying the bible, the full outline and validated state
  check -> deterministic: header, required fields, every locked descriptor and the style lock copied VERBATIM
           independent review: identity/style/continuity/instructions/structure, evidence-quoted state changes
  repair-> bounded retries per prompt; an unresolved prompt stops the run (no later prompt is written on a broken one)
  audit -> blocking whole-series audit, then one prompts.md
A checkpoint bound to the brief + pipeline version lets a resumed run skip accepted prompts.
Logs never contain prompt text.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable

from runtime.operations import OpError
from runtime.worker_base import Worker
from workers.documents.story import _call, fingerprint, parse_json

VERSION = 1
MAX_PROMPTS = 100
MAX_BRIEF_CHARS = 60_000
RETRIES = 2
CATEGORIES = ("identity", "style", "continuity", "instructions", "structure")
FIELDS = ("subject", "characters", "setting", "action", "camera", "lighting", "style", "negative")

WORKER = Worker("prompts")

SYSTEM = ("You are VISION's prompt engineer for image/video generators. Follow the user's brief exactly: count, "
          "language, target model, length and format. Output only the requested content — no commentary.")

_NUM = {"one": 1, "single": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50}


def prompt_count(brief: str, given=None) -> int:
    if given is not None:
        n = int(given) if str(given).isdigit() else 0
    else:
        m = re.search(r"\b(\d{1,3}|" + "|".join(_NUM) + r")[\s-]+(?:[\w-]+[\s,]+){0,3}?(?:prompts?|scenes?|shots?|frames?|panels?|clips?|images?)\b", brief, re.I)
        n = (int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1).lower()]) if m else 1
    if not 1 <= n <= MAX_PROMPTS:
        raise OpError(f"Prompt requests must contain 1–{MAX_PROMPTS} prompts; the requested count will not be shortened.")
    return n


def _facts(d) -> bool:
    return isinstance(d, dict) and all(isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip() for k, v in d.items())


def validate_plan(plan, count: int) -> list[str]:
    if not isinstance(plan, dict):
        return ["Plan must be an object"]
    issues = []
    bible = plan.get("bible")
    if not isinstance(bible, dict):
        return ["Missing bible"]
    chars = bible.get("characters")
    if not isinstance(chars, dict) or not _facts(chars):
        issues.append("Bible characters must map each name to a locked visual descriptor")
    for k in ("style", "setting", "negative"):
        if not isinstance(bible.get(k), str) or not bible[k].strip():
            issues.append(f"Bible needs {k}")
    state = plan.get("initial_state", {})
    if not _facts(state) and state != {}:
        issues.append("Initial state must contain string facts keyed by entity.field")
        state = {}
    state = dict(state)
    outline = plan.get("outline")
    if not isinstance(outline, list) or len(outline) != count:
        return issues + [f"Outline must contain exactly {count} prompts"]
    for i, beat in enumerate(outline, 1):
        if not isinstance(beat, dict) or beat.get("index") != i:
            issues.append(f"Prompt contract {i} is invalid or out of order"); continue
        if not isinstance(beat.get("moment"), str) or not beat["moment"].strip():
            issues.append(f"Prompt {i} needs moment")
        cast = beat.get("characters", [])
        if not isinstance(cast, list) or (isinstance(chars, dict) and any(c not in chars for c in cast)):
            issues.append(f"Prompt {i} uses a character missing from the bible")
        for key in ("before", "after"):
            facts = beat.get(key, {})
            if facts and not _facts(facts):
                issues.append(f"Prompt {i} has invalid {key} state"); continue
            if key == "before":
                issues += [f"Prompt {i} precondition does not follow prior state: {f}" for f, v in (facts or {}).items() if state.get(f) != v]
            else:
                state.update(facts or {})
    return issues


def check_prompt(text: str, index: int, plan: dict, multi: bool) -> list[str]:
    """Deterministic consistency gate: the locks must be copied word-for-word, not paraphrased."""
    problems = []
    low = text.lower()
    if multi:
        heads = [int(n) for n in re.findall(r"^\s*#{1,3}\s+prompt\s+(\d+)\b", text, re.I | re.M)]
        if heads != [index]:
            problems.append(f"Expected exactly one '## Prompt {index}' header")
    missing = [f for f in FIELDS if not re.search(rf"^\s*(?:[-*]\s*)?\**{f}\**\s*:", text, re.I | re.M)]
    if missing:
        problems.append("missing fields: " + ", ".join(missing))
    bible = plan["bible"]
    if bible["style"].strip().lower() not in low:
        problems.append("style lock not copied verbatim")
    if bible["negative"].strip().lower() not in low:
        problems.append("negative lock not copied verbatim")
    cast = plan["outline"][index - 1].get("characters", [])
    for name in cast:
        desc = bible["characters"][name].strip().lower()
        if desc not in low:
            problems.append(f"locked descriptor for {name} changed or missing")
    if len(text.strip()) < 200:
        problems.append("too short")
    return problems


def validate_review(review, text: str, state: dict, contract=None):
    candidate = dict(state)
    if not isinstance(review, dict):
        return ["Review must be an object"], candidate
    checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
    issues = ["Failed or unchecked: " + c for c in CATEGORIES if checks.get(c) is not True]
    findings = review.get("issues")
    issues += findings if isinstance(findings, list) and all(isinstance(i, str) for i in findings) else ["Review issues must be a list of strings"]
    changes = review.get("changes")
    if not isinstance(changes, list):
        return issues + ["Missing evidence-linked changes"], candidate
    for ch in changes:
        if not isinstance(ch, dict):
            issues.append("Invalid state transition"); continue
        key, before, after, ev = (ch.get(k) for k in ("key", "from", "to", "evidence"))
        if not (isinstance(key, str) and key.strip() and candidate.get(key) == before and isinstance(after, str) and after.strip()
                and isinstance(ev, str) and ev.strip() and ev in text):
            issues.append("State transition lacks valid prior state or exact evidence"); continue
        candidate[key] = after
    if contract:
        issues += ["Unmet precondition: " + k for k, v in (contract.get("before") or {}).items() if state.get(k) != v]
        issues += ["Required change not shown: " + k for k, v in (contract.get("after") or {}).items() if candidate.get(k) != v]
    return issues, candidate


REVIEW = SYSTEM + """ Independently critique the draft prompt. JSON only:
{"checks":{"identity":true,"style":true,"continuity":true,"instructions":true,"structure":true},"issues":[],"changes":[{"key":"entity.field","from":"old","to":"new","evidence":"exact quote from draft"}]}
identity: every character matches the bible (face, age, hair, build, outfit) with nothing added or contradicted.
style: same medium, palette, lens language and rendering as the bible. continuity: props, wounds, clothing, location,
time of day and weather follow validated state and the contract. instructions: the brief's count/language/model/length rules.
structure: header and every field present. false means blocked. Changes list ONLY facts this prompt changes."""


def run_prompts(params: dict, workdir: str, progress: Callable[[str], None] | None = None) -> dict:
    brief = str(params.get("prompt", "")).strip()
    if not brief or len(brief) > MAX_BRIEF_CHARS:
        raise OpError("The prompt brief is missing or too large.")
    count = prompt_count(brief, params.get("count"))
    multi = count > 1
    say = progress or (lambda s: None)
    path = os.path.join(workdir, "prompts-checkpoint.json")
    ident = fingerprint(brief + f"|{count}")
    ck = {"version": VERSION, "request": ident, "plan": None, "prompts": {}, "reviews": {}, "audit": None}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
        except ValueError:
            raise OpError("Prompt checkpoint is invalid; no saved prompts were accepted.")
        if saved.get("request") != ident or saved.get("version") != VERSION:
            raise OpError("Prompt checkpoint belongs to a different brief or pipeline version.")
        ck = saved

    def save():
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(ck, f, ensure_ascii=False)
        os.replace(path + ".tmp", path)

    if not ck["plan"]:
        say("planning")
        problems: list[str] = []
        for _ in range(1 + RETRIES):
            raw = _call([{"role": "system", "content": SYSTEM + """ Plan before writing. JSON only:
{"bible":{"characters":{"Name":"locked visual descriptor, 15-40 words: age, face, hair, build, exact outfit"},"style":"one locked style line (medium, palette, lens, lighting language)","setting":"world/location rules","negative":"one locked negative line"},
"initial_state":{"entity.field":"value"},"outline":[{"index":1,"moment":"what this prompt shows","characters":["Name"],"before":{},"after":{}}]}
outline has exactly N items in order. Each linked prompt advances one causal moment; every 'before' equals the state left by earlier prompts.
Only changes physically possible in that moment go in 'after'. No prompt text yet."""},
                         {"role": "user", "content": json.dumps({"brief": brief, "N": count, "repair": problems})}], 6000)
            try:
                plan = parse_json(raw); problems = validate_plan(plan, count)
            except (ValueError, TypeError):
                problems = ["Invalid planning JSON"]
            if not problems:
                ck["plan"] = plan; save(); break
        if not ck["plan"]:
            raise OpError("Prompt planning remains unresolved: " + "; ".join(problems[:4]))
    plan = ck["plan"]
    if validate_plan(plan, count):
        raise OpError("Saved prompt plan is invalid.")
    state = dict(plan.get("initial_state") or {})
    written: list[str] = []
    for i in range(1, count + 1):
        key, contract = str(i), plan["outline"][i - 1]
        if key in ck["prompts"]:
            text = ck["prompts"][key]
            bad, cand = validate_review(ck["reviews"].get(key), text, state, contract)
            if check_prompt(text, i, plan, multi) or bad:
                raise OpError(f"Saved prompt {i} failed revalidation; no dependent prompt was written.")
            state = cand; written.append(text); continue
        say(f"writing_prompt_{i}")
        ctx = {"brief": brief, "bible": plan["bible"], "full_outline": plan["outline"], "contract": contract,
               "validated_state": state, "previous_prompts": written[-2:]}
        head = f"Start with the line '## Prompt {i}'. " if multi else ""
        msgs = [{"role": "system", "content": SYSTEM + f" Write ONLY prompt {i}. {head}Then one line per field, labelled exactly: "
                 + ", ".join(f.capitalize() + ":" for f in FIELDS)
                 + ". Copy each present character's bible descriptor, the bible style line and the bible negative line WORD FOR WORD. "
                   "Show only this contract's moment; never reveal later events; never change locked appearance."},
                {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}]
        problems, review, cand = ["not attempted"], None, state
        for _ in range(1 + RETRIES):
            text = _call(msgs, 3000)
            problems = check_prompt(text, i, plan, multi)
            if not problems:
                say("validating_output")
                try:
                    review = parse_json(_call([{"role": "system", "content": REVIEW},
                                               {"role": "user", "content": json.dumps({**ctx, "draft": text}, ensure_ascii=False)}], 3000))
                    problems, cand = validate_review(review, text, state, contract)
                except (ValueError, TypeError):
                    problems = ["Reviewer did not return valid JSON"]
            if not problems:
                break
            msgs = msgs[:2] + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"Prompt {i} failed checks ({'; '.join(problems[:8])}). Repair ONLY this prompt."}]
        if problems:
            raise OpError(f"Prompt {i} could not be completed after retries ({'; '.join(problems[:4])}).")
        ck["prompts"][key], ck["reviews"][key] = text, review
        state = cand; written.append(text); save()

    b = plan["bible"]
    bible_md = "# Consistency Bible\n\n" + "\n".join(f"- **{n}:** {d}" for n, d in b["characters"].items()) + \
        f"\n- **Style:** {b['style']}\n- **Setting:** {b['setting']}\n- **Negative:** {b['negative']}\n"
    document = (bible_md + "\n---\n\n" if multi else "") + "\n\n---\n\n".join(w.strip() for w in written) + "\n"
    if multi and not ck.get("audit"):
        say("validating_output")
        try:
            audit = parse_json(_call([{"role": "system", "content": REVIEW + " Final whole-series audit across ALL prompts: identity drift, style drift, continuity breaks, count. Return changes:[]."},
                                      {"role": "user", "content": json.dumps({"brief": brief, "plan": plan, "series": document}, ensure_ascii=False)}], 4000))
        except (ValueError, TypeError):
            raise OpError("Final prompt audit did not return valid JSON; series not approved.")
        problems, _ = validate_review(audit, document, state)
        if audit.get("changes"):
            problems.append("Final audit cannot mutate validated state")
        if problems:
            raise OpError("Final prompt audit remains unresolved: " + "; ".join(problems[:4]))
        ck["audit"] = audit; save()
    say("assembling")
    out = os.path.join(workdir, "prompts.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(document)
    return {"output": out, "mime": "text/markdown",
            "result": {"prompts": count, "linked": multi, "pipeline_version": VERSION, "audited": multi}}


run = WORKER.run
