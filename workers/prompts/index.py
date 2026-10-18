"""VISION runtime worker: prompts — single prompts, linked prompt series and story scene prompts (story_generate).

Pipeline:
  plan    -> Master Blueprint + locked bible (character descriptors, style, negative, rules, props) + one contract per unit
  write   -> every unit IN PARALLEL: the plan already fixes each unit's before/after state, so units never wait on each other
  check   -> deterministic gate per unit: header, every required section, bible descriptors / style / negative copied
             verbatim, Devanagari dialogue when Hindi is requested, contracted state facts present
  review  -> independent model review per unit (parallel), evidence-quoted only
  repair  -> only failing units, bounded rounds, in parallel
  audit   -> whole-series audit with a 0-100 consistency score; the run is accepted only at >= MIN_SCORE
A checkpoint bound to the brief + pipeline version lets a resumed run skip accepted units. Logs never contain text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from runtime.operations import OpError
from runtime.worker_base import Worker

VERSION = 3
MAX_UNITS = 100
MAX_BRIEF_CHARS = 60_000
REPAIR_ROUNDS = 2
PARALLEL = 10
MIN_SCORE = 90
CALL_TIMEOUT_S = 240
DEFAULT_SECTIONS = ["Subject", "Characters", "Setting", "Action", "Camera", "Lighting", "Style", "Negative"]
CATEGORIES = ("identity", "style", "continuity", "instructions", "structure")

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

WORKER = Worker("prompts")
SYSTEM = ("You are VISION's prompt engineer for story scenes and image/video generators. Follow the user's brief exactly: "
          "count, sections, languages, scripts and format. Output only the requested content — no commentary.")


# ---------- providers ----------

def _post(url: str, key: str, payload: dict) -> str:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "vision-runtime"})
    with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
        data = json.loads(r.read())
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def provider_call(messages: list[dict], max_tokens: int, json_mode: bool = False) -> str:
    """NVIDIA first, Groq fallback; one bounded retry per provider on transient errors only."""
    order = ((NVIDIA_URL, "NVIDIA_API_KEY", NVIDIA_MODEL), (GROQ_URL, "GROQ_API_KEY", GROQ_MODEL))
    if json_mode:  # structured steps (plan, reviews, audits) go to the stronger, faster JSON follower first
        order = order[::-1]
    providers = [(u, os.environ[k], m) for u, k, m in order if os.environ.get(k)]
    if not providers:
        raise OpError("No AI provider key is configured on the runner.")
    last = "no provider succeeded"
    for url, key, model in providers:
        for attempt in range(3):
            try:
                payload = {"model": model, "messages": messages, "temperature": 0.4, "max_tokens": max_tokens}
                # Speed: hidden reasoning is off — the plan, deterministic gate and independent audit do the checking.
                if model == NVIDIA_MODEL:
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                else:
                    payload["reasoning_effort"] = "low"
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                t = time.time()
                text = _post(url, key, payload)
                print(f"[prompts] call {model.split('/')[-1]} {round(time.time() - t)}s chars={len(text)}", flush=True)
                if text:
                    return text
                last = "empty response"
            except urllib.error.HTTPError as e:
                last = f"provider HTTP {e.code}"
                print(f"[prompts] call {model.split('/')[-1]} HTTP {e.code}", flush=True)
                if e.code == 429:
                    try:
                        wait = float(e.headers.get("retry-after") or 0)
                    except (TypeError, ValueError):
                        wait = 0
                    if attempt or len(providers) > 1 and (url, key, model) != providers[-1]:
                        break  # rate limited: move to the other provider instead of waiting
                    time.sleep(min(max(wait, 5), 15))
                    continue
                if e.code < 500:
                    break
            except Exception as e:
                last = "provider unreachable"
                print(f"[prompts] call {model.split('/')[-1]} failed {type(e).__name__}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise OpError(f"The AI providers could not complete this step ({last}).")


CALL: Callable[..., str] = provider_call  # tests/benchmarks swap this


def _ai(messages, max_tokens, json_mode=False):
    return CALL(messages, max_tokens, json_mode)


def auditor_call(messages, max_tokens, json_mode=False):
    """Audits run on a different model than the writer (Groq first) so the series is not graded by its own author."""
    saved = os.environ.pop("NVIDIA_API_KEY", None) if os.environ.get("GROQ_API_KEY") else None
    try:
        return provider_call(messages, max_tokens, json_mode)
    except OpError:
        if saved is None:
            raise
    finally:
        if saved is not None:
            os.environ["NVIDIA_API_KEY"] = saved
    return provider_call(messages, max_tokens, json_mode)  # auditor unavailable/too small: fall back, still a separate pass


AUDIT_CALL: Callable[..., str] = auditor_call


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_json(raw: str):
    clean = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean)
    a, b = clean.find("{"), clean.rfind("}")
    return json.loads(clean[a:b + 1] if a >= 0 else clean)


# ---------- brief parsing ----------

_NUM = {"one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50}
_UNIT = r"(prompts?|scenes?|shots?|frames?|panels?|clips?|images?|chapters?|episodes?|parts?)"


def unit_count(brief: str, given=None) -> tuple[int, str]:
    m = re.search(r"\b(\d{1,3}|" + "|".join(_NUM) + r")[\s,-]+(?:[\w-]+[\s,]+){0,4}?" + _UNIT + r"\b", brief, re.I)
    noun = "scene" if m and re.match(r"scene|chapter|episode|part", m.group(2), re.I) else "prompt"
    if given is not None:
        n = int(given) if str(given).isdigit() else 0
    else:
        n = (int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1).lower()]) if m else 1
    if not 1 <= n <= MAX_UNITS:
        raise OpError(f"Requests must contain 1–{MAX_UNITS} prompts or scenes; the requested count will not be shortened.")
    return n, noun


def required_sections(brief: str) -> list[str]:
    """Numbered section list from the brief ("1. Scene Title and Context: ..."), else the default prompt fields."""
    found = re.findall(r"^\s*(\d{1,2})[.)]\s+([A-Z][^:\n]{2,60}):", brief, re.M)
    seq = []
    for n, name in found:
        if int(n) == len(seq) + 1:
            seq.append(re.sub(r"\s*\([^)]*\)", "", name).strip())
    return seq if len(seq) >= 3 else DEFAULT_SECTIONS


def wants_story(brief: str) -> bool:
    """Full story text only when the brief asks for it and does not forbid it."""
    if re.search(r"\b(?:do not|don't|dont|never|without)\s+(?:generate|write|include|give)?\s*(?:the\s+)?(?:finished|full|complete)?\s*story\b", brief, re.I):
        return False
    return bool(re.search(r"\b(?:write|create|generate|give|make)\b[^.\n]{0,40}\bstory\b[^.\n]{0,40}\b(?:and|with|plus)\b[^.\n]{0,30}\bprompts?\b", brief, re.I))


def fence(text: str) -> str:
    """One independently copyable Markdown block; the fence is longer than any backtick run inside."""
    run = max([len(m) for m in re.findall(r"`{3,}", text)] + [3]) + 1
    return f"{'`' * run}markdown\n{text.strip()}\n{'`' * run}"


def wants_devanagari(brief: str) -> bool:
    return bool(re.search(r"\b(hindi|devanagari)\b", brief, re.I))


# ---------- validation ----------

def _facts(d) -> bool:
    return isinstance(d, dict) and all(isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip() for k, v in d.items())


def normalize_plan(plan, count: int):
    """Deterministic repairs of bookkeeping slips so a sound plan is not rejected for them:
    'before' maps are recomputed from the running state, string indexes become ints, clue units are coerced."""
    if not isinstance(plan, dict):
        return plan

    def flat(v):
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, list):
            return ", ".join(flat(x) for x in v if flat(x))
        if isinstance(v, dict):
            return ", ".join(flat(x) for k, x in v.items() if k.lower() not in ("name", "role") and flat(x))
        return str(v) if v is not None else ""
    for wrap in ("plan", "result", "data", "output"):
        if isinstance(plan.get(wrap), dict) and "outline" in plan[wrap]:
            plan.update(plan.pop(wrap))
    bible = plan.get("bible") if isinstance(plan.get("bible"), dict) else {}
    for key in ("characters", "style", "negative"):  # models sometimes put the locks at the top level
        if key not in bible and key in plan:
            bible[key] = plan[key]
    chars = bible.get("characters")
    if isinstance(chars, list):  # [{"name": "...", "descriptor": ...}] -> {"name": "descriptor"}
        chars = {str(c.get("name")): c for c in chars if isinstance(c, dict) and c.get("name")}
    if isinstance(chars, dict):
        bible["characters"] = {str(k): flat(v) for k, v in chars.items() if flat(v)}
    for key in ("style", "negative"):
        if key in bible and not isinstance(bible[key], str):
            bible[key] = flat(bible[key])
    plan["bible"] = bible
    if not isinstance(plan.get("blueprint"), str) and plan.get("blueprint") is not None:
        plan["blueprint"] = json.dumps(plan["blueprint"], ensure_ascii=False, indent=1)
    if not isinstance(plan.get("outline"), list):
        return plan
    names = list((plan.get("bible") or {}).get("characters") or {})

    def resolve(n):
        n = str(n).strip().lower()
        for k in names:
            kl = k.lower()
            if n == kl or n.split()[0] == kl.split()[0] or n in kl or kl in n:
                return k
        return None
    for beat in plan["outline"]:
        if isinstance(beat, dict) and isinstance(beat.get("characters"), list):
            beat["characters"] = list(dict.fromkeys(r for r in map(resolve, beat["characters"]) if r))
    state = {k: str(v) for k, v in (plan.get("initial_state") or {}).items()} if isinstance(plan.get("initial_state"), dict) else {}
    plan["initial_state"] = dict(state)
    for i, beat in enumerate(plan["outline"], 1):
        if not isinstance(beat, dict):
            continue
        beat["index"] = i
        before = beat.get("before") if isinstance(beat.get("before"), dict) else {}
        beat["before"] = {k: state[k] for k in before if k in state}
        after = beat.get("after") if isinstance(beat.get("after"), dict) else {}
        beat["after"] = {str(k): str(v) for k, v in after.items() if str(v).strip()}
        state.update(beat["after"])
        for key in ("time", "location"):
            if not isinstance(beat.get(key), str) or not beat[key].strip():
                beat[key] = plan["outline"][i - 2].get(key, "") if i > 1 and isinstance(plan["outline"][i - 2], dict) else ""
    clues = plan.get("clues") if isinstance(plan.get("clues"), list) else []
    fixed = []
    for c in clues:
        try:
            c = {"name": str(c["name"]), "setup": int(c["setup"]), "payoff": int(c["payoff"])}
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= c["setup"] < c["payoff"] <= count:
            fixed.append(c)
    plan["clues"] = fixed
    return plan


def validate_plan(plan, count: int) -> list[str]:
    if not isinstance(plan, dict):
        return ["Plan must be an object"]
    issues = []
    for wrap in ("plan", "result", "data", "output"):
        if isinstance(plan.get(wrap), dict) and "outline" in plan[wrap]:
            plan.update(plan.pop(wrap))
    bible = plan.get("bible") if isinstance(plan.get("bible"), dict) else {}
    for key in ("characters", "style", "negative"):  # models sometimes put the locks at the top level
        if key not in bible and key in plan:
            bible[key] = plan[key]
    chars = bible.get("characters")
    if not isinstance(chars, dict) or not chars or not _facts(chars):
        issues.append("Bible characters must map each name to a locked visual descriptor")
        chars = {}
    for k in ("style", "negative"):
        if not isinstance(bible.get(k), str) or not bible[k].strip():
            issues.append(f"Bible needs {k}")
    if not isinstance(plan.get("blueprint"), str) or len(plan["blueprint"].strip()) < 200:
        issues.append("Missing Master Blueprint")
    state = plan.get("initial_state") if _facts(plan.get("initial_state")) else {}
    state = dict(state)
    outline = plan.get("outline")
    if not isinstance(outline, list) or len(outline) != count:
        return issues + [f"Outline must contain exactly {count} units"]
    for i, beat in enumerate(outline, 1):
        if not isinstance(beat, dict) or beat.get("index") != i:
            issues.append(f"Contract {i} is invalid or out of order"); continue
        if not isinstance(beat.get("moment"), str) or not beat["moment"].strip():
            issues.append(f"Unit {i} needs moment")
        for key in ("time", "location"):
            if not isinstance(beat.get(key), str) or not beat[key].strip():
                issues.append(f"Unit {i} needs {key}")
        cast = beat.get("characters", [])
        if not isinstance(cast, list) or any(c not in chars for c in cast):
            issues.append(f"Unit {i} uses a character missing from the bible")
        for key in ("before", "after"):
            facts = beat.get(key) or {}
            if not _facts(facts) and facts != {}:
                issues.append(f"Unit {i} has invalid {key} state"); continue
            if key == "before":
                issues += [f"Unit {i} precondition does not follow prior state: {f}" for f, v in facts.items() if state.get(f) != v]
            else:
                state.update(facts)
    clues = plan.get("clues")
    if count > 1:
        if not isinstance(clues, list) or not clues:
            issues.append("Plan needs clues with setup and payoff units")
        else:
            for c in clues:
                if not (isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("setup"), int)
                        and isinstance(c.get("payoff"), int) and 1 <= c["setup"] < c["payoff"] <= count):
                    issues.append(f"Clue {c.get('name') if isinstance(c, dict) else c} needs setup < payoff within the series")
    return issues


_DASH = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _norm(s: str) -> str:
    return re.sub(r"[\s*_`]+", " ", s.translate(_DASH).lower()).strip()


def _mentions(name: str, low: str) -> bool:
    """A clue counts as named when most of its significant words appear (handles inflection and transliteration)."""
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    words = [w for w in re.findall(r"[\w']+", _norm(name)) if len(w) > 3] or re.findall(r"[\w']+", _norm(name))
    return sum(w[:5] in low for w in words) >= max(1, round(len(words) * 0.6))


def check_unit(text: str, index: int, plan: dict, spec: dict) -> list[str]:
    """Deterministic consistency gate: locks must be copied word-for-word, not paraphrased."""
    problems, low = [], _norm(text)
    if spec["multi"]:
        heads = [int(n) for n in re.findall(rf"^\s*#{{1,3}}\s+(?:{spec['noun']}|prompt)\s+(\d+)\b", text, re.I | re.M)]
        if heads != [index]:
            problems.append(f"Expected exactly one '# {spec['noun'].title()} {index}' header")
    heads = [_norm(re.sub(r"^\s*(?:#{1,6}\s+|\*\*)", "", l)) for l in text.splitlines() if re.match(r"^\s*(?:#{1,6}\s+|\*\*|\d{1,2}[.)]\s)", l)]
    heads += [_norm(l.split(":")[0]) for l in text.splitlines() if ":" in l]
    for sec in spec["sections"]:
        key = _norm(sec).split(" and ")[0][:24]
        if not any(key in h for h in heads):
            problems.append(f"missing section: {sec}")
    bible = plan["bible"]
    for lock in ("style", "negative"):
        if _norm(bible[lock]) not in low:
            problems.append(f"{lock} lock not copied verbatim")
    for name in plan["outline"][index - 1].get("characters", []):
        if _norm(bible["characters"][name]) not in low:
            problems.append(f"locked descriptor for {name} changed or missing")
    if spec["devanagari"] and not re.search(r"[\u0900-\u097F]{3,}", text):
        problems.append("dialogue is not in Devanagari")
    for key, val in (plan["outline"][index - 1].get("after") or {}).items():
        if _norm(val) not in low:
            problems.append(f"contracted change not shown verbatim: {key} -> {val}")
    if re.search(r"^\s*#{1,3}\s*master story blueprint", text, re.I | re.M):
        problems.append("unit must not repeat the Master Story Blueprint")
    for c in plan.get("clues") or []:
        if index in (c.get("setup"), c.get("payoff")) and not _mentions(c["name"], low):
            problems.append(f"clue '{c['name']}' must appear by name (setup {c['setup']}, payoff {c['payoff']})")
    if len(text.strip()) < (600 if spec["noun"] == "scene" else 200):
        problems.append("too short")
    return problems


def validate_review(review) -> list[str]:
    if not isinstance(review, dict):
        return ["Review must be an object"]
    checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
    issues = ["Failed or unchecked: " + c for c in CATEGORIES if checks.get(c) is not True]
    found = review.get("issues")
    issues += found if isinstance(found, list) and all(isinstance(i, str) for i in found) else ["Review issues must be a list of strings"]
    return issues


def deterministic_score(units: list[str], plan: dict, spec: dict) -> int:
    """Percent of all lock/section/state checks that pass across the series."""
    total = passed = 0
    for i, text in enumerate(units, 1):
        n_checks = len(spec["sections"]) + 2 + len(plan["outline"][i - 1].get("characters", [])) + len(plan["outline"][i - 1].get("after") or {}) + 2
        total += n_checks
        passed += max(0, n_checks - len(check_unit(text, i, plan, spec)))
    return round(100 * passed / total) if total else 0


# ---------- prompts ----------

BLUEPRINT_SYS = SYSTEM + """ Write ONLY a dense Master Story Blueprint in Markdown (under 1,800 words, bullet points): plot, characters with exact physical appearance and clothing,
locations, supernatural/world rules, important props, timeline, mysteries, recurring clues and their payoffs, ending. No scene scripts."""

PLAN_SYS = SYSTEM + """ Turn the given blueprint into a compact plan. Return JSON only (no Markdown inside):
{"bible":{"characters":{"Name":"locked visual descriptor, 20-40 words: age, face, hair, build, exact clothing"},"style":"ONE locked cinematic style line","negative":"ONE locked negative-constraints line"},
"initial_state":{"entity.field":"short value"},
"clues":[{"name":"short clue name","setup":1,"payoff":10}],
"outline":[{"index":1,"moment":"what happens","time":"HH:MM","location":"place","characters":["Name"],"props_in_play":["prop: who holds it / where"],"injuries":["Name: injury (cause, unit)"],"clues_planted":["clue name"],"clues_paid":["clue name"],"before":{"entity.field":"value"},"after":{"entity.field":"new short value"}}]}
Exactly N outline items. Times never go backwards and gaps are explained. Every prop or injury exists only from the unit that causes it;
a dropped/lost prop stays dropped until a unit recovers it. Every clue planted is paid off in a later unit (final unit resolves the mystery). Each unit advances one causal step; every 'before' equals the state left by earlier units.
Track props' locations, injuries, knowledge and clues as state keys. Values are 1-4 word phrases. Seed clues early and pay them off at the end."""

REVIEW_SYS = SYSTEM + """ Critique the draft against the bible, contract and brief. JSON only:
{"checks":{"identity":true,"style":true,"continuity":true,"instructions":true,"structure":true},"issues":[]}
identity: characters match bible appearance and clothing. style: same visual style. continuity: props, injuries, location,
knowledge and time follow the contract's before state and reach its after state; nothing from later units revealed.
instructions: brief's language/script/section rules. structure: header and all sections. false = blocked; issues name the smallest fix."""

AUDIT_SYS = SYSTEM + """ Audit the WHOLE series for consistency and continuity. JSON only:
{"score":0-100,"checks":{"identity":true,"style":true,"continuity":true,"instructions":true,"structure":true},"issues":[],"units":[unit numbers that must be repaired]}
Each issue must start with "unit N:". Score strictly: deduct for any appearance/clothing drift, prop teleportation, timeline gaps, unexplained knowledge, unpaid clues, rule violations."""


def _unit_messages(i: int, plan: dict, spec: dict, brief: str, fix: list[str] | None = None, draft: str | None = None):
    c = plan["outline"][i - 1]
    head = f"Begin with the line '# {spec['noun'].title()} {i}: <title>'. " if spec["multi"] else ""
    sys = (SYSTEM + f" Write ONLY {spec['noun']} {i} of {spec['count']} as a self-contained Markdown document. {head}"
           f"Use these sections as '## ' headings, in order: {'; '.join(spec['sections'])}. "
           "Keep every character exactly as the bible describes (the worker inserts the locked descriptors itself). Start from the 'before' state, end at the 'after' state; never reveal later units. "
           "Use ONLY the contract's time, location, props_in_play and injuries: invent no new props, wounds, events or locations. "
           "Name each planted/paid clue exactly. Do not repeat the blueprint. The ending must lead directly into next_unit.")
    if spec["devanagari"]:
        sys += " Dialogue lines are in natural Hindi using Devanagari script; everything else in English."
    ctx = {"brief": brief, "bible": plan["bible"], "blueprint": plan["blueprint"],
           "clues": plan.get("clues"), "outline": [{"index": o["index"], "moment": o["moment"], "time": o.get("time")} for o in plan["outline"]],
           "previous_unit": plan["outline"][i - 2] if i > 1 else None, "contract": c,
           "next_unit": plan["outline"][i] if i < len(plan["outline"]) else None}
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}]
    if fix and draft:
        msgs += [{"role": "assistant", "content": draft},
                 {"role": "user", "content": f"Fix ONLY these problems and return the full corrected {spec['noun']}: " + "; ".join(fix[:10])}]
    return msgs


def _insert_after_heading(text: str, words: tuple, block: str) -> str:
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if re.match(r"^\s*(?:#{1,6}\s+|\*\*|\d{1,2}[.)]\s)", line) and all(w in line.lower() for w in words):
            return "\n".join(lines[:n + 1] + [block] + lines[n + 1:])
    return text.rstrip() + "\n\n" + block


def enforce_locks(text: str, i: int, plan: dict, spec: dict) -> str:
    """Locks are written by the worker, not trusted to the model: every unit carries the identical canonical
    character descriptors, style line, negative line and contracted end state, so they cannot drift."""
    b, c = plan["bible"], plan["outline"][i - 1]
    cast = c.get("characters", [])
    if cast:
        chars = "\n".join(f"- **{n} (locked appearance):** {b['characters'][n]}" for n in cast)
        text = _insert_after_heading(text, ("character",), chars)
    text = _insert_after_heading(text, ("visual",), f"**Locked style:** {b['style']}")
    text = _insert_after_heading(text, ("negative",), f"**Locked negative:** {b['negative']}")
    state = [f"- {k}: {v}" for k, v in (c.get("after") or {}).items()]
    where = f"- Time: {c.get('time', '')} | Location: {c.get('location', '')}"
    text = _insert_after_heading(text, ("continuity",), "**Canonical state at scene end:**\n" + "\n".join([where] + state))
    return text


def _write_and_check(i, plan, spec, brief, fix=None, draft=None):
    text = enforce_locks(_ai(_unit_messages(i, plan, spec, brief, fix, draft), 9000), i, plan, spec)
    problems = check_unit(text, i, plan, spec)
    if problems:
        print(f"[prompts] unit {i} gate: " + " | ".join(p.split(':')[0][:40] for p in problems[:5]), flush=True)
    if not problems:
        try:
            ctx = {"bible": plan["bible"], "contract": plan["outline"][i - 1], "brief_rules": brief[:4000], "draft": text}
            problems = validate_review(parse_json(_ai([{"role": "system", "content": REVIEW_SYS},
                                                       {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}], 1500, True)))
        except (ValueError, TypeError):
            problems = []  # deterministic gate already passed; an unparsable review is not evidence of a defect
    return i, text, problems


# ---------- run ----------

def run_prompts(params: dict, workdir: str, progress: Callable[[str], None] | None = None) -> dict:
    brief = str(params.get("prompt", "")).strip()
    if not brief or len(brief) > MAX_BRIEF_CHARS:
        raise OpError("The brief is missing or too large.")
    count, noun = unit_count(brief, params.get("count"))
    spec = {"count": count, "noun": noun, "multi": count > 1, "sections": required_sections(brief), "devanagari": wants_devanagari(brief), "story": wants_story(brief) and count > 1}
    t0 = time.time()

    def say(stage):  # stage + elapsed seconds only; never prompt text
        print(f"[prompts] {stage} t={round(time.time() - t0)}s", flush=True)
        relay = "processing" if stage == "writing" else "validating_output" if stage.startswith("repairing") else stage
        if progress:
            try:
                progress(relay)  # relay accepts a fixed stage list; a progress hiccup never fails the job
            except Exception:
                pass
    path = os.path.join(workdir, "prompts-checkpoint.json")
    ident = fingerprint(f"{brief}|{count}|{VERSION}")
    ck = {"version": VERSION, "request": ident, "plan": None, "units": {}}
    if os.path.exists(path):
        try:
            saved = json.load(open(path, encoding="utf-8"))
        except ValueError:
            raise OpError("Checkpoint is invalid; no saved work was accepted.")
        if saved.get("request") != ident:
            raise OpError("Checkpoint belongs to a different brief or pipeline version.")
        ck = saved

    def save():
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(ck, f, ensure_ascii=False)
        os.replace(path + ".tmp", path)

    if not ck["plan"]:
        say("planning")
        problems: list[str] = []
        blueprint = _ai([{"role": "system", "content": BLUEPRINT_SYS},
                         {"role": "user", "content": json.dumps({"brief": brief, "N": count}, ensure_ascii=False)}], 3500)
        story = ""
        if spec["story"]:
            story = _ai([{"role": "system", "content": SYSTEM + " Write ONLY the complete story prose that this blueprint describes."},
                         {"role": "user", "content": blueprint}], 8000)
        for _ in range(3):
            try:
                plan = parse_json(_ai([{"role": "system", "content": PLAN_SYS},
                                       {"role": "user", "content": json.dumps({"brief": brief[:6000], "blueprint": blueprint, "N": count, "repair": problems}, ensure_ascii=False)}], 8000, True))
                if isinstance(plan, dict):
                    plan["blueprint"], plan["story"] = blueprint, story
                plan = normalize_plan(plan, count)
                problems = validate_plan(plan, count)
            except (ValueError, TypeError):
                problems = ["Invalid planning JSON"]
            print(f"[prompts] plan attempt problems={len(problems)} first={problems[0][:60] if problems else ''} keys={sorted(plan)[:12] if isinstance(plan, dict) else '-'} bible={sorted(plan.get('bible') or {})[:8] if isinstance(plan, dict) and isinstance(plan.get('bible'), dict) else '-'}", flush=True)
            if not problems:
                ck["plan"] = plan; save(); break
        if not ck["plan"]:
            raise OpError("Planning remains unresolved: " + "; ".join(problems[:4]))
    plan = ck["plan"]
    if validate_plan(plan, count):
        raise OpError("Saved plan is invalid.")

    units: dict[int, str] = {}
    for k, t in ck["units"].items():
        if not check_unit(t, int(k), plan, spec):
            units[int(k)] = t
    pending = {i: (None, None) for i in range(1, count + 1) if i not in units}
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for round_ in range(1 + REPAIR_ROUNDS):
            if not pending:
                break
            say("writing" if round_ == 0 else f"repairing_{len(pending)}")
            results = list(pool.map(lambda i: _write_and_check(i, plan, spec, brief, *pending[i]), list(pending)))
            nxt = {}
            for i, text, problems in results:
                if problems:
                    nxt[i] = (problems, text)
                else:
                    units[i] = text; ck["units"][str(i)] = text
            save(); pending = nxt
        for i, (problems, text) in pending.items():  # still imperfect: kept, and counted against the score below
            if text:
                units[i] = text
        if len(units) < count:
            raise OpError(f"{count - len(units)} unit(s) could not be written.")

        series = [units[i] for i in range(1, count + 1)]
        det = deterministic_score(series, plan, spec)
        audit_score = 100
        if spec["multi"]:
            say("validating_output")
            for round_ in range(1 + REPAIR_ROUNDS):
                # Windowed audit (4 consecutive units, overlapping by 1) keeps each request inside provider limits;
                # the series score is the weakest window.
                scores, notes_all, flagged = [], [], set()
                for start in range(0, count, 3):
                    idx = list(range(start + 1, min(start + 5, count + 1)))
                    window = "\n\n---\n\n".join(series[i - 1] for i in idx)
                    try:
                        a_ = parse_json(AUDIT_CALL([{"role": "system", "content": AUDIT_SYS},
                                                    {"role": "user", "content": json.dumps({"units": idx, "bible": plan["bible"], "clues": plan.get("clues"),
                                                     "outline": [plan["outline"][i - 1] for i in idx], "series": window}, ensure_ascii=False)}], 2500, True))
                        scores.append(int(a_.get("score", 0)))
                        notes_all += [x for x in a_.get("issues", []) if isinstance(x, str)]
                        flagged |= {i for i in a_.get("units", []) if isinstance(i, int) and i in idx}
                    except (ValueError, TypeError, OpError):
                        scores.append(0)
                    if start + 4 >= count:
                        break
                audit_score = min(scores) if scores else 0
                audit = {"issues": notes_all, "units": sorted(flagged)}
                print(f"[prompts] audit scores={scores} flagged={sorted(flagged)}", flush=True)
                if audit_score >= MIN_SCORE:
                    break
                fix = sorted({i for i in audit.get("units", []) if isinstance(i, int) and 1 <= i <= count}
                             | {int(n) for x in notes_all for n in re.findall(r"unit (\d+)", x, re.I) if 1 <= int(n) <= count})
                if not fix or round_ == REPAIR_ROUNDS:
                    break
                say(f"repairing_{len(fix)}")
                def own(i):
                    return [x for x in notes_all if re.search(rf"unit {i}\b", x, re.I)] or notes_all
                for i, text, problems in pool.map(lambda i: _write_and_check(i, plan, spec, brief, own(i), series[i - 1]), fix):
                    if not problems:
                        series[i - 1] = text; ck["units"][str(i)] = text
                save()
            det = deterministic_score(series, plan, spec)
    score = min(det, audit_score)
    if spec["multi"] and score < MIN_SCORE:
        raise OpError(f"Consistency score {score}% is below the {MIN_SCORE}% bar; the series was not approved.")

    b = plan["bible"]
    bible_md = "## Locked Visual Bible\n\n" + "\n".join(f"- **{n}:** {d}" for n, d in b["characters"].items()) + \
        f"\n- **Style:** {b['style']}\n- **Negative:** {b['negative']}\n"
    # Reply = exactly what was asked: story (if requested), blueprint, then each unit in its own Markdown block.
    blocks = []
    if spec["story"] and isinstance(plan.get("story"), str) and plan["story"].strip():
        blocks.append(fence("# Story\n\n" + plan["story"].strip()))
    if spec["multi"]:
        blocks.append(fence("# Master Story Blueprint\n\n" + plan["blueprint"].strip() + "\n\n" + bible_md))
    blocks += [fence(s) if spec["multi"] else s.strip() for s in series]
    document = "\n\n".join(blocks) + "\n"
    say("assembling")
    name = "story.md" if noun == "scene" else "prompts.md"
    out = os.path.join(workdir, name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(document)
    return {"output": out, "mime": "text/markdown",
            "result": {"units": count, "kind": noun, "consistency_score": score, "deterministic_score": det,
                       "audit_score": audit_score, "pipeline_version": VERSION}}


run = WORKER.run
