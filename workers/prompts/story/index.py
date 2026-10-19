"""VISION runtime worker: prompts — single prompts, linked prompt series and story scene prompts (story_generate).

Pipeline:
  plan    -> Master Blueprint + locked bible (character descriptors, style, negative, rules, props) + one contract per unit
  write   -> every unit IN PARALLEL: the plan already fixes each unit's before/after state, so units never wait on each other
  check   -> deterministic gate per unit: header, every required section, bible descriptors / style / negative copied
             verbatim, Devanagari dialogue when Hindi is requested, contracted state facts present
  review  -> independent fact extraction and windowed audit, evidence-quoted only
  repair  -> only failing units, bounded rounds, in parallel
  audit   -> whole-series audit with a 0-100 consistency score; the run is accepted only at >= MIN_SCORE
A checkpoint bound to the brief + pipeline version lets a resumed run skip accepted units. Logs never contain text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from runtime.operations import OpError
from runtime.worker_base import Worker
from validators import Constraint, Contract, validate
from validators import continuity as _continuity
from validators import repair as _repair
from validators.compliance import spoken_words as _spoken_words

VERSION = 7
MAX_UNITS = 100
MAX_BRIEF_CHARS = 60_000
REPAIR_ROUNDS = 2
PARALLEL = 4
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


NVIDIA_RPM = 40          # Nemotron 3 Super quota: 40 requests per minute
NVIDIA_CONCURRENCY = 4   # at most 4 requests in flight at once


class _RateLimiter:
    """Thread-safe sliding 60s window plus a concurrency cap, shared by every thread in the job."""

    def __init__(self, rpm: int, concurrency: int):
        self.rpm, self.sem, self.lock, self.stamps = rpm, threading.BoundedSemaphore(concurrency), threading.Lock(), []

    def acquire(self):
        self.sem.acquire()
        while True:
            with self.lock:
                now = time.time()
                self.stamps = [t for t in self.stamps if now - t < 60]
                if len(self.stamps) < self.rpm:
                    self.stamps.append(now)
                    return
                wait = 60 - (now - self.stamps[0]) + 0.05
            time.sleep(wait)

    def release(self):
        self.sem.release()


NVIDIA_LIMIT = _RateLimiter(NVIDIA_RPM, NVIDIA_CONCURRENCY)


def provider_call(messages: list[dict], max_tokens: int, json_mode: bool = False, think: bool = False) -> str:
    """Nemotron 3 Super for every step, paced to its quota (40/min, 4 at once). 429 waits and retries on Nemotron;
    Groq is used only if Nemotron keeps failing with server errors or is unreachable."""
    providers = [(u, os.environ[k], m) for u, k, m in ((NVIDIA_URL, "NVIDIA_API_KEY", NVIDIA_MODEL), (GROQ_URL, "GROQ_API_KEY", GROQ_MODEL))
                 if os.environ.get(k)]
    if not providers:
        raise OpError("No AI provider key is configured on the runner.")
    last = "no provider succeeded"
    for url, key, model in providers:
        nvidia = model == NVIDIA_MODEL
        for attempt in range(5 if nvidia else 2):
            payload = {"model": model, "messages": messages, "temperature": 0.4, "max_tokens": max_tokens}
            if nvidia:
                payload["chat_template_kwargs"] = {"enable_thinking": think}
            else:
                payload["reasoning_effort"] = "medium" if think else "low"
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if nvidia:
                NVIDIA_LIMIT.acquire()
            try:
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
                    time.sleep(min(max(wait, 3 * (attempt + 1)), 30))
                    continue
                if e.code < 500:
                    break
            except Exception as e:
                last = "provider unreachable"
                print(f"[prompts] call {model.split('/')[-1]} failed {type(e).__name__}", flush=True)
            finally:
                if nvidia:
                    NVIDIA_LIMIT.release()
            time.sleep(2 * (attempt + 1))
    raise OpError(f"The AI providers could not complete this step ({last}).")


CALL: Callable[..., str] = provider_call  # tests/benchmarks swap this


def _ai(messages, max_tokens, json_mode=False, think=False):
    try:
        return CALL(messages, max_tokens, json_mode, think)
    except TypeError:  # simple test doubles
        return CALL(messages, max_tokens, json_mode)


def auditor_call(messages, max_tokens, json_mode=False):
    """Checks run as separate Nemotron passes with their own instructions; code, not the model, makes the final judgement."""
    return provider_call(messages, max_tokens, json_mode)


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
    noun = "scene" if (m and re.match(r"scene|chapter|episode|part", m.group(2), re.I)) or re.search(r"\bstory\b|\bscenes?\b", brief, re.I) else "prompt"
    if given is not None:
        n = int(given) if str(given).isdigit() else 0
    else:
        n = (int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1).lower()]) if m else max(1, (duration_seconds(brief) + 7) // 8) if duration_seconds(brief) and re.search(r"all\s+(?:the\s+)?scenes?", brief, re.I) else 1
    if not 1 <= n <= MAX_UNITS:
        raise OpError(f"Requests must contain 1–{MAX_UNITS} prompts or scenes; the requested count will not be shortened.")
    return n, noun


SECTION_PATTERNS = [("Story Line", r"story line"), ("Character Details", r"character (?:details|profiles|descriptions)"), ("Visual Prompt", r"visual prompts?"), ("Dialogue", r"dialogues?"), ("Sound Effects", r"sound (?:effects?|design)"), ("Camera Direction", r"camera(?: direction)?")]


def explicit_sections(brief: str) -> list[str]:
    # Delivery instructions, not the character/theme/constraints supplied as inputs.
    clauses = re.findall(r"(?:give|provide|return|output|deliver|show)\s+(?:me\s+)?([^.!?\n]+)", brief, re.I)
    if re.search(r"\b(?:only|just)\b", brief, re.I):
        clauses.append(brief)
    chosen = [name for name, pattern in SECTION_PATTERNS if any(re.search(pattern, c, re.I) for c in clauses)]
    return chosen


def fixed_background(brief: str) -> bool:
    return bool(re.search(r"background[^.!?\n]{0,40}(?:constant|same|unchanged|fixed)|(?:constant|same|unchanged|fixed)[^.!?\n]{0,40}background", brief, re.I))


def check_fixed_background(plan: dict, brief: str) -> list[str]:
    """Compatibility wrapper over validators.continuity (canonical room IDs, positions are not rooms)."""
    return [f.legacy() for f in fixed_background_findings(plan, brief)]


def fixed_background_findings(plan: dict, brief: str):
    return _continuity.check_fixed_background(plan, fixed_background(brief))


def dialogue_range(brief: str):
    m = re.search(r"(\d+)\s*(?:to|[-–])\s*(\d+)\s+words?\s+(?:of\s+)?dialog(?:ue|ues)", brief, re.I)
    if not m:
        m = re.search(r"dialog(?:ue|ues)[^.!?\n]{0,40}?(\d+)\s*(?:to|[-–])\s*(\d+)\s+words?", brief, re.I)
    return (int(m.group(1)), int(m.group(2))) if m else None


def dialogue_words(text: str) -> int:
    return _spoken_words(text)


def build_contract(brief: str, count: int) -> Contract:
    """Source-backed requirements compiled before any model call."""
    cs = [Constraint("units.count", "count", count, "count")]
    if explicit_sections(brief):
        cs.append(Constraint("sections", "fields", explicit_sections(brief), "delivery clause"))
    if dialogue_range(brief):
        cs.append(Constraint("dialogue.words", "range", list(dialogue_range(brief)), "dialogue word range"))
    if fixed_background(brief):
        cs.append(Constraint("background.fixed", "invariant", True, "background constant"))
    return Contract("story_generate", cs)


def required_sections(brief: str) -> list[str]:
    """Numbered section list from the brief ("1. Scene Title and Context: ..."), else the default prompt fields."""
    found = re.findall(r"^\s*(\d{1,2})[.)]\s+([A-Z][^:\n]{2,60}):", brief, re.M)
    seq = []
    for n, name in found:
        if int(n) == len(seq) + 1:
            seq.append(re.sub(r"\s*\([^)]*\)", "", name).strip())
    if len(seq) >= 3:
        return seq
    if explicit_sections(brief):
        return explicit_sections(brief)
    if re.search(r"\b(?:only|just)\b", brief, re.I):
        requested = [("Story Line", r"story line"), ("Character Details", r"character (?:details|profiles|descriptions)"), ("Visual Prompt", r"visual prompts?"), ("Dialogue", r"dialogues?"), ("Sound Effects", r"sound (?:effects?|design)"), ("Camera Direction", r"camera")]
        chosen = [name for name, pattern in requested if re.search(pattern, brief, re.I)]
        if chosen:
            return chosen
    if re.search(r"\bstory\b|\bscene", brief, re.I):
        return ["Story Line", "Character Details", "Visual Prompt", "Dialogue", "Sound Effects", "Camera Direction", "Scene Actions", "Continuity", "Negative Constraints", "Ending and Transition"]
    return DEFAULT_SECTIONS


def wants_story(brief: str) -> bool:
    """Full story text only when the brief asks for it and does not forbid it."""
    if re.search(r"\b(?:do not|don't|dont|never|without)\s+(?:generate|write|include|give)?\s*(?:the\s+)?(?:finished|full|complete)?\s*story\b", brief, re.I):
        return False
    if explicit_sections(brief) and not re.search(r"(?:give|provide|include|return)\s+(?:me\s+)?(?:the\s+)?(?:full|complete|finished)\s+story", brief, re.I):
        return False
    return bool(re.search(r"\b(?:write|create|generate|give|make)\s+(?:me\s+)?(?:a\s+|the\s+|an\s+)?(?:(?:\d+\s*[- ]?(?:seconds?|minutes?)|short|full|complete|horror|detailed|supernatural)\s+){0,4}(?:story|stories)\b", brief, re.I))


def fence(text: str) -> str:
    """One independently copyable Markdown block; the fence is longer than any backtick run inside."""
    run = max([len(m) for m in re.findall(r"`{3,}", text)] + [3]) + 1
    return f"{'`' * run}markdown\n{text.strip()}\n{'`' * run}"


def duration_seconds(brief: str) -> int:
    m = re.search(r"\b(\d+)\s*[- ]?(seconds?|secs?|minutes?|mins?)\b", brief, re.I)
    return int(m.group(1)) * (60 if m.group(2).lower().startswith("min") else 1) if m else 0


def grouped_scenes(brief: str) -> bool:
    return bool(re.search(r"(?:each|every|per)[^.\n]{0,100}(?:markdown\s+(?:box|block)|box|block)[^.\n]{0,140}(?:all|details|visual|dialogue|character)|(?:each|every|per)\s+(?:visual\s+)?prompt[^.\n]{0,60}(?:has|have|contains|includes)[^.\n]{0,80}(?:all|details|dialogue|character)|(?:all|every)[^.\n]{0,60}details[^.\n]{0,60}(?:each|per)\s+(?:scene|prompt)", brief, re.I))


def assemble_output(plan: dict, series: list[str], spec: dict, brief: str) -> str:
    blocks = []
    story = str(plan.get("story") or "").strip()
    if spec["story"] and story:
        match = re.match(r"^#\s+([^\n]+)\n+", story)
        title = match.group(1) if match else str(plan.get("title") or "Story")
        body = story[match.end():] if match else story
        blocks += [title, fence(body)]
    if spec.get("prompts", True):
        for text in series:
            selected = explicit_sections(brief)
            if selected:
                parts = re.split(r"(?m)(?=^## )", text)
                title = text.splitlines()[0] if text.startswith("# ") else ""
                text = title + "\n\n" + "\n\n".join(part.strip() for part in parts if any(re.match(r"^##\s+" + re.escape(name) + r"\b", part, re.I) for name in selected))
            if grouped_scenes(brief) or spec["noun"] != "scene":
                blocks.append(fence(text))
                continue
            lines = text.splitlines()
            heading = lines[0].lstrip("# ") if lines and lines[0].startswith("# ") else ""
            if heading:
                blocks.append(heading)
                text = "\n".join(lines[1:]).strip()
            sections = re.split(r"(?m)(?=^## )", text)
            blocks.extend(fence(section) for section in sections if section.strip())
    return "\n\n".join(blocks) + "\n"


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
    pos = dict(plan.get("start_positions") or {}) if isinstance(plan.get("start_positions"), dict) else {}
    props = dict(plan.get("start_props") or {}) if isinstance(plan.get("start_props"), dict) else {}
    pos = {resolve(k) or k: str(v) for k, v in pos.items()}
    for beat in plan["outline"]:
        if not isinstance(beat, dict):
            continue
        beat["start_positions"], beat["start_props"] = dict(pos), dict(props)
        ep = beat.get("end_positions") if isinstance(beat.get("end_positions"), dict) else {}
        pos.update({resolve(k) or k: str(v) for k, v in ep.items()})
        beat["end_positions"] = dict(pos)
        pr = beat.get("end_props") if isinstance(beat.get("end_props"), dict) else {}
        props.update({str(k): str(v) for k, v in pr.items()})
        beat["end_props"] = dict(props)
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
    last = (0, 0)
    for i, beat in enumerate(outline, 1):
        m = re.match(r"^\s*D(?:ay)?\s*(\d+)\D+(\d{1,2}):(\d{2})", str(beat.get("time", "")) if isinstance(beat, dict) else "", re.I)
        if count > 1 and not m:
            issues.append(f"Unit {i} time must be 'D<day> HH:MM'"); continue
        if m:
            t = (int(m.group(1)), int(m.group(2)) * 60 + int(m.group(3)))
            if t < last:
                # deterministic repair: a midnight crossing moves to the next day; otherwise continue 10 minutes later
                t = (last[0] + 1, t[1]) if t[0] == last[0] and t[1] < 6 * 60 and last[1] >= 18 * 60 else (last[0], last[1] + 10)
                t = (t[0] + t[1] // 1440, t[1] % 1440)
                beat["time"] = f"D{t[0]} {t[1] // 60:02d}:{t[1] % 60:02d}"
            elif i > 1 and (t[0] - last[0]) * 1440 + t[1] - last[1] > 180 and not beat.get("time_skip"):
                # deterministic repair: the writer must show the elapsed time on screen
                gap = ((t[0] - last[0]) * 1440 + t[1] - last[1]) // 60
                beat["time_skip"] = f"about {gap} hours pass since the previous unit; open by showing what happened in that time"
            last = t
    clues = plan.get("clues")
    if plan.get("requires_clues", True) and count > 1:
        if not isinstance(clues, list) or not clues:
            issues.append("Plan needs clues with setup and payoff units")
        else:
            for c in clues:
                if not (isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("setup"), int)
                        and isinstance(c.get("payoff"), int) and 1 <= c["setup"] < c["payoff"] <= count):
                    issues.append(f"Clue {c.get('name') if isinstance(c, dict) else c} needs setup < payoff within the series")
    wardrobe = r"\b(?:shirt|t-shirt|skirt|sweater|trousers|pants|shorts|shoes|socks|dress|uniform|coat|jacket)\b"
    for prop, holder in list((plan.get("start_props") or {}).items()):
        if not re.search(wardrobe, prop, re.I):
            continue
        if any((beat.get("end_props") or {}).get(prop, holder) != holder for beat in outline):
            issues.append(f"Immutable wardrobe {prop} changes owner/location; keep it worn by {holder}")
        else:
            # Clothing is already canon; tracking it as a transported prop is a category error.
            plan["start_props"].pop(prop, None)
            for beat in outline:
                for field in ("start_props", "end_props"):
                    (beat.get(field) or {}).pop(prop, None)
    return issues


_DASH = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _norm(s: str) -> str:
    return re.sub(r"[\s*_`]+", " ", s.translate(_DASH).lower()).strip()


def _mentions(name: str, low: str) -> bool:
    """A clue counts as named when most of its significant words appear (handles inflection and transliteration)."""
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", " ").replace("-", " ")
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
    bounds = spec.get("dialogue_range")
    if bounds:
        words = dialogue_words(text)
        if not bounds[0] <= words <= bounds[1]:
            problems.append(f"Dialogue has {words} spoken words; required {bounds[0]}–{bounds[1]} per scene, excluding speaker labels and delivery notes")
    bible = plan["bible"]
    visual_requested = any(re.search(r"visual|character|subject|style|setting", sec, re.I) for sec in spec["sections"])
    for lock in ("style", "negative"):
        if visual_requested and _norm(bible[lock]) not in low:
            problems.append(f"{lock} lock not copied verbatim")
    if visual_requested and spec.get("fixed_background") and _norm(bible.get("background", "")) not in low:
        problems.append("locked background description missing or changed")
    for name in plan["outline"][index - 1].get("characters", []):
        if visual_requested and _norm(bible["characters"][name]) not in low:
            problems.append(f"locked descriptor for {name} changed or missing")
    if spec["devanagari"] and not re.search(r"[\u0900-\u097F]{3,}", text):
        problems.append("dialogue is not in Devanagari")

    if re.search(r"^\s*#{1,3}\s*master story blueprint", text, re.I | re.M):
        problems.append("unit must not repeat the Master Story Blueprint")
    for c in plan.get("clues") or []:
        if index in (c.get("setup"), c.get("payoff")) and not _mentions(c["name"], low):
            problems.append(f"clue '{c['name']}' must appear by name (setup {c['setup']}, payoff {c['payoff']})")
    if len(text.strip()) < min(600 if spec["noun"] == "scene" else 200, max(80, len(spec["sections"]) * 80)):
        problems.append("too short")
    return problems


_FOREIGN = re.compile(r"[A-Za-z]*[\u00C0-\u024F]+[A-Za-z]*")
_HYBRID = re.compile(r"[\u0900-\u097F]+[-\u2011]?[A-Za-z]+|[A-Za-z]+[-\u2011]?[\u0900-\u097F]+")


def language_issues(text: str, spec: dict, allowed: set | None = None) -> list[str]:
    """Deterministic: no accented foreign Latin words; no Devanagari/Latin hybrids; Devanagari only in dialogue."""
    out = []
    bad = {w for w in _FOREIGN.findall(text) if w.lower() not in ("café", "naïve", "fiancée", "fiancé", "façade", "cliché", "décor", "résumé", "protégé", "déjà", "vu", "sautéed", "crème", "entrée", "élan", "rôle", "über") and w.lower() not in (allowed or set())}
    if bad:
        out.append("foreign-language words: " + ", ".join(sorted(bad)[:5]))
    other = re.findall(r"[\u0980-\u0DFF\u0600-\u06FF\u4E00-\u9FFF\u0400-\u04FF]+", text)
    if other:
        out.append("text in a script other than English/Devanagari (e.g. Bengali): " + " ".join(other[:3]))
    if spec.get("devanagari"):
        if _HYBRID.search(text):
            out.append("Hindi dialogue mixes Latin letters into Devanagari words")
        section, outside, latin = "", 0, set()
        for line in text.splitlines():
            if re.match(r"^\s*#{1,3}\s", line):
                section = line.lower()
            elif re.search(r"[\u0900-\u097F]{2,}", line) and "dialog" not in section:
                outside += 1
            elif "dialog" in section and re.search(r"[\u0900-\u097F]{2,}", line):
                spoken = re.sub(r"\([^)]*\)|\[[^\]]*\]|^[^:\u0900-\u097F]*:", "", line)
                latin |= {w for w in re.findall(r"[A-Za-z]{3,}", spoken) if w.lower() not in (allowed or set())}
        if latin:
            out.append("English/Latin words inside Hindi dialogue (write them in Hindi, Devanagari): " + ", ".join(sorted(latin)[:6]))
        if outside > 2:
            out.append("Devanagari used outside the Dialogue section")
    return out


EXTRACT_SYS = """Read the scene and report facts as JSON only:
{"evidence":{"prop":"exact sentence copied from the scene that shows the prop reaching its end holder/location; empty string if no sentence shows it"},"start_props":{"prop":"holder or location id when the scene BEGINS, or 'unknown'"},"end_positions":{"Name":"one id from locations, or 'unknown'"},"end_props":{"prop":"holder name or one location id, or 'unknown'"},"unexplained_moves":["Name"],"contradictions":["exact quote of any clothing, background or state contradiction"]}
Do not infer positions from metadata/locks; read the actual visual actions and dialogue. Detect incompatible clothing even when a lock elsewhere claims the right outfit. Unchanged props can be inferred from visible placement; do not invent facts. end_* = where each listed character/prop is when the scene ENDS. unexplained_moves = characters who appear in a new place without being shown moving."""


def scene_evidence(text: str) -> str:
    """Copied metadata is context, never evidence of an event in the actual scene."""
    lines = []
    ledger = False
    for line in text.splitlines():
        if line.startswith("**Canonical state at scene end:**"):
            ledger = True
        if ledger and line.startswith("## "):
            ledger = False
        if ledger or re.match(r"^\s*(?:-\s+\*\*.*\(locked appearance\)|\*\*(?:Locked style|Locked negative|Background):)", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def audit_findings(review: dict, indices: list[int]) -> tuple[int, list[str], set[int]]:
    """A numeric self-score cannot override failed or unperformed checks."""
    if not isinstance(review, dict):
        return 0, ["Independent audit unavailable"], set(indices)
    notes = validate_review(review)
    try:
        score = max(0, min(100, int(review.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    if notes:
        score = min(score, MIN_SCORE - 1)
    flagged = {i for i in review.get("units", []) if isinstance(i, int) and i in indices}
    flagged |= {int(n) for x in notes for n in re.findall(r"unit (\d+)", x, re.I) if int(n) in indices}
    if score < MIN_SCORE and not flagged:
        flagged = set(indices)
    return score, notes, flagged


def fact_issues(text: str, i: int, plan: dict) -> list[str]:
    """Independent extraction compared deterministically with the plan's ledger (the model only reads; code judges)."""
    c = plan["outline"][i - 1]
    want_pos, want_props = c.get("end_positions") or {}, c.get("end_props") or {}
    if not want_pos and not want_props:
        return []
    try:
        got = parse_json(AUDIT_CALL([{"role": "system", "content": EXTRACT_SYS},
                                     {"role": "user", "content": json.dumps({"locations": plan.get("locations"), "characters": list(want_pos),
                                      "props": sorted(set(want_props) | set(c.get("start_props") or {})), "scene": scene_evidence(text), "contract": c, "bible": plan["bible"]}, ensure_ascii=False)}], 800, True))
    except Exception:
        return ["Independent fact extraction unavailable; this scene has not passed verification"]
    if not isinstance(got, dict):
        return ["Independent fact extraction returned invalid data"]
    out = []
    same = lambda a, b: _norm(str(a)) == _norm(str(b)) 
    for n, loc in want_pos.items():
        g = (got.get("end_positions") or {}).get(n)
        if not g or g == "unknown":
            out.append(f"{n} end location was not verified")
        elif not same(g, loc):
            out.append(f"{n} must end at {loc}, scene leaves them at {g}")
    for k, v in (c.get("start_props") or {}).items():
        g = (got.get("start_props") or {}).get(k)
        if not g or g == "unknown":
            out.append(f"{k} prop location was not verified")
        elif not same(g, v):
            out.append(f"{k} must begin with/at {v} (as the previous unit left it), scene starts it with/at {g}")
    for k, v in want_props.items():
        g = (got.get("end_props") or {}).get(k)
        if not g or g == "unknown":
            out.append(f"{k} prop location was not verified")
        elif not same(g, v):
            out.append(f"{k} must end with/at {v}, scene shows {g}")
    ev = got.get("evidence") if isinstance(got.get("evidence"), dict) else {}
    low = _norm(text)
    for k, v in want_props.items():
        if v == (c.get("start_props") or {}).get(k):
            continue  # unchanged this unit: nothing to show
        q = ev.get(k)
        if not (isinstance(q, str) and len(q) > 15 and q.strip() in text):
            out.append(f"Scene Actions must show {k} reaching {v} on screen (no sentence shows it)")
    for x in got.get("contradictions") or []:
        if isinstance(x, str) and x.strip():
            out.append("Visible content contradicts canon: " + x)
    for n in got.get("unexplained_moves") or []:
        if isinstance(n, str):
            out.append(f"{n} changes place without being shown moving")
    return out


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
        passed += max(0, n_checks - len(check_unit(text, i, plan, spec)) - len(language_issues(text, spec, _allowed_words(plan))))
    return round(100 * passed / total) if total else 0


# ---------- prompts ----------

BLUEPRINT_SYS = SYSTEM + """ Write ONLY a dense Master Story Blueprint in Markdown (under 1,800 words, bullet points): plot, characters with exact physical appearance and clothing,
locations, world rules relevant to the genre, essential props, timeline, causal beats and ending. Do not invent supernatural rules, mysteries, injuries or props for unrelated stories. Keep a fixed background and unchanged clothing when required. No scene scripts."""

PLAN_SYS = SYSTEM + """ Turn the given blueprint into a compact plan. Return JSON only (no Markdown inside):
{"bible":{"background":"exact immutable background description when required; otherwise empty string","characters":{"Name":"locked visual descriptor, 20-40 words: age, face, hair, build, exact clothing"},"style":"ONE locked cinematic style line","negative":"ONE locked negative-constraints line"},
"locations":["short location id, e.g. foyer"],
"start_positions":{"Name":"location id at the start of unit 1"},
"start_props":{"prop name":"holder name or location id at the start of unit 1"},
"initial_state":{"entity.field":"short value"},
"clues":[{"name":"short clue name","setup":1,"payoff":10}],
"outline":[{"index":1,"moment":"what happens","time":"D1 22:00","location":"location id","end_positions":{"Name":"location id at unit end (EVERY character)"},"end_props":{"prop name":"holder or location id at unit end (EVERY tracked prop)"},"characters":["Name"],"props_in_play":["prop: who holds it / where"],"injuries":["Name: injury (cause, unit)"],"clues_planted":["clue name"],"clues_paid":["clue name"],"before":{"entity.field":"value"},"after":{"entity.field":"new short value"}}]}
Exactly N outline items. time is 'D<day> HH:MM' and never goes backwards; a gap over 3 hours needs "time_skip":"reason shown on screen". Destroyed props stay destroyed (value 'destroyed'). A character's start position is their previous end position; anyone who moves must be shown moving.
Times never go backwards and gaps are explained. Every prop or injury exists only from the unit that causes it;
a dropped/lost prop stays dropped until a unit recovers it. If the story has mystery clues, every planted clue is paid off in a later unit; otherwise an empty clues list is valid. Each unit advances one causal step; every 'before' equals the state left by earlier units.
Track only causally necessary props, injuries and knowledge as state keys. Do not create artificial keys for confidence, emotions or status; emotional beats belong in moment. Empty state maps and empty clue lists are valid for stories without mystery. Seed recurring motifs or mystery clues only if relevant to the user's genre; do not invent a mystery for a school, comedy or ordinary story. Outfit changes are forbidden when the brief requires fixed clothes."""

REVIEW_SYS = SYSTEM + """ Critique the draft against the bible, contract and brief. JSON only:
{"checks":{"identity":true,"style":true,"continuity":true,"instructions":true,"structure":true},"issues":[]}
identity: characters match bible appearance and clothing. style: same visual style. continuity: props, injuries, location,
knowledge and time follow the contract's before state and reach its after state; nothing from later units revealed.
instructions: brief's language/script/section rules. structure: header and all sections. false = blocked; issues name the smallest fix."""

AUDIT_SYS = SYSTEM + """ Audit the WHOLE series for consistency and continuity. JSON only:
{"score":0-100,"checks":{"identity":true,"style":true,"continuity":true,"instructions":true,"structure":true},"issues":[],"units":[unit numbers that must be repaired]}
Each issue must start with "unit N:". Score strictly: deduct for any appearance/clothing drift, prop teleportation, timeline gaps, unexplained knowledge, unpaid clues, rule violations."""


def _unit_messages(i: int, plan: dict, spec: dict, brief: str, fix: list[str] | None = None, draft: str | None = None, neighbours: dict | None = None):
    c = plan["outline"][i - 1]
    head = f"Begin with the line '# {spec['noun'].title()} {i}: <title>'. " if spec["multi"] else ""
    sys = (SYSTEM + f" Write ONLY {spec['noun']} {i} of {spec['count']} as a self-contained Markdown document. {head}"
           f"Use these sections as '## ' headings, in order: {'; '.join(spec['sections'])}. "
           "Keep every character exactly as the bible describes (the worker inserts the locked descriptors itself). Start from the 'before' state, end at the 'after' state; never reveal later units. "
           "Use ONLY the contract's time, location, props_in_play and injuries: invent no new props, wounds, events or locations. "
           "Characters begin at start_positions and finish at end_positions; anyone who changes place is SHOWN moving. Props begin at start_props and end at end_props. "
           "Characters not in this unit's cast stay off-screen where start_positions says. "
           "If the contract has time_skip, show that elapsed time in the first requested narrative/visual field, without adding a Context section. "
           "Include dialogue only when requested. Describe actual on-screen appearances within Visual Prompt; do not rely on the injected locks as a substitute for correct visible content. "
           "Name each planted/paid clue exactly. Do not repeat the blueprint. The ending must lead directly into next_unit.")
    sys += " Do not add any sections beyond the requested headings. Character details, fixed clothing, background and actions belong inside Visual Prompt when no separate section was requested."
    if spec.get("dialogue_range"):
        lo, hi = spec["dialogue_range"]
        sys += f" Dialogue must contain {lo}–{hi} spoken words TOTAL for this scene across all speakers, excluding labels and delivery notes. Count and correct before returning."
    if spec.get("fixed_background"):
        sys += " Describe the exact locked background in your actual Visual Prompt, not only metadata. The background stays identical in ALL scenes: never move outdoors or add/remove background furniture, posters, windows or decor."
    if spec.get("duration"):
        duration = spec["duration"]
        start, end = (i - 1) * duration / spec["count"], i * duration / spec["count"]
        sys += f" This scene runs from {start:g}s to {end:g}s of the {duration}s story; actions and spoken words must fit {end-start:g}s. Include this time range in the requested story/visual field, never add an unrequested section."
    if spec["devanagari"]:
        sys += (" Dialogue lines are pure natural Hindi in Devanagari only (no Latin letters, no words from other languages); "
                "delivery notes in parentheses are plain English. Everything outside dialogue is plain English with no other languages.")
    ctx = {"brief": brief, "bible": plan["bible"], "blueprint": plan["blueprint"],
           "clues": plan.get("clues"), "outline": [{"index": o["index"], "moment": o["moment"], "time": o.get("time")} for o in plan["outline"]],
           "previous_unit": plan["outline"][i - 2] if i > 1 else None, "contract": c,
           "next_unit": plan["outline"][i] if i < len(plan["outline"]) else None}
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}]
    if fix and draft:
        if neighbours:
            msgs.append({"role": "user", "content": "Approved neighbouring units (stay consistent with them, do not rewrite them): " + json.dumps(neighbours, ensure_ascii=False)})
        msgs += [{"role": "assistant", "content": draft},
                 {"role": "user", "content": f"Fix ONLY these problems and return the full corrected {spec['noun']}: " + "; ".join(fix[:10])}]
    return msgs


def _insert_after_heading(text: str, words: tuple, block: str) -> str:
    if block.splitlines()[0] in text:  # already stamped (e.g. a repaired draft): never duplicate
        return text
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if re.match(r"^\s*(?:#{1,6}\s+|\*\*|\d{1,2}[.)]\s)", line) and all(w in line.lower() for w in words):
            return "\n".join(lines[:n + 1] + [block] + lines[n + 1:])
    return text.rstrip() + "\n\n" + block


def canonical_header(text: str, i: int, spec: dict) -> str:
    """Exactly one '# Scene N: title' header; pasted blueprint sections are removed."""
    text = re.sub(r"^[ \t]*#{1,3}[ \t]*master story blueprint.*\n(?:(?![ \t]*#{1,2}[ \t])[^\n]*\n?)*", "", text, flags=re.I | re.M)
    if not spec["multi"]:
        return text
    pat = re.compile(rf"^[ \t]*#{{1,3}}[ \t]+(?:{spec['noun']}|prompt)[ \t]+\d+\b[: \t-]*(.*)$", re.I | re.M)
    titles = [m.group(1).strip(" *") for m in pat.finditer(text)]
    body = pat.sub("", text).strip()
    title = next((t for t in titles if t), "")
    return f"# {spec['noun'].title()} {i}: {title or 'Untitled'}\n\n{body}"


def enforce_locks(text: str, i: int, plan: dict, spec: dict) -> str:
    """Locks are written by the worker, not trusted to the model: every unit carries the identical canonical
    character descriptors, style line, negative line and contracted end state, so they cannot drift."""
    b, c = plan["bible"], plan["outline"][i - 1]
    if not any(re.search(r"visual|character|subject|style|setting", sec, re.I) for sec in spec["sections"]):
        return text
    cast = c.get("characters", [])
    if cast:
        chars = "\n".join(f"- **{n} (locked appearance):** {b['characters'][n]}" for n in cast
                           if f"**{n} (locked appearance):**" not in text)
        if chars:
            text = _insert_after_heading(text, ("character",) if "Character Details" in spec["sections"] else ("visual",), chars)
    text = _insert_after_heading(text, ("visual",), f"**Locked style:** {b['style']}")
    text = _insert_after_heading(text, ("negative",) if any("negative" in x.lower() for x in spec["sections"]) else ("visual",), f"**Locked negative:** {b['negative']}")
    if spec.get("fixed_background") and b.get("background"):
        text = _insert_after_heading(text, ("visual",), f"**Background:** {b['background']}")
    if "Continuity" not in spec["sections"]:
        return text
    state = [f"- {k}: {v}" for k, v in (c.get("after") or {}).items()]
    state += [f"- {n} ends at: {v}" for n, v in (c.get("end_positions") or {}).items()]
    state += [f"- {k} is with/at: {v}" for k, v in (c.get("end_props") or {}).items()]
    where = f"- Time: {c.get('time', '')} | Location: {c.get('location', '')}"
    text = _insert_after_heading(text, ("continuity",) if "Continuity" in spec["sections"] else ("visual",), "**Canonical state at scene end:**\n" + "\n".join([where] + state))
    return text


def _allowed_words(plan: dict) -> set:
    """Character names may appear in Latin letters inside dialogue."""
    src = " ".join(plan["bible"]["characters"])  # only people's names; places and objects must be said in Hindi
    return {w.lower() for w in re.findall(r"[^\W\d_]{3,}", src.replace("_", " "))}


DIALOGUE_FIX_SYS = ("Rewrite ONLY this Dialogue section. Spoken words must be pure natural Hindi in Devanagari: replace every English or other "
                    "Latin-script word inside speech with its Hindi word in Devanagari. Keep speaker names, order, meaning and the English delivery notes "
                    "in parentheses. Return only the section, starting with its heading.")


def _bounded_dialogue(text: str, bounds, devanagari=False) -> str:
    m = re.search(r"(?im)^##\s+Dialogue[^\n]*\n([\s\S]*?)(?=^##\s|\Z)", text)
    if not m:
        return text
    labels = re.findall(r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?([^:\n]+?):", m.group(1))
    labels = [re.sub(r"\([^)]*\)|\[[^\]]*\]|[*_\"]", "", x).strip() for x in labels]
    if not labels:
        return text
    target = sum(bounds) // 2
    counts = [target // len(labels) + (1 if i < target % len(labels) else 0) for i in range(len(labels))]
    instruction = "Rewrite this dialogue with the same speakers, meaning, and scene facts. JSON only: {turns:[{speaker:string,words:[string]}]}. Exactly one turn per listed speaker in order; each words array has EXACTLY its assigned count, ONE spoken word per element (contractions count as one). No notes or labels in words. Do not repeat the original overlength lines."
    if devanagari:
        instruction += " All spoken words must be natural Hindi in pure Devanagari."
    try:
        data = parse_json(_ai([{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps({"original": m.group(1), "turn_contracts": [{"speaker": label, "words": count} for label, count in zip(labels, counts)]}, ensure_ascii=False)}], 2000, True, True))
        turns = data.get("turns")
        if not isinstance(turns, list) or len(turns) != len(labels):
            return text
        lines = []
        for label, count, turn in zip(labels, counts, turns):
            words = turn.get("words") if isinstance(turn, dict) else None
            if not isinstance(words, list) or len(words) != count or any(not isinstance(w, str) or len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", w)) != 1 for w in words):
                return text
            lines.append(f"{label}: {' '.join(words)}")
        replacement = "## Dialogue\n" + "\n".join(lines) + "\n\n"
        candidate = text[:m.start()] + replacement + text[m.end():]
        return candidate if bounds[0] <= dialogue_words(candidate) <= bounds[1] else text
    except (ValueError, TypeError, AttributeError):
        return text


def _fix_dialogue(text: str, issues: list[str], bounds=None, devanagari=True) -> str:
    """Targeted, small repair for language-only failures: the rest of the unit is kept byte-for-byte."""
    m = re.search(r"^[ \t]*#{1,3}[ \t]*[^\n]*dialog[^\n]*\n(?:(?![ \t]*#{1,3}[ \t])[^\n]*\n?)*", text, re.I | re.M)
    if not m:
        return text
    instruction = DIALOGUE_FIX_SYS if devanagari else "Rewrite ONLY this Dialogue section. Preserve speakers, order, meaning and scene facts. Return only the section and its heading."
    if bounds:
        instruction += f" Exactly {sum(bounds) // 2} spoken words TOTAL across all speakers; allowed {bounds[0]}–{bounds[1]}. Speaker labels and parenthesized/bracketed delivery notes do not count. Never add commentary or word-count labels."
    new = _ai([{"role": "system", "content": instruction},
               {"role": "user", "content": "Problems: " + "; ".join(issues) + "\n\n" + m.group(0)}], 3000).strip()
    if not re.match(r"^[ \t]*#{1,3}[^\n]*dialog", new, re.I):
        return text
    return text[:m.start()] + new + "\n\n" + text[m.end():]


def _write_and_check(i, plan, spec, brief, fix=None, draft=None, neighbours=None):
    text = enforce_locks(canonical_header(_ai(_unit_messages(i, plan, spec, brief, fix, draft, neighbours), 12000, False, fix is None), i, spec), i, plan, spec)
    allowed = _allowed_words(plan)
    problems = check_unit(text, i, plan, spec) + language_issues(text, spec, allowed)
    bounds = spec.get("dialogue_range")
    if bounds and not bounds[0] <= dialogue_words(text) <= bounds[1]:
        for _ in range(2):
            text = _bounded_dialogue(text, bounds, spec["devanagari"])
            if bounds[0] <= dialogue_words(text) <= bounds[1]:
                break
        problems = check_unit(text, i, plan, spec) + language_issues(text, spec, allowed)
    lang = language_issues(text, spec, allowed)
    if lang and len(problems) == len(lang):  # only language problems: fix just the dialogue lines
        for _ in range(2):
            text = _fix_dialogue(text, lang)
            lang = language_issues(text, spec, allowed)
            if not lang:
                break
        problems = check_unit(text, i, plan, spec) + lang
    if not problems and spec["multi"]:
        problems = fact_issues(text, i, plan)
    if problems:
        print(f"[prompts] unit {i} gate: " + " | ".join(p[:120] for p in problems[:5]), flush=True)
    return i, text, problems


# ---------- run ----------

def run_prompts(params: dict, workdir: str, progress: Callable[[str], None] | None = None) -> dict:
    brief = str(params.get("prompt", "")).strip()
    if not brief or len(brief) > MAX_BRIEF_CHARS:
        raise OpError("The brief is missing or too large.")
    count, noun = unit_count(brief, params.get("count"))
    spec = {"count": count, "noun": noun, "multi": count > 1, "sections": required_sections(brief), "devanagari": wants_devanagari(brief), "story": wants_story(brief), "prompts": bool(re.search(r"\bprompts?\b", brief, re.I)) or not wants_story(brief), "duration": duration_seconds(brief), "dialogue_range": dialogue_range(brief), "fixed_background": fixed_background(brief)}
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
        plan = None
        say("planning")
        problems: list[str] = []
        blueprint = _ai([{"role": "system", "content": BLUEPRINT_SYS},
                         {"role": "user", "content": json.dumps({"brief": brief, "N": count}, ensure_ascii=False)}], 3500)
        if spec["devanagari"]:  # the blueprint is all-English: strip stray non-Latin script deterministically
            blueprint = re.sub(r"[\u0900-\u0DFF]+[^\n]*?(?=[.;\n]|$)", "", blueprint)
        story = ""
        contract = build_contract(brief, count)
        loop = _repair.RepairLoop(max_rounds=2)
        rejected = None      # previous candidate, sent back with structured findings
        findings: list[dict] = []
        for attempt in range(1 + loop.max_rounds):
            report = None
            try:
                payload = {"brief": brief, "blueprint": blueprint, "N": count, "repair": problems}
                if rejected is not None:
                    payload.update({"rejected_candidate": rejected, "findings": findings,
                                    "instruction": "Fix ONLY the listed findings in rejected_candidate; keep every other field identical."})
                plan = parse_json(_ai([{"role": "system", "content": PLAN_SYS + (" The user requires a fixed background. Keep every supplied main character present throughout, unless absence is explicitly requested. Use a simple causal story relevant to the theme, not pranks, injuries, mysteries or unrelated props. Track only essential objects; background furniture belongs to bible.background, never duplicate it as moving props. Declare locations as a list with exactly ONE short room ID, such as classroom, and use that exact ID as every scene's location. start_positions/end_positions use 'ROOM_ID: spot' (e.g. 'classroom: front desk'); the spot is a within-room position, never a new room. Add bible.background: ONE exact locked description of the complete background, furniture, decor and lighting, reused unchanged. Do not move outdoors. Clothes remain worn and unchanged, not folded, removed or changed." if spec["fixed_background"] else "")},
                                       {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], 8000, True))
                if isinstance(plan, dict):
                    plan["blueprint"], plan["story"] = blueprint, story
                    plan["requires_clues"] = bool(re.search(r"\b(?:mystery|clues?|payoffs?)\b", brief, re.I))
                    if spec["fixed_background"] and not plan.get("locations"):
                        first = next((b.get("location") for b in plan.get("outline") or [] if isinstance(b, dict) and b.get("location")), None)
                        if first:
                            plan["locations"] = [first]
                            plan.setdefault("normalizations", []).append({"path": "locations", "reason": "declared from first scene location"})
                plan = normalize_plan(plan, count)
                report = validate(plan, {"profile": "story.plan", "contract": contract, "count": count})
                problems = validate_plan(plan, count) + [f.legacy() for f in report.errors if f.code.startswith("continuity.")]
                findings = [f.to_dict() for f in report.errors]
            except (ValueError, TypeError):
                problems, findings = ["Invalid planning JSON"], [{"code": "structural.invalid_json", "message": "Invalid planning JSON"}]
            safe = report.safe_log() if report else {"status": "failed", "codes": ["structural.invalid_json"]}
            print(f"[prompts] plan attempt={attempt + 1} problems={len(problems)} report={json.dumps(safe)}", flush=True)
            if not problems:
                ck["plan"] = plan; save(); break
            rejected = plan if isinstance(plan, dict) else None
            if report:
                loop.record(report)
            if not loop.should_continue():
                break
            loop.next()
        if not ck["plan"]:
            raise OpError("Planning remains unresolved: " + "; ".join(problems[:4]))
    plan = ck["plan"]
    if validate_plan(plan, count) or check_fixed_background(plan, brief):
        raise OpError("Saved plan is invalid.")

    units: dict[int, str] = {}
    for k, t in ck["units"].items():
        if not check_unit(t, int(k), plan, spec) and not language_issues(t, spec, _allowed_words(plan)) and not fact_issues(t, int(k), plan):
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
        if pending:
            raise OpError("Scene requirements remain unresolved: " + "; ".join(f"Scene {i}: {'; '.join(problems[:3])}" for i, (problems, text) in pending.items()))
        if len(units) < count:
            raise OpError(f"{count - len(units)} unit(s) could not be written.")

        series = [units[i] for i in range(1, count + 1)]
        det = deterministic_score(series, plan, spec)
        audit_score = 100
        best = None
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
                                                     "full_outline": [{"index": o["index"], "moment": o["moment"], "time": o.get("time"), "location": o.get("location"), "after": o.get("after")} for o in plan["outline"]],
                                                     "contracts_for_these_units": [plan["outline"][i - 1] for i in idx],
                                                     "brief": brief, "requested_sections": spec["sections"], "note": "Events from units outside this window happened as the full_outline says; judge these units against it.",
                                                     "series": window, "visible_content": "\n\n".join(scene_evidence(series[i - 1]) for i in idx)}, ensure_ascii=False)}], 2500, True))
                        window_score, notes, hit = audit_findings(a_, idx)
                        scores.append(window_score)
                        notes_all += notes
                        flagged |= hit
                    except (ValueError, TypeError, OpError):
                        scores.append(0)
                    if start + 4 >= count:
                        break
                audit_score = min(scores) if scores else 0
                audit = {"issues": notes_all, "units": sorted(flagged)}
                print(f"[prompts] audit scores={scores} flagged={sorted(flagged)}", flush=True)
                if audit_score >= MIN_SCORE:
                    break
                if best is None or audit_score > best[0]:
                    best = (audit_score, list(series))
                fix = sorted({i for i in audit.get("units", []) if isinstance(i, int) and 1 <= i <= count}
                             | {int(n) for x in notes_all for n in re.findall(r"unit (\d+)", x, re.I) if 1 <= int(n) <= count})
                if not fix or round_ == REPAIR_ROUNDS:
                    break
                say(f"repairing_{len(fix)}")
                def own(i):
                    return [x for x in notes_all if re.search(rf"unit {i}\b", x, re.I)] or notes_all
                if best is None or audit_score > best[0]:
                    best = (audit_score, list(series))
                near = lambda i: {k: series[k - 1][-1500:] if k < i else series[k - 1][:1500] for k in (i - 1, i + 1) if 1 <= k <= count}
                for i, text, problems in pool.map(lambda i: _write_and_check(i, plan, spec, brief, own(i), series[i - 1], near(i)), fix):
                    if not problems:
                        series[i - 1] = text; ck["units"][str(i)] = text
                save()
            if best and best[0] > audit_score:  # a repair round never makes the delivered series worse
                audit_score, series = best
            det = deterministic_score(series, plan, spec)
        ck["units"] = {str(i): t for i, t in enumerate(series, 1)}
        save()
    final_issues = [f"Scene {i}: {problem}" for i, t in enumerate(series, 1) for problem in check_unit(t, i, plan, spec) + language_issues(t, spec, _allowed_words(plan))]
    if final_issues:
        raise OpError("Final requirements unresolved: " + "; ".join(final_issues[:8]))
    if spec.get("dialogue_range"):
        failures = [f"Scene {i}: {dialogue_words(t)} dialogue words" for i, t in enumerate(series, 1) if not spec["dialogue_range"][0] <= dialogue_words(t) <= spec["dialogue_range"][1]]
        if failures:
            raise OpError("Dialogue word limit unresolved: " + "; ".join(failures))
    score = min(det, audit_score)
    if spec["multi"] and score < MIN_SCORE:
        raise OpError(f"Consistency score {score}% is below the {MIN_SCORE}% bar; the series was not approved.")

    if spec["story"]:
        say("writing")
        plan["story"] = _ai([{"role": "system", "content": SYSTEM + " Write the full story, beginning with '# <story title>'. No blueprint, commentary or extra sections. Preserve the approved scene events and dialogue. Respect the requested duration; target spoken narration at about 2.3 words per second when duration is specified."},
                            {"role": "user", "content": json.dumps({"brief": brief, "duration_seconds": spec["duration"], "bible": plan["bible"], "approved_scenes": series}, ensure_ascii=False)}], 8000)
        # The newly written full story is checked against accepted scenes; never reuse the scene score as its approval.
        for attempt in range(2):
            review = parse_json(AUDIT_CALL([{"role": "system", "content": SYSTEM + " Compare the full story against the approved scenes and request. JSON only: {\"checks\":{\"identity\":true,\"style\":true,\"continuity\":true,\"instructions\":true,\"structure\":true},\"issues\":[]}. Fail invented contradictions, omissions, wrong requested language, duration-inappropriate narration or a different ending. Cite exact evidence."},
                                          {"role": "user", "content": json.dumps({"brief": brief, "story": plan["story"], "scenes": series}, ensure_ascii=False)}], 2500, True))
            issues = validate_review(review)
            if spec["duration"]:
                words = len(re.findall(r"\b\w+\b", plan["story"]))
                if words > int(spec["duration"] * 3.0) + 10:
                    issues.append("Narration exceeds the requested duration at 3 words per second.")
            if not issues:
                break
            if attempt == 1:
                raise OpError("The full story did not pass its consistency or duration review.")
            plan["story"] = _ai([{"role": "system", "content": SYSTEM + " Repair only the cited issues, return '# <title>' and full story prose. No commentary."},
                                  {"role": "user", "content": json.dumps({"brief": brief, "story": plan["story"], "issues": issues, "scenes": series}, ensure_ascii=False)}], 8000)
    document = assemble_output(plan, series, spec, brief)
    say("assembling")
    name = "story.md" if noun == "scene" else "prompts.md"
    out = os.path.join(workdir, name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(document)
    return {"output": out, "mime": "text/markdown",
            "result": {"runtime_sha": os.environ.get("GITHUB_SHA", "local"), "units": count, "kind": noun, "consistency_score": score, "deterministic_score": det,
                       "audit_score": audit_score, "pipeline_version": VERSION}}


run = WORKER.run
