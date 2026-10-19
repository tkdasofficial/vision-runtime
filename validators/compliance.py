"""Requested-field and spoken-word checks on the visible text of one unit."""
from __future__ import annotations

import re

from validators.contracts import Finding

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def section(text: str, name: str) -> str | None:
    m = re.search(rf"(?im)^##\s+{re.escape(name)}[^\n]*\n([\s\S]*?)(?=^##\s|\Z)", text or "")
    return m.group(1) if m else None


def spoken_words(text: str) -> int:
    """Words actually spoken in the Dialogue section: speaker labels and (directions)/[cues] excluded."""
    body = section(text, "Dialogue")
    if body is None:
        return 0
    speech = re.sub(r"(?m)^\s*(?:[-*]\s*)?[^:\n]+:\s*", "", body)
    speech = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", speech)
    return len(_WORD.findall(speech))


def check_unit(text: str, unit: int, sections: list[str] | None = None, word_range: tuple[int, int] | None = None) -> list[Finding]:
    out: list[Finding] = []
    for name in sections or []:
        body = section(text, name)
        if body is None or not body.strip():
            out.append(Finding("compliance.missing_section", f"sections.{name}", f"Missing requested section {name}", path=f"units[{unit - 1}].{name}",
                               unit=unit, expected=name, repair_scope=(f"units[{unit - 1}].{name}",)))
    if word_range:
        lo, hi = word_range
        n = spoken_words(text)
        if not lo <= n <= hi:
            out.append(Finding("compliance.dialogue_words", "dialogue.words", f"Dialogue has {n} spoken words; required {lo}-{hi}",
                               path=f"units[{unit - 1}].Dialogue", unit=unit, expected=[lo, hi], observed=n,
                               repair_scope=(f"units[{unit - 1}].Dialogue",)))
    return out
