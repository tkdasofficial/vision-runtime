"""Shape checks: exact count, order, duplicate units and references to declared characters."""
from __future__ import annotations

import re

from validators.contracts import Finding


def check_outline(plan: dict, count: int) -> list[Finding]:
    if not isinstance(plan, dict):
        return [Finding("structural.not_object", "plan.shape", "Plan must be an object", path="")]
    outline = plan.get("outline")
    if not isinstance(outline, list) or len(outline) != count:
        return [Finding("structural.count", "units.count", f"Outline must contain exactly {count} units", path="outline", expected=count,
                        observed=len(outline) if isinstance(outline, list) else None, repair_scope=("outline",))]
    chars = (plan.get("bible") or {}).get("characters") if isinstance(plan.get("bible"), dict) else None
    out: list[Finding] = []
    for i, beat in enumerate(outline, 1):
        p = f"outline[{i - 1}]"
        if not isinstance(beat, dict) or beat.get("index") != i:
            out.append(Finding("structural.order", "units.order", f"Contract {i} is invalid or out of order", path=f"{p}.index", unit=i, expected=i,
                               observed=beat.get("index") if isinstance(beat, dict) else None, repair_scope=(p,)))
            continue
        if isinstance(chars, dict):
            unknown = [c for c in beat.get("characters") or [] if c not in chars]
            if unknown:
                out.append(Finding("structural.unknown_character", "cast.declared", f"Unit {i} uses a character missing from the bible", path=f"{p}.characters",
                                   unit=i, observed=unknown, repair_scope=(f"{p}.characters",)))
    return out


def check_assembly(units: list[str], count: int) -> list[Finding]:
    """Final assembled series: exact count, each unit present once, headings in order."""
    out: list[Finding] = []
    if len(units) != count:
        out.append(Finding("structural.assembly_count", "units.count", f"Assembled output has {len(units)} of {count} units", path="units", expected=count, observed=len(units)))
    seen = set()
    for i, text in enumerate(units, 1):
        if not str(text or "").strip():
            out.append(Finding("structural.empty_unit", "units.count", f"Unit {i} is empty", path=f"units[{i - 1}]", unit=i))
            continue
        m = re.match(r"^#\s+\w+\s+(\d+)\b", text)
        if m and int(m.group(1)) != i:
            out.append(Finding("structural.heading_order", "units.order", f"Unit {i} heading says {m.group(1)}", path=f"units[{i - 1}]", unit=i, expected=i, observed=int(m.group(1))))
        h = hash(text.strip())
        if h in seen:
            out.append(Finding("structural.duplicate_unit", "units.unique", f"Unit {i} duplicates an earlier unit", path=f"units[{i - 1}]", unit=i))
        seen.add(h)
    return out
