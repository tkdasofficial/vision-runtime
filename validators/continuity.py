"""Continuity checks on canonical IDs: rooms are compared after alias resolution, and a position inside a room
("classroom: front desk", "near the window") is never mistaken for a second background."""
from __future__ import annotations

import re

from validators.contracts import Finding

OUTSIDE = re.compile(r"\b(?:outside|outdoors?|another room|hallway|corridor|playground|street|garden|yard|park)\b", re.I)
_FILLER = re.compile(r"\b(?:the|a|an|our|their|same|school|main|inside|in|at)\b", re.I)


def room_key(value) -> str:
    """Canonical comparison key: case/punctuation/articles/plurals ignored, 'class room' == 'Classroom'."""
    s = re.sub(r"[^\w\s]", " ", str(value or "").lower())
    s = _FILLER.sub(" ", s)
    s = re.sub(r"\s+", "", s)
    return re.sub(r"s$", "", s)


def split_position(value) -> tuple[str, str]:
    """'classroom: front desk' / 'classroom - by the window' / 'classroom/desk' -> (room, position)."""
    s = str(value or "").strip()
    m = re.match(r"^\s*([^:/>\-–—|]+?)\s*(?::|/|>|\||\s[-–—]\s)\s*(.+)$", s)
    return (m.group(1).strip(), m.group(2).strip()) if m else (s, "")


class RoomIndex:
    """Declared rooms with their aliases. Unknown references resolve to None so they fail precisely."""

    def __init__(self, rooms, aliases: dict | None = None):
        self.canon: dict[str, str] = {}
        for r in rooms or []:
            if str(r).strip():
                self.canon.setdefault(room_key(r), str(r).strip())
        for alias, target in (aliases or {}).items():
            tid = self.resolve(target)
            if tid:
                self.canon.setdefault(room_key(alias), tid)

    def resolve(self, value) -> str | None:
        key = room_key(value)
        if not key:
            return None
        if key in self.canon:
            return self.canon[key]
        hits = {cid for k, cid in self.canon.items() if k and (k in key or key in k)}
        return next(iter(hits)) if len(hits) == 1 else None

    def __len__(self):
        return len(set(self.canon.values()))


def plan_rooms(plan: dict) -> RoomIndex:
    declared = list(plan.get("locations") or [])
    if not declared:
        declared = [b.get("location") for b in plan.get("outline") or [] if isinstance(b, dict) and b.get("location")][:1]
    return RoomIndex(declared, plan.get("location_aliases") if isinstance(plan.get("location_aliases"), dict) else None)


def check_fixed_background(plan: dict, required: bool) -> list[Finding]:
    """Read-only. Applies only when the brief requested a fixed background."""
    if not required or not isinstance(plan, dict):
        return []
    out: list[Finding] = []
    rooms = plan_rooms(plan)
    if len(rooms) > 1:
        out.append(Finding("continuity.multiple_rooms_declared", "background.fixed", "Fixed background allows exactly one declared room",
                           path="locations", expected=1, observed=len(rooms), repair_scope=("locations",)))
    anchor = None
    outline = plan.get("outline") if isinstance(plan.get("outline"), list) else []

    def ref(path, value, unit):
        nonlocal anchor
        room, _position = split_position(value)
        if OUTSIDE.search(room) and not rooms.resolve(room):
            out.append(Finding("continuity.left_background", "background.fixed", "Fixed background requires the same location in every scene; scene moves outside", path=path,
                               unit=unit, expected=anchor, observed=room, repair_scope=(path, f"outline[{unit - 1}].moment")))
            return
        cid = rooms.resolve(room)
        if cid is None:
            if path.endswith(".location"):
                out.append(Finding("continuity.unknown_room", "background.fixed", "Scene references an undeclared room", path=path,
                                   unit=unit, expected=anchor, observed=room, repair_scope=(path,)))
            return  # an unresolved *position* value is a within-room spot, not a second background
        if anchor is None:
            anchor = cid
        elif cid != anchor:
            out.append(Finding("continuity.room_changed", "background.fixed", "Fixed background requires the same location in every scene",
                               path=path, unit=unit, expected=anchor, observed=cid, repair_scope=(path,)))

    for i, beat in enumerate(outline, 1):
        if not isinstance(beat, dict):
            continue
        ref(f"outline[{i - 1}].location", beat.get("location"), i)
        for field in ("start_positions", "end_positions"):
            for who, value in (beat.get(field) or {}).items() if isinstance(beat.get(field), dict) else ():
                ref(f"outline[{i - 1}].{field}.{who}", value, i)
    if not str((plan.get("bible") or {}).get("background") or "").strip():
        out.append(Finding("continuity.background_unlocked", "background.fixed", "Fixed background requires one locked description of furniture, decor and lighting",
                           path="bible.background", repair_scope=("bible.background",)))
    return out
