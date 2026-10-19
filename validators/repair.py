"""Scoped repair: build a repair request from the rejected candidate + findings, apply a patch only inside the
permitted paths on a copy, and decide when to stop."""
from __future__ import annotations

import copy
import re

from validators.contracts import ValidationReport

MAX_CONTENT_ROUNDS = 2
_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def parse_path(path: str) -> list:
    return [int(i) if i else k for k, i in _TOKEN.findall(path)]


def _within(path: str, scope: set[str]) -> bool:
    return any(path == s or path.startswith(s + ".") or path.startswith(s + "[") for s in scope)


def allowed_scope(report: ValidationReport) -> set[str]:
    return {p for f in report.errors for p in f.repair_scope}


def repair_request(candidate, report: ValidationReport, locked: dict | None = None) -> dict:
    """What a repair call receives: the rejected candidate, exact findings and the only paths it may change."""
    return {"rejected_candidate": candidate, "findings": [f.to_dict() for f in report.errors],
            "allowed_paths": sorted(allowed_scope(report)), "locked_canon": locked or {},
            "instruction": "Return {\"patches\": [{\"path\": ..., \"value\": ...}]} changing only allowed_paths."}


def _set(doc, parts, value):
    cur = doc
    for k in parts[:-1]:
        cur = cur[k]
    cur[parts[-1]] = value


def apply_patches(candidate, patches, scope: set[str], locked_paths: set[str] = frozenset()):
    """Returns (patched_copy, rejected_paths). The original candidate is never touched."""
    doc = copy.deepcopy(candidate)
    rejected = []
    for p in patches or []:
        path = str((p or {}).get("path") or "")
        if not path or not _within(path, scope) or _within(path, set(locked_paths)):
            rejected.append(path); continue
        try:
            _set(doc, parse_path(path), p.get("value"))
        except (KeyError, IndexError, TypeError):
            rejected.append(path)
    return doc, rejected


class RepairLoop:
    """Tracks rounds and failure signatures; stops after MAX_CONTENT_ROUNDS or two identical consecutive failures."""

    def __init__(self, max_rounds: int = MAX_CONTENT_ROUNDS):
        self.max_rounds, self.rounds, self.history = max_rounds, 0, []

    def record(self, report: ValidationReport) -> None:
        self.history.append(report.signature())

    def should_continue(self) -> bool:
        if self.rounds >= self.max_rounds:
            return False
        return not (len(self.history) >= 3 and self.history[-1] == self.history[-2] == self.history[-3])

    def next(self) -> None:
        self.rounds += 1
