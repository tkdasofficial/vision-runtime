"""Versioned validation contracts, findings and reports shared by every profile.

Validators are pure: they never mutate the artifact and never call a provider. A report is bound to the
contract and artifact hashes so a stale report can never approve different content.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

VALIDATOR_VERSION = 1
PASSED, FAILED, UNVERIFIED = "passed", "failed", "unverified"
SEVERITIES = ("error", "warning")


def stable_hash(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Finding:
    code: str                 # stable machine code, e.g. continuity.room_changed
    constraint: str           # contract constraint id this finding enforces
    message: str              # safe, content-free summary
    path: str = ""            # JSON-ish path of the offending field, e.g. outline[3].location
    unit: int | None = None   # scene/prompt index when scoped to one unit
    expected: Any = None
    observed: Any = None
    severity: str = "error"
    repair_scope: tuple[str, ...] = ()  # paths a repair may change to fix this finding

    def to_dict(self) -> dict:
        d = asdict(self)
        d["repair_scope"] = list(self.repair_scope)
        return d

    def legacy(self) -> str:
        return self.message if self.unit is None or f"{self.unit}" in self.message else f"Unit {self.unit}: {self.message}"


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str
    value: Any
    source: str = ""          # brief clause that created the constraint
    scope: str = "all"


@dataclass
class Contract:
    operation: str
    constraints: list[Constraint] = field(default_factory=list)
    version: int = VALIDATOR_VERSION

    def get(self, cid: str):
        return next((c.value for c in self.constraints if c.id == cid), None)

    def has(self, cid: str) -> bool:
        return any(c.id == cid for c in self.constraints)

    def to_dict(self) -> dict:
        return {"operation": self.operation, "version": self.version, "constraints": [asdict(c) for c in self.constraints]}

    @property
    def hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass
class ValidationReport:
    profile: str
    contract_hash: str
    artifact_hash: str
    findings: list[Finding] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)   # checks that could not run
    version: int = VALIDATOR_VERSION

    @property
    def status(self) -> str:
        if any(f.severity == "error" for f in self.findings):
            return FAILED
        return UNVERIFIED if self.unverified else PASSED

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def signature(self) -> str:
        """Identity of the failure set; two repairs with the same signature made no progress."""
        return stable_hash(sorted((f.code, f.path, str(f.observed)) for f in self.errors))

    def bound_to(self, contract_hash: str, artifact_hash: str) -> bool:
        return self.version == VALIDATOR_VERSION and self.contract_hash == contract_hash and self.artifact_hash == artifact_hash

    def to_dict(self) -> dict:
        return {"profile": self.profile, "version": self.version, "status": self.status, "contract_hash": self.contract_hash,
                "artifact_hash": self.artifact_hash, "findings": [f.to_dict() for f in self.findings], "unverified": list(self.unverified)}

    def safe_log(self) -> dict:
        """Public-log summary: codes, paths and hashes only, never content."""
        return {"profile": self.profile, "status": self.status, "artifact": self.artifact_hash[:12],
                "codes": sorted({f.code for f in self.errors}), "paths": [f.path for f in self.errors][:8], "unverified": self.unverified[:4]}
