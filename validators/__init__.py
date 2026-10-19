"""Shared, pure validators for VISION runtime outputs. Entry point: validate(artifact, context)."""
from __future__ import annotations

from validators import compliance, continuity, structural
from validators.contracts import (FAILED, PASSED, UNVERIFIED, VALIDATOR_VERSION, Constraint, Contract, Finding, ValidationReport,
                                  stable_hash)

PROFILES = ("story.plan", "story.unit", "story.assembly")


def validate(artifact, context: dict) -> ValidationReport:
    """context: {profile, contract: Contract, count?, unit?, sections?}. Never mutates artifact, never calls a model."""
    profile = context["profile"]
    contract: Contract = context["contract"]
    if profile not in PROFILES:
        raise ValueError(f"unknown validation profile {profile}")
    findings: list[Finding] = []
    unverified: list[str] = []
    count = context.get("count") or contract.get("units.count") or 1
    if profile == "story.plan":
        findings += structural.check_outline(artifact, count)
        findings += continuity.check_fixed_background(artifact, bool(contract.get("background.fixed")))
    elif profile == "story.unit":
        findings += compliance.check_unit(artifact, context.get("unit") or 1, context.get("sections") or contract.get("sections"),
                                          contract.get("dialogue.words"))
    else:
        findings += structural.check_assembly(list(artifact or []), count)
        for i, text in enumerate(artifact or [], 1):
            findings += compliance.check_unit(text, i, context.get("sections") or contract.get("sections"), contract.get("dialogue.words"))
        if contract.get("background.fixed"):
            unverified.append("semantic.background_visual")  # needs an injected evidence evaluator
    return ValidationReport(profile, contract.hash, stable_hash(artifact), findings, unverified)


__all__ = ["validate", "Contract", "Constraint", "Finding", "ValidationReport", "PASSED", "FAILED", "UNVERIFIED", "VALIDATOR_VERSION", "stable_hash"]
