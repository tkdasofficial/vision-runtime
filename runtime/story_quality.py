"""Pure story contracts and evidence-gated continuity; no provider dependencies."""
import hashlib
import json
import re

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
