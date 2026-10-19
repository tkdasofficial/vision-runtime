import copy
import json
import os

from validators import Constraint, Contract, validate
from validators.continuity import RoomIndex, check_fixed_background, split_position
from validators.repair import RepairLoop, apply_patches, repair_request

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "validation_contract.json")


def plan(locs, positions=None, background="Green board, two desks"):
    return {"locations": ["classroom"], "bible": {"background": background, "characters": {"Monkey": "brown", "Rabbit": "white"}},
            "outline": [{"index": i, "location": l, "characters": ["Monkey"],
                         "end_positions": (positions or {}).get(i, {"Monkey": "classroom"})} for i, l in enumerate(locs, 1)]}


def codes(p):
    return {f.code for f in check_fixed_background(p, True)}


def test_aliases_are_the_same_room():
    assert codes(plan(["classroom", "Classroom", "the class room", "School classroom"])) == set()


def test_within_room_positions_are_not_backgrounds():
    p = plan(["classroom"] * 3, {2: {"Monkey": "classroom: front desk"}, 3: {"Monkey": "near the window"}})
    assert codes(p) == set()


def test_real_room_change_fails_with_path():
    p = plan(["classroom", "classroom", "library"]); p["locations"] = ["classroom", "library"]
    fs = check_fixed_background(p, True)
    assert "continuity.room_changed" in {f.code for f in fs}
    assert any(f.path == "outline[2].location" and f.observed == "library" for f in fs)


def test_unknown_room_and_outdoor_fail_precisely():
    assert "continuity.unknown_room" in codes(plan(["classroom", "kitchen"]))
    assert "continuity.left_background" in codes(plan(["classroom", "outside"]))


def test_not_applied_unless_requested():
    assert check_fixed_background(plan(["classroom", "outside"]), False) == []


def test_missing_background_lock():
    assert "continuity.background_unlocked" in codes(plan(["classroom"], background=""))


def test_validation_is_read_only():
    p = plan(["classroom", "Classroom"]); before = copy.deepcopy(p)
    validate(p, {"profile": "story.plan", "contract": Contract("x", [Constraint("background.fixed", "invariant", True)]), "count": 2})
    assert p == before


def dialogue(n):
    return "# Scene 1: A\n\n## Visual Prompt\nx\n\n## Dialogue\nMonkey: " + " ".join(["word"] * n) + " (smiles)\n"


def test_spoken_word_bounds_24_25_30_31():
    c = Contract("x", [Constraint("dialogue.words", "range", [25, 30]), Constraint("sections", "fields", ["Visual Prompt", "Dialogue"])])
    status = {n: validate(dialogue(n), {"profile": "story.unit", "contract": c, "unit": 1}).status for n in (24, 25, 30, 31)}
    assert status == {24: "failed", 25: "passed", 30: "passed", 31: "failed"}


def test_assembly_with_fixed_background_is_unverified_not_passed():
    c = Contract("x", [Constraint("background.fixed", "invariant", True)])
    r = validate(["# Scene 1: A\nx"], {"profile": "story.assembly", "contract": c, "count": 1})
    assert r.status == "unverified"


def test_report_is_bound_to_hashes():
    c = Contract("x", [])
    r = validate(dialogue(26), {"profile": "story.unit", "contract": c, "unit": 1})
    assert r.bound_to(c.hash, r.artifact_hash)
    assert not r.bound_to(c.hash, "other")


def test_repair_patches_only_allowed_paths_on_copy():
    p = plan(["classroom", "library"])
    r = validate(p, {"profile": "story.plan", "contract": Contract("x", [Constraint("background.fixed", "invariant", True)]), "count": 2})
    req = repair_request(p, r)
    assert req["rejected_candidate"] is p and "outline[1].location" in req["allowed_paths"]
    fixed, rejected = apply_patches(p, [{"path": "outline[1].location", "value": "classroom"}, {"path": "bible.characters.Monkey", "value": "red"}],
                                    set(req["allowed_paths"]))
    assert fixed["outline"][1]["location"] == "classroom" and p["outline"][1]["location"] == "library"
    assert rejected == ["bible.characters.Monkey"] and fixed["bible"]["characters"]["Monkey"] == "brown"


def test_repair_loop_stops_on_identical_failures_and_round_cap():
    c = Contract("x", [Constraint("background.fixed", "invariant", True)])
    r = validate(plan(["classroom", "outside"]), {"profile": "story.plan", "contract": c, "count": 2})
    loop = RepairLoop(max_rounds=5)
    for _ in range(3):
        loop.record(r)
    assert not loop.should_continue()
    capped = RepairLoop(max_rounds=2); capped.next(); capped.next()
    assert not capped.should_continue()


def test_split_position():
    assert split_position("classroom: front desk") == ("classroom", "front desk")
    assert RoomIndex(["Classroom"]).resolve("class room") == "Classroom"


def test_shared_contract_fixture():
    """Same fixture is consumed by the VISION app's TS adapter (src/lib/validation)."""
    fx = json.load(open(FIX, encoding="utf-8"))
    c = Contract(fx["operation"], [Constraint(**k) for k in fx["constraints"]])
    for case in fx["unit_cases"]:
        r = validate(case["text"], {"profile": "story.unit", "contract": c, "unit": 1})
        assert r.status == case["status"], case["name"]
        assert sorted({f.code for f in r.errors}) == case["codes"], case["name"]
