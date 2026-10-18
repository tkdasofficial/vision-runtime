import glob
import importlib
import json
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from runtime.operations import OPERATIONS, OpError  # noqa: E402
from runtime.worker_base import WORKFLOWS, detect  # noqa: E402

WFS = sorted(glob.glob(os.path.join(ROOT, ".github", "workflows", "*.yml")))
WORKERS = sorted(glob.glob(os.path.join(ROOT, "workers", "*", "index.py")))


def test_exactly_18_one_word_workflows_one_per_worker():
    assert len(WFS) == 18 == len(WORKFLOWS)
    assert all(k == v['worker'] for k, v in WORKFLOWS.items())
    for p in WFS:
        assert os.path.basename(p)[:-4].isalpha()


def test_exactly_18_workers_load():
    assert len(WORKERS) == 18
    for p in WORKERS:
        name = p.split(os.sep)[-2]
        mod = importlib.import_module(f"workers.{name}.index")
        assert mod.WORKER.name == name and callable(mod.run)


@pytest.mark.parametrize("path", WFS)
def test_workflow_valid_and_routed(path):
    doc = yaml.safe_load(open(path))
    name = os.path.basename(path)[:-4]
    on = doc.get("on", doc.get(True))
    assert list(on) == ["workflow_dispatch"]  # never auto-runs
    reg = WORKFLOWS[name]
    env = doc["jobs"]["run"]["steps"][-1]["env"]
    assert env["WORKFLOW"] == name and env["WORKER"] == reg["worker"]
    assert os.path.exists(os.path.join(ROOT, "workers", reg["worker"], "index.py"))
    assert on["workflow_dispatch"]["inputs"]["operation"]["options"] == reg["ops"]
    text = open(path).read()
    assert ("secrets." in text) == (name == "documents")  # AI keys only in the story-capable workflow
    assert text.count("secrets.") == text.count("inputs.operation == 'story_generate' && secrets.")
    for step in doc["jobs"]["run"]["steps"]:
        if "uses" in step:
            assert len(step["uses"].split("@")[1]) == 40  # pinned to a commit SHA


def test_registry_ops_exist_and_cover_all_runtime_ops():
    used = {op for w in WORKFLOWS.values() for op in w["ops"]}
    assert used <= set(OPERATIONS) and used == set(OPERATIONS)


def test_every_format_routes_to_a_workflow_that_runs_its_op():
    reg = json.load(open(os.path.join(ROOT, "registry", "formats.json")))
    for f in reg["formats"]:
        assert f["worker"] == f["workflow"]
        rt = f["capabilities"]["convert_runtime"]
        if rt:
            assert rt["op"] in WORKFLOWS[f["workflow"]]["ops"]


def test_format_registry_accounts_for_121_formats():
    reg = json.load(open(os.path.join(ROOT, "registry", "formats.json")))
    ids = [f["id"] for f in reg["formats"]]
    assert len(ids) == 121 == len(set(ids)) and reg["categories"] == 18
    for f in reg["formats"]:
        assert f["workflow"] in WORKFLOWS and f["verification"] in {"runtime-implemented", "inline-only", "unsupported"}


def _worker(name):
    return importlib.import_module(f"workers.{name}.index").WORKER


def test_rejects_unrouted_operation(tmp_path):
    with pytest.raises(OpError):
        _worker("code").run("video_convert", None, {}, str(tmp_path))


def test_rejects_wrong_format(tmp_path):
    p = tmp_path / "input.bin"; p.write_bytes(b"x")
    with pytest.raises(OpError):
        _worker("subtitles").run("checksum", str(p), {}, str(tmp_path), "photo.png")


def test_rejects_input_outside_workdir(tmp_path):
    outside = tmp_path / "a.srt"; outside.write_text("1")
    work = tmp_path / "w"; work.mkdir()
    with pytest.raises(OpError):
        _worker("subtitles").run("checksum", str(outside), {}, str(work), "a.srt")


def test_checksum_real(tmp_path):
    p = tmp_path / "input.bin"; p.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
    res = _worker("subtitles").run("checksum", str(p), {}, str(tmp_path), "a.srt")
    assert len(res["result"]["sha256"]) == 64 and res["result"]["worker"] == "subtitles"


def test_terminal_checksum_any_file(tmp_path):
    p = tmp_path / "input.bin"; p.write_bytes(b"\x00\x01")
    assert _worker("logs").run("checksum", str(p), {}, str(tmp_path), "blob.xyz")["result"]["status"] == "ok"


def test_image_convert_real_output(tmp_path):
    from PIL import Image
    p = tmp_path / "input.bin"; Image.new("RGBA", (8, 6), (255, 0, 0, 128)).save(p, "PNG")
    res = _worker("images").run("image_convert", str(p), {"target": "jpeg"}, str(tmp_path), "a.png")
    assert detect(res["output"], "jpeg") and res["result"]["detected_format"] == "jpeg"


def test_data_convert_real_output(tmp_path):
    p = tmp_path / "input.bin"; p.write_text("a,b\n1,x\n2,y\n")
    res = _worker("data").run("data_convert", str(p), {"source": "csv", "target": "parquet"}, str(tmp_path), "t.csv")
    assert detect(res["output"], "parquet")


def test_font_convert_real_output(tmp_path):
    from fontTools.fontBuilder import FontBuilder
    fb = FontBuilder(1000, isTTF=True); fb.setupGlyphOrder([".notdef"]); fb.setupCharacterMap({})
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph()}); fb.setupHorizontalMetrics({".notdef": (500, 0)})
    fb.setupHorizontalHeader(); fb.setupNameTable({"familyName": "T", "styleName": "R"}); fb.setupOS2(); fb.setupPost()
    p = tmp_path / "input.bin"; fb.save(str(p))
    res = _worker("fonts").run("font_convert", str(p), {"target": "woff2"}, str(tmp_path), "t.ttf")
    assert detect(res["output"], "woff2")


def test_output_signature_mismatch_rejected(tmp_path, monkeypatch):
    from runtime import operations as O
    def fake(inp, params, workdir):
        out = os.path.join(workdir, "converted.pdf"); open(out, "wb").write(b"not a pdf"); return {"output": out, "mime": "application/pdf"}
    monkeypatch.setitem(O.OPERATIONS, "render", O.Operation("render", True, 1 << 20, 1 << 20, 10, fake))
    p = tmp_path / "input.bin"; p.write_text("digraph{}")
    with pytest.raises(OpError):
        _worker("diagrams").run("render", str(p), {"source": "dot", "target": "pdf"}, str(tmp_path), "g.dot")
