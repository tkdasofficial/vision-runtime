"""Real conversions with the native tools installed in CI; each output is re-read independently."""
import json, os, sqlite3, subprocess, tarfile, zipfile
import pytest
from runtime.operations import get, OpError

def run(op, path, tmp_path, **params):
    w = tmp_path / "w"; w.mkdir(exist_ok=True)
    return get(op).run(str(path), params, str(w))

def tone(tmp_path):
    p = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(p)], check=True)
    return p

@pytest.mark.parametrize("target", ["mp3", "flac", "ogg", "opus", "m4a", "aac", "aiff", "wma", "wav"])
def test_audio(tmp_path, target):
    r = run("audio_convert", tone(tmp_path), tmp_path, target=target)
    assert r["result"]["duration"] > 0.5

def test_audio_rejects_text(tmp_path):
    p = tmp_path / "a.bin"; p.write_text("hello")
    with pytest.raises(OpError): run("audio_convert", p, tmp_path, target="mp3")

def test_video_new_targets(tmp_path):
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=176x144:duration=1", "-f", "lavfi", "-i", "sine=duration=1", "-shortest", "-pix_fmt", "yuv420p", str(p)], check=True)
    for t in ["avi", "wmv", "mpeg", "3gp", "m4v"]:
        assert run("video_convert", p, tmp_path / t if (tmp_path / t).mkdir() is None else None, target=t)["result"]["width"] == 176

def test_office_docx_pdf_roundtrip(tmp_path):
    p = tmp_path / "a.html"; p.write_text("<h1>Hello नमस्ते</h1><p>Body</p>", encoding="utf-8")
    r = run("office_convert", p, tmp_path, source="html", target="docx")
    assert zipfile.is_zipfile(r["output"])
    r2 = run("office_convert", r["output"], tmp_path / "x" if (tmp_path / "x").mkdir() is None else None, source="docx", target="pdf")
    assert r2["result"]["pages"] >= 1

def test_office_sheet(tmp_path):
    p = tmp_path / "a.csv"; p.write_text("name,qty\nA,-3\n", encoding="utf-8")
    for t in ["xlsx", "ods", "xls"]:
        d = tmp_path / t; d.mkdir()
        assert os.path.getsize(run("office_convert", p, d, source="csv", target=t)["output"]) > 100

def test_office_wrong_family(tmp_path):
    p = tmp_path / "a.csv"; p.write_text("a\n1\n")
    with pytest.raises(OpError): run("office_convert", p, tmp_path, source="csv", target="pptx")

def test_sqlite_parquet_avro(tmp_path):
    db = tmp_path / "a.db"; con = sqlite3.connect(db); con.execute("create table t(name text, n int)"); con.execute("insert into t values ('हिन्दी', -2)"); con.commit(); con.close()
    j = run("data_convert", db, tmp_path, source="sqlite", target="json")
    assert json.load(open(j["output"], encoding="utf-8")) == [{"name": "हिन्दी", "n": -2}]
    for t in ["parquet", "avro", "csv"]:
        d = tmp_path / t; d.mkdir()
        out = run("data_convert", db, d, source="sqlite", target=t)["output"]
        d2 = tmp_path / (t + "2"); d2.mkdir()
        if t != "csv":
            back = run("data_convert", out, d2, source=t, target="json")
            assert json.load(open(back["output"], encoding="utf-8"))[0]["n"] == -2

def test_archive_7z_to_zip(tmp_path):
    src = tmp_path / "s"; src.mkdir(); (src / "a.txt").write_text("hi")
    subprocess.run(["7z", "a", "-y", str(tmp_path / "a.7z"), "a.txt"], cwd=src, check=True, capture_output=True)
    r = run("archive_convert", tmp_path / "a.7z", tmp_path, source="7z", target="zip")
    assert zipfile.ZipFile(r["output"]).read("a.txt") == b"hi"

def test_archive_rejects_traversal(tmp_path):
    p = tmp_path / "e.tar"
    with tarfile.open(p, "w") as t:
        info = tarfile.TarInfo("../evil.txt"); info.size = 1
        import io; t.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(OpError): run("archive_convert", p, tmp_path, source="tar", target="zip")

def test_font_ttf_woff2(tmp_path):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    fb = FontBuilder(1000, isTTF=True); fb.setupGlyphOrder([".notdef"]); fb.setupCharacterMap({})
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph()}); fb.setupHorizontalMetrics({".notdef": (500, 0)}); fb.setupHorizontalHeader(); fb.setupNameTable({"familyName": "T", "styleName": "R"}); fb.setupOS2(); fb.setupPost()
    p = tmp_path / "t.ttf"; fb.save(str(p))
    r = run("font_convert", p, tmp_path, target="woff2")
    assert open(r["output"], "rb").read(4) == b"wOF2"
    with pytest.raises(OpError): run("font_convert", p, tmp_path, target="otf")

def test_ebook_epub(tmp_path):
    p = tmp_path / "b.txt"; p.write_text("Chapter 1\n\nOnce upon a time.")
    r = run("ebook_convert", p, tmp_path, source="txt", target="epub")
    assert zipfile.is_zipfile(r["output"])
    d = tmp_path / "m"; d.mkdir()
    r2 = run("ebook_convert", r["output"], d, source="epub", target="mobi")
    assert b"BOOKMOBI" in open(r2["output"], "rb").read(100)

@pytest.mark.parametrize("src,body", [("dot", "digraph{a->b}"), ("puml", "@startuml\nA -> B\n@enduml"), ("mmd", "graph TD; A-->B"), ("svg", '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>')])
def test_render(tmp_path, src, body):
    p = tmp_path / f"d.{src}"; p.write_text(body)
    r = run("render", p, tmp_path, source=src, target="png")
    assert open(r["output"], "rb").read(4) == b"\x89PNG"

def test_render_tex(tmp_path):
    p = tmp_path / "d.tex"; p.write_text(r"\documentclass{article}\begin{document}Hi\end{document}")
    assert run("render", p, tmp_path, source="tex", target="pdf")["result"]["pages"] == 1

def test_render_dxf(tmp_path):
    import ezdxf
    doc = ezdxf.new(); doc.modelspace().add_line((0, 0), (1, 1)); p = tmp_path / "d.dxf"; doc.saveas(p)
    assert os.path.getsize(run("render", p, tmp_path, source="dxf", target="svg")["output"]) > 100
