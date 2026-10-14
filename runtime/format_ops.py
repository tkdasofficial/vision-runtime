"""Heavy file-format operations (native tools). Registered in operations.py.

Rules: fixed argument lists only (no shell strings), fixed file names inside the
work dir, size/time limits, and every output is re-opened/probed before it is
returned. No media generation: only conversion of the user's own file.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import zipfile

MB = 1024 * 1024


from runtime.operations import OpError  # one error type the runner reports safely


def _run(cmd, timeout, cwd=None, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        raise OpError("The conversion took too long and was stopped.")
    except FileNotFoundError:
        raise OpError("A required converter is not installed on the runner.")
    return r


def _src(params, allowed):
    s = str(params.get("source", "")).lower()
    if s not in allowed:
        raise OpError(f"Unsupported input format. Supported: {', '.join(sorted(allowed))}.")
    return s


def _tgt(params, allowed):
    t = str(params.get("target", "")).lower()
    if t not in allowed:
        raise OpError(f"Unsupported output format. Supported: {', '.join(sorted(allowed))}.")
    return t


def _named(inp, workdir, ext):
    p = os.path.join(workdir, f"source.{ext}")
    shutil.copyfile(inp, p)
    return p


def _nonempty(path, what="file"):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise OpError(f"The converter produced no {what}.")
    return path


# ---------- audio (decode + re-encode with ffmpeg) ----------
AUDIO_TARGETS = {
    "mp3": (["-c:a", "libmp3lame", "-b:a", "192k"], "audio/mpeg", "mp3"),
    "wav": (["-c:a", "pcm_s16le"], "audio/wav", "wav"),
    "flac": (["-c:a", "flac"], "audio/flac", "flac"),
    "ogg": (["-c:a", "libvorbis", "-q:a", "5"], "audio/ogg", "ogg"),
    "opus": (["-c:a", "libopus", "-b:a", "128k"], "audio/ogg", "opus"),
    "m4a": (["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"], "audio/mp4", "ipod"),
    "aac": (["-c:a", "aac", "-b:a", "192k"], "audio/aac", "adts"),
    "aiff": (["-c:a", "pcm_s16be"], "audio/aiff", "aiff"),
    "wma": (["-c:a", "wmav2", "-b:a", "192k"], "audio/x-ms-wma", "asf"),
}
MAX_AUDIO_SECONDS = 60 * 60


def ffprobe(path):
    r = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], 60)
    if r.returncode != 0:
        raise OpError("The file is not a valid, supported audio/video file.")
    return json.loads(r.stdout)


def op_audio_convert(inp, params, workdir):
    target = _tgt(params, AUDIO_TARGETS)
    info = ffprobe(inp)
    a = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not a:
        raise OpError("No audio stream found in the file.")
    if float(info.get("format", {}).get("duration") or 0) > MAX_AUDIO_SECONDS:
        raise OpError("Audio longer than 60 minutes is not supported.")
    args, mime, fmt = AUDIO_TARGETS[target]
    out = os.path.join(workdir, f"converted.{target}")
    r = _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", inp, "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1", *args, "-f", fmt, out], 1200)
    if r.returncode != 0:
        raise OpError("The audio conversion failed (damaged input or unsupported codec).")
    oi = ffprobe(_nonempty(out))
    oa = [s for s in oi.get("streams", []) if s.get("codec_type") == "audio"]
    if not oa:
        raise OpError("The converted file has no audio stream.")
    return {"output": out, "mime": mime, "result": {"to": target, "duration": round(float(oi["format"].get("duration") or 0), 2), "codec": oa[0].get("codec_name"), "sample_rate": oa[0].get("sample_rate"), "channels": oa[0].get("channels")}}


# ---------- office documents (LibreOffice headless) ----------
OFFICE_SOURCES = {"doc", "docx", "odt", "rtf", "txt", "html", "xls", "xlsx", "ods", "csv", "tsv", "ppt", "pptx", "odp", "md"}
OFFICE_FILTERS = {
    "pdf": ("pdf", "application/pdf"), "docx": ("docx:MS Word 2007 XML", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "doc": ("doc:MS Word 97", "application/msword"), "odt": ("odt", "application/vnd.oasis.opendocument.text"),
    "rtf": ("rtf", "application/rtf"), "txt": ("txt:Text (encoded):UTF8", "text/plain"), "html": ("html", "text/html"),
    "xlsx": ("xlsx:Calc MS Excel 2007 XML", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "xls": ("xls:MS Excel 97", "application/vnd.ms-excel"), "ods": ("ods", "application/vnd.oasis.opendocument.spreadsheet"),
    "csv": ("csv:Text - txt - csv (StarCalc):44,34,76,1", "text/csv"),
    "pptx": ("pptx:Impress MS PowerPoint 2007 XML", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    "ppt": ("ppt:MS PowerPoint 97", "application/vnd.ms-powerpoint"), "odp": ("odp", "application/vnd.oasis.opendocument.presentation"),
}
FAMILY = {
    "text": {"doc", "docx", "odt", "rtf", "txt", "html", "md"},
    "sheet": {"xls", "xlsx", "ods", "csv", "tsv"},
    "slides": {"ppt", "pptx", "odp"},
}
ALLOWED_OUT = {"text": {"pdf", "docx", "doc", "odt", "rtf", "txt", "html"}, "sheet": {"pdf", "xlsx", "xls", "ods", "csv", "html"}, "slides": {"pdf", "pptx", "ppt", "odp"}}


def op_office_convert(inp, params, workdir):
    src = _src(params, OFFICE_SOURCES)
    target = _tgt(params, OFFICE_FILTERS)
    fam = next(k for k, v in FAMILY.items() if src in v)
    if target not in ALLOWED_OUT[fam]:
        raise OpError(f"{src.upper()} can be converted to {', '.join(sorted(ALLOWED_OUT[fam])).upper()}.")
    if src == target:
        raise OpError(f"The file is already {target.upper()}.")
    ext = "txt" if src == "md" else src
    path = _named(inp, workdir, ext)
    outdir = os.path.join(workdir, "out"); os.makedirs(outdir)
    env = dict(os.environ, HOME=workdir)
    r = _run(["soffice", "--headless", "--norestore", "--nolockcheck", "-env:UserInstallation=file://" + os.path.join(workdir, "lo"), "--convert-to", OFFICE_FILTERS[target][0], "--outdir", outdir, path], 600, env=env)
    outs = [os.path.join(outdir, f) for f in os.listdir(outdir)]
    if r.returncode != 0 or not outs:
        raise OpError("LibreOffice could not convert this file (it may be damaged or password-protected).")
    out = os.path.join(workdir, f"converted.{target}")  # runner only accepts outputs directly in the work dir
    shutil.move(_nonempty(outs[0]), out)
    res = {"from": src, "to": target}
    if target == "pdf":
        from pypdf import PdfReader
        try:
            res["pages"] = len(PdfReader(out).pages)
        except Exception:
            raise OpError("The produced PDF does not open.")
        if res["pages"] < 1:
            raise OpError("The produced PDF has no pages.")
    elif target in ("docx", "xlsx", "pptx", "odt", "ods", "odp"):
        if not zipfile.is_zipfile(out):
            raise OpError("The produced document is not a valid package.")
    return {"output": out, "mime": OFFICE_FILTERS[target][1], "result": res}


# ---------- data stores: SQLite, Parquet, Avro, Shapefile ----------
DATA_SOURCES = {"sqlite", "db", "parquet", "avro", "csv", "json", "shp"}
DATA_TARGETS = {"csv": "text/csv", "json": "application/json", "parquet": "application/vnd.apache.parquet", "avro": "application/avro", "sqlite": "application/vnd.sqlite3", "geojson": "application/geo+json"}
MAX_ROWS = 200_000


def _read_rows(src, path, params):
    import csv
    if src in ("sqlite", "db"):
        import sqlite3
        with open(path, "rb") as f:
            if f.read(16) != b"SQLite format 3\x00":
                raise OpError("This is not a SQLite database.")
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            if not tables:
                raise OpError("The database has no tables.")
            want = str(params.get("table") or "")
            out = {}
            for t in tables if not want else [x for x in tables if x == want] or tables[:1]:
                cur = con.execute(f'SELECT * FROM "{t.replace(chr(34), chr(34) * 2)}" LIMIT {MAX_ROWS}')
                cols = [d[0] for d in cur.description]
                out[t] = [dict(zip(cols, [v.hex() if isinstance(v, bytes) else v for v in row])) for row in cur]
            return out
        finally:
            con.close()
    if src == "parquet":
        import pyarrow.parquet as pq
        try:
            return {"data": pq.read_table(path).slice(0, MAX_ROWS).to_pylist()}
        except Exception:
            raise OpError("This is not a readable Parquet file.")
    if src == "avro":
        import fastavro
        try:
            with open(path, "rb") as f:
                return {"data": [r for _, r in zip(range(MAX_ROWS), fastavro.reader(f))]}
        except Exception:
            raise OpError("This is not a readable Avro file.")
    if src == "csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            return {"data": [r for _, r in zip(range(MAX_ROWS), csv.DictReader(f))]}
    if src == "json":
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        if isinstance(v, dict):
            v = next((x for x in v.values() if isinstance(x, list)), [v])
        if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
            raise OpError("The JSON must be a list of objects.")
        return {"data": v[:MAX_ROWS]}
    raise OpError("Unsupported data source.")


def _shp_geojson(path, workdir):
    import shapefile
    d = os.path.join(workdir, "shp"); os.makedirs(d)
    if not zipfile.is_zipfile(path):
        raise OpError("Attach the Shapefile as a ZIP containing the .shp, .shx and .dbf files.")
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if re.search(r"\.(shp|shx|dbf|prj|cpg)$", n, re.I) and ".." not in n and not n.startswith("/")]
        shp = [n for n in names if n.lower().endswith(".shp")]
        if not shp:
            raise OpError("No .shp file was found in the ZIP.")
        stem = shp[0][:-4]
        for ext in ("shp", "shx", "dbf"):
            m = [n for n in names if n.lower() == f"{stem}.{ext}".lower()]
            if not m:
                raise OpError(f"The ZIP is missing the .{ext} part of the Shapefile.")
            with open(os.path.join(d, f"layer.{ext}"), "wb") as f:
                f.write(z.read(m[0]))
    r = shapefile.Reader(os.path.join(d, "layer"))
    return r.__geo_interface__


def op_data_convert(inp, params, workdir):
    src = _src(params, DATA_SOURCES)
    target = _tgt(params, DATA_TARGETS)
    out = os.path.join(workdir, f"converted.{target}")
    if src == "shp":
        if target != "geojson":
            raise OpError("Shapefiles can be converted to GeoJSON.")
        gj = _shp_geojson(inp, workdir)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(gj, f, ensure_ascii=False, default=str)
        return {"output": out, "mime": DATA_TARGETS[target], "result": {"from": src, "to": target, "features": len(gj.get("features", []))}}
    if target == "geojson" or src == target or (src == "db" and target == "sqlite"):
        raise OpError(f"{src.upper()} → {target.upper()} is not a supported conversion.")
    tables = _read_rows(src, inp, params)
    first = next(iter(tables.values()))
    if target == "json":
        with open(out, "w", encoding="utf-8") as f:
            json.dump(tables if len(tables) > 1 else first, f, ensure_ascii=False, indent=2, default=str)
    elif target == "csv":
        import csv
        cols = list(dict.fromkeys(k for r in first for k in r))
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(first)
    elif target == "parquet":
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(first), out)
        if pq.read_metadata(out).num_rows != len(first):
            raise OpError("The Parquet file did not verify.")
    elif target == "avro":
        import fastavro
        cols = list(dict.fromkeys(k for r in first for k in r))
        def typ(c):
            vals = [r.get(c) for r in first if r.get(c) is not None]
            if vals and all(isinstance(v, bool) for v in vals): return ["null", "boolean"]
            if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals): return ["null", "long"]
            if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals): return ["null", "double"]
            return ["null", "string"]
        schema = {"type": "record", "name": "Row", "fields": [{"name": re.sub(r"\W", "_", c) or "col", "type": typ(c)} for c in cols]}
        def conv(v, t):
            if v is None: return None
            return str(v) if t == ["null", "string"] else v
        recs = [{f["name"]: conv(r.get(c), f["type"]) for c, f in zip(cols, schema["fields"])} for r in first]
        with open(out, "wb") as f:
            fastavro.writer(f, fastavro.parse_schema(schema), recs)
    elif target == "sqlite":
        import sqlite3
        con = sqlite3.connect(out)
        for name, rows in tables.items():
            cols = list(dict.fromkeys(k for r in rows for k in r)) or ["value"]
            q = lambda s: '"' + str(s).replace('"', '""') + '"'
            con.execute(f"CREATE TABLE {q(name)} ({', '.join(q(c) for c in cols)})")
            con.executemany(f"INSERT INTO {q(name)} VALUES ({', '.join('?' * len(cols))})", [[r.get(c) if not isinstance(r.get(c), (dict, list)) else json.dumps(r.get(c)) for c in cols] for r in rows])
        con.commit(); con.close()
    _nonempty(out)
    return {"output": out, "mime": DATA_TARGETS[target], "result": {"from": src, "to": target, "tables": {k: len(v) for k, v in tables.items()}}}


# ---------- archives (7z / rar / tar → zip), safe extraction ----------
ARCHIVE_SOURCES = {"7z", "rar", "tar", "gz", "tgz", "zip", "cbr", "cbz"}
ARCHIVE_TARGETS = {"zip": "application/zip", "7z": "application/x-7z-compressed", "tar": "application/x-tar", "tgz": "application/gzip", "cbz": "application/vnd.comicbook+zip"}
MAX_ENTRIES = 5000
MAX_UNPACKED = 300 * MB


def _safe_extract(path, src, workdir):
    d = os.path.join(workdir, "x"); os.makedirs(d)
    named = _named(path, workdir, {"tgz": "tar.gz", "cbr": "rar", "cbz": "zip"}.get(src, src))
    r = _run(["7z", "l", "-slt", "-ba", named], 120)
    if r.returncode != 0:
        raise OpError("The archive can't be read (damaged, encrypted or wrong type).")
    entries, total = 0, 0
    for block in r.stdout.split("\n\n"):
        kv = dict(l.split(" = ", 1) for l in block.splitlines() if " = " in l)
        if not kv.get("Path"):
            continue
        p = kv["Path"].replace("\\", "/")
        if p.startswith("/") or ".." in p.split("/") or re.match(r"^[A-Za-z]:", p):
            raise OpError("The archive contains unsafe paths and was rejected.")
        if kv.get("Attributes", "").startswith("l") or "Symbolic" in kv.get("Attributes", ""):
            raise OpError("The archive contains links and was rejected.")
        if kv.get("Encrypted") == "+":
            raise OpError("Password-protected archives are not supported.")
        entries += 1; total += int(kv.get("Size") or 0)
    if entries > MAX_ENTRIES or total > MAX_UNPACKED:
        raise OpError("The archive is too large when unpacked.")
    r = _run(["7z", "x", "-y", "-snl-", f"-o{d}", named], 600)
    if r.returncode != 0:
        raise OpError("The archive could not be extracted.")
    if src in ("gz", "tgz"):  # gz holds a tar: unpack the inner tar too
        inner = [f for f in os.listdir(d) if f.endswith(".tar")]
        if inner:
            return _safe_extract(os.path.join(d, inner[0]), "tar", os.path.join(workdir, "t"))
    for root, dirs, files in os.walk(d):
        for n in dirs + files:
            if os.path.islink(os.path.join(root, n)):
                raise OpError("The archive contains links and was rejected.")
    return d


def op_archive_convert(inp, params, workdir):
    src = _src(params, ARCHIVE_SOURCES)
    target = _tgt(params, ARCHIVE_TARGETS)
    if src == target:
        raise OpError(f"The archive is already {target.upper()}.")
    os.makedirs(os.path.join(workdir, "t"), exist_ok=True)
    d = _safe_extract(inp, src, workdir)
    files = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs]
    if not files:
        raise OpError("The archive is empty.")
    out = os.path.join(workdir, f"converted.{target}")
    if target in ("zip", "cbz"):
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):
                z.write(f, os.path.relpath(f, d))
        with zipfile.ZipFile(out) as z:
            if z.testzip() is not None or len(z.namelist()) != len(files):
                raise OpError("The new ZIP did not verify.")
    elif target in ("tar", "tgz"):
        import tarfile
        with tarfile.open(out, "w:gz" if target == "tgz" else "w") as t:
            for f in sorted(files):
                t.add(f, os.path.relpath(f, d), recursive=False)
        with tarfile.open(out) as t:
            if len(t.getmembers()) != len(files):
                raise OpError("The new archive did not verify.")
    else:
        r = _run(["7z", "a", "-t7z", "-y", out, "."], 600, cwd=d)
        if r.returncode != 0 or _run(["7z", "t", out], 120).returncode != 0:
            raise OpError("The 7z archive did not verify.")
    return {"output": _nonempty(out), "mime": ARCHIVE_TARGETS[target], "result": {"from": src, "to": target, "files": len(files)}}


# ---------- e-books (Calibre) ----------
EBOOK_SOURCES = {"mobi", "azw3", "fb2", "epub", "cbz", "cbr", "docx", "html", "txt", "md", "rtf", "odt"}
EBOOK_TARGETS = {"epub": "application/epub+zip", "pdf": "application/pdf", "txt": "text/plain", "mobi": "application/x-mobipocket-ebook", "azw3": "application/vnd.amazon.ebook", "fb2": "application/x-fictionbook+xml", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "html": "text/html"}


def op_ebook_convert(inp, params, workdir):
    src = _src(params, EBOOK_SOURCES)
    target = _tgt(params, EBOOK_TARGETS)
    if src == target:
        raise OpError(f"The book is already {target.upper()}.")
    path = _named(inp, workdir, "txtz" if False else ("txt" if src == "md" else src))
    out = os.path.join(workdir, f"converted.{'htmlz' if target == 'html' else target}")
    r = _run(["ebook-convert", path, out], 900, env=dict(os.environ, HOME=workdir, QT_QPA_PLATFORM="offscreen"))
    if r.returncode != 0:
        raise OpError("Calibre could not convert this book (it may be DRM-protected or damaged).")
    _nonempty(out)
    if target == "html":  # htmlz is a zip with index.html; return the HTML itself
        with zipfile.ZipFile(out) as z:
            html = next((n for n in z.namelist() if n.endswith(".html")), None)
            if not html:
                raise OpError("The converted book has no HTML.")
            out2 = os.path.join(workdir, "converted.html")
            with open(out2, "wb") as f:
                f.write(z.read(html))
            out = out2
    if target == "epub" and not zipfile.is_zipfile(out):
        raise OpError("The produced EPUB is not valid.")
    return {"output": out, "mime": EBOOK_TARGETS[target], "result": {"from": src, "to": target}}


# ---------- fonts (fontTools) ----------
FONT_TARGETS = {"ttf": "font/ttf", "otf": "font/otf", "woff": "font/woff", "woff2": "font/woff2"}


def op_font_convert(inp, params, workdir):
    from fontTools.ttLib import TTFont
    target = _tgt(params, FONT_TARGETS)
    try:
        font = TTFont(inp)
    except Exception:
        raise OpError("The file is not a valid TTF/OTF/WOFF/WOFF2 font.")
    cff = "CFF " in font or "CFF2" in font
    if target == "ttf" and cff:
        raise OpError("This font uses PostScript (CFF) outlines; it can be saved as OTF, WOFF or WOFF2, not TTF.")
    if target == "otf" and not cff:
        raise OpError("This font uses TrueType outlines; it can be saved as TTF, WOFF or WOFF2, not OTF.")
    font.flavor = target if target in ("woff", "woff2") else None
    out = os.path.join(workdir, f"converted.{target}")
    font.save(out)
    check = TTFont(out)
    name = check["name"].getDebugName(4) if "name" in check else None
    return {"output": out, "mime": FONT_TARGETS[target], "result": {"to": target, "name": name, "glyphs": len(check.getGlyphOrder())}}


# ---------- diagrams & TeX rendering ----------
RENDER_SOURCES = {"dot", "mmd", "puml", "dxf", "tex", "svg"}
RENDER_TARGETS = {"svg": "image/svg+xml", "png": "image/png", "pdf": "application/pdf"}


def op_render(inp, params, workdir):
    src = _src(params, RENDER_SOURCES)
    target = _tgt(params, RENDER_TARGETS)
    out = os.path.join(workdir, f"converted.{target}")
    if src == "dot":
        r = _run(["dot", f"-T{target}", "-o", out, inp], 120)
    elif src == "puml":
        path = _named(inp, workdir, "puml")
        r = _run(["plantuml", f"-t{target}", "-o", workdir, path], 180, env=dict(os.environ, PLANTUML_SECURITY_PROFILE="SANDBOX"))
        made = os.path.join(workdir, f"source.{target}")
        if os.path.exists(made):
            shutil.move(made, out)
    elif src == "mmd":
        path = _named(inp, workdir, "mmd")
        cfg = os.path.join(workdir, "pp.json")
        with open(cfg, "w") as f:
            json.dump({"args": ["--no-sandbox"], "executablePath": shutil.which("google-chrome") or shutil.which("chromium") or ""}, f)
        r = _run(["mmdc", "-q", "-p", cfg, "-i", path, "-o", out], 240)
    elif src == "dxf":
        import ezdxf, matplotlib
        matplotlib.use("Agg")
        from ezdxf.addons.drawing import matplotlib as dm
        try:
            doc = ezdxf.readfile(inp)
        except Exception:
            raise OpError("The file is not a valid DXF drawing.")
        dm.qsave(doc.modelspace(), out)
        r = subprocess.CompletedProcess([], 0)
    elif src == "tex":
        if target != "pdf":
            raise OpError("TeX documents are compiled to PDF.")
        path = _named(inp, workdir, "tex")
        for _ in range(2):  # second pass resolves references
            r = _run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "source.tex"], 180, cwd=workdir, env=dict(os.environ, openout_any="p", openin_any="p"))
        made = os.path.join(workdir, "source.pdf")
        if os.path.exists(made):
            shutil.move(made, out)
    else:  # svg → png/pdf
        if target == "svg":
            raise OpError("The file is already SVG.")
        import cairosvg
        try:
            (cairosvg.svg2png if target == "png" else cairosvg.svg2pdf)(url=inp, write_to=out, unsafe=False)
        except Exception:
            raise OpError("The SVG could not be rendered.")
        r = subprocess.CompletedProcess([], 0)
    if r.returncode != 0 or not os.path.exists(out):
        raise OpError(f"The {src.upper()} source has errors and could not be rendered.")
    _nonempty(out)
    res = {"from": src, "to": target}
    if target == "pdf":
        from pypdf import PdfReader
        res["pages"] = len(PdfReader(out).pages)
    return {"output": out, "mime": RENDER_TARGETS[target], "result": res}
