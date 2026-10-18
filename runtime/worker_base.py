"""Shared worker contract. Every workers/<name>/index.py builds one Worker; no generation logic is duplicated:
workers validate scope (operation + format) and delegate to the operations in runtime/operations.py."""
from __future__ import annotations

import json
import os

from runtime.operations import OpError, get

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "registry", "workflows.json")) as _f:
    WORKFLOWS: dict = json.load(_f)

# Magic bytes for outputs whose container can be checked cheaply.
SIGNATURES = {
    "pdf": [b"%PDF"], "png": [b"\x89PNG"], "jpeg": [b"\xff\xd8\xff"], "gif": [b"GIF8"], "bmp": [b"BM"],
    "tiff": [b"II*\x00", b"MM\x00*"], "ico": [b"\x00\x00\x01\x00"], "webp": [b"RIFF"], "wav": [b"RIFF"], "avi": [b"RIFF"],
    "flac": [b"fLaC"], "ogg": [b"OggS"], "opus": [b"OggS"], "mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
    "aiff": [b"FORM"], "zip": [b"PK\x03\x04"], "cbz": [b"PK\x03\x04"], "docx": [b"PK\x03\x04"], "xlsx": [b"PK\x03\x04"],
    "pptx": [b"PK\x03\x04"], "odt": [b"PK\x03\x04"], "ods": [b"PK\x03\x04"], "odp": [b"PK\x03\x04"], "epub": [b"PK\x03\x04"],
    "7z": [b"7z\xbc\xaf\x27\x1c"], "tgz": [b"\x1f\x8b"], "sqlite": [b"SQLite format 3\x00"], "parquet": [b"PAR1"],
    "avro": [b"Obj\x01"], "woff": [b"wOFF"], "woff2": [b"wOF2"], "ttf": [b"\x00\x01\x00\x00", b"true"], "otf": [b"OTTO"],
    "doc": [b"\xd0\xcf\x11\xe0"], "xls": [b"\xd0\xcf\x11\xe0"], "ppt": [b"\xd0\xcf\x11\xe0"], "rtf": [b"{\\rtf"],
    "webm": [b"\x1a\x45\xdf\xa3"], "mkv": [b"\x1a\x45\xdf\xa3"], "wmv": [b"\x30\x26\xb2\x75"], "wma": [b"\x30\x26\xb2\x75"],
}


def ext_of(name) -> str:
    base = str(name or "").lower().replace("\\", "/").rsplit("/", 1)[-1]
    if base == "dockerfile":
        return "dockerfile"
    return base.rsplit(".", 1)[1] if "." in base else ""


def detect(path: str, fmt: str) -> bool:
    sigs = SIGNATURES.get(fmt)
    if not sigs:
        return True
    with open(path, "rb") as f:
        head = f.read(16)
    return any(head.startswith(s) for s in sigs)


class Worker:
    def __init__(self, name: str):
        self.name = name
        self.workflows = {k: v for k, v in WORKFLOWS.items() if v["worker"] == name}
        if not self.workflows:
            raise ValueError(f"no workflow routes to worker {name}")
        self.ops = {op for w in self.workflows.values() for op in w["ops"]}
        self.formats = {s for w in self.workflows.values() for s in w.get("sources", [])}

    def run(self, op_id: str, inp: str | None, params: dict, workdir: str, input_name: str | None = None) -> dict:
        if op_id not in self.ops:
            raise OpError(f"The {self.name} worker does not run '{op_id}'.")
        op = get(op_id)
        source = str(params.get("source") or ext_of(input_name)).lower()
        catch_all = any(not w.get("sources") and op_id in w["ops"] for w in self.workflows.values())
        if op.needs_input and self.formats and source not in self.formats and not catch_all:
            raise OpError(f"The {self.name} worker does not accept .{source or '?'} files.")
        if inp is not None:
            real = os.path.realpath(inp)
            if os.path.dirname(real) != os.path.realpath(workdir):
                raise OpError("Input path rejected.")
        res = op.run(inp, params, workdir)
        out = res.get("output")
        target = str(params.get("target", "")).lower().replace("jpg", "jpeg")
        if out:
            if os.path.dirname(os.path.realpath(out)) != os.path.realpath(workdir):
                raise OpError("Output path rejected.")
            if os.path.getsize(out) == 0:
                raise OpError("The worker produced an empty file.")
            if target and not detect(out, target):
                raise OpError(f"The output is not a valid {target.upper()} file.")
        result = dict(res.get("result") or {})
        result.update({"worker": self.name, "status": "ok", "detected_format": target or source or None})
        res["result"] = result
        return res
