"""Registry of permitted operations. Nothing outside this file can run on the runner.

Each operation declares accepted inputs, limits, expected outputs and a timeout.
Commands are always built from fixed argument lists (never a shell string), so
user values can't inject commands.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

MB = 1024 * 1024


class OpError(Exception):
    """Safe, user-facing failure."""


@dataclass(frozen=True)
class Operation:
    id: str
    needs_input: bool
    max_input: int
    max_output: int
    timeout_s: int
    run: Callable[[str | None, dict, str], dict]


# ---------- checksums ----------
ALGOS = {"sha256": hashlib.sha256, "sha512": hashlib.sha512}


def file_digest(path: str, algo: str) -> str:
    h = ALGOS[algo]()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def op_checksum(inp, params, workdir):
    algos = [a for a in params.get("algorithms", ["sha256"]) if a in ALGOS] or ["sha256"]
    out = {a: file_digest(inp, a) for a in algos}
    expected = str(params.get("expected", "")).strip().lower()
    if expected:
        if not re.fullmatch(r"[0-9a-f]{64}|[0-9a-f]{128}", expected):
            raise OpError("Expected checksum must be a 64 (SHA-256) or 128 (SHA-512) character hex string.")
        algo = "sha256" if len(expected) == 64 else "sha512"
        actual = out.get(algo) or file_digest(inp, algo)
        out["verify"] = {"algorithm": algo, "expected": expected, "actual": actual, "match": actual == expected}
    return {"result": out}


# ---------- image format conversion (no editing) ----------
IMAGE_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP", "gif": "GIF", "bmp": "BMP", "tiff": "TIFF", "ico": "ICO", "avif": "AVIF", "pdf": "PDF"}
IMAGE_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp", "tiff": "image/tiff", "ico": "image/x-icon", "avif": "image/avif", "pdf": "application/pdf"}
NO_ALPHA = {"jpeg", "bmp", "pdf"}


def op_image_convert(inp, params, workdir):
    from PIL import Image, UnidentifiedImageError

    target = str(params.get("target", "")).lower().replace("jpg", "jpeg")
    if target not in IMAGE_FORMATS:
        raise OpError(f"Unsupported output format. Supported: {', '.join(sorted(IMAGE_FORMATS))}.")
    Image.MAX_IMAGE_PIXELS = 80_000_000  # decompression-bomb guard
    try:  # HEIC/HEIF (and AVIF on older Pillow) decoders
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass
    with open(inp, "rb") as fh:
        head = fh.read(512).lstrip()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head):
        import cairosvg
        png = os.path.join(workdir, "svg.png")
        try:
            cairosvg.svg2png(url=inp, write_to=png, unsafe=False)
        except Exception:
            raise OpError("The SVG could not be rendered.")
        inp = png
    try:
        with Image.open(inp) as probe:
            probe.verify()
        img = Image.open(inp)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        raise OpError("The file is not a valid, supported image.")
    src = (img.format or "").lower()
    if src == target.upper().lower() and src != "":
        raise OpError(f"The image is already {target.upper()}.")
    w, h = img.size
    warnings = []
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if target in NO_ALPHA:
        if has_alpha:
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            img = bg
            warnings.append(f"{target.upper()} has no transparency; transparent areas were filled with white.")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    elif target == "ico" and (w > 256 or h > 256):
        raise OpError("ICO images can be at most 256×256 pixels; this image is larger.")
    elif img.mode in ("CMYK", "I;16", "I", "F") and target in ("webp", "gif", "png"):
        img = img.convert("RGBA" if has_alpha else "RGB")
    out = os.path.join(workdir, f"converted.{'jpg' if target == 'jpeg' else target}")
    kw = {"quality": 92} if target in ("jpeg", "webp") else {}
    if target == "webp" and src in ("png", "gif", "bmp", "tiff"):
        kw = {"lossless": True}
    img.save(out, IMAGE_FORMATS[target], **kw)
    if target == "pdf":
        from pypdf import PdfReader
        if len(PdfReader(out).pages) != 1:
            raise OpError("The produced PDF did not verify.")
        return {"output": out, "mime": IMAGE_MIME[target], "result": {"from": src, "to": target, "width": w, "height": h, "warnings": warnings}}
    with Image.open(out) as check:
        if check.size != (w, h):
            raise OpError("Converted image dimensions did not match the original.")
    return {"output": out, "mime": IMAGE_MIME[target], "result": {"from": src, "to": target, "width": w, "height": h, "warnings": warnings}}


# ---------- video format conversion (no editing) ----------
VIDEO_TARGETS = {
    "mp4": (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"], "video/mp4"),
    "webm": (["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "33", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-c:a", "libopus", "-b:a", "96k"], "video/webm"),
    "mov": (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"], "video/quicktime"),
    "mkv": (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k"], "video/x-matroska"),
    "m4v": (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-f", "mp4"], "video/x-m4v"),
    "avi": (["-c:v", "mpeg4", "-q:v", "4", "-c:a", "libmp3lame", "-b:a", "160k"], "video/x-msvideo"),
    "wmv": (["-c:v", "wmv2", "-q:v", "4", "-c:a", "wmav2", "-b:a", "160k"], "video/x-ms-wmv"),
    "mpeg": (["-c:v", "mpeg2video", "-q:v", "4", "-c:a", "mp2", "-b:a", "192k", "-f", "mpeg"], "video/mpeg"),
    "3gp": (["-c:v", "libx264", "-profile:v", "baseline", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "22050", "-f", "3gp"], "video/3gpp"),
}
MAX_VIDEO_SECONDS = 15 * 60


def probe(path: str) -> dict:
    import json
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise OpError("The file is not a valid, supported video.")
    return json.loads(r.stdout)


def op_video_convert(inp, params, workdir):
    target = str(params.get("target", "")).lower()
    if target not in VIDEO_TARGETS:
        raise OpError(f"Unsupported output format. Supported: {', '.join(VIDEO_TARGETS)}.")
    info = probe(inp)
    streams = info.get("streams", [])
    v = [s for s in streams if s.get("codec_type") == "video"]
    a = [s for s in streams if s.get("codec_type") == "audio"]
    if not v:
        raise OpError("No video stream found in the file.")
    dur = float(info.get("format", {}).get("duration") or 0)
    if dur > MAX_VIDEO_SECONDS:
        raise OpError("Videos longer than 15 minutes are not supported.")
    args, mime = VIDEO_TARGETS[target]
    out = os.path.join(workdir, f"converted.{target}")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", inp, "-map", "0:v:0"]
    if a:
        cmd += ["-map", "0:a:0"]
    cmd += ["-sn", "-dn", "-map_metadata", "-1", *args, out]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=params.get("_timeout", 1500))
    except subprocess.TimeoutExpired:
        raise OpError("Conversion took too long and was stopped.")
    if r.returncode != 0:
        msg = (r.stderr or "").lower()
        if "unknown encoder" in msg or "encoder not found" in msg or "decoder" in msg:
            raise OpError("A required codec is not available on the runner for this conversion.")
        raise OpError("The conversion failed (the input may be damaged or use an unsupported codec).")
    oi = probe(out)
    ov = [s for s in oi.get("streams", []) if s.get("codec_type") == "video"]
    oa = [s for s in oi.get("streams", []) if s.get("codec_type") == "audio"]
    if not ov or (a and not oa):
        raise OpError("The converted file is missing expected streams.")
    return {"output": out, "mime": mime, "result": {
        "to": target, "duration": round(float(oi["format"].get("duration") or 0), 2),
        "width": ov[0].get("width"), "height": ov[0].get("height"),
        "video_codec": ov[0].get("codec_name"), "audio_codec": oa[0].get("codec_name") if oa else None,
    }}


from runtime import format_ops as F  # noqa: E402

def _story(inp, params, workdir):
    from runtime.story import run_story
    return run_story(params, workdir, params.get("_progress"))


OPERATIONS: dict[str, Operation] = {
    "audio_convert": Operation("audio_convert", True, 25 * MB, 150 * MB, 1200, F.op_audio_convert),
    "office_convert": Operation("office_convert", True, 25 * MB, 60 * MB, 600, F.op_office_convert),
    "data_convert": Operation("data_convert", True, 25 * MB, 100 * MB, 600, F.op_data_convert),
    "archive_convert": Operation("archive_convert", True, 25 * MB, 100 * MB, 600, F.op_archive_convert),
    "ebook_convert": Operation("ebook_convert", True, 25 * MB, 60 * MB, 900, F.op_ebook_convert),
    "font_convert": Operation("font_convert", True, 25 * MB, 30 * MB, 300, F.op_font_convert),
    "render": Operation("render", True, 5 * MB, 30 * MB, 600, F.op_render),
    "checksum": Operation("checksum", True, 25 * MB, 0, 300, op_checksum),
    "story_generate": Operation("story_generate", False, 0, 5 * MB, 1500, _story),
    "image_convert": Operation("image_convert", True, 25 * MB, 60 * MB, 300, op_image_convert),
    "video_convert": Operation("video_convert", True, 25 * MB, 100 * MB, 1500, op_video_convert),
}


def get(op_id: str) -> Operation:
    if op_id not in OPERATIONS:
        raise OpError("Operation is not permitted.")
    return OPERATIONS[op_id]
