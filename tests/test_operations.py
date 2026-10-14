import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.operations import OPERATIONS, OpError, get  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def write(tmp, name, data):
    p = tmp / name
    p.write_bytes(data)
    return str(p)


# ---- checksums (known vectors) ----
def test_sha256_sha512_vectors(tmp_path):
    p = write(tmp_path, "a", b"abc")
    r = get("checksum").run(p, {"algorithms": ["sha256", "sha512"]}, str(tmp_path))["result"]
    assert r["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert r["sha512"].startswith("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a")


def test_checksum_verify(tmp_path):
    p = write(tmp_path, "a", b"abc")
    ok = get("checksum").run(p, {"expected": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}, str(tmp_path))["result"]
    assert ok["verify"]["match"] is True
    bad = get("checksum").run(p, {"expected": "0" * 64}, str(tmp_path))["result"]
    assert bad["verify"]["match"] is False
    with pytest.raises(OpError):
        get("checksum").run(p, {"expected": "xyz"}, str(tmp_path))


# ---- registry / security ----
def test_unknown_operation_rejected():
    for bad in ["bash", "../checksum", "checksum; rm -rf /", ""]:
        with pytest.raises(OpError):
            get(bad)
    assert set(OPERATIONS) == {"checksum", "image_convert", "video_convert", "audio_convert", "office_convert", "data_convert", "archive_convert", "ebook_convert", "font_convert", "render"}


# ---- images ----
def make_png(tmp, alpha=True):
    from PIL import Image
    img = Image.new("RGBA" if alpha else "RGB", (40, 30), (255, 0, 0, 0) if alpha else (0, 255, 0))
    if alpha:
        img.putpixel((5, 5), (0, 0, 255, 255))
    p = str(tmp / "in.png")
    img.save(p)
    return p


@pytest.mark.parametrize("target,mime", [("jpeg", "image/jpeg"), ("webp", "image/webp"), ("gif", "image/gif"), ("bmp", "image/bmp"), ("tiff", "image/tiff")])
def test_png_to_formats(tmp_path, target, mime):
    from PIL import Image
    r = get("image_convert").run(make_png(tmp_path), {"target": target}, str(tmp_path))
    assert r["mime"] == mime
    with Image.open(r["output"]) as im:
        assert im.size == (40, 30)
        assert im.format.lower() == target


def test_transparency(tmp_path):
    from PIL import Image
    r = get("image_convert").run(make_png(tmp_path), {"target": "webp"}, str(tmp_path))
    with Image.open(r["output"]) as im:
        assert im.mode == "RGBA" and im.getpixel((0, 0))[3] == 0
    j = get("image_convert").run(make_png(tmp_path), {"target": "jpg"}, str(tmp_path))
    assert j["result"]["warnings"]


def test_jpeg_to_png_roundtrip(tmp_path):
    j = get("image_convert").run(make_png(tmp_path, alpha=False), {"target": "jpeg"}, str(tmp_path))
    r = get("image_convert").run(j["output"], {"target": "png"}, str(tmp_path))
    assert r["result"]["width"] == 40


def test_invalid_image_and_format(tmp_path):
    with pytest.raises(OpError):
        get("image_convert").run(write(tmp_path, "x.png", b"not an image"), {"target": "png"}, str(tmp_path))
    with pytest.raises(OpError):
        get("image_convert").run(make_png(tmp_path), {"target": "exe"}, str(tmp_path))
    with pytest.raises(OpError):
        get("image_convert").run(make_png(tmp_path), {"target": "png"}, str(tmp_path))


# ---- video ----
def make_video(tmp):
    p = str(tmp / "in.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1", "-f", "lavfi", "-i", "sine=duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", p], check=True)
    return p


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize("target,vcodec", [("webm", "vp9"), ("mov", "h264"), ("mkv", "h264")])
def test_video_convert(tmp_path, target, vcodec):
    r = get("video_convert").run(make_video(tmp_path), {"target": target}, str(tmp_path))
    assert r["result"]["video_codec"] == vcodec and r["result"]["audio_codec"]
    assert r["result"]["width"] == 160


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_video_invalid(tmp_path):
    with pytest.raises(OpError):
        get("video_convert").run(write(tmp_path, "v.mp4", b"garbage" * 100), {"target": "webm"}, str(tmp_path))
    with pytest.raises(OpError):
        get("video_convert").run(write(tmp_path, "v.mp4", b"x"), {"target": "avi"}, str(tmp_path))
