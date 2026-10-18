"""Runs one VISION job on a GitHub-hosted runner.

Flow: GitHub OIDC token -> claim job at the relay -> download + verify input ->
run the registered operation -> upload output -> report completion.
Logs only contain the job id, operation and stage — never file names, contents or tokens.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.operations import OpError, get  # noqa: E402

RELAY = "https://vision.tusharkantidasofficial.workers.dev/runtime"
CHUNK = 700_000
ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def log(msg: str) -> None:
    print(f"[runtime] {msg}", flush=True)


def http(method: str, url: str, token: str, body: bytes | None = None, ctype: str = "application/json", tries: int = 3) -> bytes:
    for attempt in range(tries):
        req = urllib.request.Request(url, data=body, method=method, headers={"Authorization": f"Bearer {token}", "Content-Type": ctype, "User-Agent": "vision-runtime"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"relay {e.code}")
        except urllib.error.URLError:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("relay unreachable")


def oidc_token() -> str:
    url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"] + "&audience=vision-runtime-relay"
    req = urllib.request.Request(url, headers={"Authorization": f"bearer {os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["value"]


def main() -> int:
    job_id = os.environ.get("JOB_ID", "")
    op_id = os.environ.get("OPERATION", "")
    if not ID_RE.match(job_id):
        log("invalid job id"); return 2
    op = get(op_id)  # rejects anything not registered
    claim = json.loads(http("POST", f"{RELAY}/claim", oidc_token(), json.dumps({"job_id": job_id}).encode()))
    sess = claim["session"]
    print(f"::add-mask::{sess}", flush=True)
    if claim["operation"] != op_id:
        log("operation mismatch"); return 2
    log(f"claimed job {job_id} ({op_id})")
    work = tempfile.mkdtemp(prefix="vision-")
    try:
        inp = None
        if op.needs_input:
            meta = claim.get("input")
            if not meta or meta["size"] > op.max_input:
                raise OpError("Input file is missing or larger than allowed.")
            http("POST", f"{RELAY}/progress", sess, json.dumps({"stage": "validating_input"}).encode())
            inp = os.path.join(work, "input.bin")  # fixed name: user file names never touch the filesystem
            h = hashlib.sha256(); size = 0
            with open(inp, "wb") as f:
                for i in range(int(meta["chunks"])):
                    data = base64.b64decode(http("GET", f"{RELAY}/input/{i}", sess, ctype="text/plain"), validate=True)
                    size += len(data); h.update(data); f.write(data)
                    if size > op.max_input:
                        raise OpError("Input file is larger than allowed.")
            if size != meta["size"] or h.hexdigest() != meta["sha256"]:
                raise OpError("Input file failed its integrity check.")
        http("POST", f"{RELAY}/progress", sess, json.dumps({"stage": "processing"}).encode())
        params = dict(claim.get("params") or {}); params["_timeout"] = op.timeout_s
        params["_progress"] = lambda stage: http("POST", f"{RELAY}/progress", sess, json.dumps({"stage": stage}).encode())
        res = op.run(inp, params, work)
        output = None
        if res.get("output"):
            http("POST", f"{RELAY}/progress", sess, json.dumps({"stage": "validating_output"}).encode())
            path = res["output"]
            if os.path.dirname(os.path.realpath(path)) != os.path.realpath(work):
                raise OpError("Output path rejected.")
            size = os.path.getsize(path)
            if size == 0 or size > op.max_output:
                raise OpError("The converted file is empty or larger than allowed.")
            http("POST", f"{RELAY}/progress", sess, json.dumps({"stage": "uploading_output"}).encode())
            h = hashlib.sha256(); n = 0
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(CHUNK), b""):
                    h.update(block)
                    http("POST", f"{RELAY}/output/{n}", sess, base64.b64encode(block), ctype="text/plain")
                    n += 1
            output = {"name": os.path.basename(path), "mime": res["mime"], "size": size, "sha256": h.hexdigest(), "chunks": n}
        http("POST", f"{RELAY}/complete", sess, json.dumps({"ok": True, "result": res.get("result"), "output": output}).encode())
        log("completed")
        return 0
    except OpError as e:
        http("POST", f"{RELAY}/complete", sess, json.dumps({"ok": False, "error": str(e)}).encode())
        log("failed (reported)")
        return 1
    except Exception as e:  # never leak internals: report a generic error
        log(f"internal error: {type(e).__name__}")
        try:
            http("POST", f"{RELAY}/complete", sess, json.dumps({"ok": False, "error": "The runner hit an internal error."}).encode())
        except Exception:
            pass
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
