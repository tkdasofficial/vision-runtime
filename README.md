# Vision-Runtime

Background execution runtime for VISION. The backend dispatches one of **28 workflows**
(`.github/workflows/<name>.yml`, generated from `registry/workflows.json`) with only a job id
and operation. Each workflow installs only its worker's dependencies and runs
`runtime/runner.py`, which:

1. Gets a GitHub OIDC token; the VISION relay verifies repo, workflow file, `main` and
   `workflow_dispatch`, and only lets that workflow claim jobs routed to it.
2. Downloads the input in chunks and checks size + SHA-256.
3. Calls the routed worker (`workers/<worker>/index.py`, 18 total), which checks the operation
   and format are in scope, runs the operation from `runtime/operations.py`, and verifies the
   output signature.
4. Uploads the output in chunks; VISION re-verifies the hash.

`registry/formats.json` maps all 121 canonical formats (18 categories) to workflow, worker,
dependencies, capabilities and verification status (`runtime-implemented`, `inline-only`
= handled in the Cloudflare Worker, or `unsupported`).

No workflow runs on push or schedule. Tests: `python3 -m pytest -q tests` (run locally).

## Operations

| id | input | limits | output |
|---|---|---|---|
| `checksum` | any file ≤ 25 MB | 5 min | SHA-256 / SHA-512, optional verify |
| `image_convert` | image ≤ 25 MB, ≤ 80 MP | 5 min | png, jpeg, webp, gif, bmp, tiff, ico (format conversion only) |
| `video_convert` | video ≤ 25 MB, ≤ 15 min | 25 min | mp4 (H.264/AAC), webm (VP9/Opus), mov, mkv |

Image and video modules convert formats only — no editing, enhancement or generation.

## Usage limits

Public repositories get free GitHub-hosted minutes; private repos draw from the account's
monthly quota. VISION caps concurrent jobs per user and overall, and never auto-retries.

## Tests

`python3 -m pytest -q tests` (run locally; no workflow runs automatically).
