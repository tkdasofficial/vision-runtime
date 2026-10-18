# Vision-Runtime

Background execution runtime for VISION. The VISION backend dispatches `.github/workflows/terminal.yml` (file operations) or `.github/workflows/story.yml` (stories, code in `workers/story/index.py`)
with only a job id and an operation name; the runner then:

1. Gets a GitHub OIDC token (audience `vision-runtime-relay`).
2. Claims the job at the VISION relay (Cloudflare Worker), which verifies the token is from
   **this repo, this workflow file, the `main` branch, and a `workflow_dispatch` run**, and
   binds the job to that run id (one claim per job).
3. Downloads the input in chunks and checks size + SHA-256.
4. Runs the operation from `runtime/operations.py` (the only allowed operations).
5. Uploads the output in chunks and reports completion; VISION re-verifies the hash.

No secrets are stored in this repo or passed to workflows. User files never go into
workflow inputs, logs or GitHub artifacts (this repo is public, so logs only show job id,
operation and stage).

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
