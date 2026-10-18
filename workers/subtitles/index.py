"""VISION runtime worker: subtitles. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("subtitles")
run = WORKER.run
