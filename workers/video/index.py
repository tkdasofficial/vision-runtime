"""VISION runtime worker: video. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("video")
run = WORKER.run
