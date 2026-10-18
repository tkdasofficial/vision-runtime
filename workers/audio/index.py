"""VISION runtime worker: audio. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("audio")
run = WORKER.run
