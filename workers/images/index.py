"""VISION runtime worker: images. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("images")
run = WORKER.run
