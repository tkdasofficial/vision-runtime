"""VISION runtime worker: data. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("data")
run = WORKER.run
