"""VISION runtime worker: maps. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("maps")
run = WORKER.run
