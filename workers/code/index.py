"""VISION runtime worker: code. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("code")
run = WORKER.run
