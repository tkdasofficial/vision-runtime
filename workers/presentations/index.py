"""VISION runtime worker: presentations. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("presentations")
run = WORKER.run
