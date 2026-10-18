"""VISION runtime worker: config. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("config")
run = WORKER.run
