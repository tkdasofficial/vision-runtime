"""VISION runtime worker: logs. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("logs")
run = WORKER.run
