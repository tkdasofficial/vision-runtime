"""VISION runtime worker: fonts. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("fonts")
run = WORKER.run
