"""VISION runtime worker: archives. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("archives")
run = WORKER.run
