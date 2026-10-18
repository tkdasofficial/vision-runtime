"""VISION runtime worker: contacts. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("contacts")
run = WORKER.run
