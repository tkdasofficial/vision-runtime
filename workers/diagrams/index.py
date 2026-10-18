"""VISION runtime worker: diagrams. Scope (operations + formats) comes from registry/workflows.json."""
from runtime.worker_base import Worker

WORKER = Worker("diagrams")
run = WORKER.run
